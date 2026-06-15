"""Ask a vision LLM to detect the injected visual regressions.

Two backends are supported:

* ``claude``      — shells out to ``claude -p`` (Claude Code headless) with ``--model haiku`` and lets it
                    read the image with the Read tool.
* ``openrouter``  — POSTs the image to OpenRouter's OpenAI-compatible chat endpoint, so any vision model
                    on OpenRouter (e.g. ``meta-llama/llama-3.2-11b-vision-instruct``) can be evaluated.
                    Needs ``OPENROUTER_API_KEY`` in the environment.

Reads the dataset ``manifest.json``, runs each plot through the chosen backend, and writes
``detections.json``. Run ``python -m visual_regression.detect --help`` for options.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_CLAUDE_MODEL = "haiku"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.2-11b-vision-instruct"
REQUEST_TIMEOUT_S = 180


def build_prompt(taxonomy: list[dict[str, str]]) -> str:
    """Build the detection prompt listing the closed set of defect categories.

    Args:
        taxonomy: Defect catalogue (name/description dicts) from the dataset manifest.

    Returns:
        A prompt instructing the model to return strict JSON of detected category names.
    """
    catalogue = "\n".join(f"- {d['name']}: {d['description']}" for d in taxonomy)
    return (
        "You are a meticulous data-visualisation QA reviewer. You are shown a single retail chart image.\n"
        "Decide which, if any, of the following visual problems are present in the image. Only report a\n"
        "category if you can actually see the problem in the chart; a clean, well-formatted chart should\n"
        "return an empty list.\n\n"
        f"Possible problems:\n{catalogue}\n\n"
        'Respond with ONLY a JSON object of the form {"defects": ["category_name", ...]} using category\n'
        "names from the list above. No prose, no markdown, no explanation."
    )


def _run_claude(image_path: str, prompt: str, model: str) -> str:
    """Detect via ``claude -p`` (Claude Code headless); returns the model's raw text response.

    Uses ``--output-format stream-json``, which emits newline-delimited JSON events and works across
    Claude Code environments (some inject ``--include-partial-messages``, which only ``stream-json``
    accepts). The final ``{"type": "result", ...}`` event carries the assistant's text.
    """
    full_prompt = f"{prompt}\n\nRead the chart image at this path and analyse it: {image_path}"
    result = subprocess.run(
        [
            "claude",
            "-p",
            full_prompt,
            "--model",
            model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--allowedTools",
            "Read",
        ],
        capture_output=True,
        text=True,
        timeout=REQUEST_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0:
        msg = f"claude exited {result.returncode}: {result.stderr.strip()[:300]}"
        raise RuntimeError(msg)
    return _extract_stream_json_result(result.stdout)


def _extract_stream_json_result(stdout: str) -> str:
    """Pull the assistant text from the final ``type=result`` event of a stream-json run."""
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if len(stripped) == 0:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            return str(event.get("result", ""))
    msg = "no result event found in claude stream-json output"
    raise RuntimeError(msg)


def _run_openrouter(image_path: str, prompt: str, model: str, api_key: str) -> str:
    """Detect via OpenRouter's chat-completions API; returns the model's raw text response."""
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                },
            ],
        },
    ).encode("utf-8")
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        payload = json.loads(response.read())
    return str(payload["choices"][0]["message"]["content"])


def _extract_json(raw: str) -> dict | None:
    """Best-effort parse of a JSON object from a model response that may wrap it in prose/markdown."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json", "", text.strip(), flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


def _parse_detections(raw: str, valid_names: set[str]) -> list[str]:
    """Extract the detected category names from a raw response, keeping only known categories."""
    obj = _extract_json(raw)
    if obj is None:
        return []
    items = obj.get("defects", [])
    if not isinstance(items, list):
        return []
    names: set[str] = set()
    for item in items:
        name = item.get("name") if isinstance(item, dict) else item
        normalised = str(name).strip().lower()
        if normalised in valid_names:
            names.add(normalised)
    return sorted(names)


def detect_all(
    plots: list[dict], base_dir: Path | str, backend: Callable[[str, str], str], taxonomy: list[dict[str, str]]
) -> list[dict]:
    """Run every plot through ``backend`` and return per-plot detection records.

    Args:
        plots: The manifest's ``plots`` list.
        base_dir: Directory the plot ``file`` paths are relative to (the manifest's directory).
        backend: Callable ``(image_path, prompt) -> raw_text`` performing one detection.
        taxonomy: Defect catalogue, used to build the prompt and validate reported categories.

    Returns:
        One record per plot: ``{file, detected, has_defect, raw, error}``.
    """
    base_dir = Path(base_dir)
    prompt = build_prompt(taxonomy)
    valid_names = {d["name"] for d in taxonomy}

    results: list[dict] = []
    for plot in plots:
        image_path = base_dir / plot["file"]
        raw, error = "", None
        try:
            raw = backend(str(image_path), prompt)
            detected = _parse_detections(raw, valid_names)
        except Exception as exc:
            detected, error = [], str(exc)
        results.append(
            {"file": plot["file"], "detected": detected, "has_defect": len(detected) > 0, "raw": raw, "error": error},
        )
    return results


def _make_backend(args: argparse.Namespace) -> Callable[[str, str], str]:
    """Build the detection callable for the selected backend, binding model/credentials."""
    if args.backend == "claude":
        return lambda image_path, prompt: _run_claude(image_path, prompt, args.model or DEFAULT_CLAUDE_MODEL)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key is None or len(api_key) == 0:
        raise SystemExit("OPENROUTER_API_KEY is not set; export it before using --backend openrouter.")
    model = args.model or DEFAULT_OPENROUTER_MODEL
    return lambda image_path, prompt: _run_openrouter(image_path, prompt, model, api_key)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run a vision LLM over a visual-regression dataset.")
    parser.add_argument("--manifest", required=True, help="Path to the dataset manifest.json.")
    parser.add_argument("--backend", choices=("claude", "openrouter"), default="claude", help="Detection backend.")
    parser.add_argument("--model", default=None, help="Model id/alias (backend default if omitted).")
    parser.add_argument("--out", default=None, help="Output detections JSON (defaults next to the manifest).")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N plots (cost control).")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plots = manifest["plots"][: args.limit] if args.limit is not None else manifest["plots"]
    model = args.model or (DEFAULT_CLAUDE_MODEL if args.backend == "claude" else DEFAULT_OPENROUTER_MODEL)

    backend = _make_backend(args)
    print(f"Detecting with backend={args.backend} model={model} over {len(plots)} plots...")
    results = detect_all(plots, manifest_path.parent, backend, manifest["taxonomy"])

    out_path = Path(args.out) if args.out is not None else manifest_path.parent / "detections.json"
    out_path.write_text(
        json.dumps({"backend": args.backend, "model": model, "detections": results}, indent=2),
        encoding="utf-8",
    )
    errors = sum(1 for r in results if r["error"] is not None)
    print(f"Wrote {len(results)} detections ({errors} errors) to {out_path}")


if __name__ == "__main__":
    main()
