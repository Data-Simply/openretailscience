"""Ask a vision LLM to detect the injected visual regressions.

Two backends are supported:

* ``claude``      — shells out to ``claude -p`` (Claude Code headless) with ``--model haiku`` and lets it
                    read the image with the Read tool.
* ``openrouter``  — POSTs the image to OpenRouter's OpenAI-compatible chat endpoint, so any vision model
                    on OpenRouter (e.g. ``meta-llama/llama-3.2-11b-vision-instruct``) can be evaluated.
                    Needs ``OPENROUTER_API_KEY`` in the environment. Requests set
                    ``provider.data_collection = "deny"`` and ``provider.zdr = true`` so only
                    zero-data-retention providers see the prompt or chart image.

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
# Only route to zero-data-retention providers that will not store/train on the prompt or chart image.
OPENROUTER_PROVIDER_PREFS = {"data_collection": "deny", "zdr": True}


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


def build_freetext_prompt() -> str:
    """Build the open-ended prompt asking the model to describe any problems in natural language.

    Returns:
        A prompt instructing the model to either say ``NO ISSUES`` or briefly describe what is wrong.
    """
    return (
        "You are a meticulous data-visualisation QA reviewer. You are shown a single retail chart image.\n"
        "Describe any visual formatting or layout problems you can see — for example text cut off at an "
        "edge, overlapping or unreadable labels, a legend covering the data, a title in the wrong place, "
        "wrong size or low contrast, a large gap or overlap between an axis and its labels, or a distorted "
        "aspect ratio. Judge only the rendering, not whether the underlying numbers are right.\n"
        "If the chart looks clean and correctly formatted, reply with exactly: NO ISSUES\n"
        "Otherwise reply with a brief description (1-3 sentences) of what is wrong."
    )


def _claude_invoke(prompt: str, model: str, *, allow_read: bool) -> str:
    """Run ``claude -p`` and return the assistant's text from the stream-json result event.

    Uses ``--output-format stream-json``, which works across Claude Code environments (some inject
    ``--include-partial-messages``, which only ``stream-json`` accepts). ``allow_read`` whitelists the
    Read tool so image prompts can open the file; text-only judge prompts leave it off.
    """
    command = ["claude", "-p", prompt, "--model", model, "--output-format", "stream-json", "--verbose"]
    if allow_read:
        command += ["--allowedTools", "Read"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=REQUEST_TIMEOUT_S, check=False)
    if result.returncode != 0:
        msg = f"claude exited {result.returncode}: {result.stderr.strip()[:300]}"
        raise RuntimeError(msg)
    return _extract_stream_json_result(result.stdout)


def _run_claude(image_path: str, prompt: str, model: str) -> str:
    """Detect via ``claude -p`` (Claude Code headless), letting it read the image file."""
    full_prompt = f"{prompt}\n\nRead the chart image at this path and analyse it: {image_path}"
    return _claude_invoke(full_prompt, model, allow_read=True)


def _run_claude_text(prompt: str, model: str) -> str:
    """Run a text-only ``claude -p`` prompt (used by the free-text scoring judge)."""
    return _claude_invoke(prompt, model, allow_read=False)


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


def _openrouter_body(content: list[dict], model: str) -> dict:
    """Build the OpenRouter chat-completions request body.

    Sets ``provider.data_collection = "deny"`` and ``provider.zdr = true`` so OpenRouter only routes to
    zero-data-retention providers that will not store or train on the prompt and chart image.
    """
    return {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
        "provider": OPENROUTER_PROVIDER_PREFS,
    }


def _openrouter_request(content: list[dict], model: str, api_key: str) -> str:
    """POST one user message to OpenRouter's chat-completions API and return the reply text."""
    body = json.dumps(_openrouter_body(content, model)).encode("utf-8")
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        payload = json.loads(response.read())
    return str(payload["choices"][0]["message"]["content"])


def _run_openrouter(image_path: str, prompt: str, model: str, api_key: str) -> str:
    """Detect via OpenRouter's chat-completions API with an image; returns the raw text response."""
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
    ]
    return _openrouter_request(content, model, api_key)


def _run_openrouter_text(prompt: str, model: str, api_key: str) -> str:
    """Run a text-only OpenRouter prompt (used by the free-text scoring judge)."""
    return _openrouter_request([{"type": "text", "text": prompt}], model, api_key)


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


def _categories_record(file: str, raw: str, error: str | None, valid_names: set[str]) -> dict:
    """Build a closed-set detection record from a raw response."""
    detected = _parse_detections(raw, valid_names)
    return {"file": file, "detected": detected, "has_defect": len(detected) > 0, "raw": raw, "error": error}


def _freetext_record(file: str, raw: str, error: str | None) -> dict:
    """Build an open-ended detection record carrying the model's free-text description."""
    return {"file": file, "description": raw.strip(), "error": error}


def detect_all(
    plots: list[dict],
    base_dir: Path | str,
    backend: Callable[[str, str], str],
    taxonomy: list[dict[str, str]],
    mode: str = "categories",
) -> list[dict]:
    """Run every plot through ``backend`` and return per-plot detection records.

    Args:
        plots: The manifest's ``plots`` list.
        base_dir: Directory the plot ``file`` paths are relative to (the manifest's directory).
        backend: Callable ``(image_path, prompt) -> raw_text`` performing one detection.
        taxonomy: Defect catalogue, used to build the prompt and validate reported categories.
        mode: ``"categories"`` for closed-set category detection, ``"freetext"`` for open-ended
            natural-language descriptions.

    Returns:
        One record per plot. In ``categories`` mode: ``{file, detected, has_defect, raw, error}``.
        In ``freetext`` mode: ``{file, description, error}``.
    """
    base_dir = Path(base_dir)
    prompt = build_prompt(taxonomy) if mode == "categories" else build_freetext_prompt()
    valid_names = {d["name"] for d in taxonomy}

    results: list[dict] = []
    for plot in plots:
        image_path = base_dir / plot["file"]
        raw, error = "", None
        try:
            raw = backend(str(image_path), prompt)
        except Exception as exc:
            error = str(exc)
        if mode == "categories":
            results.append(_categories_record(plot["file"], raw, error, valid_names))
        else:
            results.append(_freetext_record(plot["file"], raw, error))
    return results


def _resolve_openrouter_key() -> str:
    """Return the OpenRouter API key from the environment or exit with a clear message."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key is None or len(api_key) == 0:
        raise SystemExit("OPENROUTER_API_KEY is not set; export it before using the openrouter backend.")
    return api_key


def make_image_backend(backend: str, model: str | None = None) -> Callable[[str, str], str]:
    """Build an image detection callable ``(image_path, prompt) -> raw_text`` for ``backend``."""
    if backend == "claude":
        claude_model = model or DEFAULT_CLAUDE_MODEL
        return lambda image_path, prompt: _run_claude(image_path, prompt, claude_model)
    router_model = model or DEFAULT_OPENROUTER_MODEL
    api_key = _resolve_openrouter_key()
    return lambda image_path, prompt: _run_openrouter(image_path, prompt, router_model, api_key)


def make_text_backend(backend: str, model: str | None = None) -> Callable[[str], str]:
    """Build a text-only callable ``(prompt) -> raw_text`` for ``backend`` (used by the judge)."""
    if backend == "claude":
        claude_model = model or DEFAULT_CLAUDE_MODEL
        return lambda prompt: _run_claude_text(prompt, claude_model)
    router_model = model or DEFAULT_OPENROUTER_MODEL
    api_key = _resolve_openrouter_key()
    return lambda prompt: _run_openrouter_text(prompt, router_model, api_key)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run a vision LLM over a visual-regression dataset.")
    parser.add_argument("--manifest", required=True, help="Path to the dataset manifest.json.")
    parser.add_argument("--backend", choices=("claude", "openrouter"), default="claude", help="Detection backend.")
    parser.add_argument("--model", default=None, help="Model id/alias (backend default if omitted).")
    parser.add_argument(
        "--mode",
        choices=("categories", "freetext"),
        default="categories",
        help="categories: pick from the closed defect list. freetext: describe problems in prose.",
    )
    parser.add_argument("--out", default=None, help="Output detections JSON (defaults next to the manifest).")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N plots (cost control).")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plots = manifest["plots"][: args.limit] if args.limit is not None else manifest["plots"]
    model = args.model or (DEFAULT_CLAUDE_MODEL if args.backend == "claude" else DEFAULT_OPENROUTER_MODEL)

    backend = make_image_backend(args.backend, args.model)
    print(f"Detecting with backend={args.backend} model={model} mode={args.mode} over {len(plots)} plots...")
    results = detect_all(plots, manifest_path.parent, backend, manifest["taxonomy"], mode=args.mode)

    out_path = Path(args.out) if args.out is not None else manifest_path.parent / "detections.json"
    out_path.write_text(
        json.dumps({"backend": args.backend, "model": model, "mode": args.mode, "detections": results}, indent=2),
        encoding="utf-8",
    )
    errors = sum(1 for r in results if r["error"] is not None)
    print(f"Wrote {len(results)} detections ({errors} errors) to {out_path}")


if __name__ == "__main__":
    main()
