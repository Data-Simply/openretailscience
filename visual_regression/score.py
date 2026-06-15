"""Score detector output against the ground-truth manifest.

Computes two views:

* **binary** — did the detector flag *any* defect on a plot that actually had one? (precision / recall /
  F1 / accuracy over the has-defect-or-not question, including the clean plots).
* **per_category** — multi-label precision / recall / F1 for each defect category, plus micro/macro
  averages, so you can see which regressions a model is good or bad at spotting.

Run ``python -m visual_regression.score --help`` for options.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from visual_regression.detect import _extract_json, make_text_backend

if TYPE_CHECKING:
    from collections.abc import Callable


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    """Return precision, recall, F1 and support for one confusion triple."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}


def evaluate(plots: list[dict], detections: list[dict], taxonomy: list[dict[str, str]]) -> dict:
    """Compare ``detections`` against the manifest ``plots`` ground truth.

    Args:
        plots: The manifest's ``plots`` list (ground truth).
        detections: Detector output records (each with ``file`` and ``detected``).
        taxonomy: Defect catalogue, defining which categories to score.

    Returns:
        A report dict with ``binary``, ``per_category``, ``micro``, ``macro`` and ``num_plots`` keys.

    Raises:
        ValueError: If no plot files are shared between the manifest and the detections.
    """
    truth = {p["file"]: {d["name"] for d in p["defects"]} for p in plots}
    pred = {r["file"]: set(r["detected"]) for r in detections}
    files = [f for f in truth if f in pred]
    if len(files) == 0:
        raise ValueError("No overlapping plot files between manifest and detections.")

    binary_tp = sum(1 for f in files if len(truth[f]) > 0 and len(pred[f]) > 0)
    binary_fp = sum(1 for f in files if len(truth[f]) == 0 and len(pred[f]) > 0)
    binary_fn = sum(1 for f in files if len(truth[f]) > 0 and len(pred[f]) == 0)
    binary_tn = sum(1 for f in files if len(truth[f]) == 0 and len(pred[f]) == 0)
    binary = _prf(binary_tp, binary_fp, binary_fn)
    binary["accuracy"] = (binary_tp + binary_tn) / len(files)
    binary["tn"] = binary_tn

    per_category: dict[str, dict[str, float | int]] = {}
    for entry in taxonomy:
        category = entry["name"]
        tp = sum(1 for f in files if category in truth[f] and category in pred[f])
        fp = sum(1 for f in files if category not in truth[f] and category in pred[f])
        fn = sum(1 for f in files if category in truth[f] and category not in pred[f])
        per_category[category] = _prf(tp, fp, fn)

    micro = _prf(
        sum(int(c["tp"]) for c in per_category.values()),
        sum(int(c["fp"]) for c in per_category.values()),
        sum(int(c["fn"]) for c in per_category.values()),
    )
    scored = [c for c in per_category.values() if int(c["support"]) > 0]
    macro = {
        "precision": sum(c["precision"] for c in scored) / len(scored) if len(scored) > 0 else 0.0,
        "recall": sum(c["recall"] for c in scored) / len(scored) if len(scored) > 0 else 0.0,
        "f1": sum(c["f1"] for c in scored) / len(scored) if len(scored) > 0 else 0.0,
    }

    return {"num_plots": len(files), "binary": binary, "per_category": per_category, "micro": micro, "macro": macro}


def format_report(report: dict, backend: str | None = None, model: str | None = None) -> str:
    """Render ``report`` as a human-readable text table."""
    lines: list[str] = []
    header = "Visual-regression detection report"
    if backend is not None and model is not None:
        header += f"  (backend={backend}, model={model})"
    lines.append(header)
    lines.append(f"Plots scored: {report['num_plots']}")

    binary = report["binary"]
    lines.append("")
    lines.append("Binary (any defect detected vs present):")
    lines.append(
        f"  accuracy={binary['accuracy']:.3f}  precision={binary['precision']:.3f}  "
        f"recall={binary['recall']:.3f}  f1={binary['f1']:.3f}",
    )
    lines.append(f"  tp={binary['tp']} fp={binary['fp']} fn={binary['fn']} tn={binary['tn']}")

    lines.append("")
    lines.append(f"{'category':<26}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}")
    for category, stats in sorted(report["per_category"].items(), key=lambda kv: kv[1]["f1"], reverse=True):
        lines.append(
            f"{category:<26}{stats['precision']:>7.2f}{stats['recall']:>8.2f}{stats['f1']:>7.2f}{stats['support']:>9}",
        )

    micro, macro = report["micro"], report["macro"]
    lines.append("")
    lines.append(f"micro  precision={micro['precision']:.3f} recall={micro['recall']:.3f} f1={micro['f1']:.3f}")
    lines.append(f"macro  precision={macro['precision']:.3f} recall={macro['recall']:.3f} f1={macro['f1']:.3f}")
    return "\n".join(lines)


def build_judge_prompt(is_clean: bool, truth_descriptions: list[str], model_description: str) -> str:
    """Build the prompt asking an LLM judge to grade one free-text review against the ground truth.

    Args:
        is_clean: Whether the chart actually has no injected defect.
        truth_descriptions: The injected defect description(s); empty when the chart is clean.
        model_description: The detector's free-text review of the chart.

    Returns:
        A prompt instructing the judge to answer two booleans as JSON.
    """
    if is_clean:
        actual = "The chart is actually clean and correctly formatted; no defect was injected."
    else:
        actual = f"The chart actually has this known formatting defect: {'; '.join(truth_descriptions)}"
    return (
        "You are grading a vision model's visual-QA review of a chart.\n"
        f"{actual}\n\n"
        f'The model\'s review was:\n"""\n{model_description}\n"""\n\n'
        "Answer two true/false questions about the review:\n"
        "1. identifies_real_issue: does the review correctly describe the actual known defect above? "
        "If the chart is clean, this must be false.\n"
        "2. claims_problem: does the review assert the chart has at least one visual/formatting problem? "
        "Replies like 'NO ISSUES' or 'looks fine' are false.\n"
        'Respond with ONLY JSON: {"identifies_real_issue": true|false, "claims_problem": true|false}'
    )


def _parse_judge(raw: str) -> dict[str, bool]:
    """Parse the judge's JSON verdict, defaulting missing/garbled fields to False."""
    obj = _extract_json(raw) or {}
    return {
        "identifies_real_issue": bool(obj.get("identifies_real_issue", False)),
        "claims_problem": bool(obj.get("claims_problem", False)),
    }


def make_judge(text_backend: Callable[[str], str]) -> Callable[[bool, list[str], str], dict[str, bool]]:
    """Wrap a text backend into a per-plot judge ``(is_clean, truth, description) -> verdict``."""
    return lambda is_clean, truth, desc: _parse_judge(text_backend(build_judge_prompt(is_clean, truth, desc)))


def _judge_one(
    item: tuple[str, bool, list[str], str],
    judge: Callable[[bool, list[str], str], dict[str, bool]],
) -> dict:
    """Grade one ``(file, is_clean, truth, description)`` and classify it into a TP/FP/FN/TN verdict."""
    file, is_clean, truth, description = item
    verdict = judge(is_clean, truth, description)
    if is_clean:
        outcome = "fp" if verdict["claims_problem"] else "tn"
    else:
        outcome = "tp" if verdict["identifies_real_issue"] else "fn"
    return {"file": file, "is_clean": is_clean, "outcome": outcome, "description": description}


def evaluate_freetext(
    plots: list[dict],
    detections: list[dict],
    judge: Callable[[bool, list[str], str], dict[str, bool]],
    workers: int = 1,
) -> dict:
    """Score open-ended detections with an LLM ``judge`` into binary detection metrics.

    Each plot becomes one of TP/FP/FN/TN: a defective plot is a true positive when the judge says the
    review identified the known defect (else a false negative); a clean plot is a false positive when the
    judge says the review still claimed a problem (else a true negative). Errored detections are skipped.

    Args:
        plots: The manifest's ``plots`` list (ground truth).
        detections: Free-text detector records (each with ``file`` and ``description``).
        judge: Per-plot grader returning ``{"identifies_real_issue", "claims_problem"}``.
        workers: Number of judge calls to run concurrently (the judging is I/O-bound).

    Returns:
        A report dict with ``binary`` metrics, per-plot ``verdicts`` and ``num_plots``.

    Raises:
        ValueError: If no non-errored plot files are shared between the manifest and the detections.
    """
    is_clean_map = {p["file"]: len(p["defects"]) == 0 for p in plots}
    truth = {p["file"]: [d["description"] for d in p["defects"]] for p in plots}
    described = {r["file"]: r.get("description", "") for r in detections if r.get("error") is None}
    items = [(f, is_clean_map[f], truth[f], described[f]) for f in truth if f in described]
    if len(items) == 0:
        raise ValueError("No overlapping (non-errored) plot files between manifest and detections.")

    judge_one = partial(_judge_one, judge=judge)
    if workers <= 1:
        verdicts = [judge_one(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            verdicts = list(executor.map(judge_one, items))

    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for verdict in verdicts:
        counts[verdict["outcome"]] += 1
    binary = _prf(counts["tp"], counts["fp"], counts["fn"])
    binary["accuracy"] = (counts["tp"] + counts["tn"]) / len(items)
    binary["tn"] = counts["tn"]
    return {"num_plots": len(items), "mode": "freetext", "binary": binary, "verdicts": verdicts}


def format_freetext_report(report: dict, detector: str | None = None, judge_model: str | None = None) -> str:
    """Render a free-text ``report`` as a human-readable text block."""
    lines: list[str] = ["Free-text visual-regression detection report"]
    annotations = [a for a in (detector, f"judge={judge_model}" if judge_model is not None else None) if a is not None]
    if len(annotations) > 0:
        lines[0] += "  (" + ", ".join(annotations) + ")"
    lines.append(f"Plots scored: {report['num_plots']}")
    binary = report["binary"]
    lines.append("")
    lines.append("Binary (judge: did the model correctly flag the known issue, and stay quiet on clean charts?):")
    lines.append(
        f"  accuracy={binary['accuracy']:.3f}  precision={binary['precision']:.3f}  "
        f"recall={binary['recall']:.3f}  f1={binary['f1']:.3f}",
    )
    lines.append(f"  tp={binary['tp']} fp={binary['fp']} fn={binary['fn']} tn={binary['tn']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Score detector output against the dataset manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the dataset manifest.json.")
    parser.add_argument("--detections", required=True, help="Path to the detections.json from detect.py.")
    parser.add_argument("--out", default=None, help="Output report JSON (defaults next to the detections).")
    parser.add_argument(
        "--judge-backend",
        choices=("claude", "openrouter"),
        default="claude",
        help="LLM judge backend for free-text scoring.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model (free-text mode only; a stronger model such as 'sonnet' grades more reliably).",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of judge calls to run concurrently.")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    detections_doc = json.loads(Path(args.detections).read_text(encoding="utf-8"))
    detections = detections_doc["detections"]
    mode = detections_doc.get("mode", "categories")
    detector = f"detector={detections_doc.get('backend')}/{detections_doc.get('model')}"

    if mode == "freetext":
        judge = make_judge(make_text_backend(args.judge_backend, args.judge_model))
        print(
            f"Judging {len(detections)} free-text reviews with {args.judge_backend}/{args.judge_model or 'default'}..."
        )
        report = evaluate_freetext(manifest["plots"], detections, judge, workers=args.workers)
        print(format_freetext_report(report, detector, args.judge_model or f"{args.judge_backend} default"))
    else:
        report = evaluate(manifest["plots"], detections, manifest["taxonomy"])
        print(format_report(report, detections_doc.get("backend"), detections_doc.get("model")))

    out_path = Path(args.out) if args.out is not None else Path(args.detections).with_name("report.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
