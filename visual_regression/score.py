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
from pathlib import Path


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


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Score detector output against the dataset manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the dataset manifest.json.")
    parser.add_argument("--detections", required=True, help="Path to the detections.json from detect.py.")
    parser.add_argument("--out", default=None, help="Output report JSON (defaults next to the detections).")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    detections_doc = json.loads(Path(args.detections).read_text(encoding="utf-8"))
    detections = detections_doc["detections"]

    report = evaluate(manifest["plots"], detections, manifest["taxonomy"])
    print(format_report(report, detections_doc.get("backend"), detections_doc.get("model")))

    out_path = Path(args.out) if args.out is not None else Path(args.detections).with_name("report.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
