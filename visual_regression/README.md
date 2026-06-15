# Visual Regression Detection Harness

An experiment to see whether a small/open vision LLM (Claude **Haiku**, or any vision model on
**OpenRouter**) can detect visual regressions in OpenRetailScience charts.

It works in three stages:

1. **Generate** a few hundred charts using the real `openretailscience.plots` API. A quarter are left
   clean; the rest get exactly one deliberate defect injected (title clipped off the edge, overlapping
   tick labels, legend dumped on the data, a big gap between the x-ticks and the x-axis label, …). A
   single `manifest.json` records every plot and its issue.
2. **Detect** — run each image through a vision LLM and ask which defects it sees.
3. **Score** — compare the model's detections against the manifest ground truth.

The charts are genuine OpenRetailScience editorial charts: each defect is produced by letting the
library render the chart normally, freezing its chrome layout engine, then mutating the matplotlib
artists so the breakage survives `savefig`. See `defects.py` for the full catalogue.

## Quick start

```bash
# 1. Generate 300 labelled charts into visual_regression/output/
python -m visual_regression.generate --count 300

# 2a. Detect with Claude Haiku (uses the local `claude` CLI in headless mode)
python -m visual_regression.detect --manifest visual_regression/output/manifest.json \
    --backend claude --model haiku

# 2b. ...or with an open model via OpenRouter
export OPENROUTER_API_KEY=sk-or-...
python -m visual_regression.detect --manifest visual_regression/output/manifest.json \
    --backend openrouter --model meta-llama/llama-3.2-11b-vision-instruct

# 3. Score the detections against ground truth
python -m visual_regression.score \
    --manifest visual_regression/output/manifest.json \
    --detections visual_regression/output/detections.json
```

Use `--limit N` on `detect` to try a handful of images first (the detection step makes one model call
per image, so a full 300-image run costs 300 calls).

## Outputs

- `output/images/plot_NNNN.png` (`generate`) — the rendered charts.
- `output/manifest.json` (`generate`) — the authoritative record: every plot, its `chart_type`, and the
  injected defect(s). Also embeds the defect `taxonomy`.
- `output/manifest.csv` (`generate`) — the same, human-readable, one row per plot.
- `output/detections.json` (`detect`) — the model's detected categories per plot, plus the raw response.
- `output/report.json` (`score`) — binary and per-category precision/recall/F1.

The `output/` directory is git-ignored: the dataset is large and fully reproducible from a seed, so
regenerate it with `generate` rather than committing the images.

## Defect catalogue

The defects live in `visual_regression/defects.py` as a registry of `Defect` objects (name,
human-readable description, severity, the chart features it needs, and the matplotlib mutation). The
same catalogue is embedded in every manifest, shown to the detector as the closed set of categories it
may report, and used by the scorer — so adding a new regression is a single entry in that list.

## Notes & limitations (v1)

- Base charts come from `openretailscience.plots`; defects are injected by mutating the rendered
  matplotlib figure. This is the pragmatic mix the task asked for — it keeps the charts realistic while
  giving precise, labelled control over what is broken.
- Each defective plot carries exactly one defect, which keeps the ground truth clean for scoring.
- The detector is given a closed set of categories, so this measures multi-label classification, not
  open-ended description.
