# Visual Regression Detection Harness

An experiment to see whether a small/open vision LLM (Claude **Haiku**, or any vision model on
**OpenRouter**) can detect visual regressions in OpenRetailScience charts.

It works in three stages:

1. **Generate** a few hundred charts using the real `openretailscience.plots` API. A quarter are left
   clean; the rest get exactly one deliberate defect injected (title clipped off the edge, overlapping
   tick labels, legend dumped on the data, a big gap between the x-ticks and the x-axis label, …). A
   single `manifest.json` records every plot and its issue.
2. **Detect** — run each image through a vision LLM and ask what is wrong, in one of two modes:
   - `categories` (default): the model picks from the closed defect list → multi-label classification.
   - `freetext`: the model describes problems in prose → open-ended detection.
3. **Score** — compare against the manifest ground truth. `categories` is scored by exact category
   match; `freetext` is graded by an LLM judge (did the prose identify the known issue, and stay quiet
   on clean charts?).

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

### Free-text mode

```bash
# Detect: ask the model to describe problems in prose instead of picking categories
python -m visual_regression.detect --manifest visual_regression/output/manifest.json \
    --backend claude --model haiku --mode freetext

# Score: an LLM judge grades each prose review against the known issue.
# A stronger judge (e.g. --judge-model sonnet) grades more reliably.
python -m visual_regression.score \
    --manifest visual_regression/output/manifest.json \
    --detections visual_regression/output/detections.json \
    --judge-backend claude --judge-model haiku
```

`score` auto-detects the mode from `detections.json`. Free-text scoring makes one judge call per plot
(on top of the detection calls), so prefer `--limit N` while iterating.

## Outputs

- `output/images/plot_NNNN.png` (`generate`) — the rendered charts.
- `output/manifest.json` (`generate`) — the authoritative record: every plot, its `chart_type`, and the
  injected defect(s). Also embeds the defect `taxonomy`.
- `output/manifest.csv` (`generate`) — the same, human-readable, one row per plot.
- `output/detections.json` (`detect`) — the model's output per plot (detected categories, or a free-text
  `description`), plus the run's `mode`.
- `output/report.json` (`score`) — binary + per-category precision/recall/F1 (`categories`), or binary
  metrics plus per-plot judge verdicts (`freetext`).

The `output/` directory is git-ignored: the dataset is large and fully reproducible from a seed, so
regenerate it with `generate` rather than committing the images.

OpenRouter requests opt out of provider data collection and require zero data retention
(`provider: {data_collection: "deny", zdr: true}`), so the charts and prompts are not stored or trained
on.

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
- `categories` mode gives the detector the closed list (multi-label classification); `freetext` mode is
  open-ended and graded by an LLM judge, which is more realistic but only as reliable as the judge.
