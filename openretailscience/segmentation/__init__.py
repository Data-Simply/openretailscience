"""Segment KPIs and customer segmentations, one class per question.

- spend, transactions, customers and units by segment or any dimension, with subtotals via `cube` / `rollup`
  → `SegTransactionStats`
- recency, frequency and monetary scores per customer → `RFMSegmentation`
- heavy, medium, light and zero spenders → `HMLSegmentation`
- customers split at your own spend percentiles → `ThresholdSegmentation`
- new, repeating and lapsed customers between two periods → `NLRSegmentation`

Call help() on a class or function, such as `help(openretailscience.segmentation.SegTransactionStats)`, for its
parameters, output and an example.
"""

from openretailscience.segmentation.hml import HMLSegmentation as HMLSegmentation
from openretailscience.segmentation.nlr import NLRSegmentation as NLRSegmentation
from openretailscience.segmentation.rfm import RFMSegmentation as RFMSegmentation
from openretailscience.segmentation.segstats import SegTransactionStats as SegTransactionStats
from openretailscience.segmentation.segstats import cube as cube
from openretailscience.segmentation.segstats import rollup as rollup
from openretailscience.segmentation.threshold import ThresholdSegmentation as ThresholdSegmentation

# No `__all__`: pydoc would render every listed class in full on this package's help() page.
_EXPORTS = (
    "HMLSegmentation",
    "NLRSegmentation",
    "RFMSegmentation",
    "SegTransactionStats",
    "ThresholdSegmentation",
    "cube",
    "rollup",
)


def __dir__() -> list[str]:
    return list(_EXPORTS)
