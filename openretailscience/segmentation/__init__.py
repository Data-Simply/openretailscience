"""Segment KPIs and customer segmentations, one class per question.

- spend, transactions, customers and units by segment or any dimension, with subtotals via `cube` / `rollup`
  → `SegTransactionStats`
- recency, frequency and monetary scores per customer → `RFMSegmentation`
- heavy, medium, light and zero spenders → `HMLSegmentation`
- customers split at your own spend percentiles → `ThresholdSegmentation`
- new, repeating and lapsed customers between two periods → `NLRSegmentation`
"""

from openretailscience.segmentation.hml import HMLSegmentation
from openretailscience.segmentation.nlr import NLRSegmentation
from openretailscience.segmentation.rfm import RFMSegmentation
from openretailscience.segmentation.segstats import SegTransactionStats, cube, rollup
from openretailscience.segmentation.threshold import ThresholdSegmentation

__all__ = [
    "HMLSegmentation",
    "NLRSegmentation",
    "RFMSegmentation",
    "SegTransactionStats",
    "ThresholdSegmentation",
    "cube",
    "rollup",
]
