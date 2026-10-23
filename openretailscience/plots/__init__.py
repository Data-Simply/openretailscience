"""Charts, one module per chart type, each drawn with its `plot()` function.

- general charts of pre-aggregated data → `bar`, `line`, `area`, `scatter`, `histogram`
- a value over time from raw transactions → `time`; several periods overlaid → `period_on_period`
- what added up to a change → `waterfall`
- overlap between 2-3 sets → `venn`
- a 2D grid of values → `heatmap`; a cohort retention matrix → `cohort`
- groups against a baseline where 100 is parity → `index`
- price-band distribution across categories → `price`
- gaps in data availability → `broken_timeline`
- colors, axis formatting and trend lines → `styles`
- the tree layout behind `RevenueTree.draw_tree` → `tree_diagram`
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

# Lazy (PEP 562): every module imports matplotlib, so each loads only when used.
__all__ = [
    "area",
    "bar",
    "broken_timeline",
    "cohort",
    "heatmap",
    "histogram",
    "index",
    "line",
    "period_on_period",
    "price",
    "scatter",
    "styles",
    "time",
    "tree_diagram",
    "venn",
    "waterfall",
]


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return importlib.import_module(f"{__name__}.{name}")


def __dir__() -> list[str]:
    return __all__
