"""OpenRetailScience: tools for retail data science.

- `segmentation`: KPIs by segment (`SegTransactionStats`) and customer segmentations (RFM, HML, NLR, thresholds)
- `analysis`: revenue trees, gain-loss switching, cross-shop overlap, cohorts, market baskets and more
- `plots`: one module per chart type, each with a `plot()` function
- `utils`: Ibis helpers for labelling rows and splitting transactions into time periods
- `options`: map your column names once with `set_option`
- `experimental`: APIs that may change without notice
- `skills`: install the bundled AI-agent skill with `install_skills()`
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

# Lazy (PEP 562) so `import openretailscience` stays cheap.
__all__ = ["analysis", "experimental", "options", "plots", "segmentation", "skills", "utils"]


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return importlib.import_module(f"{__name__}.{name}")


def __dir__() -> list[str]:
    return __all__
