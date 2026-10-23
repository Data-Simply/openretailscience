"""Retail analyses, one class per question.

- why revenue changed between two periods (customers, visits, spend per visit) → `RevenueTree`
  (`calc_tree_kpis` is its KPI math as a function)
- customers gained, lost and switched between brands or stores across two periods → `GainLoss`
- customer overlap between 2-3 segments (Venn) → `CrossShop`
- retention of customers by first-purchase cohort → `CohortAnalysis`
- products bought together (market-basket rules) → `ProductAssociation`
- which products customers treat as substitutes → `CustomerDecisionHierarchy`
- purchases per customer, days between purchases, churn by purchase number → `PurchasesPerCustomer`,
  `DaysBetweenPurchases`, `TransactionChurn`
- one rank across several metrics → `CompositeRank`
- distance between two lat/lon points → `haversine_distance`

Spend, transactions and customers by segment live in `openretailscience.segmentation.SegTransactionStats`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import FunctionType

    from openretailscience.analysis.cohort import CohortAnalysis
    from openretailscience.analysis.composite_rank import CompositeRank
    from openretailscience.analysis.cross_shop import CrossShop
    from openretailscience.analysis.customer import DaysBetweenPurchases, PurchasesPerCustomer, TransactionChurn
    from openretailscience.analysis.customer_decision_hierarchy import CustomerDecisionHierarchy
    from openretailscience.analysis.gain_loss import GainLoss
    from openretailscience.analysis.haversine import haversine_distance
    from openretailscience.analysis.product_association import ProductAssociation
    from openretailscience.analysis.revenue_tree import RevenueTree, calc_tree_kpis

# Lazy (PEP 562): some modules pull in matplotlib and scipy, so each loads only when one of its names is used.
_EXPORTS = {
    "CohortAnalysis": "cohort",
    "CompositeRank": "composite_rank",
    "CrossShop": "cross_shop",
    "DaysBetweenPurchases": "customer",
    "PurchasesPerCustomer": "customer",
    "TransactionChurn": "customer",
    "CustomerDecisionHierarchy": "customer_decision_hierarchy",
    "GainLoss": "gain_loss",
    "haversine_distance": "haversine",
    "ProductAssociation": "product_association",
    "RevenueTree": "revenue_tree",
    "calc_tree_kpis": "revenue_tree",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> type | FunctionType:
    if name not in _EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    try:
        module = importlib.import_module(f"{__name__}.{_EXPORTS[name]}")
    except AttributeError as err:
        # Escaping as AttributeError makes `from x import name` report only "cannot import name".
        msg = f"failed to import {name!r} from {__name__!r}"
        raise ImportError(msg) from err
    return getattr(module, name)


def __dir__() -> list[str]:
    return __all__
