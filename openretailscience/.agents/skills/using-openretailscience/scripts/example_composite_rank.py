"""Example script for CompositeRank analysis."""

import numpy as np
import pandas as pd

from openretailscience.analysis.composite_rank import CompositeRank

rng = np.random.default_rng(42)

n_products = 50
product_ids = np.arange(1, n_products + 1)
df = pd.DataFrame({
    "product_id": product_ids,
    "weekly_sales": rng.uniform(100, 1000, size=n_products),
    "margin_pct": rng.uniform(10, 50, size=n_products),
    "stock_days": rng.uniform(5, 60, size=n_products),
    "return_rate": rng.uniform(0, 15, size=n_products),
})

# Example 1: Product Range Review
ranker = CompositeRank(
    df=df,
    rank_cols=[
        ("weekly_sales", "desc"),  # Higher is better
        ("margin_pct", "desc"),  # Higher is better
        ("stock_days", "asc"),  # Lower is better
        ("return_rate", "asc"),  # Lower is better
    ],
    agg_func="mean",
)

# Example 2: Conservative Ranking (Min Aggregation)
ranker_min = CompositeRank(
    df=df, rank_cols=[("weekly_sales", "desc"), ("margin_pct", "desc")], agg_func="min", ignore_ties=False
)
