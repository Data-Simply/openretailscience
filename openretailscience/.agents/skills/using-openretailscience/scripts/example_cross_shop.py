"""Example script for CrossShop analysis."""

import numpy as np
import pandas as pd

from openretailscience.analysis.cross_shop import CrossShop

rng = np.random.default_rng(42)

# Generate sample data: some customers shop both categories
n_customers = 150
customer_ids = np.arange(1, n_customers + 1)
branches = [
    (rng.random(n_customers) < 0.4, "Electronics", rng.uniform(100, 500, n_customers)),
    (rng.random(n_customers) < 0.5, "Clothing", rng.uniform(50, 300, n_customers)),
    (rng.random(n_customers) < 0.3, "Home", rng.uniform(75, 400, n_customers)),
]
df = pd.concat(
    [
        pd.DataFrame({"customer_id": customer_ids[mask], "category": category, "unit_spend": spend[mask]})
        for mask, category, spend in branches
    ],
    ignore_index=True,
)

# Example 1: 2-Way Cross-Shopping
cs = CrossShop(
    df=df,
    group_1_col="category",
    group_1_val="Electronics",
    group_2_val="Clothing",
    labels=["Electronics", "Clothing"],
)

# Example 2: 3-Way Cross-Shopping
cs3 = CrossShop(
    df=df,
    group_1_col="category",
    group_1_val="Electronics",
    group_2_val="Clothing",
    group_3_val="Home",
    labels=["Electronics", "Clothing", "Home"],
)
