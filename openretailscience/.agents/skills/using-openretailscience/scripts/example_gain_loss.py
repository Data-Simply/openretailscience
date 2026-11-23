"""Example script demonstrating Gain/Loss analysis with openretailscience."""

import numpy as np
import pandas as pd

from openretailscience.analysis.gain_loss import GainLoss

rng = np.random.default_rng(42)


# Example 1: Basic Brand Switching Analysis
n = 100
customer_ids = np.arange(1, n + 1)
branches = [
    (rng.random(n) < 0.6, "Brand A", "P1", rng.uniform(50, 200, n)),  # 60% buy Brand A in P1
    (rng.random(n) < 0.5, "Brand B", "P1", rng.uniform(40, 180, n)),  # 50% buy Brand B in P1
    (rng.random(n) < 0.55, "Brand A", "P2", rng.uniform(60, 220, n)),  # 55% buy Brand A in P2 (some switching)
    (rng.random(n) < 0.55, "Brand B", "P2", rng.uniform(45, 190, n)),  # 55% buy Brand B in P2
]
df = pd.concat(
    [
        pd.DataFrame({"customer_id": customer_ids[mask], "brand": brand, "unit_spend": spend[mask], "period": period})
        for mask, brand, period, spend in branches
    ],
    ignore_index=True,
)

gl = GainLoss(
    df=df,
    period_col="period",
    p1_value="P1",
    p2_value="P2",
    compare_col="brand",
    focus_value="Brand A",
    comparison_value="Brand B",
)
# gl.df columns: focus/comparison/total for _p1, _p2 and _diff, then new, lost, increased_focus,
# decreased_focus, switch_from_comparison, switch_to_comparison

# Example 2: Category Migration Analysis, Broken Down by Region
regions = ["North", "South", "East", "West"]
n_cat = 150
cat_customer_ids = np.arange(1, n_cat + 1)
region = rng.choice(regions, size=n_cat)
cat_branches = [
    (rng.random(n_cat) < 0.5, "Premium", "P1", rng.uniform(100, 300, n_cat)),
    (rng.random(n_cat) < 0.6, "Standard", "P1", rng.uniform(30, 150, n_cat)),
    (rng.random(n_cat) < 0.45, "Premium", "P2", rng.uniform(110, 320, n_cat)),  # some migration
    (rng.random(n_cat) < 0.65, "Standard", "P2", rng.uniform(35, 160, n_cat)),
]
cat_df = pd.concat(
    [
        pd.DataFrame({
            "customer_id": cat_customer_ids[mask],
            "category": category,
            "unit_spend": spend[mask],
            "period": period,
            "region": region[mask],
        })
        for mask, category, period, spend in cat_branches
    ],
    ignore_index=True,
)

gl_regional = GainLoss(
    df=cat_df,
    period_col="period",
    p1_value="P1",
    p2_value="P2",
    compare_col="category",
    focus_value="Premium",
    comparison_value="Standard",
    group_col="region",
)

# Example 3: Customer Count Analysis (each customer counts once per group and period)
gl_count = GainLoss(
    df=df,
    period_col="period",
    p1_value="P1",
    p2_value="P2",
    compare_col="brand",
    focus_value="Brand A",
    comparison_value="Brand B",
    value_col="customer_id",
    agg_func="nunique",
)

# Example 4: Customer-Level Details. customer_table is lazy: filter before executing on large data
customers = gl.customer_table
new_customers = customers.filter(customers.new > 0).execute()
switchers_to_a = customers.filter(customers.switch_from_comparison > 0).execute()

# Example 5: Visualizing Results
gl.plot(
    title="Revenue Flow: Brand A vs Brand B",
    x_label="Revenue ($)",
    y_label="Category",
    source_text="Source: Sample Transaction Data",
    move_legend_outside=True,
)
