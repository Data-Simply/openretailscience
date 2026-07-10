"""Example script demonstrating Gain/Loss analysis with openretailscience."""

import numpy as np
import pandas as pd

from openretailscience.analysis.gain_loss import GainLoss

# Generate reproducible sample data
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

# Create boolean indexes
p1_index = df["period"] == "P1"
p2_index = df["period"] == "P2"
brand_a_index = df["brand"] == "Brand A"
brand_b_index = df["brand"] == "Brand B"

gl = GainLoss(
    df=df,
    p1_index=p1_index,
    p2_index=p2_index,
    focus_group_index=brand_a_index,
    focus_group_name="Brand A",
    comparison_group_index=brand_b_index,
    comparison_group_name="Brand B",
)
# gl.gain_loss_table_df columns: new, lost, increased_focus, decreased_focus,
# switch_from_comparison, switch_to_comparison

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
    p1_index=cat_df["period"] == "P1",
    p2_index=cat_df["period"] == "P2",
    focus_group_index=cat_df["category"] == "Premium",
    focus_group_name="Premium",
    comparison_group_index=cat_df["category"] == "Standard",
    comparison_group_name="Standard",
    group_col="region",
)

# Example 3: Customer Count Analysis
# Count customers instead of summing spend
gl_count = GainLoss(
    df=df,
    p1_index=p1_index,
    p2_index=p2_index,
    focus_group_index=brand_a_index,
    focus_group_name="Brand A",
    comparison_group_index=brand_b_index,
    comparison_group_name="Brand B",
    value_col="customer_id",
    agg_func="count",
)

# Example 4: Customer-Level Details
# Access customer-level data
customer_details = gl.gain_loss_df

# Identify different customer segments
new_customers = customer_details[customer_details["new"] > 0]
lost_customers = customer_details[customer_details["lost"] < 0]
switchers_to_a = customer_details[customer_details["switch_from_comparison"] > 0]
switchers_to_b = customer_details[customer_details["switch_to_comparison"] < 0]

# Example 5: Visualizing Results
gl.plot(
    title="Revenue Flow: Brand A vs Brand B",
    x_label="Revenue ($)",
    y_label="Category",
    source_text="Source: Sample Transaction Data",
    move_legend_outside=True,
)
