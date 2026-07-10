"""Example script demonstrating index plot usage.

This script shows how to create index plots for comparing
categories against a baseline value (typically 100).
"""

import numpy as np
import pandas as pd
from openretailscience.plots import index

# Create sample data - sales by region and category
rng = np.random.default_rng(42)
regions = ["North", "South", "East", "West"]
categories = ["Electronics", "Clothing", "Food", "Home"]
months = ["Jan", "Feb", "Mar"]

df = pd.MultiIndex.from_product(
    [regions, categories, months],
    names=["region", "category", "month"],
).to_frame(index=False)
df["sales"] = rng.integers(5000, 50000, size=len(df))

# Example 1: Basic index plot - regions vs Electronics baseline
index.plot(
    df=df,
    value_col="sales",
    group_col="region",
    index_col="category",
    value_to_index="Electronics",
    title="Regional Sales Performance vs Electronics Category",
    y_label="Region",
    x_label="Index (Electronics = 100)",
    source_text="Source: Sales Data 2024",
)

# Example 2: With series - compare across months
index.plot(
    df=df,
    value_col="sales",
    group_col="category",
    index_col="region",
    value_to_index="North",
    series_col="month",
    title="Category Performance by Region (Indexed to North)",
    y_label="Category",
    legend_title="Month",
    sort_by="group",
)

# Example 3: Sorted by value, descending
index.plot(
    df=df,
    value_col="sales",
    group_col="region",
    index_col="category",
    value_to_index="Food",
    sort_by="value",
    sort_order="descending",
    title="Regional Performance Ranked (Food = 100)",
    highlight_range=(90, 110),
)

# Example 4: Filter top and bottom performers
index.plot(
    df=df,
    value_col="sales",
    group_col="category",
    index_col="region",
    value_to_index="North",
    top_n=2,
    bottom_n=2,
    title="Top 2 and Bottom 2 Categories",
    sort_by="value",
    sort_order="descending",
)

# Example 5: Filter by index value
index.plot(
    df=df,
    value_col="sales",
    group_col="region",
    index_col="category",
    value_to_index="Electronics",
    filter_below=100,
    title="Underperforming Regions (Index < 100)",
    highlight_range=None,
)
