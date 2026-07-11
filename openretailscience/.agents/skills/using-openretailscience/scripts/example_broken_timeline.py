"""Example script demonstrating broken timeline plots for data availability and gaps over time."""

import numpy as np
import pandas as pd
from openretailscience.plots import broken_timeline

rng = np.random.default_rng(42)
date_range = pd.date_range("2024-01-01", "2024-03-31", freq="D")
categories = ["Electronics", "Clothing", "Food", "Home"]

df = pd.MultiIndex.from_product(
    [date_range, categories],
    names=["transaction_date", "category"],
).to_frame(index=False)
df["sales"] = rng.integers(100, 5000, size=len(df))
# Drop ~30% of rows to create gaps
df = df[rng.random(len(df)) > 0.3].reset_index(drop=True)

# Example 1: Daily timeline
broken_timeline.plot(
    df=df,
    category_col="category",
    value_col="sales",
    title="Daily Sales Data Availability by Category",
    x_label="Date",
    y_label="Category",
    period="D",
    source_text="Source: Sales Data Q1 2024",
)

# Example 2: Weekly aggregation
broken_timeline.plot(
    df=df,
    category_col="category",
    value_col="sales",
    title="Weekly Sales Patterns",
    x_label="Week",
    y_label="Category",
    period="W",
    agg_func="sum",
)

# Example 3: With threshold - low sales days
broken_timeline.plot(
    df=df,
    category_col="category",
    value_col="sales",
    threshold_value=1000,  # Days with < 1000 shown as gaps
    title="Sales Activity (Days with $1000+ Sales)",
    x_label="Date",
    y_label="Category",
    period="D",
    source_text="Source: Threshold = $1000",
)

# Example 4: Store-level analysis
stores = ["Store A", "Store B", "Store C", "Store D", "Store E"]
store_df = pd.MultiIndex.from_product(
    [date_range, stores],
    names=["transaction_date", "store"],
).to_frame(index=False)
store_df["transactions"] = rng.integers(10, 100, size=len(store_df))
store_df = store_df[rng.random(len(store_df)) > 0.25].reset_index(drop=True)

broken_timeline.plot(
    df=store_df,
    category_col="store",
    value_col="transactions",
    title="Store Data Availability Timeline",
    x_label="Date",
    y_label="Store",
    period="D",
    bar_height=0.6,
)
