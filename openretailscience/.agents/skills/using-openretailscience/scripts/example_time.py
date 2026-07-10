"""Example script demonstrating time plot usage.

This script shows how to create timeline plots with automatic
period aggregation for transactional data.
"""

import numpy as np
import pandas as pd
from openretailscience.plots import time

# Create sample transaction data
rng = np.random.default_rng(42)
date_range = pd.date_range("2023-01-01", "2023-12-31", freq="D")
categories = ["Electronics", "Clothing", "Food"]

df = pd.MultiIndex.from_product(
    [date_range, categories],
    names=["transaction_date", "category"],
).to_frame(index=False)
df["num_transactions"] = rng.integers(5, 50, size=len(df))
df = df.loc[df.index.repeat(df["num_transactions"])].drop(columns="num_transactions").reset_index(drop=True)
df["total_price"] = rng.uniform(10, 500, size=len(df))

# Example 1: Monthly total sales
time.plot(
    df=df,
    value_col="total_price",
    period="M",
    agg_func="sum",
    title="Monthly Total Sales",
    y_label="Sales ($)",
    source_text="Source: Transaction Data 2023",
)

# Example 2: Weekly sales by category
time.plot(
    df=df,
    value_col="total_price",
    period="W",
    group_col="category",
    agg_func="sum",
    title="Weekly Sales by Product Category",
    y_label="Sales ($)",
    legend_title="Category",
    move_legend_outside=True,
    source_text="Source: Sales Data 2023",
)

# Example 3: Daily transaction count
time.plot(
    df=df,
    value_col="transaction_date",  # Count transactions
    period="D",
    agg_func="count",
    title="Daily Transaction Volume",
    y_label="Number of Transactions",
    linewidth=2,
)

# Example 4: Quarterly average transaction value
time.plot(
    df=df,
    value_col="total_price",
    period="Q",
    agg_func="mean",
    title="Quarterly Average Transaction Value",
    y_label="Average Value ($)",
    source_text="Source: 2023 Transactions",
)

# Example 6: Yearly summary
# Add another year of data
date_range_2024 = pd.date_range("2024-01-01", "2024-12-31", freq="D")
df_2024 = pd.MultiIndex.from_product(
    [date_range_2024, categories],
    names=["transaction_date", "category"],
).to_frame(index=False)
df_2024["num_transactions"] = rng.integers(5, 55, size=len(df_2024))
df_2024 = (
    df_2024.loc[df_2024.index.repeat(df_2024["num_transactions"])]
    .drop(columns="num_transactions")
    .reset_index(drop=True)
)
df_2024["total_price"] = rng.uniform(10, 520, size=len(df_2024))

df_multi_year = pd.concat([df, df_2024])

time.plot(
    df=df_multi_year,
    value_col="total_price",
    period="Y",
    agg_func="sum",
    title="Yearly Total Sales (2023-2024)",
    y_label="Total Sales ($)",
    source_text="Source: Multi-Year Data",
)
