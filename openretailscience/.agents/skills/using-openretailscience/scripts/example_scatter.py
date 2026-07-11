"""Example script demonstrating scatter plot usage."""

import numpy as np
import pandas as pd
from openretailscience.plots import scatter

rng = np.random.default_rng(42)
categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

index = pd.MultiIndex.from_product([months, categories], names=["month", "category"])
n_rows = len(index)
df = pd.DataFrame({
    "sales": rng.integers(500, 5000, n_rows),
    "profit": rng.integers(100, 2000, n_rows),
    "customers": rng.integers(50, 500, n_rows),
    "avg_transaction": rng.uniform(50, 200, n_rows),
}, index=index).reset_index()

# Example 1: Basic scatter - single series
scatter.plot(
    df=df,
    value_col="sales",
    x_col="profit",
    title="Sales vs Profit Relationship",
    x_label="Profit ($)",
    y_label="Sales ($)",
    alpha=0.6,
    s=100,
)

# Example 2: Grouped by category
scatter.plot(
    df=df,
    value_col="sales",
    x_col="profit",
    group_col="category",
    title="Sales vs Profit by Category",
    x_label="Profit ($)",
    y_label="Sales ($)",
    move_legend_outside=True,
    alpha=0.7,
    s=120,
)

# Example 3: Multiple value columns
scatter.plot(
    df=df,
    value_col=["sales", "profit", "customers"],
    x_col="month",
    title="Monthly Metrics Comparison",
    x_label="Month",
    y_label="Value",
    move_legend_outside=True,
    alpha=0.6,
    s=80,
)

# Example 5: Time series scatter
time_df = df[df["category"] == "Electronics"].copy()

scatter.plot(
    df=time_df,
    value_col="sales",
    x_col="month",
    title="Electronics Sales Pattern Over Months",
    x_label="Month",
    y_label="Sales ($)",
    alpha=0.7,
    s=200,
    marker="D",
)

# Example 6: Index-based plotting
indexed_df = df.groupby("category").agg({
    "sales": "sum",
    "profit": "sum"
}).reset_index()
indexed_df = indexed_df.set_index("category")

scatter.plot(
    df=indexed_df,
    value_col="profit",
    title="Total Profit by Category",
    y_label="Total Profit ($)",
    alpha=0.7,
    s=200,
)

# Example 7: Custom colors and markers
scatter.plot(
    df=df,
    value_col=["sales", "profit"],
    x_col="customers",
    title="Sales and Profit vs Customers",
    x_label="Number of Customers",
    y_label="Amount ($)",
    color=["#FF5733", "#33FF57"],
    marker="^",
    s=100,
    alpha=0.6,
    move_legend_outside=True,
)
