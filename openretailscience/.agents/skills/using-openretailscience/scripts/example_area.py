"""Example script demonstrating area plot usage."""

import numpy as np
import pandas as pd
from openretailscience.plots import area

# Create sample data
rng = np.random.default_rng(42)
months = pd.date_range("2023-01-01", periods=12, freq="ME")
df = pd.DataFrame({
    "month": months,
    "Electronics": rng.integers(15000, 25000, 12),
    "Clothing": rng.integers(10000, 20000, 12),
    "Food": rng.integers(8000, 15000, 12),
    "Home": rng.integers(5000, 12000, 12)
})

# Example 1: Stacked area chart (multiple columns)
area.plot(
    df=df,
    value_col=["Electronics", "Clothing", "Food", "Home"],
    x_col="month",
    title="Monthly Sales Trends by Category",
    x_label="Month",
    y_label="Sales ($)",
    move_legend_outside=True,
    alpha=0.7,
    source_text="Source: Sales Data 2023",
)

# Example 2: Non-stacked (overlapping) areas
area.plot(
    df=df,
    value_col=["Electronics", "Clothing"],
    x_col="month",
    title="Electronics vs Clothing Sales",
    x_label="Month",
    y_label="Sales ($)",
    stacked=False,
    alpha=0.4,
    move_legend_outside=True,
)

# Example 3: Using group_col for automatic pivoting
df_long = df.melt(id_vars=["month"], var_name="category", value_name="sales")

area.plot(
    df=df_long,
    value_col="sales",
    x_col="month",
    group_col="category",
    title="Category Sales Over Time",
    x_label="Month",
    y_label="Sales ($)",
    alpha=0.6,
    move_legend_outside=True,
)

# Example 4: Index-based plotting
df_indexed = df.set_index("month")

area.plot(
    df=df_indexed,
    value_col=["Electronics", "Food"],
    title="Electronics and Food Sales Trends",
    y_label="Sales ($)",
    alpha=0.5,
    move_legend_outside=True,
)

# Example 5: Single area
area.plot(
    df=df,
    value_col="Electronics",
    x_col="month",
    title="Electronics Sales Trend",
    x_label="Month",
    y_label="Sales ($)",
    source_text="Source: Electronics Department 2023",
)
