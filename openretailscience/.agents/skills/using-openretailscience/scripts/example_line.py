"""Example: Line Plots from a DataFrame, a Series, and grouped long-format data."""

import pandas as pd
from openretailscience.plots import line

# Example 1: DataFrame input
df = pd.DataFrame({
    "days_since_launch": range(-5, 6),
    "daily_sales": [100, 120, 130, 150, 160, 180, 200, 190, 210, 220, 240]
})

line.plot(
    df=df,
    value_col="daily_sales",
    x_col="days_since_launch",
    x_label="Days Since Product Launch",
    y_label="Daily Sales (units)",
    title="Sales Performance Around Launch Date",
    source_text="Source: OpenRetailScience - 2024",
)

# Example 2: Series input - the index becomes the x-axis and the values become the y-axis;
# value_col, x_col, and group_col must all be None
revenue_series = pd.Series(
    data=[10000, 12000, 15000, 17000, 20000],
    index=[-2, -1, 0, 1, 2],
    name="revenue"
)

line.plot(
    df=revenue_series,
    x_label="Days Since Event",
    y_label="Revenue (£)",
    title="Revenue Impact Around Event",
    source_text="Source: OpenRetailScience - 2024",
)

# Example 3: Grouped lines from long-format data - each unique value in group_col becomes
# its own line; data is pivoted internally
grouped_df = pd.DataFrame({
    "week": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
    "product_category": ["Electronics"] * 4 + ["Clothing"] * 4 + ["Home"] * 4,
    "weekly_sales": [5000, 5200, 5500, 5800, 3000, 3200, 3100, 3400, 2000, 2100, 2200, 2300]
})

line.plot(
    df=grouped_df,
    value_col="weekly_sales",
    x_col="week",
    group_col="product_category",
    x_label="Week",
    y_label="Sales (£)",
    title="Weekly Sales by Category",
    legend_title="Category",
    source_text="Source: OpenRetailScience - 2024",
    move_legend_outside=True,
    fill_na_value=0,  # Optional: fill missing values after pivot
)
