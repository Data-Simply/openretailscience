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

# Example 2: Series input (index -> x, values -> y; value_col, x_col, group_col must all be None)
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

# Example 3: Grouped lines from long-format data (each group_col value becomes a line; pivoted internally)
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
    fill_na_value=0,  # fill missing values after pivot
)

# Example 4: Highlight a single group value (other groups render muted behind it)
store_df = pd.DataFrame({
    "week": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
    "store": ["Riverside"] * 4 + ["Midtown"] * 4 + ["Harborview"] * 4,
    "weekly_sales": [4200, 4400, 4600, 4900, 3100, 3300, 3200, 3500, 2400, 2500, 2700, 2900]
})

line.plot(
    df=store_df,
    value_col="weekly_sales",
    x_col="week",
    group_col="store",
    highlight="Riverside",
    x_label="Week",
    y_label="Sales (£)",
    title="Weekly Sales by Store (Riverside Highlighted)",
    legend_title="Store",
    source_text="Source: OpenRetailScience - 2024",
)

# Example 5: Highlight specific value columns (other columns render muted)
metrics_df = pd.DataFrame({
    "day": range(1, 6),
    "revenue": [10000, 11200, 10800, 12400, 13100],
    "units_sold": [520, 545, 538, 602, 640],
    "avg_order_value": [19.2, 20.6, 20.1, 20.6, 20.5]
})

line.plot(
    df=metrics_df,
    value_col=["revenue", "units_sold", "avg_order_value"],
    x_col="day",
    highlight=["revenue", "avg_order_value"],
    x_label="Day",
    y_label="Value",
    title="Daily Store Metrics (Revenue & AOV Highlighted)",
    source_text="Source: OpenRetailScience - 2024",
)
