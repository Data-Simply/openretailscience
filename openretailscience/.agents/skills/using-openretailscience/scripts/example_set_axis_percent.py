"""Example script demonstrating set_axis_percent formatting."""

import pandas as pd
from openretailscience.plots import bar, line, scatter
from openretailscience.plots.styles.graph_utils import set_axis_percent

# Example 1: Basic Y-axis percentage formatting
df = pd.DataFrame({
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "retention": [1.0, 0.85, 0.72, 0.65, 0.60, 0.58],
})
ax = line.plot(df=df, value_col="retention", x_col="month", title="Monthly Retention Rate", y_label="Retention")
set_axis_percent(ax.yaxis)

# Example 2: With decimal precision
df = pd.DataFrame({
    "week": ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"],
    "conversion_rate": [0.0325, 0.0412, 0.0389, 0.0456, 0.0501],
})
ax = bar.plot(df=df, value_col="conversion_rate", x_col="week", title="Weekly Conversion Rates")
set_axis_percent(ax.yaxis, decimals=2)  # Show as 3.25%, 4.12%, etc.

# Example 3: Data already in 0-100 range
df = pd.DataFrame({
    "company": ["Company A", "Company B", "Company C", "Company D"],
    "market_share": [45, 30, 15, 10],
})
ax = bar.plot(df=df, value_col="market_share", x_col="company", title="Market Share")
set_axis_percent(ax.yaxis, xmax=100)  # Data is 0-100, not 0-1

# Example 4: Both axes as percentages
df = pd.DataFrame({
    "email_open_rate": [0.22, 0.28, 0.25, 0.30, 0.27, 0.32],
    "click_through_rate": [0.045, 0.052, 0.048, 0.058, 0.051, 0.062],
})
ax = scatter.plot(
    df=df,
    value_col="click_through_rate",
    x_col="email_open_rate",
    x_label="Email Open Rate",
    y_label="Click-Through Rate",
    title="Email Campaign Performance",
)
set_axis_percent(ax.xaxis, decimals=0)
set_axis_percent(ax.yaxis, decimals=1)

# Example 6: Stacked percentage distribution (horizontal bars)
df = pd.DataFrame({
    "segment": ["Heavy", "Medium", "Light"],
    "revenue_share": [0.60, 0.25, 0.15],
})
ax = bar.plot(
    df=df,
    value_col="revenue_share",
    x_col="segment",
    orientation="horizontal",
    title="Customer Segment Revenue Distribution",
    x_label="Revenue Share",
)
set_axis_percent(ax.xaxis, decimals=0)  # horizontal bars put the value axis on x

# Example 7: Without percentage symbol
df = pd.DataFrame({
    "period": range(1, 6),
    "value": [0.1, 0.2, 0.3, 0.4, 0.5],
})
ax = line.plot(df=df, value_col="value", x_col="period", title="Custom Formatting (No % Symbol)")
set_axis_percent(ax.yaxis, symbol=None)  # Remove % symbol
