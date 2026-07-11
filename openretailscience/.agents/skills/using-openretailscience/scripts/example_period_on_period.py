"""Example script demonstrating period-on-period plot usage."""

import numpy as np
import pandas as pd
from openretailscience.plots import period_on_period

rng = np.random.default_rng(42)
date_range = pd.date_range("2022-01-01", "2024-12-31", freq="D")

days = np.arange(len(date_range))
seasonal = 100 + 30 * np.sin(2 * np.pi * days / 365)  # Annual seasonality
trend = days * 0.05  # Slight upward trend
sales = seasonal + trend

df = pd.DataFrame({"date": date_range, "sales": sales})

# Example 1: Compare Q1 across three years
q1_periods = [
    ("2022-01-01", "2022-03-31"),  # Q1 2022
    ("2023-01-01", "2023-03-31"),  # Q1 2023
    ("2024-01-01", "2024-03-31"),  # Q1 2024
]

period_on_period.plot(
    df=df,
    x_col="date",
    value_col="sales",
    periods=q1_periods,
    title="Q1 Sales Comparison Across Years",
    y_label="Daily Sales ($)",
    x_label="Days Since Period Start",
    legend_title="Quarter",
    move_legend_outside=True,
)

# Example 2: Compare holiday shopping seasons
holiday_periods = [
    ("2022-11-01", "2022-12-31"),  # Holiday 2022
    ("2023-11-01", "2023-12-31"),  # Holiday 2023
    ("2024-11-01", "2024-12-31"),  # Holiday 2024
]

period_on_period.plot(
    df=df,
    x_col="date",
    value_col="sales",
    periods=holiday_periods,
    title="Holiday Season Performance Comparison",
    y_label="Daily Sales ($)",
    legend_title="Year",
    move_legend_outside=False,
)

# Example 3: Compare summer months
summer_periods = [
    ("2022-06-01", "2022-08-31"),
    ("2023-06-01", "2023-08-31"),
    ("2024-06-01", "2024-08-31"),
]

period_on_period.plot(
    df=df,
    x_col="date",
    value_col="sales",
    periods=summer_periods,
    title="Summer Sales Trends",
    y_label="Daily Sales ($)",
    source_text="Source: Simulated retail data",
    legend_title="Summer Period",
)
