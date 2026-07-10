"""Example script demonstrating cohort plot usage."""

import numpy as np
import pandas as pd
from openretailscience.plots import cohort

# Create sample cohort data (typically from CohortAnalysis)
# Rows = cohort start periods, Columns = periods since start
rng = np.random.default_rng(42)

cohort_periods = ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06"]
periods = list(range(6))  # 0 to 5 months since start

# Create retention cohort data (percentages)
base_retention = 0.75 - np.arange(len(cohort_periods)) * 0.02  # Slight degradation for later cohorts
cohort_data = {
    cohort_label: np.cumprod(np.concatenate(([1.0], base_retention[i] * rng.uniform(0.85, 0.95, size=5))))
    for i, cohort_label in enumerate(cohort_periods)
}

cohort_df = pd.DataFrame(cohort_data, index=periods).T

# Example 1: Basic cohort retention plot
cohort.plot(
    df=cohort_df,
    x_label="Months Since First Purchase",
    y_label="Cohort (First Purchase Month)",
    title="Customer Retention Analysis",
    cbar_label="Retention Rate",
    percentage=True,
    source_text="Source: Customer Transaction Data 2023",
)

# Example 2: Revenue cohort (absolute values)
base_revenue = 50000 + np.arange(len(cohort_periods)) * 5000  # Growing cohorts
revenue_data = {
    cohort_label: base_revenue[i] * np.cumprod(np.concatenate(([1.0], rng.uniform(0.8, 1.2, size=5))))
    for i, cohort_label in enumerate(cohort_periods)
}

revenue_df = pd.DataFrame(revenue_data, index=periods).T

cohort.plot(
    df=revenue_df,
    x_label="Months Since First Purchase",
    y_label="Cohort Month",
    title="Monthly Revenue by Cohort",
    cbar_label="Revenue ($)",
    percentage=False,
    source_text="Source: Revenue Data 2023",
)
