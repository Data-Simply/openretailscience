"""Example script for CohortAnalysis."""

import numpy as np
import pandas as pd

from openretailscience.analysis.cohort import CohortAnalysis

rng = np.random.default_rng(42)

dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")
n_customers = 200
customer_ids = np.arange(1, n_customers + 1)
first_purchases = rng.choice(dates[:300], size=n_customers)  # First purchase in first 300 days
n_additional = rng.integers(0, 10, size=n_customers)

first_purchase_df = pd.DataFrame(
    {
        "customer_id": customer_ids,
        "transaction_date": first_purchases,
        "unit_spend": rng.uniform(50, 200, size=n_customers),
    }
)

# Additional purchases: offset each customer's first-purchase date by a random 1-90 day gap.
repeat_customer_ids = np.repeat(customer_ids, n_additional)
repeat_first_purchases = np.repeat(first_purchases, n_additional)
next_purchases = repeat_first_purchases + pd.to_timedelta(rng.integers(1, 90, size=repeat_customer_ids.size), unit="D")
in_range = next_purchases <= dates[-1]

additional_purchases_df = pd.DataFrame(
    {
        "customer_id": repeat_customer_ids,
        "transaction_date": next_purchases,
        "unit_spend": rng.uniform(50, 200, size=repeat_customer_ids.size),
    }
)[in_range]

df = pd.concat([first_purchase_df, additional_purchases_df], ignore_index=True)

# Example 1: Monthly customer retention
cohort = CohortAnalysis(
    df=df, aggregation_column="customer_id", agg_func="nunique", period="month", percentage=True
)
results = cohort.df

# Example 2: Revenue by cohort
cohort_revenue = CohortAnalysis(
    df=df, aggregation_column="unit_spend", agg_func="sum", period="month", percentage=False
)
results_revenue = cohort_revenue.df

# Example 3: Weekly cohorts
cohort_weekly = CohortAnalysis(
    df=df, aggregation_column="customer_id", agg_func="nunique", period="week", percentage=True
)
results_weekly = cohort_weekly.df
