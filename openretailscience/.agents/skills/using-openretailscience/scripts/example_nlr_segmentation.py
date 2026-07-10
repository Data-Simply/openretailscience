"""Example script demonstrating NLR (New-Lapsed-Repeating) customer segmentation.

This script shows various ways to use NLRSegmentation from openretailscience to
classify customers into lifecycle stages (New, Repeating, Lapsed) by comparing their
activity across two time periods (P1 and P2).

A customer is considered active in a period only if their aggregated value (e.g., total
spend) is strictly positive in that period:
  - New:       active in P2 only
  - Repeating: active in both P1 and P2
  - Lapsed:    active in P1 only
"""

import numpy as np
import pandas as pd

from openretailscience.segmentation.nlr import NLRSegmentation

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Create sample transaction data spanning two periods (P1 and P2)
num_customers = 120
stores = ["Store_A", "Store_B", "Store_C"]

customer_ids = np.arange(1, num_customers + 1)
customer_stores = rng.choice(stores, size=num_customers)

# Decide which periods each customer is active in, biasing toward a realistic mix
# of repeating, lapsed, and new customers.
activity = rng.choice(["both", "p1_only", "p2_only"], size=num_customers, p=[0.45, 0.30, 0.25])
is_both = activity == "both"
single_period = np.where(activity == "p1_only", "P1", "P2")

# Expand customers into one row per (customer, active period); "both" customers get
# a P1 row and a P2 row, the rest get a single row for their one active period.
customer_periods = pd.concat([
    pd.DataFrame({
        "customer_id": customer_ids[~is_both],
        "store_id": customer_stores[~is_both],
        "period": single_period[~is_both],
    }),
    pd.DataFrame({
        "customer_id": np.repeat(customer_ids[is_both], 2),
        "store_id": np.repeat(customer_stores[is_both], 2),
        "period": np.tile(["P1", "P2"], is_both.sum()),
    }),
], ignore_index=True)

# Each customer makes 1-6 transactions in a period they are active in
num_transactions = rng.integers(1, 7, size=len(customer_periods))
transactions = customer_periods.loc[customer_periods.index.repeat(num_transactions)].reset_index(drop=True)
transactions["transaction_id"] = range(1000, 1000 + len(transactions))
transactions["unit_spend"] = rng.uniform(10, 500, size=len(transactions)).round(2)
transactions["units"] = rng.integers(1, 10, size=len(transactions))

# Example 1: Basic NLR Segmentation (default value_col=unit_spend, agg_func=sum)
nlr_basic = NLRSegmentation(
    df=transactions,
    period_col="period",
    p1_value="P1",
    p2_value="P2",
)
results_basic = nlr_basic.df
# results_basic columns include segment_name, unit_spend_p1, unit_spend_p2 (from value_col=unit_spend)

# Example 2: Segment by Visit Count (value_col=transaction_id, agg_func=nunique)
nlr_visits = NLRSegmentation(
    df=transactions,
    period_col="period",
    p1_value="P1",
    p2_value="P2",
    value_col="transaction_id",
    agg_func="nunique",
)
results_visits = nlr_visits.df

# Example 3: NLR Flow Broken Down by Store (group_col=store_id)
nlr_grouped = NLRSegmentation(
    df=transactions,
    period_col="period",
    p1_value="P1",
    p2_value="P2",
    group_col="store_id",
)
results_grouped = nlr_grouped.df
