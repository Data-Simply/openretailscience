"""Example script demonstrating HML (Heavy-Medium-Light) segmentation.

This script shows various ways to use HMLSegmentation from openretailscience to classify
customers into value tiers based on the Pareto principle (80/20 rule).
"""

import numpy as np
import pandas as pd

from openretailscience.segmentation.hml import HMLSegmentation

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Create sample transaction data with realistic customer value distribution
num_customers = 50
num_transactions_per_customer = rng.integers(1, 11, size=num_customers)
customer_ids = np.repeat(np.arange(1, num_customers + 1), num_transactions_per_customer)
# Pareto distribution: most customers spend little, few spend a lot (minimum spend of $10)
unit_spends = (rng.pareto(2.0, size=customer_ids.size) * 50 + 10).round(2)

transactions = pd.DataFrame({
    "customer_id": customer_ids,
    "unit_spend": unit_spends,
})

# Add some zero-spend customers (inactive/churned)
zero_customers = pd.DataFrame({
    "customer_id": [num_customers + 1, num_customers + 2, num_customers + 3],
    "unit_spend": [0.0, 0.0, 0.0],
})

transactions = pd.concat([transactions, zero_customers], ignore_index=True)

# Example 1: Basic HML Segmentation (Heavy/Medium/Light/Zero)
seg_basic = HMLSegmentation(
    df=transactions,
    zero_value_customers="separate_segment"
)
results_basic = seg_basic.df

# Example 2: Include Zero-Spend Customers with Light
seg_include = HMLSegmentation(
    df=transactions,
    zero_value_customers="include_with_light"
)
results_include = seg_include.df

# Example 3: Exclude Zero-Spend Customers
seg_exclude = HMLSegmentation(
    df=transactions,
    zero_value_customers="exclude"
)
results_exclude = seg_exclude.df

# Example 4: Add Segments Back to Transaction Data
seg = HMLSegmentation(df=transactions, zero_value_customers="include_with_light")

# Merge segment labels back onto transactions on customer_id.
# seg.df is indexed by customer_id with a segment_name column.
transactions_with_segments = transactions.merge(
    seg.df["segment_name"],
    left_on="customer_id",
    right_index=True,
    how="left",
)

# Example 5: Segment by Transaction Frequency
transactions_with_id = transactions.copy()
transactions_with_id["transaction_id"] = range(1, len(transactions_with_id) + 1)

seg_frequency = HMLSegmentation(
    df=transactions_with_id,
    value_col="transaction_id",
    agg_func="count"
)
results_frequency = seg_frequency.df
