"""Example script demonstrating Threshold Segmentation for flexible custom classification.

This script shows how to use ThresholdSegmentation from openretailscience to create
custom percentile-based segments with user-defined thresholds and names.
"""

import numpy as np
import pandas as pd

from openretailscience.segmentation.threshold import ThresholdSegmentation

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Create sample transaction data: 60 customers with varying spend
num_customers = 60
num_transactions_per_customer = rng.integers(1, 8, size=num_customers)
customer_ids = np.repeat(np.arange(1, num_customers + 1), num_transactions_per_customer)
unit_spends = rng.uniform(5, 500, size=customer_ids.size).round(2)

transactions = pd.DataFrame({
    "customer_id": customer_ids,
    "unit_spend": unit_spends,
})

# Add zero-spend customers
zero_customers = pd.DataFrame({"customer_id": [61, 62, 63], "unit_spend": [0.0, 0.0, 0.0]})
transactions = pd.concat([transactions, zero_customers], ignore_index=True)

# Example 1: Quartile Segmentation (25% splits)
seg_quartile = ThresholdSegmentation(
    df=transactions,
    thresholds=[0.25, 0.50, 0.75, 1.0],
    segments=["Q1_Bottom", "Q2", "Q3", "Q4_Top"],
    zero_value_customers="exclude"
)
results = seg_quartile.df

# Example 2: Segment by Transaction Count
transactions_with_id = transactions.copy()
transactions_with_id["transaction_id"] = range(1, len(transactions_with_id) + 1)

seg_frequency = ThresholdSegmentation(
    df=transactions_with_id,
    value_col="transaction_id",
    agg_func="count",
    thresholds=[0.33, 0.67, 1.0],
    segments=["Infrequent", "Regular", "Frequent"]
)
results = seg_frequency.df

# Example 3: Segment by Product Diversity
transactions_with_product = transactions.copy()
transactions_with_product["product_id"] = rng.integers(1, 20, size=len(transactions))

seg_variety = ThresholdSegmentation(
    df=transactions_with_product,
    value_col="product_id",
    agg_func="nunique",
    thresholds=[0.50, 1.0],
    segments=["Low_Variety", "High_Variety"]
)
results = seg_variety.df

# Example 4: Handling Zero-Spend Customers
# Option 1: Separate segment
seg_separate = ThresholdSegmentation(
    df=transactions,
    thresholds=[0.50, 0.80, 1.0],
    segments=["Low", "Medium", "High"],
    zero_value_customers="separate_segment"
)

# Option 2: Exclude
seg_exclude = ThresholdSegmentation(
    df=transactions,
    thresholds=[0.50, 0.80, 1.0],
    segments=["Low", "Medium", "High"],
    zero_value_customers="exclude"
)

# Option 3: Include with lowest
seg_include = ThresholdSegmentation(
    df=transactions,
    thresholds=[0.50, 0.80, 1.0],
    segments=["Low", "Medium", "High"],
    zero_value_customers="include_with_light"
)

# Example 5: Add Segments Back to Transactions
seg = ThresholdSegmentation(
    df=transactions,
    thresholds=[0.33, 0.67, 1.0],
    segments=["Low", "Medium", "High"]
)

# Merge segment labels back onto transactions on customer_id.
# seg.df is indexed by customer_id with a segment_name column.
transactions_with_segments = transactions.merge(
    seg.df["segment_name"],
    left_on="customer_id",
    right_index=True,
    how="left",
)
