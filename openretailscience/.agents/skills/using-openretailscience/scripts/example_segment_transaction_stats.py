"""Example script demonstrating Segment Transaction Statistics."""

import numpy as np
import pandas as pd

from openretailscience.segmentation.hml import HMLSegmentation
from openretailscience.segmentation.segstats import SegTransactionStats

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Create sample transaction data
num_transactions = 500
transactions = pd.DataFrame({
    "customer_id": rng.integers(1, 51, size=num_transactions),
    "transaction_id": range(1000, 1000 + num_transactions),
    "unit_spend": rng.uniform(10, 500, size=num_transactions).round(2),
    "unit_quantity": rng.integers(1, 10, size=num_transactions),
    "store_id": rng.choice(["Store_A", "Store_B", "Store_C"], size=num_transactions),
    "category": rng.choice(["Electronics", "Clothing", "Home"], size=num_transactions),
    "product_id": rng.integers(100, 200, size=num_transactions),
})

# Example 1: Basic HML Segment Statistics
# Create HML segments (seg.df is indexed by customer_id with a segment_name column)
seg = HMLSegmentation(df=transactions)
df_with_segments = transactions.merge(
    seg.df["segment_name"],
    left_on="customer_id",
    right_index=True,
    how="left",
)

stats = SegTransactionStats(data=df_with_segments, segment_col="segment_name", calc_total=True)
results = stats.df
# results columns include segment_name, spend, transactions, customers,
# spend_per_customer, transactions_per_customer

# Example 2: Multi-Dimensional Analysis (Store x Category)
stats_multi = SegTransactionStats(
    data=transactions,
    segment_col=["store_id", "category"],
    calc_total=True
)
results_multi = stats_multi.df
# calc_total=True adds a store_id="Total"/category="Total" row for the grand total

# Example 3: Add Custom Aggregations
stats_custom = SegTransactionStats(
    data=df_with_segments,
    segment_col="segment_name",
    extra_aggs={
        "unique_products": ("product_id", "nunique"),
        "unique_stores": ("store_id", "nunique"),
        "avg_quantity": ("unit_quantity", "mean")
    },
    calc_total=True
)
results_custom = stats_custom.df

# Example 4: Hierarchical Rollups (Store -> Category)
stats_rollup = SegTransactionStats(
    data=transactions,
    segment_col=["store_id", "category"],
    calc_rollup=True,
    calc_total=True,
    rollup_value=["All Stores", "All Categories"]
)
results_rollup = stats_rollup.df
# rollup rows use rollup_value ("All Stores"/"All Categories") as placeholders for the
# rolled-up dimension

# Example 5: Separate Metrics for Unknown/Guest Customers
# Add some unknown customers (customer_id = -1)
transactions_with_unknown = transactions.copy()
unknown_mask = rng.random(len(transactions_with_unknown)) < 0.2  # 20% unknown
transactions_with_unknown.loc[unknown_mask, "customer_id"] = -1

# Re-segment on the data that includes unknown customers, then merge labels back
seg_unknown = HMLSegmentation(df=transactions_with_unknown)
df_with_segments_unknown = transactions_with_unknown.merge(
    seg_unknown.df["segment_name"],
    left_on="customer_id",
    right_index=True,
    how="left",
)

stats_unknown = SegTransactionStats(
    data=df_with_segments_unknown,
    segment_col="segment_name",
    unknown_customer_value=-1,
    calc_total=True
)
results_unknown = stats_unknown.df
# results_unknown adds spend_unknown/spend_total and transactions_unknown/transactions_total
# columns alongside the regular (known-customer) spend/transactions columns

# Example 6: Segment Stats Without Total Row
stats_no_total = SegTransactionStats(
    data=df_with_segments,
    segment_col="segment_name",
    calc_total=False
)
results_no_total = stats_no_total.df
