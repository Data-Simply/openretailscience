"""Example script demonstrating RFM (Recency, Frequency, Monetary) segmentation."""

import numpy as np
import pandas as pd

from openretailscience.segmentation.rfm import RFMSegmentation

rng = np.random.default_rng(42)

num_customers = 100
num_transactions_per_customer = rng.integers(1, 21, size=num_customers)

days_since_last = rng.integers(1, 365, size=num_customers)
last_purchase_dates = np.datetime64("2024-07-01") - days_since_last.astype("timedelta64[D]")

customer_ids = np.repeat(np.arange(1, num_customers + 1), num_transactions_per_customer)
last_purchase_per_txn = np.repeat(last_purchase_dates, num_transactions_per_customer)
max_days_back = np.repeat(np.minimum(days_since_last + 1, 365), num_transactions_per_customer)

# Random transaction date up to each customer's last purchase date
days_back = rng.integers(0, max_days_back)
transaction_dates = (last_purchase_per_txn - days_back.astype("timedelta64[D]")).astype(str)

transaction_ids = np.arange(1000, 1000 + customer_ids.size)
unit_spends = rng.uniform(10, 1000, size=customer_ids.size).round(2)

transactions = pd.DataFrame({
    "customer_id": customer_ids,
    "transaction_id": transaction_ids,
    "transaction_date": transaction_dates,
    "unit_spend": unit_spends,
})

# Example 1: Basic RFM Segmentation (10 bins per dimension)
rfm_basic = RFMSegmentation(
    df=transactions,
    current_date="2024-07-01"
)
results_basic = rfm_basic.df

# Example 2: Custom Bin Counts (5 bins per dimension)
rfm_5bins = RFMSegmentation(
    df=transactions,
    current_date="2024-07-01",
    r_segments=5,
    f_segments=5,
    m_segments=5
)
results_5bins = rfm_5bins.df
# r_score/f_score/m_score now range 0-4 (5 bins instead of the default 10)

# Example 3: Custom Percentile Cut Points
# Create 4 monetary segments: bottom 50%, 50-80%, 80-95%, top 5%
rfm_custom = RFMSegmentation(
    df=transactions,
    current_date="2024-07-01",
    m_segments=[0.50, 0.80, 0.95]  # Creates 4 segments
)
results_custom = rfm_custom.df

# Example 4: Filter to Focus on High-Value Customers
rfm_filtered = RFMSegmentation(
    df=transactions,
    current_date="2024-07-01",
    min_monetary=500.0
)
results_filtered = rfm_filtered.df

# Example 5: Filter to Repeat Customers Only
rfm_repeat = RFMSegmentation(
    df=transactions,
    current_date="2024-07-01",
    min_frequency=5
)
results_repeat = rfm_repeat.df

# Example 6: Combined Filters with Custom Bins
rfm_midtier = RFMSegmentation(
    df=transactions,
    current_date="2024-07-01",
    min_frequency=3,
    max_frequency=10,
    min_monetary=200.0,
    max_monetary=2000.0,
    r_segments=3,
    f_segments=3,
    m_segments=3
)
results_midtier = rfm_midtier.df
