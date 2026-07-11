"""Example script for PurchasesPerCustomer, DaysBetweenPurchases, and TransactionChurn analysis."""

import numpy as np
import pandas as pd

from openretailscience.analysis.customer import DaysBetweenPurchases, PurchasesPerCustomer, TransactionChurn

rng = np.random.default_rng(42)

# Generate sample data: each customer makes 1-14 purchases on distinct random days.
dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")
n_customers = 200
n_dates = len(dates)
n_purchases = rng.integers(1, 15, size=n_customers)

# Distinct dates per customer without replacement: rank dates by a random key, keep the top n_purchases per row.
date_ranks = np.argsort(rng.random((n_customers, n_dates)), axis=1)
keep_mask = np.arange(n_dates) < n_purchases[:, None]
customer_idx, _ = np.nonzero(keep_mask)
date_idx = date_ranks[keep_mask]

df = pd.DataFrame(
    {
        "customer_id": customer_idx + 1,
        "transaction_date": dates[date_idx],
    }
).sort_values(["customer_id", "transaction_date"], ignore_index=True)
df["transaction_id"] = np.arange(1, len(df) + 1)

# Example 1: Purchases per customer
ppc = PurchasesPerCustomer(df=df)
median = ppc.purchases_percentile(0.5)
pct_one_time = ppc.find_purchase_percentile(1, comparison="equal_to")

# Example 2: Days between purchases
dbp = DaysBetweenPurchases(df=df)
median_days = dbp.purchases_percentile(0.5)
p25 = dbp.purchases_percentile(0.25)
p75 = dbp.purchases_percentile(0.75)

# Example 3: Transaction churn
tc = TransactionChurn(df=df, churn_period=90)
