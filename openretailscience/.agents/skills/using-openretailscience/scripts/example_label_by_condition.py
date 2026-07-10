"""Example script demonstrating label_by_condition usage."""

import ibis
import numpy as np
import pandas as pd
from openretailscience.utils.label import label_by_condition

# Create sample transaction data
rng = np.random.default_rng(42)
n_transactions = 2000

df = pd.DataFrame({
    "customer_id": rng.integers(1, 300, n_transactions),
    "category": rng.choice(["clothing", "footwear", "electronics", "toys", "books"], n_transactions),
    "brand": rng.choice(["Nike", "Adidas", "Generic", "Premium", "Budget"], n_transactions),
    "is_promo": rng.choice([True, False], n_transactions, p=[0.35, 0.65]),
    "channel": rng.choice(["online", "in_store"], n_transactions, p=[0.6, 0.4]),
    "unit_price": rng.lognormal(mean=3.5, sigma=0.9, size=n_transactions),
    "is_return": rng.choice([True, False], n_transactions, p=[0.08, 0.92]),
})

table = ibis.memtable(df)

# Example 1: Binary labeling - footwear buyers
footwear_buyers = label_by_condition(
    table,
    condition=table["category"] == "footwear",
    label_col="customer_id",
    return_col="bought_footwear",
    labeling_strategy="binary",
    contains_label="yes",
    not_contains_label="no"
)
footwear_result = footwear_buyers.execute()

# Example 2: Extended labeling - promo purchase behavior
promo_behavior = label_by_condition(
    table,
    condition=table["is_promo"] == True,  # noqa: E712
    label_col="customer_id",
    return_col="promo_behavior",
    labeling_strategy="extended",
    contains_label="all_promo",
    mixed_label="mixed",
    not_contains_label="no_promo"
)
promo_result = promo_behavior.execute()

# Example 8: Comprehensive customer profile - combining multiple labels
# Get multiple labels
promo_labels = label_by_condition(
    table,
    condition=table["is_promo"] == True,  # noqa: E712
    label_col="customer_id",
    return_col="promo_type",
    labeling_strategy="binary"
)

channel_labels = label_by_condition(
    table,
    condition=table["channel"] == "online",
    label_col="customer_id",
    return_col="channel_type",
    labeling_strategy="binary"
)

# Join them together for comprehensive profile
customer_profile = (
    promo_labels
    .join(channel_labels, "customer_id")
    .select(
        promo_labels["customer_id"],
        promo_labels["promo_type"],
        channel_labels["channel_type"]
    )
)
profile_result = customer_profile.execute()

# Example 9: Premium Nike buyers (brand and price combination)
premium_nike = label_by_condition(
    table,
    condition=(table["brand"] == "Nike") & (table["unit_price"] >= 60),
    label_col="customer_id",
    return_col="premium_nike_buyer",
    labeling_strategy="extended",
    contains_label="only_premium_nike",
    mixed_label="mixed_nike",
    not_contains_label="no_premium_nike"
)
premium_nike_result = premium_nike.execute()
