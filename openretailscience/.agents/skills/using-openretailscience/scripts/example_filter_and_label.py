"""Example script demonstrating filter_and_label_by_condition usage."""

import ibis
import numpy as np
import pandas as pd
from openretailscience.utils.filter_and_label import filter_and_label_by_condition

# Create sample transaction data
rng = np.random.default_rng(42)
n_transactions = 1000

df = pd.DataFrame({
    "transaction_id": range(1, n_transactions + 1),
    "category": rng.choice(["toys", "shoes", "electronics", "clothing", "books"], n_transactions),
    "amount": rng.lognormal(mean=4, sigma=0.8, size=n_transactions),
    "quantity": rng.integers(1, 20, n_transactions),
    "has_promo": rng.choice([True, False], n_transactions, p=[0.3, 0.7]),
})

table = ibis.memtable(df)

# Example 1: Filter by category
category_filtered = filter_and_label_by_condition(
    table,
    conditions={
        "toys": table["category"] == "toys",
        "electronics": table["category"] == "electronics"
    }
)
category_result = category_filtered.execute()

# Example 2: Price tier segmentation
price_tiers = filter_and_label_by_condition(
    table,
    conditions={
        "budget": table["amount"] < 30,
        "mid_range": (table["amount"] >= 30) & (table["amount"] < 100),
        "premium": table["amount"] >= 100
    }
)
price_tier_result = price_tiers.execute()

# Example 3: Transaction type classification
transaction_types = filter_and_label_by_condition(
    table,
    conditions={
        "high_value": table["amount"] >= 150,
        "promotional": (table["has_promo"] == True) & (table["amount"] < 150),  # noqa: E712
        "bulk_purchase": (table["quantity"] >= 10) & (table["has_promo"] == False) & (table["amount"] < 150),  # noqa: E712
        "regular": (table["amount"] < 150) & (table["quantity"] < 10) & (table["has_promo"] == False)  # noqa: E712
    }
)
transaction_type_result = transaction_types.execute()

# Example 4: Multi-criteria product filtering
product_df = pd.DataFrame({
    "product_id": range(1, 501),
    "category": rng.choice(["electronics", "clothing", "toys"], 500),
    "units_sold": rng.integers(10, 2000, 500),
    "margin_percent": rng.uniform(10, 70, 500),
    "is_clearance": rng.choice([True, False], 500, p=[0.1, 0.9])
})

product_table = ibis.memtable(product_df)

products_of_interest = filter_and_label_by_condition(
    product_table,
    conditions={
        "top_seller_electronics": (product_table["category"] == "electronics") &
                                  (product_table["units_sold"] >= 1000),
        "high_margin_clothing": (product_table["category"] == "clothing") &
                                (product_table["margin_percent"] >= 50),
        "clearance_items": product_table["is_clearance"] == True  # noqa: E712
    }
)
product_result = products_of_interest.execute()

# Example 6: Seasonal transaction labeling
seasonal_df = pd.DataFrame({
    "transaction_id": range(1, 1001),
    "transaction_date": pd.date_range("2024-01-01", periods=1000, freq="D"),
})

seasonal_table = ibis.memtable(seasonal_df)

seasonal_data = filter_and_label_by_condition(
    seasonal_table,
    conditions={
        "holiday_season": seasonal_table["transaction_date"].month().isin([11, 12]),
        "back_to_school": seasonal_table["transaction_date"].month().isin([7, 8]),
        "spring": seasonal_table["transaction_date"].month().isin([3, 4, 5]),
        "summer": seasonal_table["transaction_date"].month() == 6
    }
)
seasonal_result = seasonal_data.execute()
