"""Example script demonstrating Product Association analysis with openretailscience."""

import numpy as np
import pandas as pd

from openretailscience.analysis.product_association import ProductAssociation

rng = np.random.default_rng(42)


# Example 1: Basic Market Basket Analysis
# Products co-occur within a shared category so the metrics have real signal (random products give uplift ~1.0).
n_transactions = 500
categories = {
    "breakfast": (["Milk", "Bread", "Butter", "Eggs"], 0.3),
    "beverage": (["Coffee", "Tea", "Sugar"], 0.25),
    "snack": (["Chips", "Soda", "Chocolate"], 0.2),
    "alcohol": (["Beer", "Wine", "Chips"], 0.15),
}

category_frames = []
for items, category_prob in categories.values():
    active_transactions = np.nonzero(rng.random(n_transactions) < category_prob)[0] + 1
    category_frames.append(pd.DataFrame({
        "transaction_id": np.repeat(active_transactions, len(items)),
        "product": np.tile(items, len(active_transactions)),
    }))

df = pd.concat(category_frames, ignore_index=True).drop_duplicates(ignore_index=True)
df["customer_id"] = rng.integers(1, 200, size=len(df))
pa = ProductAssociation(df=df, value_col="product", group_col="transaction_id")

top_assoc = pa.df.nlargest(10, "uplift")

# Example 2: Filtered Analysis - Strong Associations Only
pa_filtered = ProductAssociation(
    df=df,
    value_col="product",
    group_col="transaction_id",
    min_occurrences=20,  # Product in at least 20 transactions
    min_cooccurrences=5,  # Pair together at least 5 times
    min_support=0.02,  # In at least 2% of transactions
    min_confidence=0.2,  # 20% conditional probability
    min_uplift=1.2,  # 20% more likely than random
)

# Example 3: Targeted Analysis - What Goes With Milk?
pa_milk = ProductAssociation(
    df=df, value_col="product", group_col="transaction_id", target_item="Milk", min_confidence=0.15
)

# Example 4: Customer-Level vs Transaction-Level Analysis
# Transaction-level: same basket
pa_transaction = ProductAssociation(
    df=df, value_col="product", group_col="transaction_id", target_item="Coffee", min_confidence=0.15
)

# Customer-level: same customer across all purchases
pa_customer = ProductAssociation(
    df=df, value_col="product", group_col="customer_id", target_item="Coffee", min_confidence=0.15
)

# Example 5: Deep Dive - Specific Product Pair
pa_all = ProductAssociation(df=df, value_col="product", group_col="transaction_id")

# Find a specific pair regardless of column order
bread_butter = pa_all.df[
    ((pa_all.df["product_1"] == "Bread") & (pa_all.df["product_2"] == "Butter"))
    | ((pa_all.df["product_1"] == "Butter") & (pa_all.df["product_2"] == "Bread"))
]
