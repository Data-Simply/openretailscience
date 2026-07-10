"""Example script demonstrating Customer Decision Hierarchy analysis with openretailscience."""

import numpy as np
import pandas as pd

from openretailscience.analysis.customer_decision_hierarchy import CustomerDecisionHierarchy

# Generate reproducible sample data
rng = np.random.default_rng(42)


# Example 1: Basic Substitutability Analysis - Beverages
beverages = {
    "sodas": ["Coke", "Pepsi", "RC Cola"],
    "diet_sodas": ["Diet Coke", "Diet Pepsi"],
    "juices": ["Orange Juice", "Apple Juice", "Cranberry Juice"],
    "water": ["Spring Water", "Sparkling Water"],
}

# Each customer makes 3-8 purchases
customer_ids = np.arange(1, 101)
n_transactions = rng.integers(3, 9, size=customer_ids.size)

# Customers typically stick to one category but may try others
primary_category = np.select(
    [customer_ids <= 40, customer_ids <= 60, customer_ids <= 80],
    ["sodas", "juices", "diet_sodas"],
    default="water",
)
secondary_category = np.select(
    [customer_ids <= 40, customer_ids <= 60, customer_ids <= 80],
    ["diet_sodas", "water", "sodas"],
    default="juices",
)

customer_id_col = np.repeat(customer_ids, n_transactions)
transaction_id_col = np.arange(1, customer_id_col.size + 1)
primary_col = np.repeat(primary_category, n_transactions)
secondary_col = np.repeat(secondary_category, n_transactions)

# Pick from primary category 70% of time
use_primary = rng.random(customer_id_col.size) < 0.7
category_col = np.where(use_primary, primary_col, secondary_col)

# Pick a product from that category
product_col = np.empty(customer_id_col.size, dtype=object)
for category, items in beverages.items():
    mask = category_col == category
    product_col[mask] = rng.choice(items, size=mask.sum())

beverage_df = pd.DataFrame({"customer_id": customer_id_col, "transaction_id": transaction_id_col, "product": product_col})
cdh = CustomerDecisionHierarchy(df=beverage_df, product_col="product", exclude_same_transaction_products=True)

# Example 2: Plotting Vertical Dendrogram
cdh.plot(title="Beverage Product Substitutability Hierarchy", orientation="top", x_label="Products", y_label="Distance")

# Example 3: Horizontal Dendrogram Layout
cdh.plot(title="Product Hierarchy", orientation="left", x_label="Distance", y_label="Beverages")

# Example 4: Yogurt Analysis - Flavors vs Types
# Each customer prefers 1-2 of the 3 types (switching flavors freely within them), and
# sometimes buys a second preferred type in the same transaction.
yogurt_flavors = ["Strawberry", "Vanilla", "Blueberry", "Peach"]
yogurt_types = np.array(["Regular", "Greek", "Kids"])

n_yogurt_customers = 80
n_txn = rng.integers(4, 10, size=n_yogurt_customers)
yogurt_customer_id_col = np.repeat(np.arange(1, n_yogurt_customers + 1), n_txn)
yogurt_transaction_id_col = np.arange(1, yogurt_customer_id_col.size + 1)

# Rank the 3 types randomly per customer and keep the top 2 as that customer's preferred
# slots; num_preferred decides whether only slot 0 is ever used (1 preferred type) or both.
num_preferred = rng.integers(1, 3, size=n_yogurt_customers)
preferred_slots = yogurt_types[np.argsort(rng.random((n_yogurt_customers, yogurt_types.size)), axis=1)[:, :2]]

num_preferred_txn = np.repeat(num_preferred, n_txn)
slots_txn = np.repeat(preferred_slots, n_txn, axis=0)
slot_idx = rng.integers(0, num_preferred_txn)
rows = np.arange(yogurt_customer_id_col.size)
yogurt_type_col = slots_txn[rows, slot_idx]
flavor_col = rng.choice(yogurt_flavors, size=yogurt_customer_id_col.size)

yogurt_df = pd.DataFrame({
    "customer_id": yogurt_customer_id_col,
    "transaction_id": yogurt_transaction_id_col,
    "product": yogurt_type_col + " " + flavor_col,
})

# Sometimes buy the other preferred type in the same transaction (only possible with 2)
extra_mask = (rng.random(yogurt_customer_id_col.size) < 0.3) & (num_preferred_txn > 1)
other_type_col = slots_txn[rows, 1 - slot_idx]
other_flavor_col = rng.choice(yogurt_flavors, size=yogurt_customer_id_col.size)
yogurt_extra_df = pd.DataFrame({
    "customer_id": yogurt_customer_id_col[extra_mask],
    "transaction_id": yogurt_transaction_id_col[extra_mask],
    "product": other_type_col[extra_mask] + " " + other_flavor_col[extra_mask],
})
yogurt_df = pd.concat([yogurt_df, yogurt_extra_df], ignore_index=True)
cdh_yogurt = CustomerDecisionHierarchy(
    df=yogurt_df, product_col="product", exclude_same_transaction_products=True, random_state=42
)

# Example 5: Comparing With and Without Transaction Exclusion
# Uses yogurt_df, not beverage_df: yogurt_df has multi-item transactions (Example 4's
# extra_mask rows), so this parameter actually changes the result here. Every beverage_df
# transaction has exactly one product, so the flag would be a no-op on that data.
cdh_no_exclusion = CustomerDecisionHierarchy(
    df=yogurt_df, product_col="product", exclude_same_transaction_products=False, random_state=42
)

# Example 6: Access Distance Matrix for Custom Analysis
# Access the distance matrix directly
distance_matrix = cdh.distances
products = cdh.pairs_df["product"].cat.categories
