"""Example script demonstrating Revenue Tree analysis with openretailscience."""

import numpy as np
import pandas as pd

from openretailscience.analysis.revenue_tree import RevenueTree

rng = np.random.default_rng(42)


def generate_sample_data():
    """Generate sample transaction data for two periods."""
    n_customers_p1 = 100
    n_customers_p2 = 120

    p1_customers = rng.choice(range(1, n_customers_p1 + 1), size=200, replace=True)
    p1_data = {
        "customer_id": p1_customers,
        "transaction_id": range(1, 201),
        "unit_spend": rng.uniform(20, 200, size=200),
        "unit_quantity": rng.integers(1, 10, size=200),
        "period": ["2023-Q1"] * 200,
    }

    # Period 2: more customers, slightly higher spend
    p2_customers = rng.choice(range(1, n_customers_p2 + 1), size=250, replace=True)
    p2_data = {
        "customer_id": p2_customers,
        "transaction_id": range(201, 451),
        "unit_spend": rng.uniform(25, 210, size=250),
        "unit_quantity": rng.integers(1, 12, size=250),
        "period": ["2023-Q2"] * 250,
    }

    df = pd.concat([pd.DataFrame(p1_data), pd.DataFrame(p2_data)], ignore_index=True)
    return df


# Example 1: Basic Period Comparison
df = generate_sample_data()
revenue_tree = RevenueTree(df=df, period_col="period", p1_value="2023-Q1", p2_value="2023-Q2")
# revenue_tree.df has *_p1/_p2/_diff/_pct_diff columns per driver, plus *_contrib columns showing
# each driver's share of the total spend_diff. unit_quantity in the input also unlocks price/volume
# metrics (price_per_unit, price_elasticity, units_per_transaction, frequency_elasticity).

# Example 2: Category-Level Analysis
categories = ["Electronics", "Clothing", "Home & Garden"]
n_trans_by_period = {"2023-Q1": 150, "2023-Q2": 180}
category_df = pd.concat(
    [
        pd.DataFrame({
            "customer_id": rng.integers(1, 100, size=n_trans),
            "unit_spend": rng.uniform(30, 300, size=n_trans),
            "unit_quantity": rng.integers(1, 8, size=n_trans),
            "product_category": rng.choice(categories, size=n_trans),
            "period": period,
        })
        for period, n_trans in n_trans_by_period.items()
    ],
    ignore_index=True,
)
category_df["transaction_id"] = np.arange(1, len(category_df) + 1)

revenue_tree_cat = RevenueTree(
    df=category_df, period_col="period", p1_value="2023-Q1", p2_value="2023-Q2", group_col="product_category"
)

# Example 3: Drawing Revenue Tree Visualization
df_simple = generate_sample_data()
revenue_tree_viz = RevenueTree(df=df_simple, period_col="period", p1_value="2023-Q1", p2_value="2023-Q2")

revenue_tree_viz.draw_tree(
    # value_labels is (current, previous); current is p2_value ("2023-Q2"), previous is p1_value ("2023-Q1").
    value_labels=("Q2 2023", "Q1 2023"),
    unit_spend_label="Revenue",
    customer_id_label="Customers",
    spend_per_customer_label="Revenue / Customer",
    transactions_per_customer_label="Orders / Customer",
    spend_per_transaction_label="Basket Size",
    units_per_transaction_label="Items / Order",
    price_per_unit_label="Price / Item",
)
