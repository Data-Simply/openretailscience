"""Example script demonstrating OpenRetailScience options configuration.

This script shows the three main approaches to handling column name mismatches:
1. TOML configuration file (recommended for projects)
2. option_context() for temporary overrides
3. Column renaming (simple but modifies data)

Also demonstrates utility functions for inspecting and managing options.
"""

import numpy as np
import pandas as pd

from openretailscience.options import (
    describe_option,
    get_option,
    list_options,
    option_context,
    reset_option,
    set_option,
)

# Create sample data with non-standard column names
rng = np.random.default_rng(42)
df = pd.DataFrame(
    {
        "cust_id": rng.integers(1, 51, size=100),
        "trans_id": range(1, 101),
        "trans_dt": pd.date_range("2024-01-01", periods=100, freq="D"),
        "SKU": rng.integers(1, 21, size=100),
        "qty": rng.integers(1, 6, size=100),
        "revenue": rng.uniform(10, 100, size=100).round(2),
    }
)

# Example 1: Inspecting current configuration
get_option("column.customer_id")  # "customer_id"
describe_option("column.customer_id")
# "column.customer_id: The name of the column containing customer IDs. (current value: customer_id)"
column_opts = [opt for opt in list_options() if opt.startswith("column.")][:10]

# Example 2: option_context() for temporary configuration
get_option("column.customer_id")  # "customer_id" (default)
with option_context(
    "column.customer_id", "cust_id",
    "column.transaction_id", "trans_id",
    "column.transaction_date", "trans_dt",
    "column.product_id", "SKU",
    "column.unit_quantity", "qty",
    "column.unit_spend", "revenue",
):
    get_option("column.customer_id")  # "cust_id" while inside the context
get_option("column.customer_id")  # back to "customer_id" after the context exits

# Example 3: set_option() for global changes, though creating a toml or using option_context() is preferred
set_option("column.customer_id", "cust_id")
get_option("column.customer_id")  # "cust_id" -- affects all subsequent code
reset_option("column.customer_id")
get_option("column.customer_id")  # back to "customer_id"

# Example 4: Column renaming approach
df_renamed = df.rename(
    columns={
        "cust_id": "customer_id",
        "trans_id": "transaction_id",
        "trans_dt": "transaction_date",
        "SKU": "product_id",
        "qty": "unit_quantity",
        "revenue": "unit_spend",
    }
)  # now matches OpenRetailScience default column names

# Example 5: Aggregation and suffix options
get_option("column.agg.customer_id")  # "customers"
get_option("column.agg.unit_spend")  # "spend"
get_option("column.agg.transaction_id")  # "transactions"
get_option("column.suffix.period_1")  # "p1"
get_option("column.suffix.period_2")  # "p2"
get_option("column.suffix.difference")  # "diff"
get_option("column.suffix.percent_difference")  # "pct_diff"
get_option("column.suffix.contribution")  # "contrib"
get_option("column.calc.spend_per_customer")  # "spend_per_customer"
get_option("column.calc.transactions_per_customer")  # "transactions_per_customer"

# Example 6: Nested option_context() calls
with option_context("column.customer_id", "cust_id"):
    get_option("column.customer_id")  # "cust_id"
    with option_context("column.product_id", "SKU"):
        get_option("column.customer_id")  # still "cust_id" from the outer context
        get_option("column.product_id")  # "SKU" from the inner context
    get_option("column.product_id")  # back to "product_id" -- inner context exited
get_option("column.customer_id")  # back to "customer_id" -- outer context exited

# Example 7: Example TOML configuration (place in project root, alongside .git, as
# openretailscience.toml -- OpenRetailScience loads it automatically on import)
#
# [column]
# customer_id = "cust_id"
# transaction_id = "trans_id"
# transaction_date = "trans_dt"
# product_id = "SKU"
# unit_quantity = "qty"
# unit_spend = "revenue"
#
# [column.agg]
# customer_id = "customers"
# unit_spend = "total_revenue"
# transaction_id = "transactions"
#
# [column.suffix]
# percent = "pct"
# difference = "diff"
# period_1 = "ly"  # Last year
# period_2 = "ty"  # This year
