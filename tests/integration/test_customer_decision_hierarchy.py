"""Unified integration tests for Customer Decision Hierarchy Analysis with multiple database backends."""

import numpy as np
import pytest

from openretailscience.analysis.customer_decision_hierarchy import CustomerDecisionHierarchy


@pytest.mark.parametrize(
    ("method", "exclude_same_transaction"),
    [
        ("yules_q", True),
        ("yules_q", False),
    ],
)
def test_customer_decision_hierarchy_integration(
    transactions_table,
    method,
    exclude_same_transaction,
):
    """CustomerDecisionHierarchy runs natively against a backend ibis.Table.

    Runs against the parameterized BigQuery/PySpark/Snowflake backends. The table is passed
    directly (no ``.execute()`` to pandas first), so this exercises the native-Ibis pushdown
    path and asserts the result is a valid square distance matrix over the products present.

    Args:
        transactions_table: Parameterized fixture providing a backend transactions table.
        method: Distance method for the analysis.
        exclude_same_transaction: Whether to exclude same-transaction products.
    """
    limited_table = transactions_table.limit(5000)

    customer_decision_hierarchy = CustomerDecisionHierarchy(
        df=limited_table,
        product_col="product_name",
        exclude_same_transaction_products=exclude_same_transaction,
        method=method,
    )

    distances = customer_decision_hierarchy.distances
    n_products = len(customer_decision_hierarchy.products)
    assert distances.shape == (n_products, n_products)
    # A valid distance matrix: symmetric, zero diagonal, values in [0, 1], no NaNs.
    assert not np.isnan(distances).any()
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), np.zeros(n_products))
    assert distances.min() >= 0.0
    assert distances.max() <= 1.0
