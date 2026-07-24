"""Cross-backend integration tests for openretailscience.experimental.cache.

cache() must return correct results on whatever backend an expression is bound to. These run
through the shared ``transactions_table`` fixture, so each backend's integration CI
(``pytest tests/integration -k <backend>``) exercises them on a live connection.

The Spark Connect (Databricks) workaround that cache() selects only on Databricks Runtime is not
reachable from a local classic PySpark session (``_is_spark_connect`` is False there, so cache()
takes the native path). It is validated separately against a live Databricks connection, not with a
local stand-in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pandas.testing import assert_frame_equal

from openretailscience.experimental.cache import cache
from openretailscience.options import ColumnHelper

if TYPE_CHECKING:
    import pandas as pd
    from ibis.expr.types import Table

cols = ColumnHelper()


def _spend_by_store(transactions: Table) -> Table:
    """Total spend per store, ordered by store.

    Args:
        transactions (Table): The transactions table.

    Returns:
        Table: A ``store_id`` / total-spend aggregation ordered by ``store_id``.
    """
    return (
        transactions.group_by(cols.store_id)
        .aggregate(**{cols.agg.unit_spend: transactions[cols.unit_spend].sum()})
        .order_by(cols.store_id)
    )


def _by_store(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by store so a cached (order-not-guaranteed) frame compares equal to its source.

    Args:
        df (pd.DataFrame): A frame containing a store column.

    Returns:
        pd.DataFrame: The frame sorted by store with a reset index.
    """
    return df.sort_values(cols.store_id).reset_index(drop=True)


class TestCache:
    """cache() on a live backend: correct data on the native path, transparent no-op when disabled."""

    def test_materializes_source_data(self, transactions_table: Table):
        """A cached expression returns the same rows as the uncached expression."""
        expr = _spend_by_store(transactions_table)
        expected = _by_store(expr.to_pandas())
        cached = cache(expr)
        try:
            assert_frame_equal(_by_store(cached.to_pandas()), expected)
        finally:
            cached.release()

    def test_context_manager_yields_source_data(self, transactions_table: Table):
        """Used as a context manager, cache() yields the source data and releases on exit."""
        expr = _spend_by_store(transactions_table)
        expected = _by_store(expr.to_pandas())
        with cache(expr) as cached:
            assert_frame_equal(_by_store(cached.to_pandas()), expected)

    def test_disabled_returns_source_unchanged(self, transactions_table: Table):
        """With caching disabled, cache() returns the expression's data unchanged."""
        expr = _spend_by_store(transactions_table)
        cached = cache(expr, enabled=False)
        assert_frame_equal(_by_store(cached.to_pandas()), _by_store(expr.to_pandas()))
