"""Cross-backend integration test for openretailscience.experimental.cache.

cache() must materialize correct results on whatever backend an expression is bound to. This runs
through the shared ``transactions_table`` fixture, so each backend's integration CI
(``pytest tests/integration -k <backend>``) exercises it on a live connection.

Backend-agnostic behavior (dispatch, passthrough, context manager, validation, Spark Connect
detection) is covered by the unit tests; this asserts only what needs a real backend: that the native
cache path returns the source data unchanged and does not silently fall back to the passthrough.

The Spark Connect (Databricks) workaround that cache() selects only on Databricks Runtime is not
reachable from a local classic PySpark session -- cache() takes the native path there -- so it is
validated separately against a live Databricks connection, not with a local stand-in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pandas.testing import assert_frame_equal

from openretailscience.experimental import cache as cache_module
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
    """cache() materializes correct data on a live backend."""

    def test_materializes_source_data(self, transactions_table: Table):
        """cache() returns the source rows and dispatches to a real cache, not the passthrough."""
        expr = _spend_by_store(transactions_table)
        expected = _by_store(expr.to_pandas())
        cached = cache(expr)
        try:
            assert not isinstance(cached, cache_module._PassthroughCachedTable)
            assert_frame_equal(_by_store(cached.to_pandas()), expected)
        finally:
            cached.release()
