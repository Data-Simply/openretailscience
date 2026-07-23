"""PySpark integration tests for openretailscience.experimental.cache.

Local PySpark runs in classic (non-Spark-Connect) mode, so it exercises two things a live
Spark session is needed for: the Spark Connect workaround implementation directly (which
functions on classic PySpark too), and the generic cache() dispatch. They are named so
``pytest -k pyspark`` selects them, matching the PySpark integration CI workflow.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import ibis
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from openretailscience.experimental import cache as cache_module
from openretailscience.experimental.cache import DatabricksCachedTable, cache
from openretailscience.options import ColumnHelper

if TYPE_CHECKING:
    from ibis.backends.pyspark import Backend as PySparkBackend
    from ibis.expr.types import Table

cols = ColumnHelper()

_TRANSACTIONS_VIEW = "transactions"

# A store with two transactions in the fixture data, used to check a cached table stays queryable.
_QUERY_STORE_ID = 102
_QUERY_STORE_SPEND = 54.50


@pytest.fixture(scope="module")
def pyspark_transactions() -> tuple[PySparkBackend, Table]:
    """A local PySpark connection seeded with a small transactions table.

    Returns:
        tuple[PySparkBackend, Table]: The PySpark connection and its transactions table.
    """
    con = ibis.pyspark.connect()
    pdf = pd.DataFrame(
        {
            cols.customer_id: [1, 2, 3, 1, 2, 3],
            cols.store_id: [101, 101, _QUERY_STORE_ID, _QUERY_STORE_ID, 103, 103],
            cols.product_id: [10, 20, 30, 40, 50, 60],
            cols.unit_qty: [1, 2, 1, 3, 2, 1],
            cols.unit_spend: [19.99, 5.49, 42.00, 12.50, 8.75, 30.00],
        }
    )
    con._session.createDataFrame(pdf).createOrReplaceTempView(_TRANSACTIONS_VIEW)
    return con, con.table(_TRANSACTIONS_VIEW)


def _spend_by_store(transactions: Table) -> Table:
    """Total spend per store, ordered by store, for deterministic comparisons.

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
    """Sort rows by store so cached (order-not-guaranteed) and source frames compare equal.

    Args:
        df (pd.DataFrame): A frame containing a store column.

    Returns:
        pd.DataFrame: The frame sorted by store with a reset index.
    """
    return df.sort_values(cols.store_id).reset_index(drop=True)


class TestSparkConnectCacheOnPySpark:
    """The Spark Connect workaround implementation, exercised directly on a live Spark session."""

    def test_returns_same_data_and_is_directly_queryable(
        self,
        pyspark_transactions: tuple[PySparkBackend, Table],
    ):
        """The cached handle IS a table: it materializes the source data and builds further expressions."""
        con, transactions = pyspark_transactions
        expr = _spend_by_store(transactions)
        cached = cache_module._cache_on_spark_connect(con, expr)
        try:
            assert isinstance(cached, DatabricksCachedTable)
            assert_frame_equal(_by_store(cached.execute()), _by_store(expr.execute()))
            store_spend = cached.filter(cached[cols.store_id] == _QUERY_STORE_ID)[cols.agg.unit_spend].sum().to_pandas()
            assert store_spend == pytest.approx(_QUERY_STORE_SPEND)
        finally:
            cached.release()

    def test_release_drops_the_temp_view(
        self,
        pyspark_transactions: tuple[PySparkBackend, Table],
    ):
        """release() removes the backing temporary view from the catalog."""
        con, transactions = pyspark_transactions
        cached = cache_module._cache_on_spark_connect(con, transactions)
        view_name = cached.op().name
        assert con._session.catalog.tableExists(view_name) is True

        cached.release()
        assert con._session.catalog.tableExists(view_name) is False

    def test_context_manager_cleans_up_on_exit(
        self,
        pyspark_transactions: tuple[PySparkBackend, Table],
    ):
        """Used as a context manager, the handle drops its view once the block exits."""
        con, transactions = pyspark_transactions
        expr = _spend_by_store(transactions)

        with cache_module._cache_on_spark_connect(con, expr) as cached:
            view_name = cached.op().name
            assert con._session.catalog.tableExists(view_name) is True
            assert_frame_equal(_by_store(cached.execute()), _by_store(expr.execute()))

        assert con._session.catalog.tableExists(view_name) is False

    def test_finalizer_drops_view_when_handle_is_garbage_collected(
        self,
        pyspark_transactions: tuple[PySparkBackend, Table],
    ):
        """A forgotten cache is cleaned up: the temp view is dropped when the handle is GC'd."""
        con, transactions = pyspark_transactions
        cached = cache_module._cache_on_spark_connect(con, transactions)
        view_name = cached.op().name
        assert con._session.catalog.tableExists(view_name) is True

        del cached
        gc.collect()
        assert con._session.catalog.tableExists(view_name) is False


class TestCacheDispatchOnPySpark:
    """The generic cache() dispatcher against a live PySpark session."""

    def test_uses_native_cache_on_classic_pyspark(
        self,
        pyspark_transactions: tuple[PySparkBackend, Table],
    ):
        """Classic (non-Connect) PySpark takes the native path, not the Databricks workaround."""
        _con, transactions = pyspark_transactions
        expr = _spend_by_store(transactions)
        cached = cache(expr)
        try:
            assert not isinstance(cached, DatabricksCachedTable)
            assert_frame_equal(_by_store(cached.execute()), _by_store(expr.execute()))
        finally:
            cached.release()

    def test_routes_to_workaround_when_spark_connect_detected(
        self,
        pyspark_transactions: tuple[PySparkBackend, Table],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """When the backend is detected as Spark Connect, cache() uses the temp-view workaround.

        Local Spark cannot run in Connect mode, so the Connect *detection* is overridden while the real
        Spark session performs the actual caching.
        """
        con, transactions = pyspark_transactions
        monkeypatch.setattr(cache_module, "_is_spark_connect", lambda _con: True)
        expr = _spend_by_store(transactions)

        cached = cache(expr)
        try:
            assert isinstance(cached, DatabricksCachedTable)
            assert con._session.catalog.tableExists(cached.op().name) is True
            assert_frame_equal(_by_store(cached.execute()), _by_store(expr.execute()))
        finally:
            cached.release()
