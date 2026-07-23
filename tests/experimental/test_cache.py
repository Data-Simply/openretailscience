"""Tests for openretailscience.experimental.cache.

The Spark Connect / Databricks code path needs a live Spark session and is exercised in
``tests/integration/test_cache.py``. These unit tests cover the backend-agnostic dispatch
(passthrough, native delegation, validation) and the Spark Connect detector, all of which
run without Spark.
"""

from __future__ import annotations

import ibis
import ibis.expr.types as ir
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from openretailscience.experimental import cache as cache_module
from openretailscience.experimental.cache import DatabricksCachedTable, cache
from openretailscience.options import ColumnHelper, option_context

cols = ColumnHelper()


@pytest.fixture
def spend_by_store() -> ir.Table:
    """A DuckDB-backed total-spend-per-store aggregation over realistic transaction data.

    Returns:
        ir.Table: An ordered ``store_id`` / spend aggregation bound to a DuckDB backend.
    """
    con = ibis.duckdb.connect()
    pdf = pd.DataFrame(
        {
            cols.store_id: [101, 101, 102, 102, 103],
            cols.unit_spend: [19.99, 5.49, 42.00, 12.50, 8.75],
        }
    )
    transactions = con.create_table("transactions", pdf)
    return (
        transactions.group_by(cols.store_id)
        .aggregate(spend=transactions[cols.unit_spend].sum())
        .order_by(cols.store_id)
    )


class TestCacheDispatch:
    """cache() returns the right handle for each situation, always with correct data."""

    def test_disabled_returns_unchanged_passthrough(self, spend_by_store: ir.Table):
        """With caching off, cache() returns the expression's data unchanged and release() is a no-op."""
        cached = cache(spend_by_store, enabled=False)
        assert isinstance(cached, cache_module._PassthroughCachedTable)
        assert_frame_equal(cached.execute(), spend_by_store.execute())

        cached.release()
        # Passthrough materialized nothing, so the expression is still queryable after release.
        assert_frame_equal(cached.execute(), spend_by_store.execute())

    def test_disabled_via_option(self, spend_by_store: ir.Table):
        """The caching.enabled option gates cache() the same way the per-call flag does."""
        with option_context("caching.enabled", False):
            cached = cache(spend_by_store)
        assert isinstance(cached, cache_module._PassthroughCachedTable)
        assert_frame_equal(cached.execute(), spend_by_store.execute())

    def test_enabled_delegates_to_native_cache(self, spend_by_store: ir.Table):
        """On a non-Spark-Connect backend, cache() delegates to Ibis's native cache, not passthrough."""
        cached = cache(spend_by_store)
        try:
            assert isinstance(cached, ir.CachedTable)
            assert not isinstance(cached, (cache_module._PassthroughCachedTable, DatabricksCachedTable))
            assert_frame_equal(cached.execute(), spend_by_store.execute())
        finally:
            cached.release()

    @pytest.mark.parametrize("bad_expr", [pd.DataFrame({cols.unit_spend: [1.0, 2.0]}), "transactions", 42])
    def test_rejects_non_ibis_expr(self, bad_expr: object):
        """A non-Ibis expression raises TypeError."""
        with pytest.raises(TypeError, match="expr must be an Ibis table"):
            cache(bad_expr)

    def test_rejects_non_bool_enabled(self, spend_by_store: ir.Table):
        """A non-bool enabled override raises TypeError."""
        with pytest.raises(TypeError, match=r"enabled must be a bool or None.*Got str"):
            cache(spend_by_store, enabled="yes")


class _ConnectSession:
    """Stub Spark Connect session (its class module marks it as Spark Connect)."""


class _ClassicSession:
    """Stub classic Spark session."""


_ConnectSession.__module__ = "pyspark.sql.connect.session"
_ClassicSession.__module__ = "pyspark.sql.session"


class _StubBackend:
    """Minimal stand-in for an Ibis backend, exposing only what the detector reads."""

    def __init__(self, name: str, session: object) -> None:
        self.name = name
        self._session = session


class TestSparkConnectDetection:
    """_is_spark_connect only flags PySpark backends backed by a Spark Connect session."""

    @pytest.mark.parametrize(
        ("con", "expected"),
        [
            (_StubBackend("pyspark", _ConnectSession()), True),
            (_StubBackend("pyspark", _ClassicSession()), False),
            (_StubBackend("pyspark", None), False),
            (_StubBackend("duckdb", None), False),
        ],
        ids=["pyspark-connect", "pyspark-classic", "pyspark-no-session", "duckdb"],
    )
    def test_detection(self, con: _StubBackend, expected: bool):
        """Only a PySpark backend with a Spark Connect session is detected."""
        assert cache_module._is_spark_connect(con) is expected
