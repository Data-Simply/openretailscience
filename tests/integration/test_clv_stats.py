"""Unified integration tests for CLVStats across multiple database backends.

These exercise the date arithmetic in CLVStats (day-granularity ``delta``, ``cast('date')``
and same-day collapse) on real backends, since date/time handling is where portable SQL
most often diverges between engines (Oracle, SQL Server, BigQuery, Snowflake, PySpark).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from openretailscience.experimental.clv import CLVStats
from openretailscience.options import ColumnHelper

if TYPE_CHECKING:
    from ibis.expr.types import Table

cols = ColumnHelper()

_SUBSET_CUSTOMERS = 25
_SAMPLE_CUSTOMERS = 10


def _subset(transactions_table: Table) -> Table:
    """Cast transaction_date to a date and restrict to a handful of complete customer histories.

    A plain row limit could split a customer mid-history, so filter on a distinct-customer subset to
    keep each selected customer whole while keeping the cross-backend run cheap.
    """
    table = transactions_table.mutate(**{cols.transaction_date: transactions_table[cols.transaction_date].cast("date")})
    subset_ids = table.select(cols.customer_id).distinct().order_by(cols.customer_id).limit(_SUBSET_CUSTOMERS)
    return table.filter(table[cols.customer_id].isin(subset_ids[cols.customer_id]))


def test_clv_stats_integration(transactions_table):
    """CLVStats produces a correct BTYD summary on each parameterized backend.

    The seeded ``transaction_date`` is a string, so it is cast to a date first. The summary
    is checked against an independent distinct-purchase-days query and the BTYD invariants
    that would break if the date delta ran wrong on a given backend.

    Args:
        transactions_table: Parameterized fixture providing the transactions table on each
            configured database backend.
    """
    subset = _subset(transactions_table)

    result = CLVStats(subset, period="week").df.set_index(cols.customer_id).sort_index()

    # Independent expected frequency: distinct purchase days per customer minus the birth day.
    expected_frequency = (
        (
            subset.group_by(cols.customer_id)
            .aggregate(purchase_days=subset[cols.transaction_date].nunique())
            .execute()
            .set_index(cols.customer_id)["purchase_days"]
            - 1
        )
        .astype("int64")
        .sort_index()
    )

    assert len(result) == _SUBSET_CUSTOMERS
    pd.testing.assert_series_equal(result["frequency"], expected_frequency, check_names=False)

    # Independent recency/T in weeks. A wrong-unit, wrong-anchor, or wrong-sign date delta on a backend
    # would diverge from these exact expected values.
    span = (
        subset.group_by(cols.customer_id)
        .aggregate(first=subset[cols.transaction_date].min(), last=subset[cols.transaction_date].max())
        .execute()
        .set_index(cols.customer_id)
        .sort_index()
    )
    first_day = pd.to_datetime(span["first"])
    observation_end = pd.Timestamp(subset[cols.transaction_date].max().execute())
    expected_recency = ((pd.to_datetime(span["last"]) - first_day).dt.days / 7).astype("float64")
    expected_t = ((observation_end - first_day).dt.days / 7).astype("float64")
    pd.testing.assert_series_equal(result["recency"], expected_recency, check_names=False)
    pd.testing.assert_series_equal(result["T"], expected_t, check_names=False)
    # monetary_value is defined exactly for repeat buyers.
    assert (result["monetary_value"].isna() == (result["frequency"] == 0)).all()


def test_clv_stats_customer_attributes_and_one_hot_integration(transactions_table):
    """A customer_attributes join and one_hot_col produce correct per-customer covariates on each backend.

    one_hot emits a ``CASE WHEN`` per dummy; a boolean-as-value form (``COALESCE(col = v, ...)``) is
    invalid SQL on Oracle and SQL Server, so this guards that the encoding stays portable. The attributes
    table is built on the same backend so the left join runs in-database.

    Args:
        transactions_table: Parameterized fixture providing the transactions table on each backend.
    """
    subset = _subset(transactions_table)

    # Build the per-customer attributes on the same backend (one row per customer): a numeric covariate
    # joined as-is, plus a categorical reduced to its per-customer max for one-hot encoding.
    customer_attributes = subset.group_by(cols.customer_id).aggregate(
        distinct_stores=subset[cols.store_id].nunique(),
        store_id=subset[cols.store_id].max(),
    )
    clv = CLVStats(subset, period="week", customer_attributes=customer_attributes, one_hot_col=cols.store_id)
    result = clv.df.set_index(cols.customer_id).sort_index()

    # customer_attributes join: distinct stores per customer, computed independently on the backend.
    expected_stores = (
        subset.group_by(cols.customer_id)
        .aggregate(n=subset[cols.store_id].nunique())
        .execute()
        .set_index(cols.customer_id)["n"]
        .astype("int64")
        .sort_index()
    )
    pd.testing.assert_series_equal(result["distinct_stores"], expected_stores, check_names=False)

    # one_hot: the store_id attribute (its per-customer max) is encoded. Each store_id_<v> dummy must
    # equal (max_store == v) as 0/1 int8, with exactly one level dropped as the reference.
    max_store = (
        subset.group_by(cols.customer_id)
        .aggregate(m=subset[cols.store_id].max())
        .execute()
        .set_index(cols.customer_id)["m"]
        .sort_index()
    )
    store_dummies = [c for c in clv.covariate_cols if c.startswith(f"{cols.store_id}_")]
    assert len(store_dummies) == max_store.nunique() - 1  # one reference level dropped
    for dummy in store_dummies:
        value = int(dummy[len(cols.store_id) + 1 :])
        pd.testing.assert_series_equal(result[dummy], (max_store == value).astype("int8"), check_names=False)
    # A row flags at most one dummy (its max store); reference-level customers flag none.
    assert (result[store_dummies].sum(axis=1) <= 1).all()


def test_clv_stats_sample_integration(transactions_table):
    """CLVStats.sample draws a deterministic customer subset on each backend.

    The sampling key compiles to a different hash function per engine (``ORA_HASH``, ``CHECKSUM``,
    ``FARM_FINGERPRINT``, ``HASH``), and the ``frac`` path adds a portable ``ABS(MOD(...))``, so this
    guards that selection stays valid SQL and stays stable across executions everywhere.

    Args:
        transactions_table: Parameterized fixture providing the transactions table on each backend.
    """
    clv = CLVStats(_subset(transactions_table), period="week")
    all_customers = set(clv.df[cols.customer_id])

    sampled = clv.sample(n=_SAMPLE_CUSTOMERS)
    sampled_ids = set(sampled.df[cols.customer_id])

    assert len(sampled.df) == _SAMPLE_CUSTOMERS
    assert sampled_ids <= all_customers
    # Same random_state, same customers: a re-executed random() predicate would not hold here.
    assert set(clv.sample(n=_SAMPLE_CUSTOMERS).df[cols.customer_id]) == sampled_ids
    assert set(clv.sample(n=_SUBSET_CUSTOMERS * 2).df[cols.customer_id]) == all_customers
    # frac=1.0 exercises the modulo/abs predicate (rather than the order-by-hash path) portably.
    assert set(clv.sample(frac=1.0).df[cols.customer_id]) == all_customers
