"""Unified integration tests for CLVStats across multiple database backends.

These exercise the date arithmetic in CLVStats (day-granularity ``delta``, ``cast('date')``
and same-day collapse) on real backends, since date/time handling is where portable SQL
most often diverges between engines (Oracle, SQL Server, BigQuery, Snowflake, PySpark).
"""

import pandas as pd

from openretailscience.experimental.clv import CLVStats
from openretailscience.options import ColumnHelper

cols = ColumnHelper()

_SUBSET_CUSTOMERS = 25


def test_clv_stats_integration(transactions_table):
    """CLVStats produces a correct BTYD summary on each parameterized backend.

    The seeded ``transaction_date`` is a string, so it is cast to a date first. The summary
    is checked against an independent distinct-purchase-days query and the BTYD invariants
    that would break if the date delta ran wrong on a given backend.

    Args:
        transactions_table: Parameterized fixture providing the transactions table on each
            configured database backend.
    """
    table = transactions_table.mutate(**{cols.transaction_date: transactions_table[cols.transaction_date].cast("date")})

    # Restrict to a handful of customers so every customer's history stays complete (a plain
    # row limit could split a customer) while keeping the cross-backend run cheap.
    subset_ids = table.select(cols.customer_id).distinct().order_by(cols.customer_id).limit(_SUBSET_CUSTOMERS)
    subset = table.filter(table[cols.customer_id].isin(subset_ids[cols.customer_id]))

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
    # Age reaches at least to the last purchase, and recency is a real elapsed time.
    assert (result["T"] >= result["recency"]).all()
    assert (result["recency"] >= 0).all()
    # monetary_value is defined exactly for repeat buyers.
    assert (result["monetary_value"].isna() == (result["frequency"] == 0)).all()
