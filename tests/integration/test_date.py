"""Unified integration tests for date utility functions with multiple database backends."""

from datetime import UTC, date, datetime, timedelta

import ibis
import pandas as pd
import pytest

from openretailscience.options import ColumnHelper
from openretailscience.utils.date import (
    BEFORE_EVENT,
    filter_and_label_by_event_periods,
    filter_and_label_by_event_to_date,
    filter_and_label_by_periods,
)

cols = ColumnHelper()
WEEK = timedelta(weeks=1)
DAY = timedelta(days=1)
OPENINGS = pd.DataFrame(
    {cols.store_id: [1, 2, 3], "event_date": pd.to_datetime(["2023-04-03", "2023-06-05", "2023-09-04"])},
)
WINDOWS = {"pre": (-8 * WEEK, -DAY), "post": (0 * WEEK, 8 * WEEK - DAY)}
LATE_OPENINGS = pd.DataFrame(
    {cols.store_id: [3, 4], "event_date": pd.to_datetime(["2023-11-20", "2023-12-04"])},
)
# _labels_to_date removes this store's sales before it opened, so require_every_window drops it
STORE_WITHOUT_PRE, ITS_OPENING = 4, date(2023, 12, 4)


def test_filter_and_label_by_periods_integration(transactions_table):
    """Integration test for filter_and_label_by_periods using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
    """
    limited_table = transactions_table.limit(1000)
    period_ranges = {
        "Q1": (datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 3, 31, tzinfo=UTC)),
        "Q2": (datetime(2023, 4, 1, tzinfo=UTC), datetime(2023, 6, 30, tzinfo=UTC)),
    }

    result = filter_and_label_by_periods(limited_table, period_ranges)
    assert result is not None

    df = result.execute()
    assert df is not None


def _labels(transactions: ibis.Table) -> pd.DataFrame:
    labelled = filter_and_label_by_event_periods(transactions, events=OPENINGS, on=cols.store_id, windows=WINDOWS)
    return labelled.select(cols.transaction_id, "period_name").order_by(cols.transaction_id).to_pandas()


def test_filter_and_label_by_event_periods_matches_duckdb(transactions_table, duckdb_transactions_table):
    """The backend labels the same transactions with the same windows as DuckDB."""
    expected = _labels(duckdb_transactions_table)

    pd.testing.assert_frame_equal(_labels(transactions_table), expected)


def _labels_to_date(transactions: ibis.Table, require_every_window: bool) -> pd.DataFrame:
    no_pre = (transactions[cols.store_id] != STORE_WITHOUT_PRE) | (
        transactions[cols.transaction_date].cast("date") >= ITS_OPENING
    )
    labelled = filter_and_label_by_event_to_date(
        transactions.filter(no_pre),
        events=LATE_OPENINGS,
        on=cols.store_id,
        end_date="2023-12-30",
        windows={"post": 0 * WEEK, "pre": BEFORE_EVENT, "post_pq": -13 * WEEK},
        include_duration=True,
        require_every_window=require_every_window,
    )
    return labelled.select(cols.transaction_id, "period_name", "period_days").order_by(cols.transaction_id).to_pandas()


@pytest.mark.parametrize("require_every_window", [False, True])
def test_filter_and_label_by_event_to_date_matches_duckdb(
    transactions_table, duckdb_transactions_table, require_every_window
):
    """The backend labels the same transactions, with the same period lengths, as DuckDB."""
    expected = _labels_to_date(duckdb_transactions_table, require_every_window)

    pd.testing.assert_frame_equal(_labels_to_date(transactions_table, require_every_window), expected)
