"""Unified integration tests for date utility functions with multiple database backends."""

from datetime import UTC, datetime, timedelta

import ibis
import pandas as pd

from openretailscience.options import ColumnHelper
from openretailscience.utils.date import filter_and_label_by_event_periods, filter_and_label_by_periods

cols = ColumnHelper()
WEEK = timedelta(weeks=1)
DAY = timedelta(days=1)
OPENINGS = pd.DataFrame(
    {cols.store_id: [1, 2, 3], "event_date": pd.to_datetime(["2023-04-03", "2023-06-05", "2023-09-04"])},
)
WINDOWS = {"pre": (-8 * WEEK, -DAY), "post": (0 * WEEK, 8 * WEEK - DAY)}


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


def test_filter_and_label_by_event_periods_matches_duckdb(transactions_table):
    """The backend labels the same transactions with the same windows as DuckDB."""
    expected = _labels(ibis.memtable(pd.read_parquet("data/transactions.parquet")))

    pd.testing.assert_frame_equal(_labels(transactions_table), expected)
