"""Unified integration tests for date utility functions with multiple database backends."""

from datetime import datetime, timezone

from openretailscience.utils.date import filter_and_label_by_periods

# Mirrors the period_ranges parameter type of filter_and_label_by_periods, matching the
# invariant dict value type the function expects.
PeriodRanges = dict[str, "tuple[datetime, datetime] | tuple[str, str]"]


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
    period_ranges: PeriodRanges = {
        "Q1": (datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 3, 31, tzinfo=timezone.utc)),
        "Q2": (datetime(2023, 4, 1, tzinfo=timezone.utc), datetime(2023, 6, 30, tzinfo=timezone.utc)),
    }

    result = filter_and_label_by_periods(limited_table, period_ranges)
    assert result is not None

    df = result.execute()
    assert df is not None
