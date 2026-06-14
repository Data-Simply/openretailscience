"""Tests for the time_utils module."""

# flake8: noqa: DTZ001
import datetime
from typing import cast

import ibis
import pandas as pd
import pytest

from openretailscience.options import option_context
from openretailscience.utils.date import filter_and_label_by_periods, find_overlapping_periods

# Mirrors the period_ranges parameter type of filter_and_label_by_periods. Used to annotate
# test dicts so they match the invariant dict value type expected by the function.
PeriodRanges = dict[str, "tuple[datetime.datetime, datetime.datetime] | tuple[str, str]"]


class TestFilterAndLabelByPeriods:
    """Test cases for filtering and labeling periods function."""

    @pytest.fixture
    def sample_transactions_table(self):
        """Fixture to provide a sample transactions table for testing."""
        data = {
            "transaction_id": [1, 2, 3, 4, 5, 6],
            "transaction_date": pd.to_datetime(
                [
                    "2023-01-15",
                    "2023-02-10",
                    "2023-04-05",
                    "2023-05-20",
                    "2023-07-10",
                    "2023-08-25",
                ],
            ),
            "amount": [100.0, 200.0, 150.0, 300.0, 250.0, 175.0],
        }
        df = pd.DataFrame(data)
        return ibis.memtable(df)

    def test_filter_and_label_by_periods_with_datetime_objects(self, sample_transactions_table):
        """Test filtering and labeling periods using datetime objects."""
        # Define expected transaction counts
        q1_count = 2
        q2_count = 2
        total_count = q1_count + q2_count

        period_ranges: PeriodRanges = {
            "Q1": (
                datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2023, 3, 31, tzinfo=datetime.timezone.utc),
            ),
            "Q2": (
                datetime.datetime(2023, 4, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2023, 6, 30, tzinfo=datetime.timezone.utc),
            ),
        }
        result = filter_and_label_by_periods(sample_transactions_table, period_ranges)
        result_df = result.execute()

        # Verify the correct number of transactions
        assert len(result_df) == total_count, f"Should return {total_count} transactions from Q1 and Q2"

        # Verify the period labels are correct
        q1_transactions = result_df[result_df["period_name"] == "Q1"]
        q2_transactions = result_df[result_df["period_name"] == "Q2"]

        assert len(q1_transactions) == q1_count, f"Should have {q1_count} transactions in Q1"
        assert len(q2_transactions) == q2_count, f"Should have {q2_count} transactions in Q2"

        # Verify the transaction IDs are correctly assigned to periods
        assert sorted(q1_transactions["transaction_id"].tolist()) == [1, 2], "Q1 should have transaction IDs 1 and 2"
        assert sorted(q2_transactions["transaction_id"].tolist()) == [3, 4], "Q2 should have transaction IDs 3 and 4"

    def test_filter_and_label_by_periods_with_string_dates(self, sample_transactions_table):
        """Test filtering and labeling periods using string dates."""
        # Define expected transaction counts
        q1_count = 2
        q2_count = 2
        q3_count = 2
        total_count = q1_count + q2_count + q3_count

        period_ranges: PeriodRanges = {
            "Q1": ("2023-01-01", "2023-03-31"),
            "Q2": ("2023-04-01", "2023-06-30"),
            "Q3": ("2023-07-01", "2023-09-30"),
        }

        result = filter_and_label_by_periods(sample_transactions_table, period_ranges)
        result_df = result.execute()

        # Verify the correct number of transactions
        assert len(result_df) == total_count, f"Should return all {total_count} transactions across Q1, Q2, and Q3"

        # Verify the period labels are correct for each quarter
        q1_transactions = result_df[result_df["period_name"] == "Q1"]
        q2_transactions = result_df[result_df["period_name"] == "Q2"]
        q3_transactions = result_df[result_df["period_name"] == "Q3"]

        assert len(q1_transactions) == q1_count, f"Should have {q1_count} transactions in Q1"
        assert len(q2_transactions) == q2_count, f"Should have {q2_count} transactions in Q2"
        assert len(q3_transactions) == q3_count, f"Should have {q3_count} transactions in Q3"

        # Verify specific transaction IDs are in the correct periods
        assert sorted(q1_transactions["transaction_id"].tolist()) == [1, 2], "Q1 should have transaction IDs 1 and 2"
        assert sorted(q2_transactions["transaction_id"].tolist()) == [3, 4], "Q2 should have transaction IDs 3 and 4"
        assert sorted(q3_transactions["transaction_id"].tolist()) == [5, 6], "Q3 should have transaction IDs 5 and 6"

    def test_filter_and_label_by_periods_invalid_period_format(self, sample_transactions_table):
        """Test that invalid period formats raise ValueError."""
        # Single date instead of tuple. Intentionally malformed to exercise validation.
        period_ranges = cast("PeriodRanges", {"Invalid": "2023-01-01"})

        with pytest.raises(ValueError, match="Period 'Invalid' must have a \\(start_date, end_date\\) tuple"):
            filter_and_label_by_periods(sample_transactions_table, period_ranges)

        # Tuple with wrong length. Intentionally malformed to exercise validation.
        period_ranges = cast("PeriodRanges", {"Invalid": ("2023-01-01", "2023-03-31", "extra-value")})

        with pytest.raises(ValueError, match="Period 'Invalid' must have a \\(start_date, end_date\\) tuple"):
            filter_and_label_by_periods(sample_transactions_table, period_ranges)

    def test_filter_and_label_by_periods_no_matches(self, sample_transactions_table):
        """Test behavior when no transactions match the given periods."""
        period_ranges: PeriodRanges = {
            "Future": ("2024-01-01", "2024-12-31"),
        }

        result = filter_and_label_by_periods(sample_transactions_table, period_ranges)
        result_df = result.execute()

        # Should return an empty dataframe since no transactions match the period
        assert len(result_df) == 0, "Should return 0 transactions for future period"

    @pytest.mark.parametrize(
        ("period_ranges", "match"),
        [
            (
                {"Invalid_Q1": ("2023-03-31", "2023-01-01")},
                r"Period 'Invalid_Q1': start date \(2023-03-31\) must be <= end date \(2023-01-01\)",
            ),
            (
                {"Invalid_Q1": (datetime.datetime(2023, 3, 31), datetime.datetime(2023, 1, 1))},
                r"Period 'Invalid_Q1': start date \(2023-03-31 00:00:00\) must be <= end date \(2023-01-01 00:00:00\)",
            ),
            (
                {"Invalid_Mixed": (datetime.datetime(2023, 6, 30), "2023-04-01")},
                r"Period 'Invalid_Mixed': start date \(2023-06-30 00:00:00\) must be <= end date \(2023-04-01\)",
            ),
        ],
        ids=["string_dates", "datetime_objects", "mixed_types"],
    )
    def test_invalid_date_order_raises(self, sample_transactions_table, period_ranges, match):
        """Test that start date > end date raises ValueError for various date types."""
        with pytest.raises(ValueError, match=match):
            filter_and_label_by_periods(sample_transactions_table, period_ranges)

    def test_equal_start_end_dates_valid(self, sample_transactions_table):
        """Test that start date == end date is valid (single day period)."""
        period_ranges: PeriodRanges = {
            "Single_Day": ("2023-01-15", "2023-01-15"),
        }

        result = filter_and_label_by_periods(sample_transactions_table, period_ranges)
        result_df = result.execute()

        assert len(result_df) == 1
        assert result_df.iloc[0]["transaction_id"] == 1

    @pytest.mark.parametrize(
        ("period_ranges", "match"),
        [
            (
                {"Q1": ("2023-01-01", "2023-06-30"), "Q2": ("2023-02-01", "2023-05-31")},
                r"Periods 'Q1' \(2023-01-01-2023-06-30\) and 'Q2' \(2023-02-01-2023-05-31\) overlap",
            ),
            (
                {"Q1": ("2023-01-01", "2023-04-15"), "Q2": ("2023-04-01", "2023-06-30")},
                r"Periods 'Q1' \(2023-01-01-2023-04-15\) and 'Q2' \(2023-04-01-2023-06-30\) overlap",
            ),
            (
                {"Q1": ("2023-01-01", "2023-03-31"), "Q2": ("2023-03-31", "2023-06-30")},
                r"Periods 'Q1' \(2023-01-01-2023-03-31\) and 'Q2' \(2023-03-31-2023-06-30\) overlap",
            ),
            (
                {
                    "Q1": ("2023-01-01", "2023-04-30"),
                    "Q2": ("2023-03-01", "2023-06-30"),
                    "Q3": ("2023-05-01", "2023-08-31"),
                },
                r"Periods 'Q1' \(2023-01-01-2023-04-30\) and 'Q2' \(2023-03-01-2023-06-30\) overlap",
            ),
            (
                {
                    "Period_A": (datetime.datetime(2023, 1, 1), datetime.datetime(2023, 3, 31)),
                    "Period_B": (datetime.datetime(2023, 3, 15), datetime.datetime(2023, 6, 30)),
                },
                r"Periods 'Period_A' \(2023-01-01-2023-03-31\) and 'Period_B' \(2023-03-15-2023-06-30\) overlap",
            ),
        ],
        ids=["complete_overlap", "partial_overlap", "touching_boundaries", "multiple_overlaps", "datetime_objects"],
    )
    def test_overlapping_periods_raises(self, sample_transactions_table, period_ranges, match):
        """Test that overlapping periods raise ValueError."""
        with pytest.raises(ValueError, match=match):
            filter_and_label_by_periods(sample_transactions_table, period_ranges)

    def test_with_custom_column_names(self, sample_transactions_table):
        """Test filter_and_label_by_periods with custom column names."""
        custom_table = sample_transactions_table.mutate(my_date_col=sample_transactions_table.transaction_date).drop(
            "transaction_date",
        )
        period_ranges: PeriodRanges = {"Test_Period": ("2023-01-01", "2023-12-31")}

        with option_context("column.transaction_date", "my_date_col"):
            result = filter_and_label_by_periods(custom_table, period_ranges)
            result_df = result.execute()

            assert isinstance(result_df, pd.DataFrame)
            assert "my_date_col" in result_df.columns

    def test_custom_period_column_name(self, sample_transactions_table):
        """Test filter_and_label_by_periods with a custom period column name."""
        # Define expected transaction counts
        q1_count = 2
        q2_count = 2

        period_ranges: PeriodRanges = {
            "Q1": ("2023-01-01", "2023-03-31"),
            "Q2": ("2023-04-01", "2023-06-30"),
        }

        # Test with custom column name
        result = filter_and_label_by_periods(sample_transactions_table, period_ranges, period_col="quarter")
        result_df = result.execute()

        # Verify the custom column exists
        assert "quarter" in result_df.columns, "Custom period column 'quarter' should exist"
        assert "period_name" not in result_df.columns, "Default 'period_name' column should not exist"

        # Verify the values are correct
        assert len(result_df) == q1_count + q2_count, f"Should return {q1_count + q2_count} transactions from Q1 and Q2"
        q1_transactions = result_df[result_df["quarter"] == "Q1"]
        q2_transactions = result_df[result_df["quarter"] == "Q2"]

        assert len(q1_transactions) == q1_count, f"Should have {q1_count} transactions in Q1"
        assert len(q2_transactions) == q2_count, f"Should have {q2_count} transactions in Q2"


class TestFindOverlappingPeriods:
    """Test cases for the find_overlapping_periods function."""

    def test_valid_range_same_year(self):
        """Test case where the start and end dates are in the same year."""
        start_date = datetime.datetime(2024, 1, 1)
        end_date = datetime.datetime(2024, 12, 31)
        expected = []
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_valid_range_multiple_years(self):
        """Test case where the start and end dates span multiple years."""
        start_date = datetime.datetime(2022, 6, 15)
        end_date = datetime.datetime(2025, 3, 10)
        expected = [
            ("2022-06-15", "2023-03-10"),
            ("2023-06-15", "2024-03-10"),
            ("2024-06-15", "2025-03-10"),
        ]
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_valid_range_full_years(self):
        """Test case with a date range spanning full years."""
        start_date = datetime.datetime(2020, 1, 1)
        end_date = datetime.datetime(2023, 12, 31)
        expected = [
            ("2020-01-01", "2021-12-31"),
            ("2021-01-01", "2022-12-31"),
            ("2022-01-01", "2023-12-31"),
        ]
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_invalid_date_range(self):
        """Test case where the start date is after the end date."""
        start_date = datetime.datetime(2024, 12, 31)
        end_date = datetime.datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="Start date must be before end date"):
            find_overlapping_periods(start_date, end_date)

    def test_same_start_and_end_date(self):
        """Test case where the start and end dates are the same."""
        start_date = datetime.datetime(2024, 6, 1)
        end_date = datetime.datetime(2024, 6, 1)
        expected = []
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_large_date_range(self):
        """Test case with a very large date range spanning many years."""
        start_date = datetime.datetime(2000, 1, 1)
        end_date = datetime.datetime(2025, 12, 31)
        expected = [(f"{year}-01-01", f"{year + 1}-12-31") for year in range(2000, 2025)]
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    @pytest.mark.parametrize(
        ("start_date", "end_date"),
        [
            (datetime.datetime(2021, 5, 10), datetime.datetime(2024, 8, 20)),
            ("2021-05-10", "2024-08-20"),
        ],
        ids=["datetime_input", "string_input"],
    )
    def test_valid_range_multiple_years_return_datetime(self, start_date, end_date):
        """Test return_str=False with datetime and string inputs returns datetime tuples."""
        expected = [
            (datetime.datetime(2021, 5, 10), datetime.datetime(2022, 8, 20)),
            (datetime.datetime(2022, 5, 10), datetime.datetime(2023, 8, 20)),
            (datetime.datetime(2023, 5, 10), datetime.datetime(2024, 8, 20)),
        ]
        result = find_overlapping_periods(start_date, end_date, return_str=False)
        assert result == expected

    @pytest.mark.parametrize(
        ("start_date", "end_date"),
        [
            (datetime.datetime(2021, 3, 1), datetime.datetime(2023, 6, 15)),
            ("2021-03-01", "2023-06-15"),
        ],
        ids=["naive_datetime", "string_input"],
    )
    def test_naive_input_returns_naive_datetimes(self, start_date, end_date):
        """Test that naive datetime and string inputs produce naive datetime outputs."""
        result = find_overlapping_periods(start_date, end_date, return_str=False)
        for start, end in result:
            # return_str=False yields datetime objects; narrow from the str | datetime union.
            assert isinstance(start, datetime.datetime)
            assert isinstance(end, datetime.datetime)
            assert start.tzinfo is None
            assert end.tzinfo is None

    def test_aware_datetime_returns_aware_datetimes(self):
        """Test that tz-aware datetime inputs produce tz-aware datetime outputs."""
        utc = datetime.timezone.utc
        start_date = datetime.datetime(2021, 3, 1, tzinfo=utc)
        end_date = datetime.datetime(2023, 6, 15, tzinfo=utc)
        expected = [
            (datetime.datetime(2021, 3, 1, tzinfo=utc), datetime.datetime(2022, 6, 15, tzinfo=utc)),
            (datetime.datetime(2022, 3, 1, tzinfo=utc), datetime.datetime(2023, 6, 15, tzinfo=utc)),
        ]
        result = find_overlapping_periods(start_date, end_date, return_str=False)
        assert result == expected
        for start, end in result:
            # return_str=False yields datetime objects; narrow from the str | datetime union.
            assert isinstance(start, datetime.datetime)
            assert isinstance(end, datetime.datetime)
            assert start.tzinfo is not None
            assert end.tzinfo is not None

    def test_non_utc_aware_datetime_preserves_timezone(self):
        """Test that non-UTC tz-aware inputs preserve the original timezone in outputs."""
        est = datetime.timezone(datetime.timedelta(hours=-5))
        start_date = datetime.datetime(2021, 3, 1, tzinfo=est)
        end_date = datetime.datetime(2023, 6, 15, tzinfo=est)
        expected = [
            (datetime.datetime(2021, 3, 1, tzinfo=est), datetime.datetime(2022, 6, 15, tzinfo=est)),
            (datetime.datetime(2022, 3, 1, tzinfo=est), datetime.datetime(2023, 6, 15, tzinfo=est)),
        ]
        result = find_overlapping_periods(start_date, end_date, return_str=False)
        assert result == expected

    @pytest.mark.parametrize(
        ("start_date", "end_date"),
        [
            ("2021-03-01", datetime.datetime(2023, 6, 15, tzinfo=datetime.timezone.utc)),
            (datetime.datetime(2021, 3, 1, tzinfo=datetime.timezone.utc), "2023-06-15"),
            (datetime.datetime(2021, 3, 1), datetime.datetime(2023, 6, 15, tzinfo=datetime.timezone.utc)),
            (datetime.datetime(2021, 3, 1, tzinfo=datetime.timezone.utc), datetime.datetime(2023, 6, 15)),
        ],
        ids=["string_start_aware_end", "aware_start_string_end", "naive_start_aware_end", "aware_start_naive_end"],
    )
    def test_mismatched_timezone_awareness_raises(self, start_date, end_date):
        """Test that mismatched timezone awareness between start and end dates raises TypeError."""
        with pytest.raises(TypeError, match="matching timezone awareness"):
            find_overlapping_periods(start_date, end_date)

    @pytest.mark.parametrize(
        ("start_date", "end_date"),
        [
            (datetime.date(2022, 6, 15), datetime.datetime(2024, 8, 20)),
            (datetime.datetime(2022, 6, 15), datetime.date(2024, 8, 20)),
            (datetime.date(2022, 6, 15), datetime.date(2024, 8, 20)),
            (12345, datetime.datetime(2024, 8, 20)),
            (datetime.datetime(2022, 6, 15), 12345),
        ],
        ids=["date_start", "date_end", "both_date", "int_start", "int_end"],
    )
    def test_invalid_type_raises_type_error(self, start_date, end_date):
        """Test that non-str/non-datetime inputs raise TypeError with a clear message."""
        with pytest.raises(TypeError, match="Expected str or datetime"):
            find_overlapping_periods(start_date, end_date)
