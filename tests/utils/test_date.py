"""Tests for the time_utils module."""

# flake8: noqa: DTZ001
import datetime as dt
from zoneinfo import ZoneInfo

import ibis
import numpy as np
import pandas as pd
import pytest

from openretailscience.options import ColumnHelper, option_context
from openretailscience.utils.date import (
    _MAX_BAD_EVENT_ROWS,
    BEFORE_EVENT,
    filter_and_label_by_event_periods,
    filter_and_label_by_event_to_date,
    filter_and_label_by_periods,
    find_overlapping_periods,
)

cols = ColumnHelper()
WEEK = dt.timedelta(weeks=1)
DAY = dt.timedelta(days=1)


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

    @pytest.mark.parametrize(
        ("to_input", "period_ranges", "expected_ids", "expected_periods"),
        [
            (
                lambda table: table,
                {
                    "Q1": ("2023-01-01", "2023-03-31"),
                    "Q2": ("2023-04-01", "2023-06-30"),
                    "Q3": ("2023-07-01", "2023-09-30"),
                },
                [1, 2, 3, 4, 5, 6],
                ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3"],
            ),
            (
                lambda table: table.to_pandas(),
                {
                    "Q1": ("2023-01-01", "2023-03-31"),
                    "Q2": ("2023-04-01", "2023-06-30"),
                    "Q3": ("2023-07-01", "2023-09-30"),
                },
                [1, 2, 3, 4, 5, 6],
                ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3"],
            ),
            (
                lambda table: table,
                {
                    "Q1": (dt.datetime(2023, 1, 1, tzinfo=dt.UTC), dt.datetime(2023, 3, 31, tzinfo=dt.UTC)),
                    "Q2": (dt.datetime(2023, 4, 1, tzinfo=dt.UTC), dt.datetime(2023, 6, 30, tzinfo=dt.UTC)),
                },
                [1, 2, 3, 4],
                ["Q1", "Q1", "Q2", "Q2"],
            ),
        ],
        ids=["ibis_string_dates", "pandas_string_dates", "datetime_bounds"],
    )
    def test_labels_each_transaction_with_its_period(
        self,
        sample_transactions_table,
        to_input,
        period_ranges,
        expected_ids,
        expected_periods,
    ):
        """Transactions inside a period get its name in period_col; the rest are dropped."""
        result = filter_and_label_by_periods(to_input(sample_transactions_table), period_ranges, period_col="quarter")

        pd.testing.assert_frame_equal(
            result.select(cols.transaction_id, "quarter").order_by(cols.transaction_id).to_pandas(),
            pd.DataFrame({cols.transaction_id: expected_ids, "quarter": expected_periods}),
        )

    @pytest.mark.parametrize(
        ("to_input", "period_ranges", "error", "match"),
        [
            (
                lambda table: table,
                {"Invalid": "2023-01-01"},
                ValueError,
                r"Period 'Invalid' must have a \(start_date, end_date\) tuple",
            ),
            (
                lambda table: table,
                {"Invalid": ("2023-01-01", "2023-03-31", "extra-value")},
                ValueError,
                r"Period 'Invalid' must have a \(start_date, end_date\) tuple",
            ),
            (lambda table: table, {}, ValueError, "period_ranges must contain at least one period"),
            (
                lambda table: table,
                [("2023-01-01", "2023-03-31")],
                TypeError,
                r"period_ranges must be a mapping of period names to \(start_date, end_date\) tuples\. Got list\.",
            ),
            (
                lambda table: table.drop(cols.transaction_date),
                {"Q1": ("2023-01-01", "2023-03-31")},
                ValueError,
                rf"missing required columns: \['{cols.transaction_date}'\]",
            ),
            (
                lambda table: [100.0, 200.0, 150.0],
                {"Q1": ("2023-01-01", "2023-03-31")},
                TypeError,
                "transactions must be either a pandas DataFrame or an Ibis Table",
            ),
            (
                lambda table: table,
                {"Q1": ("2023/01/01", "2023-03-31")},
                ValueError,
                r"^Period 'Q1' start '2023/01/01' is not a valid 'YYYY-MM-DD' date\.",
            ),
            (
                lambda table: table,
                {"Q1": ("2023-01-01", "2023-02-30")},
                ValueError,
                r"^Period 'Q1' end '2023-02-30' is not a valid 'YYYY-MM-DD' date\.",
            ),
            (
                lambda table: table.mutate(period_name=ibis.literal("Q0")),
                {"Q1": ("2023-01-01", "2023-03-31")},
                ValueError,
                r"^period_col 'period_name' is already a column in transactions\.",
            ),
            (
                lambda table: table,
                {"Q1": (20230101, "2023-03-31")},
                TypeError,
                "Expected str or datetime, got <class 'int'>",
            ),
        ],
        ids=[
            "single_date",
            "three_values",
            "empty",
            "not_a_mapping",
            "missing_date_column",
            "list_transactions",
            "malformed_start",
            "impossible_end",
            "period_col_exists",
            "integer_bound",
        ],
    )
    def test_invalid_input_raises(self, sample_transactions_table, to_input, period_ranges, error, match):
        """Malformed periods or transactions raise a specific error naming what is wrong."""
        with pytest.raises(error, match=match):
            filter_and_label_by_periods(to_input(sample_transactions_table), period_ranges)

    @pytest.mark.parametrize(
        ("period_ranges", "match"),
        [
            (
                {"Invalid_Q1": ("2023-03-31", "2023-01-01")},
                r"Period 'Invalid_Q1': start date \(2023-03-31\) must be <= end date \(2023-01-01\)",
            ),
            (
                {"Invalid_Q1": (dt.datetime(2023, 3, 31), dt.datetime(2023, 1, 1))},
                r"Period 'Invalid_Q1': start date \(2023-03-31 00:00:00\) must be <= end date \(2023-01-01 00:00:00\)",
            ),
            (
                {"Invalid_Mixed": (dt.datetime(2023, 6, 30), "2023-04-01")},
                r"Period 'Invalid_Mixed': start date \(2023-06-30 00:00:00\) must be <= end date \(2023-04-01\)",
            ),
            (
                {"Invalid_Q1": (dt.date(2023, 3, 31), dt.date(2023, 1, 1))},
                r"Period 'Invalid_Q1': start date \(2023-03-31\) must be <= end date \(2023-01-01\)",
            ),
        ],
        ids=["string_dates", "datetime_objects", "mixed_types", "date_objects"],
    )
    def test_invalid_date_order_raises(self, sample_transactions_table, period_ranges, match):
        """Test that start date > end date raises ValueError for various date types."""
        with pytest.raises(ValueError, match=match):
            filter_and_label_by_periods(sample_transactions_table, period_ranges)

    def test_equal_start_end_dates_valid(self, sample_transactions_table):
        """Test that start date == end date is valid (single day period)."""
        period_ranges = {
            "Single_Day": ("2023-01-15", "2023-01-15"),
        }

        result = filter_and_label_by_periods(sample_transactions_table, period_ranges)
        result_df = result.execute()

        assert len(result_df) == 1
        assert result_df.iloc[0]["transaction_id"] == 1

    @pytest.mark.parametrize(
        ("q1_start", "q1_end", "transaction_dates"),
        [
            ("2023-01-01", "2023-03-31", pd.to_datetime(["2023-03-31 00:00", "2023-03-31 18:45", "2023-04-01 09:00"])),
            ("2023-01-01", "2023-3-31", pd.to_datetime(["2023-03-31 00:00", "2023-03-31 18:45", "2023-04-01 09:00"])),
            ("2023-3-31", "2023-3-31", pd.to_datetime(["2023-03-31 00:00", "2023-03-31 18:45", "2023-04-01 09:00"])),
            (
                dt.date(2023, 1, 1),
                dt.date(2023, 3, 31),
                pd.to_datetime(["2023-03-31 00:00", "2023-03-31 18:45", "2023-04-01 00:00"]),
            ),
            (
                dt.date(2023, 1, 1),
                dt.date(2023, 3, 31),
                [dt.date(2023, 3, 31), dt.date(2023, 3, 31), dt.date(2023, 4, 1)],
            ),
            ("2023-1-1", "2023-3-31", [dt.date(2023, 3, 31), dt.date(2023, 3, 31), dt.date(2023, 4, 1)]),
        ],
        ids=["padded", "unpadded_end", "unpadded_start", "date_objects", "date_column", "unpadded_on_date_column"],
    )
    def test_date_only_bounds_cover_whole_days(self, q1_start, q1_end, transaction_dates):
        """Date-only bounds (strings, padded or not, or date objects) keep every sale on the end day."""
        transactions = ibis.memtable(
            pd.DataFrame({cols.transaction_id: [101, 102, 103], cols.transaction_date: transaction_dates}),
        )

        result = filter_and_label_by_periods(transactions, {"Q1": (q1_start, q1_end)}).execute()

        assert sorted(result[cols.transaction_id]) == [101, 102]

    def test_aware_bounds_filter_on_wall_clock(self):
        """A timezone-aware bound is read as wall-clock time on the naive column; its offset is ignored."""
        transactions = pd.DataFrame(
            {
                cols.transaction_id: [101, 102, 103],
                cols.transaction_date: pd.to_datetime(["2023-03-31 21:00", "2023-03-31 23:00", "2023-04-01 03:00"]),
            },
        )
        tokyo = ZoneInfo("Asia/Tokyo")
        late_night = (dt.datetime(2023, 3, 31, 22, tzinfo=tokyo), dt.datetime(2023, 4, 1, 2, tzinfo=tokyo))

        result = filter_and_label_by_periods(transactions, {"Late night": late_night}).execute()

        assert result[cols.transaction_id].tolist() == [102]

    @pytest.mark.parametrize(
        ("transaction_dates", "period_ranges"),
        [
            (
                pd.to_datetime(["2023-03-31 09:00", "2023-03-31 18:45"]),
                {"Q1": ("2023-01-01", "2023-03-31"), "Q2": (dt.datetime(2023, 3, 31, 12), "2023-06-30")},
            ),
            (
                [dt.date(2023, 3, 31), dt.date(2023, 4, 1)],
                {
                    "Q1": (dt.datetime(2023, 1, 1), dt.datetime(2023, 3, 31, 11)),
                    "Q2": (dt.datetime(2023, 3, 31, 12), dt.datetime(2023, 6, 30)),
                },
            ),
            (
                # 22:30 in New York is already Apr 1 in UTC, but the naive column holds wall-clock times
                pd.to_datetime(["2023-03-31 09:00", "2023-03-31 23:00"]),
                {
                    "Q1": ("2023-01-01", "2023-03-31"),
                    "Q2": (dt.datetime(2023, 3, 31, 22, 30, tzinfo=ZoneInfo("America/New_York")), "2023-06-30"),
                },
            ),
        ],
        ids=["string_end_on_timestamp_column", "datetimes_on_date_column", "aware_start_on_wall_clock"],
    )
    def test_bounds_sharing_a_day_overlap(self, transaction_dates, period_ranges):
        """Periods sharing a day overlap, whether via a whole-day end, a date column, or an aware wall-clock bound."""
        transactions = ibis.memtable(
            pd.DataFrame({cols.transaction_id: [101, 102], cols.transaction_date: transaction_dates}),
        )

        match = r"Periods 'Q1' \[2023-01-01, 2023-03-31\] and 'Q2' \[2023-03-31, .*overlap"
        with pytest.raises(ValueError, match=match):
            filter_and_label_by_periods(transactions, period_ranges)

    @pytest.mark.parametrize(
        ("period_ranges", "match"),
        [
            (
                {"Q1": ("2023-01-01", "2023-06-30"), "Q2": ("2023-02-01", "2023-05-31")},
                r"Periods 'Q1' \[2023-01-01, 2023-06-30\] and 'Q2' \[2023-02-01, 2023-05-31\] overlap",
            ),
            (
                {"Q1": ("2023-01-01", "2023-04-15"), "Q2": ("2023-04-01", "2023-06-30")},
                r"Periods 'Q1' \[2023-01-01, 2023-04-15\] and 'Q2' \[2023-04-01, 2023-06-30\] overlap",
            ),
            (
                {"Q1": ("2023-01-01", "2023-03-31"), "Q2": ("2023-03-31", "2023-06-30")},
                (
                    r"Periods 'Q1' \[2023-01-01, 2023-03-31\] and 'Q2' \[2023-03-31, 2023-06-30\] overlap\. "
                    r"Both ends are included, so end 'Q1' before 'Q2' starts\.$"
                ),
            ),
            (
                {
                    "Q1": ("2023-01-01", "2023-04-30"),
                    "Q2": ("2023-03-01", "2023-06-30"),
                    "Q3": ("2023-05-01", "2023-08-31"),
                },
                r"Periods 'Q1' \[2023-01-01, 2023-04-30\] and 'Q2' \[2023-03-01, 2023-06-30\] overlap",
            ),
            (
                {
                    "Period_A": (dt.datetime(2023, 1, 1), dt.datetime(2023, 3, 31)),
                    "Period_B": (dt.datetime(2023, 3, 15), dt.datetime(2023, 6, 30)),
                },
                r"Periods 'Period_A' \[2023-01-01, 2023-03-31\] and 'Period_B' \[2023-03-15, 2023-06-30\] overlap",
            ),
            (
                {
                    "Period_A": (dt.datetime(2023, 1, 1), dt.datetime(2023, 3, 31)),
                    "Period_B": (dt.datetime(2023, 3, 31), dt.datetime(2023, 6, 30)),
                },
                r"overlap\. Both ends are included, so end 'Period_A' before 'Period_B' starts\.$",
            ),
        ],
        ids=[
            "complete_overlap",
            "partial_overlap",
            "touching_boundaries",
            "multiple_overlaps",
            "datetime_objects",
            "datetimes_sharing_an_instant",
        ],
    )
    def test_overlapping_periods_raises(self, sample_transactions_table, period_ranges, match):
        """Test that overlapping periods raise ValueError."""
        with pytest.raises(ValueError, match=match):
            filter_and_label_by_periods(sample_transactions_table, period_ranges)

    def test_with_custom_column_names(self, sample_transactions_table):
        """The date column comes from the column.transaction_date option."""
        custom_table = sample_transactions_table.mutate(my_date_col=sample_transactions_table.transaction_date).drop(
            "transaction_date",
        )
        with option_context("column.transaction_date", "my_date_col"):
            result = filter_and_label_by_periods(custom_table, {"H1": ("2023-01-01", "2023-06-30")})

        pd.testing.assert_frame_equal(
            result.select(cols.transaction_id, "period_name").order_by(cols.transaction_id).to_pandas(),
            pd.DataFrame({cols.transaction_id: [1, 2, 3, 4], "period_name": ["H1", "H1", "H1", "H1"]}),
        )


class TestFindOverlappingPeriods:
    """Test cases for the find_overlapping_periods function."""

    def test_valid_range_same_year(self):
        """Test case where the start and end dates are in the same year."""
        start_date = dt.datetime(2024, 1, 1)
        end_date = dt.datetime(2024, 12, 31)
        expected = []
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_valid_range_multiple_years(self):
        """Test case where the start and end dates span multiple years."""
        start_date = dt.datetime(2022, 6, 15)
        end_date = dt.datetime(2025, 3, 10)
        expected = [
            ("2022-06-15", "2023-03-10"),
            ("2023-06-15", "2024-03-10"),
            ("2024-06-15", "2025-03-10"),
        ]
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_valid_range_full_years(self):
        """Test case with a date range spanning full years."""
        start_date = dt.datetime(2020, 1, 1)
        end_date = dt.datetime(2023, 12, 31)
        expected = [
            ("2020-01-01", "2021-12-31"),
            ("2021-01-01", "2022-12-31"),
            ("2022-01-01", "2023-12-31"),
        ]
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_invalid_date_range(self):
        """Test case where the start date is after the end date."""
        start_date = dt.datetime(2024, 12, 31)
        end_date = dt.datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="Start date must be before end date"):
            find_overlapping_periods(start_date, end_date)

    def test_same_start_and_end_date(self):
        """Test case where the start and end dates are the same."""
        start_date = dt.datetime(2024, 6, 1)
        end_date = dt.datetime(2024, 6, 1)
        expected = []
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_large_date_range(self):
        """Test case with a very large date range spanning many years."""
        start_date = dt.datetime(2000, 1, 1)
        end_date = dt.datetime(2025, 12, 31)
        expected = [(f"{year}-01-01", f"{year + 1}-12-31") for year in range(2000, 2025)]
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    @pytest.mark.parametrize(
        ("start_date", "end_date"),
        [
            (dt.datetime(2021, 5, 10), dt.datetime(2024, 8, 20)),
            ("2021-05-10", "2024-08-20"),
        ],
        ids=["datetime_input", "string_input"],
    )
    def test_valid_range_multiple_years_return_datetime(self, start_date, end_date):
        """Test return_str=False with datetime and string inputs returns datetime tuples."""
        expected = [
            (dt.datetime(2021, 5, 10), dt.datetime(2022, 8, 20)),
            (dt.datetime(2022, 5, 10), dt.datetime(2023, 8, 20)),
            (dt.datetime(2023, 5, 10), dt.datetime(2024, 8, 20)),
        ]
        result = find_overlapping_periods(start_date, end_date, return_str=False)
        assert result == expected

    @pytest.mark.parametrize(
        ("start_date", "end_date"),
        [
            (dt.datetime(2021, 3, 1), dt.datetime(2023, 6, 15)),
            ("2021-03-01", "2023-06-15"),
        ],
        ids=["naive_datetime", "string_input"],
    )
    def test_naive_input_returns_naive_datetimes(self, start_date, end_date):
        """Test that naive datetime and string inputs produce naive datetime outputs."""
        result = find_overlapping_periods(start_date, end_date, return_str=False)
        for start, end in result:
            assert start.tzinfo is None
            assert end.tzinfo is None

    def test_aware_datetime_returns_aware_datetimes(self):
        """Test that tz-aware datetime inputs produce tz-aware datetime outputs."""
        utc = dt.UTC
        start_date = dt.datetime(2021, 3, 1, tzinfo=utc)
        end_date = dt.datetime(2023, 6, 15, tzinfo=utc)
        expected = [
            (dt.datetime(2021, 3, 1, tzinfo=utc), dt.datetime(2022, 6, 15, tzinfo=utc)),
            (dt.datetime(2022, 3, 1, tzinfo=utc), dt.datetime(2023, 6, 15, tzinfo=utc)),
        ]
        result = find_overlapping_periods(start_date, end_date, return_str=False)
        assert result == expected
        for start, end in result:
            assert start.tzinfo is not None
            assert end.tzinfo is not None

    def test_non_utc_aware_datetime_preserves_timezone(self):
        """Test that non-UTC tz-aware inputs preserve the original timezone in outputs."""
        est = dt.timezone(dt.timedelta(hours=-5))
        start_date = dt.datetime(2021, 3, 1, tzinfo=est)
        end_date = dt.datetime(2023, 6, 15, tzinfo=est)
        expected = [
            (dt.datetime(2021, 3, 1, tzinfo=est), dt.datetime(2022, 6, 15, tzinfo=est)),
            (dt.datetime(2022, 3, 1, tzinfo=est), dt.datetime(2023, 6, 15, tzinfo=est)),
        ]
        result = find_overlapping_periods(start_date, end_date, return_str=False)
        assert result == expected

    @pytest.mark.parametrize(
        ("start_date", "end_date"),
        [
            ("2021-03-01", dt.datetime(2023, 6, 15, tzinfo=dt.UTC)),
            (dt.datetime(2021, 3, 1, tzinfo=dt.UTC), "2023-06-15"),
            (dt.datetime(2021, 3, 1), dt.datetime(2023, 6, 15, tzinfo=dt.UTC)),
            (dt.datetime(2021, 3, 1, tzinfo=dt.UTC), dt.datetime(2023, 6, 15)),
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
            (dt.date(2022, 6, 15), dt.datetime(2024, 8, 20)),
            (dt.datetime(2022, 6, 15), dt.date(2024, 8, 20)),
            (dt.date(2022, 6, 15), dt.date(2024, 8, 20)),
            (12345, dt.datetime(2024, 8, 20)),
            (dt.datetime(2022, 6, 15), 12345),
        ],
        ids=["date_start", "date_end", "both_date", "int_start", "int_end"],
    )
    def test_invalid_type_raises_type_error(self, start_date, end_date):
        """Test that non-str/non-datetime inputs raise TypeError with a clear message."""
        with pytest.raises(TypeError, match="Expected str or datetime"):
            find_overlapping_periods(start_date, end_date)


class TestFilterAndLabelByEventPeriods:
    """Windows are placed around each key's own event date."""

    @pytest.fixture
    def transactions(self):
        """Store 101 opens a competitor on 10 Mar 2025, store 102 on 2 Jun 2025, store 205 never."""
        return pd.DataFrame(
            {
                cols.transaction_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                cols.store_id: [101, 101, 101, 101, 101, 101, 102, 102, 102, 205],
                cols.transaction_date: pd.to_datetime(
                    [
                        "2025-02-23 18:00",  # 15 days before 101's event: before every window
                        "2025-02-24 08:15",  # first day of pre
                        "2025-03-09 23:40",  # last day of pre
                        "2025-03-10 09:00",  # event day, before the 15:30 opening: first day of post
                        "2025-03-23 21:30",  # last day of post, late in the day
                        "2025-03-24 07:45",  # one day after post
                        "2025-03-10 12:00",  # 101's event day, but 12 weeks before 102's
                        "2025-06-01 17:20",  # last day of 102's pre
                        "2024-06-03 10:05",  # first day of 102's post_py1
                        "2025-03-10 11:00",  # store with no event
                    ],
                ),
                cols.unit_spend: [12.40, 33.10, 8.75, 54.20, 19.99, 27.30, 41.00, 15.60, 22.45, 64.80],
            },
        )

    @pytest.fixture
    def events(self):
        """Competitor openings for two of the three stores; 101's is timestamped mid-afternoon."""
        return pd.DataFrame(
            {cols.store_id: [101, 102], "event_date": pd.to_datetime(["2025-03-10 15:30", "2025-06-02 00:00"])},
        )

    @pytest.mark.parametrize(
        ("to_transactions", "to_events"),
        [
            (lambda df: df, lambda df: df),
            (
                lambda df: ibis.memtable(df).mutate(**{cols.transaction_date: ibis._[cols.transaction_date].date()}),
                lambda df: ibis.memtable(df).mutate(event_date=ibis._.event_date.date()),
            ),
            (
                lambda df: ibis.memtable(df).mutate(**{cols.transaction_date: ibis._[cols.transaction_date].date()}),
                lambda df: df,
            ),
            (lambda df: df, lambda df: ibis.memtable(df).mutate(event_date=ibis._.event_date.date())),
            (lambda df: df.assign(__window_0_start=pd.Timestamp("2000-01-01")), lambda df: df),
        ],
        ids=[
            "pandas_timestamps",
            "ibis_dates",
            "dates_with_timestamp_events",
            "timestamps_with_date_events",
            "column_named_like_a_window_bound",
        ],
    )
    def test_labels_rows_by_their_own_keys_event(self, transactions, events, to_transactions, to_events):
        """Each row gets the window around its store's event; both ends of a window are whole days included."""
        result = filter_and_label_by_event_periods(
            to_transactions(transactions),
            events=to_events(events),
            on=cols.store_id,
            windows={
                "pre": (-2 * WEEK, -DAY),
                "post": (0 * WEEK, 2 * WEEK - DAY),
                "post_py1": (-52 * WEEK, -50 * WEEK - DAY),
            },
            period_col="window",
        )

        pd.testing.assert_frame_equal(
            result.select(cols.transaction_id, cols.store_id, "window").order_by(cols.transaction_id).to_pandas(),
            pd.DataFrame(
                {
                    cols.transaction_id: [2, 3, 4, 5, 8, 9],
                    cols.store_id: [101, 101, 101, 101, 102, 102],
                    "window": ["pre", "pre", "post", "post", "pre", "post_py1"],
                },
            ),
        )

    @pytest.mark.parametrize("weeks", [range(-2, 2), np.arange(-2, 2)], ids=["range", "numpy_arange"])
    def test_integer_labels_give_an_integer_column(self, transactions, events, weeks):
        """Integer window keys come back as an int64 label, ready to sort and plot as week numbers."""
        result = filter_and_label_by_event_periods(
            transactions,
            events=events,
            on=cols.store_id,
            windows={n: (n * WEEK, (n + 1) * WEEK - DAY) for n in weeks},
        )

        pd.testing.assert_frame_equal(
            result.select(cols.transaction_id, "period_name").order_by(cols.transaction_id).to_pandas(),
            pd.DataFrame({cols.transaction_id: [2, 3, 4, 5, 8], "period_name": [-2, -1, 0, 1, -1]}),
        )

    def test_require_every_window_keeps_keys_seen_in_every_window(self, transactions, events):
        """Store 101 sold in pre and post and is kept; 102 sold only in pre and is dropped."""
        result = filter_and_label_by_event_periods(
            transactions,
            events=events,
            on=cols.store_id,
            windows={"pre": (-2 * WEEK, -DAY), "post": (0 * WEEK, 2 * WEEK - DAY)},
            require_every_window=True,
        )

        pd.testing.assert_frame_equal(
            result.select(cols.transaction_id, "period_name").order_by(cols.transaction_id).to_pandas(),
            pd.DataFrame({cols.transaction_id: [2, 3, 4, 5], "period_name": ["pre", "pre", "post", "post"]}),
        )

    def test_reads_the_configured_transaction_date_column(self, transactions, events):
        """The transaction date column comes from the column.transaction_date option."""
        renamed = transactions.rename(columns={cols.transaction_date: "sold_at"})
        with option_context("column.transaction_date", "sold_at"):
            result = filter_and_label_by_event_periods(
                renamed, events=events, on=cols.store_id, windows={"pre": (-2 * WEEK, -DAY)}
            )

        pd.testing.assert_frame_equal(
            result.select(cols.transaction_id, "period_name").order_by(cols.transaction_id).to_pandas(),
            pd.DataFrame({cols.transaction_id: [2, 3, 8], "period_name": ["pre", "pre", "pre"]}),
        )

    def test_composite_key_drops_rows_without_an_event(self):
        """With a store and brand key, other brands in the same store and guest rows with a null key are dropped."""
        transactions = pd.DataFrame(
            {
                cols.transaction_id: [1, 2, 3, 4],
                cols.store_id: [101, 101, 102, None],
                "brand_id": [7, 9, 7, 7],
                cols.transaction_date: pd.to_datetime(["2025-04-01", "2025-04-01", "2025-04-01", "2025-04-01"]),
            },
        )
        rangings = pd.DataFrame(
            {
                cols.store_id: [101, 102],
                "brand_id": [7, 7],
                "event_date": pd.to_datetime(["2025-03-10", "2025-06-02"]),
            },
        )

        result = filter_and_label_by_event_periods(
            transactions,
            events=rangings,
            on=[cols.store_id, "brand_id"],
            windows={"pre": (-12 * WEEK, -DAY), "post": (0 * WEEK, 12 * WEEK - DAY)},
        )

        pd.testing.assert_frame_equal(
            result.select(cols.transaction_id, "period_name").order_by(cols.transaction_id).to_pandas(),
            pd.DataFrame({cols.transaction_id: [1, 3], "period_name": ["post", "pre"]}),
        )

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"windows": [("pre", (-WEEK, -DAY))]}, TypeError, r"^windows must be a mapping of labels"),
            ({"windows": {}}, ValueError, r"^windows must contain at least one window\.$"),
            (
                {"windows": {"pre": (-WEEK, -DAY), 0: (0 * WEEK, WEEK)}},
                ValueError,
                r"^windows labels must be all strings or all integers\.",
            ),
            (
                {"windows": {"pre": (-7, -1)}},
                TypeError,
                r"^Window 'pre' must be a \(start, end\) tuple of timedeltas\.",
            ),
            (
                {"windows": {"pre": (-WEEK,)}},
                TypeError,
                r"^Window 'pre' must be a \(start, end\) tuple of timedeltas\.",
            ),
            (
                {"windows": {"pre": (-DAY, -WEEK)}},
                ValueError,
                r"^Window 'pre' starts after it ends: start -1d must be <= end -7d\.$",
            ),
            (
                {"windows": {"pre": (-WEEK, -dt.timedelta(hours=12))}},
                ValueError,
                r"^Window 'pre' offsets must be whole days\.",
            ),
            (
                {"windows": {"pre": (-52 * WEEK, -DAY), "post_py1": (-50 * WEEK, -38 * WEEK - DAY)}},
                ValueError,
                r"^Windows 'pre' \[-364d, -1d\] and 'post_py1' \[-350d, -267d\] overlap\.$",
            ),
            (
                {"windows": {"pre": (-WEEK, 0 * WEEK), "post": (0 * WEEK, WEEK - DAY)}},
                ValueError,
                (
                    r"^Windows 'pre' \[-7d, 0d\] and 'post' \[0d, 6d\] overlap\. Both ends are included, so end "
                    r"'pre' before 'post' starts\.$"
                ),
            ),
            (
                {"windows": {"opening_day": (0 * DAY, 0 * DAY), "post": (0 * DAY, WEEK - DAY)}},
                ValueError,
                r"^Windows 'opening_day' \[0d, 0d\] and 'post' \[0d, 6d\] overlap\.$",
            ),
            ({"on": []}, ValueError, r"^on must not be an empty list\.$"),
            ({"on": None}, TypeError, r"^on must be a string or list of strings\. Got NoneType\.$"),
            (
                {"on": cols.customer_id},
                ValueError,
                rf"^on references columns not present in the data: \['{cols.customer_id}'\]\.$",
            ),
            (
                {"on": cols.transaction_id},
                ValueError,
                rf"^on \(in events\) references columns not present in the data: \['{cols.transaction_id}'\]\.$",
            ),
            (
                {"event_date_col": "opened_on"},
                ValueError,
                r"^event_date_col references columns not present in the data: \['opened_on'\]\.$",
            ),
            (
                {"period_col": cols.unit_spend},
                ValueError,
                rf"^period_col '{cols.unit_spend}' is already a column in transactions\.",
            ),
            (
                {
                    "events": pd.DataFrame(
                        {
                            cols.store_id: [101, 101, 102],
                            "event_date": pd.to_datetime(["2025-03-10", "2025-09-01", "2025-06-02"]),
                        }
                    ),
                },
                ValueError,
                (
                    rf"^events must have one row per {cols.store_id}, with no null keys or event dates\. "
                    rf"Rows to fix \(at most \d+ shown\): {{'{cols.store_id}': 101, 'rows': 2, 'null_event_dates': 0}}$"
                ),
            ),
            (
                {
                    "events": pd.DataFrame(
                        {cols.store_id: [101, 102], "event_date": pd.to_datetime(["2025-03-10", None])}
                    )
                },
                ValueError,
                rf"Rows to fix \(at most \d+ shown\): {{'{cols.store_id}': 102, 'rows': 1, 'null_event_dates': 1}}$",
            ),
            (
                {
                    "events": pd.DataFrame(
                        {cols.store_id: [101, None], "event_date": pd.to_datetime(["2025-03-10", "2025-06-02"])}
                    ),
                },
                ValueError,
                rf"Rows to fix \(at most \d+ shown\): {{'{cols.store_id}': .*, 'rows': 1, 'null_event_dates': 0}}$",
            ),
            (
                {"events": pd.DataFrame({cols.store_id: [101, 102], "event_date": ["2025-03-10", "2025-06-02"]})},
                TypeError,
                r"^Column 'event_date' must be a date or datetime type\. Got string\.",
            ),
            (
                {
                    "events": pd.DataFrame(
                        {
                            cols.store_id: [101, 102],
                            "event_date": pd.to_datetime(["2025-03-10", "2025-06-02"]).tz_localize("Europe/London"),
                        }
                    ),
                },
                ValueError,
                r"^Column 'event_date' is timezone-aware \(Europe/London\)\.",
            ),
        ],
        ids=[
            "windows_not_a_mapping",
            "no_windows",
            "mixed_label_types",
            "integer_offsets",
            "one_offset",
            "start_after_end",
            "part_day_offset",
            "overlap",
            "shared_boundary_day",
            "shared_start_day",
            "empty_on",
            "on_not_a_column_name",
            "on_missing",
            "on_missing_from_events",
            "event_date_col_missing",
            "period_col_exists",
            "two_events_for_a_store",
            "null_event_date",
            "null_key",
            "string_event_dates",
            "timezone_aware_event_dates",
        ],
    )
    def test_invalid_arguments_raise(self, transactions, events, kwargs, error, match):
        """Malformed windows, keys, column names or events raise an error that says what is wrong.

        Duplicate keys, null keys and null event dates list the rows to fix.
        """
        arguments = {"events": events, "on": cols.store_id, "windows": {"pre": (-WEEK, -DAY)}} | kwargs
        with pytest.raises(error, match=match):
            filter_and_label_by_event_periods(transactions, **arguments)

    def test_lists_at_most_a_few_rows_to_fix(self, transactions):
        """With many duplicated stores, the error lists only the first few rather than every bad key."""
        stores = list(range(101, 102 + _MAX_BAD_EVENT_ROWS))
        events = pd.DataFrame({cols.store_id: stores * 2, "event_date": pd.Timestamp("2025-03-10")})
        with pytest.raises(ValueError, match="Rows to fix") as raised:
            filter_and_label_by_event_periods(
                transactions, events=events, on=cols.store_id, windows={"pre": (-WEEK, -DAY)}
            )

        assert str(raised.value).count("'rows': 2") == _MAX_BAD_EVENT_ROWS


class TestFilterAndLabelByEventToDate:
    """Each key's test period runs from its own event date to one end date; other windows are moved copies of it."""

    @pytest.fixture
    def transactions(self):
        """Customer 1042 signs up on 20 Jan 2025, 3378 on 10 May 2025 and 5001 never; the data ends 30 Jun 2025."""
        return pd.DataFrame(
            {
                cols.transaction_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                cols.customer_id: [1042, 1042, 1042, 1042, 1042, 1042, 1042, 1042, 3378, 3378, 3378, 3378, 5001],
                cols.transaction_date: pd.to_datetime(
                    [
                        "2024-08-10 20:00",  # the day before 1042's pre
                        "2024-08-11 09:00",  # first day of 1042's pre: 162 days before sign-up
                        "2025-01-19 22:30",  # last day of 1042's pre
                        "2025-01-20 08:00",  # sign-up day, before the 15:30 sign-up: first day of post
                        "2025-06-30 21:45",  # end_date, late in the day: last day of post
                        "2025-07-01 10:00",  # after end_date
                        "2024-01-22 12:00",  # first day of 1042's post_py1: sign-up day minus 52 weeks
                        "2024-07-01 18:00",  # last day of 1042's post_py1: end_date minus 52 weeks
                        "2025-03-18 11:00",  # the day before 3378's pre
                        "2025-03-19 10:00",  # first day of 3378's pre: 52 days before sign-up
                        "2025-05-10 09:30",  # 3378's sign-up day: first day of post
                        "2024-07-02 09:00",  # the day after 3378's post_py1
                        "2025-02-01 13:15",  # customer with no sign-up
                    ],
                ),
                cols.unit_spend: [
                    18.40,
                    42.10,
                    9.75,
                    61.20,
                    24.99,
                    33.30,
                    15.60,
                    27.45,
                    12.80,
                    55.00,
                    38.90,
                    21.15,
                    47.60,
                ],
            },
        )

    @pytest.fixture
    def signups(self):
        """Loyalty sign-ups; 1042's is timestamped mid-afternoon."""
        return pd.DataFrame(
            {
                cols.customer_id: [1042, 3378],
                "event_date": pd.to_datetime(["2025-01-20 15:30", "2025-05-10 00:00"]),
            },
        )

    @pytest.mark.parametrize(
        ("to_transactions", "to_events", "end_date"),
        [
            (lambda df: df, lambda df: df, "2025-06-30"),
            (
                lambda df: ibis.memtable(df).mutate(**{cols.transaction_date: ibis._[cols.transaction_date].date()}),
                lambda df: ibis.memtable(df).mutate(event_date=ibis._.event_date.date()),
                dt.date(2025, 6, 30),
            ),
        ],
        ids=["pandas_timestamps", "ibis_dates"],
    )
    @pytest.mark.parametrize(
        ("require_every_window", "expected"),
        [
            (
                False,
                {
                    cols.transaction_id: [2, 3, 4, 5, 7, 8, 10, 11],
                    "period_name": ["pre", "pre", "post", "post", "post_py1", "post_py1", "pre", "post"],
                },
            ),
            (
                True,
                {
                    cols.transaction_id: [2, 3, 4, 5, 7, 8],
                    "period_name": ["pre", "pre", "post", "post", "post_py1", "post_py1"],
                },
            ),
        ],
        ids=["every_key", "keys_in_every_window"],
    )
    def test_labels_each_keys_test_period_and_its_moved_copies(
        self,
        transactions,
        signups,
        to_transactions,
        to_events,
        end_date,
        require_every_window,
        expected,
    ):
        """Post runs from sign-up to end_date; pre is as long and ends the day before; post_py1 is 52 weeks earlier.

        require_every_window drops 3378, who bought nothing in post_py1.
        """
        result = filter_and_label_by_event_to_date(
            to_transactions(transactions),
            events=to_events(signups),
            on=cols.customer_id,
            end_date=end_date,
            windows={"post": 0 * WEEK, "pre": BEFORE_EVENT, "post_py1": -52 * WEEK},
            require_every_window=require_every_window,
        )

        pd.testing.assert_frame_equal(
            # Drop the inputs rather than select the label, so a duration column added by default fails the match.
            result.drop(cols.customer_id, cols.transaction_date, cols.unit_spend)
            .order_by(cols.transaction_id)
            .to_pandas(),
            pd.DataFrame(expected),
        )

    @pytest.mark.parametrize(
        ("kwargs", "duration_col"),
        [({"include_duration": True}, "period_days"), ({"include_duration": True, "duration_col": "days"}, "days")],
        ids=["default_name", "custom_name"],
    )
    def test_include_duration_adds_each_keys_period_length(self, transactions, signups, kwargs, duration_col):
        """Every window of a key is as long as its test period, counting both ends: 162 days for 1042, 52 for 3378."""
        result = filter_and_label_by_event_to_date(
            transactions,
            events=signups,
            on=cols.customer_id,
            end_date="2025-06-30",
            windows={"post": 0 * WEEK, "pre": BEFORE_EVENT},
            **kwargs,
        )

        pd.testing.assert_frame_equal(
            result.select(cols.transaction_id, duration_col).order_by(cols.transaction_id).to_pandas(),
            pd.DataFrame({cols.transaction_id: [2, 3, 4, 5, 10, 11], duration_col: [162, 162, 162, 162, 52, 52]}),
        )

    @pytest.mark.parametrize(
        ("customer", "event_date", "windows", "expected"),
        [
            (3378, "2024-12-31", {"pre": BEFORE_EVENT, "post_py1": -52 * WEEK}, (12, "pre", 182)),
            (1042, "2025-06-30", {"post": 0 * WEEK}, (5, "post", 1)),
        ],
        ids=["event_on_the_cutoff_day", "event_on_end_date"],
    )
    def test_events_on_the_date_limits_are_accepted(self, transactions, customer, event_date, windows, expected):
        """The earliest event the windows allow gets the longest test period; an event on end_date gets one day."""
        result = filter_and_label_by_event_to_date(
            transactions,
            events=pd.DataFrame({cols.customer_id: [customer], "event_date": pd.to_datetime([event_date])}),
            on=cols.customer_id,
            end_date="2025-06-30",
            windows=windows,
            include_duration=True,
        )

        transaction_id, period_name, period_days = expected
        pd.testing.assert_frame_equal(
            result.select(cols.transaction_id, "period_name", "period_days").to_pandas(),
            pd.DataFrame(
                {cols.transaction_id: [transaction_id], "period_name": [period_name], "period_days": [period_days]}
            ),
        )

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"windows": [("post", 0 * WEEK)]}, TypeError, r"^windows must be a mapping of labels to timedelta"),
            ({"windows": {}}, ValueError, r"^windows must contain at least one window\.$"),
            (
                {"windows": {"post": 0 * WEEK, 1: -52 * WEEK}},
                ValueError,
                r"^windows labels must be all strings or all integers\.",
            ),
            (
                {"windows": {"post": 0 * WEEK, "pre": "before"}},
                TypeError,
                r"^Window 'pre' must be a timedelta or BEFORE_EVENT\. Got 'before'\.$",
            ),
            (
                {"windows": {"post": dt.timedelta(hours=-12)}},
                ValueError,
                r"^Window 'post' must move by whole days\. Got -1 day, 12:00:00\.$",
            ),
            (
                {"windows": {"post": 0 * WEEK, "also_post": 0 * WEEK}},
                ValueError,
                r"^Windows 'post' and 'also_post' move by the same amount, so they always overlap\.$",
            ),
            (
                {"windows": {"pre": BEFORE_EVENT, "also_pre": BEFORE_EVENT}},
                ValueError,
                r"^Windows 'pre' and 'also_pre' are both BEFORE_EVENT, so they always overlap\.$",
            ),
            (
                {"windows": {"pre": BEFORE_EVENT, "lag": -DAY}},
                ValueError,
                r"^Windows 'pre' and 'lag' both hold the day before the event, so they always overlap\.$",
            ),
            (
                {"end_date": "2025/06/30"},
                ValueError,
                r"^end_date must be an ISO 'YYYY-MM-DD' date string\. Got '2025/06/30'\.$",
            ),
            (
                {"include_duration": True, "duration_col": cols.unit_spend},
                ValueError,
                rf"^duration_col '{cols.unit_spend}' is already a column in transactions\.",
            ),
            (
                {"include_duration": True, "duration_col": "period_name"},
                ValueError,
                r"^duration_col and period_col are both 'period_name'\. Pass a different duration_col\.$",
            ),
            (
                {"on": cols.store_id},
                ValueError,
                rf"^on references columns not present in the data: \['{cols.store_id}'\]",
            ),
            (
                {
                    "events": pd.DataFrame(
                        {cols.customer_id: [1042, 3378], "event_date": pd.to_datetime(["2025-01-20", "2025-07-01"])},
                    ),
                },
                ValueError,
                (
                    rf"event dates from 2024-12-31 to 2025-06-30.*Rows to fix \(at most \d+ shown\): "
                    rf"{{'{cols.customer_id}': 3378, 'rows': 1, 'null_event_dates': 0, 'dates_outside': 1}}$"
                ),
            ),
            (
                {
                    "windows": {"post": 0 * WEEK, "pre": BEFORE_EVENT},
                    "events": pd.DataFrame(
                        {cols.customer_id: [1042, 3378], "event_date": pd.to_datetime(["2025-01-20", "2025-07-01"])},
                    ),
                },
                ValueError,
                (
                    rf"event dates on or before 2025-06-30\. Rows to fix \(at most \d+ shown\): "
                    rf"{{'{cols.customer_id}': 3378, 'rows': 1, 'null_event_dates': 0, 'dates_outside': 1}}$"
                ),
            ),
            (
                {
                    "events": pd.DataFrame(
                        {cols.customer_id: [1042, 3378], "event_date": pd.to_datetime(["2024-12-30", "2025-05-10"])},
                    ),
                },
                ValueError,
                (
                    rf"^events must have one row per {cols.customer_id}, with no null keys or event dates, and event "
                    r"dates from 2024-12-31 to 2025-06-30\. An earlier event gives a test period over 182 days, "
                    r"where windows 'pre' and 'post_py1' overlap\. Rows to fix"
                ),
            ),
        ],
        ids=[
            "windows_not_a_mapping",
            "no_windows",
            "mixed_label_types",
            "string_instead_of_constant",
            "part_day_shift",
            "same_shift_twice",
            "before_event_twice",
            "moved_back_one_day_with_before_event",
            "malformed_end_date",
            "duration_col_exists",
            "duration_col_is_period_col",
            "on_missing",
            "event_after_end_date",
            "event_after_end_date_without_window_limit",
            "event_too_early_for_windows",
        ],
    )
    def test_invalid_arguments_raise(self, transactions, signups, kwargs, error, match):
        """Malformed windows, dates or columns, and events the windows can't fit, raise an error naming the problem."""
        arguments = {
            "events": signups,
            "on": cols.customer_id,
            "end_date": "2025-06-30",
            "windows": {"post": 0 * WEEK, "pre": BEFORE_EVENT, "post_py1": -52 * WEEK},
        } | kwargs
        with pytest.raises(error, match=match):
            filter_and_label_by_event_to_date(transactions, **arguments)
