"""Tests for the Cohort Analysis module."""

import datetime

import pandas as pd
import pandas.testing as pdt
import pytest

from openretailscience.analysis.cohort import CohortAnalysis, _periods_between
from openretailscience.options import option_context


class TestCohortAnalysis:
    """Tests for the Cohort Analysis module."""

    @pytest.fixture
    def transactions_df(self) -> pd.DataFrame:
        """Returns a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "transaction_id": list(range(12)),
                "customer_id": [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4],
                "unit_spend": [3.23, 3.35, 6.00, 4.50, 5.10, 7.20, 3.80, 4.90, 6.50, 2.10, 8.00, 3.50],
                "transaction_date": [
                    datetime.date(2023, 1, 15),
                    datetime.date(2023, 1, 20),
                    datetime.date(2023, 2, 5),
                    datetime.date(2023, 2, 10),
                    datetime.date(2023, 3, 1),
                    datetime.date(2023, 3, 15),
                    datetime.date(2023, 3, 20),
                    datetime.date(2023, 4, 10),
                    datetime.date(2023, 4, 25),
                    datetime.date(2023, 5, 5),
                    datetime.date(2023, 5, 20),
                    datetime.date(2023, 6, 10),
                ],
            },
        )

    @pytest.fixture
    def expected_results_df(self) -> pd.DataFrame:
        """Expected cohort result DataFrame for comparison."""
        expected_df = pd.DataFrame(
            {
                0: [2.0, 1.0, 0.0, 0.0, 2.0],
                1: [1.0, 1.0, 0.0, 0.0, 1.0],
                2: [2.0, 1.0, 0.0, 0.0, 0.0],
                3: [1.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=pd.date_range("2023-01-01", periods=5, freq="MS"),
        )

        expected_df.index.name = "min_period_shopped"
        expected_df.columns.name = "period_since"

        return expected_df

    def test_cohort_computation(self, transactions_df, expected_results_df):
        """Tests cohort computation logic and compares output with expected DataFrame."""
        cohort = CohortAnalysis(
            df=transactions_df,
            aggregation_column="unit_spend",
            agg_func="nunique",
            period="month",
            percentage=False,
        )
        result = cohort.df
        pdt.assert_frame_equal(result, expected_results_df)

    def test_missing_columns(self):
        """Test if missing columns raise an error."""
        df = pd.DataFrame({"customer_id": [1, 2, 3], "unit_spend": [10, 20, 30]})
        with pytest.raises(ValueError, match="Input data is missing required columns"):
            CohortAnalysis(
                df=df,
                aggregation_column="unit_spend",
            )

    def test_invalid_period(self, transactions_df):
        """Test if an invalid period raises an error."""
        with pytest.raises(ValueError, match=r"period must be one of .*'m'"):
            CohortAnalysis(
                df=transactions_df,
                aggregation_column="unit_spend",
                period="m",
            )

    @pytest.mark.parametrize("period", ["MONTH", "Month", "MoNtH"])
    def test_period_case_insensitive_matches_lowercase(self, transactions_df, period):
        """Mixed-case period values produce identical output to the lowercase form."""
        lower = CohortAnalysis(df=transactions_df, aggregation_column="unit_spend", period=period.lower()).df
        upper = CohortAnalysis(df=transactions_df, aggregation_column="unit_spend", period=period).df
        pdt.assert_frame_equal(upper, lower)

    def test_cohort_percentage_normalizes_each_cohort_to_own_period_zero(self, transactions_df):
        """Tests that percentage=True normalizes each cohort row by its own period-0 value.

        Rows for months with no first-time customers (2023-03, 2023-04) are gap-filled
        with zeros after the percentage calculation.
        """
        cohort = CohortAnalysis(
            df=transactions_df,
            aggregation_column="unit_spend",
            agg_func="nunique",
            period="month",
            percentage=True,
        )
        result = cohort.df

        expected_df = pd.DataFrame(
            {
                0: [1.0, 1.0, 0.0, 0.0, 1.0],
                1: [0.5, 1.0, 0.0, 0.0, 0.5],
                2: [1.0, 1.0, 0.0, 0.0, 0.0],
                3: [0.5, 0.0, 0.0, 0.0, 0.0],
            },
            index=pd.date_range("2023-01-01", periods=5, freq="MS"),
        )
        expected_df.index.name = "min_period_shopped"
        expected_df.columns.name = "period_since"

        pdt.assert_frame_equal(result, expected_df)

    def test_cohort_percentage_with_zero_period_zero_sum_produces_zeros(self):
        """Tests that percentage=True produces zeros when a cohort's period-0 sum is zero.

        When using agg_func='sum', a cohort's period-0 value can be zero if all
        transactions in the first period have zero spend. The replace(0, np.nan)
        guard prevents division-by-zero errors (NaN/inf), producing zeros instead.
        """
        df = pd.DataFrame(
            {
                "transaction_id": list(range(5)),
                "customer_id": [1, 2, 1, 3, 3],
                "unit_spend": [10.0, 20.0, 5.0, 0.0, 15.0],
                "transaction_date": [
                    datetime.date(2023, 1, 15),  # Customer 1, cohort Jan
                    datetime.date(2023, 1, 20),  # Customer 2, cohort Jan
                    datetime.date(2023, 2, 10),  # Customer 1, period 1
                    datetime.date(2023, 2, 5),  # Customer 3, cohort Feb (zero spend)
                    datetime.date(2023, 3, 1),  # Customer 3, period 1
                ],
            },
        )

        cohort = CohortAnalysis(
            df=df,
            aggregation_column="unit_spend",
            agg_func="sum",
            period="month",
            percentage=True,
        )
        result = cohort.df

        expected_df = pd.DataFrame(
            {
                0: [1.0, 0.0],
                1: [0.17, 0.0],
            },
            index=pd.date_range("2023-01-01", periods=2, freq="MS"),
        )
        expected_df.index.name = "min_period_shopped"
        expected_df.columns.name = "period_since"

        pdt.assert_frame_equal(result, expected_df)

    def test_with_custom_column_names(self, transactions_df):
        """Test CohortAnalysis with custom column names to ensure column overrides work correctly."""
        custom_transactions_df = transactions_df.rename(
            columns={
                "customer_id": "cust_id",
                "transaction_date": "txn_date",
                "unit_spend": "spend_amount",
            },
        )

        with option_context("column.customer_id", "cust_id", "column.transaction_date", "txn_date"):
            cohort = CohortAnalysis(
                df=custom_transactions_df,
                aggregation_column="spend_amount",
                agg_func="nunique",
                period="month",
            )

            result = cohort.df
            assert isinstance(result, pd.DataFrame), "Should return DataFrame with custom columns"
            assert not result.empty, "Should produce results with custom column names"


class TestPeriodsBetween:
    """Tests for _periods_between, which derives the cohort period_since column."""

    @pytest.mark.parametrize(
        ("period", "start", "end", "expected"),
        [
            ("day", "2023-03-01", "2023-03-08", 7),
            ("week", "2023-01-02", "2023-02-06", 5),  # Mondays, five weeks apart
            ("month", "2023-01-01", "2023-04-01", 3),
            ("month", "2022-11-01", "2023-02-01", 3),  # crosses the year boundary
            ("quarter", "2023-01-01", "2023-10-01", 3),
            ("quarter", "2022-10-01", "2023-04-01", 2),  # crosses the year boundary
            ("year", "2021-01-01", "2023-01-01", 2),
            ("month", "2023-06-01", "2023-06-01", 0),  # same period -> the cohort's own period 0
        ],
    )
    def test_counts_whole_periods_between_aligned_dates(self, period, start, end, expected):
        """Each period unit returns the count of whole periods elapsed, including across years."""
        result = _periods_between(pd.Series([pd.Timestamp(start)]), pd.Series([pd.Timestamp(end)]), period)
        assert result.tolist() == [expected]
        assert result.dtype == "int64"

    def test_operates_elementwise_over_a_column(self):
        """The helper computes period_since for every row of the cohort table at once."""
        starts = pd.Series([pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")])
        ends = pd.Series([pd.Timestamp("2023-01-01"), pd.Timestamp("2023-03-01"), pd.Timestamp("2023-05-01")])
        result = _periods_between(starts, ends, "month")
        assert result.tolist() == [0, 2, 3]

    def test_unsupported_period_raises(self):
        """An unrecognized period unit raises rather than returning a wrong count."""
        with pytest.raises(ValueError, match="Unsupported period"):
            _periods_between(
                pd.Series([pd.Timestamp("2023-01-01")]),
                pd.Series([pd.Timestamp("2023-02-01")]),
                "fortnight",
            )
