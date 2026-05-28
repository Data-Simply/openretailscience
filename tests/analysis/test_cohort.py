"""Tests for the Cohort Analysis module."""

import datetime

import pandas as pd
import pandas.testing as pdt
import pytest

from openretailscience.analysis.cohort import CohortAnalysis
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
        with pytest.raises(ValueError, match="The following columns are required but missing"):
            CohortAnalysis(
                df=df,
                aggregation_column="unit_spend",
            )

    def test_invalid_period(self, transactions_df):
        """Test if an invalid period raises an error."""
        invalid_period = "m"
        with pytest.raises(
            ValueError,
            match=f"Invalid period '{invalid_period}'. Allowed values: {CohortAnalysis.VALID_PERIODS}",
        ):
            CohortAnalysis(
                df=transactions_df,
                aggregation_column="unit_spend",
                period=invalid_period,
            )

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
