"""Tests for the NLRSegmentation class."""

import ibis
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from openretailscience.options import ColumnHelper
from openretailscience.segmentation.nlr import (
    SEGMENT_LAPSED,
    SEGMENT_NEW,
    SEGMENT_REPEATING,
    NLRSegmentation,
)

cols = ColumnHelper()


class TestNLRSegmentation:
    """Tests for the NLRSegmentation class."""

    @pytest.fixture
    def transaction_df(self):
        """Return a DataFrame with transactions across two periods, including multiple transactions per customer."""
        return pd.DataFrame(
            {
                cols.customer_id: [1001, 1001, 1002, 1003, 1001, 1003, 1003, 1004],
                cols.unit_spend: [60.00, 40.00, 200.00, 150.00, 120.00, 80.00, 100.00, 90.00],
                "year": [2023, 2023, 2023, 2023, 2024, 2024, 2024, 2024],
            },
        )

    def test_basic_segmentation(self, transaction_df):
        """Test customers are classified as New, Repeating, or Lapsed with correct aggregated values."""
        seg = NLRSegmentation(
            df=transaction_df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
        )
        result = seg.df.sort_index()
        p1_col = f"{cols.unit_spend}_p1"
        p2_col = f"{cols.unit_spend}_p2"

        expected = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1004],
                "segment_name": [SEGMENT_REPEATING, SEGMENT_LAPSED, SEGMENT_REPEATING, SEGMENT_NEW],
                p1_col: [100.00, 200.00, 150.00, 0.00],
                p2_col: [120.00, 0.00, 180.00, 90.00],
            },
        ).set_index(cols.customer_id)

        assert_frame_equal(result, expected, check_dtype=False)

    def test_segment_names_are_exhaustive(self, transaction_df):
        """Test that every customer receives a known segment — no NULLs or unknown values."""
        seg = NLRSegmentation(
            df=transaction_df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
        )
        result = seg.df
        valid_segments = {SEGMENT_NEW, SEGMENT_REPEATING, SEGMENT_LAPSED}

        assert result["segment_name"].notna().all()
        assert set(result["segment_name"].unique()) == valid_segments

    def test_zero_spend_not_counted_as_bought(self):
        """Test that a customer with zero aggregated spend in a period is not considered to have bought."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1001, 1002],
                cols.unit_spend: [0.00, 50.00, 100.00],
                "quarter": ["Q1", "Q2", "Q1"],
            },
        )
        seg = NLRSegmentation(
            df=df,
            period_col="quarter",
            p1_value="Q1",
            p2_value="Q2",
        )
        result = seg.df

        # 1001 has zero spend in Q1 but positive in Q2 -> New
        assert result.loc[1001, "segment_name"] == SEGMENT_NEW
        # 1002 only in Q1 with positive spend -> Lapsed
        assert result.loc[1002, "segment_name"] == SEGMENT_LAPSED

    def test_zero_spend_both_periods_excluded(self):
        """Test that a customer with zero spend in both periods is excluded from results."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1001, 1002, 1002],
                cols.unit_spend: [0.00, 0.00, 50.00, 75.00],
                "period": ["P1", "P2", "P1", "P2"],
            },
        )
        seg = NLRSegmentation(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
        )
        result = seg.df

        zero_spend_customer = 1001
        positive_spend_customer = 1002
        # Customer with zero spend in both periods should not appear
        assert zero_spend_customer not in result.index
        # Customer with positive spend in both should be Repeating
        assert result.loc[positive_spend_customer, "segment_name"] == SEGMENT_REPEATING

    def test_negative_spend_not_counted_as_bought(self):
        """Test that a customer with negative aggregated spend in a period is not considered to have bought."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1001, 1001],
                cols.unit_spend: [-50.00, 30.00, 80.00],
                "period": ["P1", "P1", "P2"],
            },
        )
        seg = NLRSegmentation(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
        )
        result = seg.df

        # 1001 has negative aggregated spend (-50+30=-20) in P1 but positive in P2 -> New
        assert result.loc[1001, "segment_name"] == SEGMENT_NEW

    @pytest.mark.parametrize(
        "columns",
        [
            {cols.unit_spend: [100.00], "year": [2023]},
            {cols.customer_id: [1001], "year": [2023]},
            {cols.customer_id: [1001], cols.unit_spend: [100.00]},
        ],
        ids=["missing_customer_id", "missing_value_col", "missing_period_col"],
    )
    def test_raises_on_missing_required_column(self, columns):
        """Test that ValueError naming required_cols is raised when a required column is missing."""
        df = pd.DataFrame(columns)
        with pytest.raises(ValueError, match="required_cols references columns not present in the DataFrame"):
            NLRSegmentation(df=df, period_col="year", p1_value=2023, p2_value=2024)

    @pytest.mark.parametrize(
        ("p1_value", "p2_value"),
        [
            (2023, 2023),
            (ibis.literal(2023), ibis.literal(2023)),
            (2023, ibis.literal(2023)),
            (ibis.literal(2023), 2023),
        ],
        ids=["plain_values", "ibis_scalars", "mixed_plain_ibis", "mixed_ibis_plain"],
    )
    def test_raises_when_p1_equals_p2(self, transaction_df, p1_value, p2_value):
        """Test that ValueError is raised when p1_value and p2_value are the same."""
        with pytest.raises(ValueError, match="must be different"):
            NLRSegmentation(df=transaction_df, period_col="year", p1_value=p1_value, p2_value=p2_value)

    def test_empty_result_when_no_transactions_match_periods(self):
        """Test that an empty DataFrame is returned when no transactions match the given periods."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002],
                cols.unit_spend: [50.00, 75.00],
                "quarter": ["Q1", "Q2"],
            },
        )
        seg = NLRSegmentation(
            df=df,
            period_col="quarter",
            p1_value="Q3",
            p2_value="Q4",
        )
        result = seg.df
        assert len(result) == 0

    @pytest.mark.parametrize(
        "agg_func",
        ["foobar", "min"],
        ids=["unknown_func", "min_not_supported"],
    )
    def test_raises_on_invalid_agg_func(self, transaction_df, agg_func):
        """Test that ValueError is raised when agg_func is not a supported aggregation."""
        with pytest.raises(ValueError, match="agg_func"):
            NLRSegmentation(df=transaction_df, period_col="year", p1_value=2023, p2_value=2024, agg_func=agg_func)

    def test_alternate_value_col(self):
        """Test segmentation with a non-default value column."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1001],
                cols.unit_qty: [5, 10, 8],
                "year": [2023, 2023, 2024],
            },
        )
        seg = NLRSegmentation(
            df=df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
            value_col=cols.unit_qty,
        )
        result = seg.df

        assert result.loc[1001, "segment_name"] == SEGMENT_REPEATING
        assert result.loc[1002, "segment_name"] == SEGMENT_LAPSED

    def test_agg_func_count_segments_by_transaction_count(self):
        """Test that agg_func='count' segments based on transaction counts rather than spend."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1001, 1001, 1002, 1003],
                cols.unit_spend: [10.00, 20.00, 30.00, 50.00, 75.00],
                "year": [2023, 2023, 2024, 2023, 2024],
            },
        )
        seg = NLRSegmentation(
            df=df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
            agg_func="count",
        )
        result = seg.df.sort_index()
        p1_col = f"{cols.unit_spend}_p1"
        p2_col = f"{cols.unit_spend}_p2"

        expected = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003],
                "segment_name": [SEGMENT_REPEATING, SEGMENT_LAPSED, SEGMENT_NEW],
                p1_col: [2, 1, 0],
                p2_col: [1, 0, 1],
            },
        ).set_index(cols.customer_id)

        assert_frame_equal(result, expected, check_dtype=False)

    def test_input_dataframe_not_mutated(self, transaction_df):
        """Test that the original DataFrame is not modified."""
        original_df = transaction_df.copy()
        seg = NLRSegmentation(
            df=transaction_df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
        )
        _ = seg.df
        assert original_df.equals(transaction_df)

    def test_table_property_returns_ibis_table_with_expected_columns(self, transaction_df):
        """Test that the table property returns an ibis Table with the expected schema."""
        seg = NLRSegmentation(
            df=transaction_df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
        )
        table = seg.table
        assert isinstance(table, ibis.Table)
        assert cols.customer_id in table.columns
        assert "segment_name" in table.columns
        assert f"{cols.unit_spend}_p1" in table.columns
        assert f"{cols.unit_spend}_p2" in table.columns

    def test_accepts_ibis_table_input(self, transaction_df):
        """Test that an ibis Table can be passed directly as input."""
        table = ibis.memtable(transaction_df)
        seg = NLRSegmentation(
            df=table,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
        )
        result = seg.df

        assert result.loc[1001, "segment_name"] == SEGMENT_REPEATING
        assert result.loc[1002, "segment_name"] == SEGMENT_LAPSED
        assert result.loc[1004, "segment_name"] == SEGMENT_NEW

    @pytest.mark.parametrize(
        ("p1_value", "p2_value"),
        [
            (ibis.literal(2023), ibis.literal(2024)),
            (ibis.literal(2023), 2024),
            (2023, ibis.literal(2024)),
        ],
        ids=["both_ibis", "mixed_ibis_p1", "mixed_ibis_p2"],
    )
    def test_ibis_scalar_period_values(self, transaction_df, p1_value, p2_value):
        """Test that ibis Scalar and mixed ibis/plain period values produce correct segments."""
        seg = NLRSegmentation(
            df=transaction_df,
            period_col="year",
            p1_value=p1_value,
            p2_value=p2_value,
        )
        result = seg.df

        assert result.loc[1001, "segment_name"] == SEGMENT_REPEATING
        assert result.loc[1002, "segment_name"] == SEGMENT_LAPSED
        assert result.loc[1004, "segment_name"] == SEGMENT_NEW

    def test_df_property_caches_result(self, transaction_df):
        """Test that the df property returns the same cached DataFrame on subsequent calls."""
        seg = NLRSegmentation(
            df=transaction_df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
        )
        first_call = seg.df
        second_call = seg.df
        assert first_call is second_call


class TestNLRSegmentationGroupCol:
    """Tests for NLRSegmentation group_col functionality."""

    @pytest.fixture
    def store_transaction_df(self):
        """Return a DataFrame with transactions across stores and periods."""
        return pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1001, 1003, 1001, 1002, 1003, 1004],
                cols.unit_spend: [100.00, 200.00, 150.00, 80.00, 120.00, 90.00, 70.00, 60.00],
                cols.store_id: [2001, 2001, 2001, 2001, 2002, 2002, 2002, 2002],
                "year": [2023, 2023, 2024, 2024, 2023, 2023, 2024, 2024],
            },
        )

    @pytest.mark.parametrize(
        "group_col",
        [cols.store_id, [cols.store_id]],
        ids=["string", "list"],
    )
    def test_segments_calculated_within_each_group(self, store_transaction_df, group_col):
        """Test that NLR segments are calculated independently within each group."""
        seg = NLRSegmentation(
            df=store_transaction_df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
            group_col=group_col,
        )
        result = seg.df.sort_index()
        p1_col = f"{cols.unit_spend}_p1"
        p2_col = f"{cols.unit_spend}_p2"

        expected = pd.DataFrame(
            {
                cols.customer_id: [1001, 1001, 1002, 1002, 1003, 1003, 1004],
                cols.store_id: [2001, 2002, 2001, 2002, 2001, 2002, 2002],
                "segment_name": [
                    SEGMENT_REPEATING,
                    SEGMENT_LAPSED,
                    SEGMENT_LAPSED,
                    SEGMENT_LAPSED,
                    SEGMENT_NEW,
                    SEGMENT_NEW,
                    SEGMENT_NEW,
                ],
                p1_col: [100.00, 120.00, 200.00, 90.00, 0.00, 0.00, 0.00],
                p2_col: [150.00, 0.00, 0.00, 0.00, 80.00, 70.00, 60.00],
            },
        ).set_index([cols.customer_id, cols.store_id])

        assert_frame_equal(result, expected, check_dtype=False)

    def test_raises_on_missing_group_column(self):
        """Test that ValueError is raised when group_col column is missing from the DataFrame."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001],
                cols.unit_spend: [100.00],
                "year": [2023],
            },
        )
        with pytest.raises(ValueError, match="group_col references columns not present in the DataFrame"):
            NLRSegmentation(
                df=df,
                period_col="year",
                p1_value=2023,
                p2_value=2024,
                group_col=cols.store_id,
            )
