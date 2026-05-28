"""Tests for the ThresholdSegmentation class."""

import pandas as pd
import pytest

from openretailscience.options import ColumnHelper, get_option, option_context
from openretailscience.segmentation.threshold import ThresholdSegmentation

cols = ColumnHelper()


class TestThresholdSegmentation:
    """Tests for the ThresholdSegmentation class."""

    def test_correct_segmentation(self):
        """Test that the method correctly segments customers based on given thresholds and segments."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4],
                cols.unit_spend: [100, 200, 300, 400],
            },
        )
        thresholds = [0.5, 1]
        segments = ["Low", "High"]
        seg = ThresholdSegmentation(
            df=df,
            thresholds=thresholds,
            segments=segments,
            value_col=cols.unit_spend,
            zero_value_customers="exclude",
        )
        result_df = seg.df
        assert result_df.loc[1, "segment_name"] == "Low"
        assert result_df.loc[2, "segment_name"] == "Low"
        assert result_df.loc[3, "segment_name"] == "High"
        assert result_df.loc[4, "segment_name"] == "High"

    def test_single_customer(self):
        """Test that the method correctly segments a DataFrame with only one customer."""
        df = pd.DataFrame({get_option("column.customer_id"): [1], cols.unit_spend: [100]})
        thresholds = [0.5, 1]
        segments = ["Low"]
        with pytest.raises(ValueError):
            ThresholdSegmentation(
                df=df,
                thresholds=thresholds,
                segments=segments,
            )

    def test_correct_aggregation_function(self):
        """Test that the correct aggregation function is applied for product_id custom segmentation."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5],
                "product_id": [3, 4, 4, 6, 1, 5, 7, 2, 2, 3, 2, 3, 4, 1],
            },
        )
        value_col = "product_id"
        agg_func = "nunique"

        my_seg = ThresholdSegmentation(
            df=df,
            value_col=value_col,
            agg_func=agg_func,
            thresholds=[0.2, 0.8, 1],
            segments=["Low", "Medium", "High"],
            zero_value_customers="separate_segment",
        )

        expected_result = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5],
                "product_id": [1, 4, 2, 2, 3],
                "segment_name": ["Low", "High", "Medium", "Medium", "Medium"],
            },
        )
        pd.testing.assert_frame_equal(my_seg.df.sort_values(cols.customer_id).reset_index(), expected_result)

    def test_thresholds_not_unique(self):
        """Test that the method raises an error when the thresholds are not unique."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100, 200, 300, 400, 500],
            },
        )
        thresholds = [0.5, 0.5, 0.8, 1]
        segments = ["Low", "Medium", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

    def test_thresholds_too_few_segments(self):
        """Test that the method raises an error when there are too few/many segments for the number of thresholds."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100, 200, 300, 400, 500],
            },
        )
        thresholds = [0.4, 0.6, 0.8, 1]
        segments = ["Low", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

        segments = ["Low", "Medium", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

    def test_thresholds_too_too_few_thresholds(self):
        """Test that the method raises an error when there are too few/many thresholds for the number of segments."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100, 200, 300, 400, 500],
            },
        )
        thresholds = [0.4, 1]
        segments = ["Low", "Medium", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

        thresholds = [0.2, 0.5, 0.6, 0.8, 1]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

    def test_with_custom_column_names(self):
        """Test ThresholdSegmentation with custom column names."""
        df = pd.DataFrame(
            {
                "buyer_id": [1, 2, 3, 4, 5, 1, 2, 3],
                "spend_amount": [100, 200, 50, 150, 75, 50, 100, 300],
                "purchase_date": pd.date_range("2023-01-01", periods=8),
            },
        )

        thresholds = [0.5, 1.0]
        segments = ["Low", "High"]

        with option_context("column.customer_id", "buyer_id", "column.unit_spend", "spend_amount"):
            seg = ThresholdSegmentation(
                df=df,
                thresholds=thresholds,
                segments=segments,
            )

            result_df = seg.df
            assert isinstance(result_df, pd.DataFrame), "Should execute successfully with custom column names"
            assert result_df.index.name == "buyer_id", "Should use custom customer_id column name as index"

    def test_segmentation_with_tied_spend_values(self):
        """Test customers with identical spend are segmented consistently using customer_id as tiebreaker."""
        # Create dataset with multiple customers having identical spend values
        df = pd.DataFrame(
            {
                cols.customer_id: [101, 102, 103, 104, 105, 106, 107, 108],
                cols.unit_spend: [100, 100, 100, 200, 200, 300, 300, 300],
            },
        )

        thresholds = [0.5, 1.0]
        segments = ["Low", "High"]

        seg = ThresholdSegmentation(
            df=df,
            thresholds=thresholds,
            segments=segments,
            zero_value_customers="exclude",
        )
        result = seg.df

        # Expected behavior: customers ordered by (spend, customer_id)
        # With 8 customers and threshold at 0.5:
        # - Percentile ranks: 0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0
        # - First 4 customers (ranks ≤ 0.5) → "Low", last 4 (ranks > 0.5) → "High"
        # - Ordering: (100, 101), (100, 102), (100, 103), (200, 104), (200, 105), (300, 106), (300, 107), (300, 108)
        expected_df = pd.DataFrame(
            {
                cols.customer_id: [101, 102, 103, 104, 105, 106, 107, 108],
                cols.unit_spend: [100, 100, 100, 200, 200, 300, 300, 300],
                "segment_name": ["Low", "Low", "Low", "Low", "High", "High", "High", "High"],
            },
        ).set_index(cols.customer_id)

        pd.testing.assert_frame_equal(result.sort_index(), expected_df)


class TestThresholdSegmentationGroupCol:
    """Tests for ThresholdSegmentation group_col functionality."""

    @pytest.fixture
    def store_transaction_df(self):
        """Return a DataFrame with customer transactions across multiple stores."""
        return pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1001, 1002, 1003],
                cols.unit_spend: [500.00, 100.00, 200.00, 50.00, 400.00, 300.00],
                cols.store_id: [2001, 2001, 2001, 2002, 2002, 2002],
            },
        )

    def test_segments_calculated_within_each_group(self, store_transaction_df):
        """Test that segments are calculated independently within each store group."""
        seg = ThresholdSegmentation(
            df=store_transaction_df,
            value_col=cols.unit_spend,
            thresholds=[0.5, 1.0],
            segments=["Low", "High"],
            group_col=cols.store_id,
            zero_value_customers="exclude",
        )

        # With percent_rank and 3 customers per store: rank 0->0.0 (Low), rank 1->0.5 (Low), rank 2->1.0 (High)
        # Store 2001: 1002=100 (Low), 1003=200 (Low), 1001=500 (High)
        # Store 2002: 1001=50 (Low), 1003=300 (Low), 1002=400 (High)
        expected_df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1001, 1002, 1003],
                cols.store_id: [2001, 2001, 2001, 2002, 2002, 2002],
                cols.unit_spend: [500.00, 100.00, 200.00, 50.00, 400.00, 300.00],
                "segment_name": ["High", "Low", "Low", "Low", "High", "Low"],
            },
        ).set_index([cols.customer_id, cols.store_id])

        pd.testing.assert_frame_equal(seg.df.sort_index(), expected_df.sort_index())

    def test_group_col_as_list(self, store_transaction_df):
        """Test that group_col works when provided as a list of columns."""
        store_transaction_df["category_id"] = [301, 301, 301, 301, 301, 301]

        seg = ThresholdSegmentation(
            df=store_transaction_df,
            value_col=cols.unit_spend,
            thresholds=[0.5, 1.0],
            segments=["Low", "High"],
            group_col=[cols.store_id, "category_id"],
            zero_value_customers="exclude",
        )

        result_df = seg.df
        assert result_df.index.names == [cols.customer_id, cols.store_id, "category_id"]

    def test_zero_value_separate_segment_within_groups(self):
        """Test that zero-value customers are correctly segmented within each group."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1001, 1002, 1003],
                cols.unit_spend: [500.00, 100.00, 0.00, 50.00, 400.00, 0.00],
                cols.store_id: [2001, 2001, 2001, 2002, 2002, 2002],
            },
        )

        seg = ThresholdSegmentation(
            df=df,
            value_col=cols.unit_spend,
            thresholds=[0.5, 1.0],
            segments=["Low", "High"],
            group_col=cols.store_id,
            zero_value_customers="separate_segment",
        )

        expected_df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1001, 1002, 1003],
                cols.store_id: [2001, 2001, 2001, 2002, 2002, 2002],
                cols.unit_spend: [500.00, 100.00, 0.00, 50.00, 400.00, 0.00],
                "segment_name": ["High", "Low", "Zero", "Low", "High", "Zero"],
            },
        ).set_index([cols.customer_id, cols.store_id])

        pd.testing.assert_frame_equal(seg.df.sort_index(), expected_df.sort_index())

    def test_missing_group_col_raises_error(self):
        """Test that missing group_col column raises ValueError."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003],
                cols.unit_spend: [100.00, 200.00, 300.00],
            },
        )

        with pytest.raises(ValueError, match="group_col references columns not present in the DataFrame"):
            ThresholdSegmentation(
                df=df,
                value_col=cols.unit_spend,
                thresholds=[0.5, 1.0],
                segments=["Low", "High"],
                group_col=cols.store_id,
            )
