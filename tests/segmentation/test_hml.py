"""Tests for the HMLSegmentation class."""

import pandas as pd
import pytest

from openretailscience.options import ColumnHelper, get_option
from openretailscience.segmentation.hml import HMLSegmentation

cols = ColumnHelper()


class TestHMLSegmentation:
    """Tests for the HMLSegmentation class."""

    @pytest.fixture
    def base_df(self):
        """Return a base DataFrame for testing."""
        return pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [1000, 200, 0, 500, 300],
            },
        )

    @pytest.mark.parametrize(
        ("zero_value_customers", "expected_segments"),
        [
            ("exclude", {1: "Heavy", 2: "Light", 4: "Medium", 5: "Light"}),
            ("include_with_light", {1: "Heavy", 2: "Light", 3: "Light", 4: "Medium", 5: "Light"}),
            ("separate_segment", {1: "Heavy", 2: "Light", 3: "Zero", 4: "Medium", 5: "Light"}),
        ],
    )
    def test_zero_spend_customer_handling(self, base_df, zero_value_customers, expected_segments):
        """Zero-spend customer 3 is dropped, ranked with Light, or put in its own Zero segment."""
        result_df = HMLSegmentation(base_df, zero_value_customers=zero_value_customers).df

        expected_df = base_df.merge(pd.DataFrame(expected_segments.items(), columns=[cols.customer_id, "segment_name"]))
        pd.testing.assert_frame_equal(result_df.sort_values(cols.customer_id, ignore_index=True), expected_df)

    # Raises ValueError if required columns are missing
    def test_raises_value_error_if_required_columns_missing(self, base_df):
        """Test that the method raises an error when the DataFrame is missing a required column."""
        with pytest.raises(ValueError):
            HMLSegmentation(base_df.drop(columns=[get_option("column.customer_id")]))

    # Validate that the input dataframe is not changed
    def test_input_dataframe_not_changed(self, base_df):
        """Test that the method does not alter the original DataFrame."""
        original_df = base_df.copy()

        hml_segmentation = HMLSegmentation(base_df)
        _ = hml_segmentation.df

        assert original_df.equals(base_df)  # Check if the original dataframe is not changed

    def test_alternate_value_col(self, base_df):
        """Test that the method correctly segments a DataFrame with an alternate value column."""
        base_df = base_df.rename(columns={cols.unit_spend: cols.unit_qty})
        hml_segmentation = HMLSegmentation(base_df, value_col=cols.unit_qty)
        result_df = hml_segmentation.df

        expected_df = base_df.assign(segment_name=["Heavy", "Light", "Zero", "Medium", "Light"])
        pd.testing.assert_frame_equal(result_df.sort_values(cols.customer_id, ignore_index=True), expected_df)


class TestHMLSegmentationGroupCol:
    """Tests for HMLSegmentation group_col functionality."""

    @pytest.fixture
    def store_transaction_df(self):
        """Return a DataFrame with customer transactions across multiple stores for HML testing."""
        # 5 customers per store to test HML distribution
        return pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1004, 1005, 1001, 1002, 1003, 1004, 1005],
                cols.unit_spend: [
                    # Store 2001: 1001 highest, 1005 lowest
                    1000.00,
                    500.00,
                    300.00,
                    200.00,
                    100.00,
                    # Store 2002: 1005 highest, 1001 lowest (reversed)
                    100.00,
                    200.00,
                    300.00,
                    500.00,
                    1000.00,
                ],
                cols.store_id: [2001] * 5 + [2002] * 5,
            },
        )

    def test_hml_segments_calculated_within_each_store(self, store_transaction_df):
        """Test that HML segments are calculated independently within each store."""
        hml = HMLSegmentation(
            df=store_transaction_df,
            value_col=cols.unit_spend,
            group_col=cols.store_id,
            zero_value_customers="exclude",
        )

        expected_df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1004, 1005, 1001, 1002, 1003, 1004, 1005],
                cols.store_id: [2001] * 5 + [2002] * 5,
                cols.unit_spend: [
                    1000.00,
                    500.00,
                    300.00,
                    200.00,
                    100.00,
                    100.00,
                    200.00,
                    300.00,
                    500.00,
                    1000.00,
                ],
                # Store 2001: 1001=Heavy, 1002=Medium, 1003/1004/1005=Light
                # Store 2002: 1005=Heavy, 1004=Medium, 1001/1002/1003=Light
                "segment_name": [
                    "Heavy",
                    "Medium",
                    "Light",
                    "Light",
                    "Light",
                    "Light",
                    "Light",
                    "Light",
                    "Medium",
                    "Heavy",
                ],
            },
        )

        pd.testing.assert_frame_equal(
            hml.df.sort_values([cols.store_id, cols.customer_id], ignore_index=True), expected_df
        )

    def test_hml_with_zero_value_and_group_col(self):
        """Test HML segmentation handles zero-value customers correctly within groups."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1004, 1005, 1001, 1002, 1003, 1004, 1005],
                cols.unit_spend: [
                    1000.00,
                    500.00,
                    300.00,
                    200.00,
                    0.00,
                    0.00,
                    200.00,
                    300.00,
                    500.00,
                    1000.00,
                ],
                cols.store_id: [2001] * 5 + [2002] * 5,
            },
        )

        hml = HMLSegmentation(
            df=df,
            value_col=cols.unit_spend,
            group_col=cols.store_id,
            zero_value_customers="separate_segment",
        )

        expected_df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1004, 1005, 1001, 1002, 1003, 1004, 1005],
                cols.store_id: [2001] * 5 + [2002] * 5,
                cols.unit_spend: [
                    1000.00,
                    500.00,
                    300.00,
                    200.00,
                    0.00,
                    0.00,
                    200.00,
                    300.00,
                    500.00,
                    1000.00,
                ],
                "segment_name": [
                    "Heavy",
                    "Medium",
                    "Light",
                    "Light",
                    "Zero",
                    "Zero",
                    "Light",
                    "Light",
                    "Medium",
                    "Heavy",
                ],
            },
        )

        pd.testing.assert_frame_equal(
            hml.df.sort_values([cols.store_id, cols.customer_id], ignore_index=True), expected_df
        )
