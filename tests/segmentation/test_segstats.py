"""Tests for the SegTransactionStats class."""

import warnings

import ibis
import numpy as np
import pandas as pd
import pytest
from sqlglot import exp

from openretailscience.options import ColumnHelper, get_option, option_context
from openretailscience.segmentation.segstats import SegTransactionStats, _resolve_group_key, cube, rollup

cols = ColumnHelper()


@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestCalcSegStats:
    """Tests for the _calc_seg_stats method."""

    @pytest.fixture
    def base_df(self):
        """Return a base DataFrame for testing."""
        return pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5],
                cols.unit_spend: [100.0, 200.0, 150.0, 300.0, 250.0],
                cols.transaction_id: [101, 102, 103, 104, 105],
                "segment_name": ["Premium", "Standard", "Premium", "Standard", "Premium"],
                cols.unit_qty: [10, 20, 15, 30, 25],
            },
        )

    def test_correctly_calculates_revenue_transactions_customers_per_segment(self, base_df):
        """Test that the method correctly calculates at the transaction-item level."""
        expected_output = pd.DataFrame(
            {
                "segment_name": ["Premium", "Standard", "Total"],
                cols.agg.unit_spend: [500.0, 500.0, 1000.0],
                cols.agg.transaction_id: [3, 2, 5],
                cols.agg.customer_id: [3, 2, 5],
                cols.agg.unit_qty: [50, 50, 100],
                cols.calc.spend_per_cust: [166.666667, 250.0, 200.0],
                cols.calc.spend_per_trans: [166.666667, 250.0, 200.0],
                cols.calc.trans_per_cust: [1.0, 1.0, 1.0],
                cols.calc.price_per_unit: [10.0, 10.0, 10.0],
                cols.calc.units_per_trans: [16.666667, 25.0, 20.0],
            },
        )
        segment_stats = (
            SegTransactionStats(base_df, "segment_name").df.sort_values("segment_name").reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_correctly_calculates_revenue_transactions_customers(self):
        """Test that the method correctly calculates at the transaction level."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100.0, 200.0, 150.0, 300.0, 250.0],
                cols.transaction_id: [101, 102, 103, 104, 105],
                "segment_name": ["Premium", "Standard", "Premium", "Standard", "Premium"],
            },
        )

        expected_output = pd.DataFrame(
            {
                "segment_name": ["Premium", "Standard", "Total"],
                cols.agg.unit_spend: [500.0, 500.0, 1000.0],
                cols.agg.transaction_id: [3, 2, 5],
                cols.agg.customer_id: [3, 2, 5],
                cols.calc.spend_per_cust: [166.666667, 250.0, 200.0],
                cols.calc.spend_per_trans: [166.666667, 250.0, 200.0],
                cols.calc.trans_per_cust: [1.0, 1.0, 1.0],
            },
        )

        segment_stats = SegTransactionStats(df, "segment_name").df.sort_values("segment_name").reset_index(drop=True)
        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_handles_dataframe_with_one_segment(self, base_df):
        """Test that the method correctly handles a DataFrame with only one segment."""
        df = base_df.copy()
        df["segment_name"] = "Premium"

        expected_output = pd.DataFrame(
            {
                "segment_name": ["Premium", "Total"],
                cols.agg.unit_spend: [1000.0, 1000.0],
                cols.agg.transaction_id: [5, 5],
                cols.agg.customer_id: [5, 5],
                cols.agg.unit_qty: [100, 100],
                cols.calc.spend_per_cust: [200.0, 200.0],
                cols.calc.spend_per_trans: [200.0, 200.0],
                cols.calc.trans_per_cust: [1.0, 1.0],
                cols.calc.price_per_unit: [10.0, 10.0],
                cols.calc.units_per_trans: [20.0, 20.0],
            },
        )

        segment_stats = SegTransactionStats(df, "segment_name").df.sort_values("segment_name").reset_index(drop=True)
        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_handles_dataframe_with_zero_net_units(self, base_df):
        """Test that the method correctly handles a DataFrame with a segment with net zero units."""
        df = base_df.copy()
        df[cols.unit_qty] = [10, 20, 15, 30, -25]

        expected_output = pd.DataFrame(
            {
                "segment_name": ["Premium", "Standard", "Total"],
                cols.agg.unit_spend: [500.0, 500.0, 1000.0],
                cols.agg.transaction_id: [3, 2, 5],
                cols.agg.customer_id: [3, 2, 5],
                cols.agg.unit_qty: [0, 50, 50],
                cols.calc.spend_per_cust: [166.666667, 250.0, 200.0],
                cols.calc.spend_per_trans: [166.666667, 250.0, 200.0],
                cols.calc.trans_per_cust: [1.0, 1.0, 1.0],
                cols.calc.price_per_unit: [np.nan, 10.0, 20.0],
                cols.calc.units_per_trans: [0, 25.0, 10.0],
            },
        )
        segment_stats = SegTransactionStats(df, "segment_name").df.sort_values("segment_name").reset_index(drop=True)

        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_excludes_total_row_when_calc_total_false(self, base_df):
        """Test that the method excludes the total row when calc_total=False."""
        expected_output = pd.DataFrame(
            {
                "segment_name": ["Premium", "Standard"],
                cols.agg.unit_spend: [500.0, 500.0],
                cols.agg.transaction_id: [3, 2],
                cols.agg.customer_id: [3, 2],
                cols.agg.unit_qty: [50, 50],
                cols.calc.spend_per_cust: [166.666667, 250.0],
                cols.calc.spend_per_trans: [166.666667, 250.0],
                cols.calc.trans_per_cust: [1.0, 1.0],
                cols.calc.price_per_unit: [10.0, 10.0],
                cols.calc.units_per_trans: [16.666667, 25.0],
            },
        )

        segment_stats = (
            SegTransactionStats(base_df, "segment_name", calc_total=False)
            .df.sort_values("segment_name")
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_sort_by_true_orders_segment_column_ascending(self, base_df):
        """Test that sort_by=True returns rows ordered ascending by the segment column."""
        result = SegTransactionStats(base_df, "segment_name", sort_by=True).df

        assert result["segment_name"].tolist() == ["Premium", "Standard", "Total"]

    def test_sort_by_true_orders_multiple_segment_columns_ascending(self, base_df):
        """Test that sort_by=True orders rows ascending by each segment column in turn."""
        df = base_df.assign(region=["North", "South", "East", "South", "East"])

        result = SegTransactionStats(df, ["segment_name", "region"], sort_by=True).df

        expected_pairs = [
            ("Premium", "East"),
            ("Premium", "North"),
            ("Standard", "South"),
            ("Total", "Total"),
        ]
        assert list(zip(result["segment_name"], result["region"], strict=True)) == expected_pairs

    def test_calculates_segment_stats_without_customer_data(self, base_df):
        """Test that the method correctly calculates segment statistics without customer data."""
        df_without_customer = base_df.drop(columns=[cols.customer_id])

        expected_output = pd.DataFrame(
            {
                "segment_name": ["Premium", "Standard", "Total"],
                cols.agg.unit_spend: [500.0, 500.0, 1000.0],
                cols.agg.transaction_id: [3, 2, 5],
                cols.agg.unit_qty: [50, 50, 100],
                cols.calc.spend_per_trans: [166.666667, 250.0, 200.0],
                cols.calc.price_per_unit: [10.0, 10.0, 10.0],
                cols.calc.units_per_trans: [16.666667, 25.0, 20.0],
            },
        )
        segment_stats = (
            SegTransactionStats(df_without_customer, "segment_name")
            .df.sort_values("segment_name")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(segment_stats, expected_output)


@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestSegTransactionStats:
    """Tests for the SegTransactionStats class."""

    def test_handles_empty_dataframe_with_errors(self):
        """Test that the method raises an error when the DataFrame is missing a required column."""
        df = pd.DataFrame(
            columns=[cols.unit_spend, cols.transaction_id, cols.unit_qty],
        )

        with pytest.raises(ValueError):
            SegTransactionStats(df, "segment_name")

    def test_raises_error_when_segment_col_is_empty_list(self):
        """Test that a ValueError is raised when segment_col is an empty list."""
        df = pd.DataFrame(
            {
                cols.customer_id: [101, 102, 103],
                cols.unit_spend: [150.0, 200.0, 175.0],
                cols.transaction_id: [1001, 1002, 1003],
            },
        )

        with pytest.raises(ValueError) as excinfo:
            SegTransactionStats(df, segment_col=[])

        assert "segment_col must not be an empty list" in str(excinfo.value)

    def test_multiple_segment_columns(self):
        """Test that the class correctly handles multiple segment columns."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 3, 3],
                cols.unit_spend: [100.0, 150.0, 200.0, 250.0, 300.0, 350.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "segment_name": [
                    "High Value",
                    "High Value",
                    "Medium Value",
                    "Medium Value",
                    "High Value",
                    "High Value",
                ],
                "region": ["North", "North", "South", "South", "East", "East"],
            },
        )

        # Test with a list of segment columns
        seg_stats = SegTransactionStats(df, ["segment_name", "region"])

        # Create expected DataFrame with the combinations actually produced
        expected_output = pd.DataFrame(
            {
                "segment_name": ["High Value", "High Value", "Medium Value", "Total"],
                "region": ["East", "North", "South", "Total"],
                cols.agg.unit_spend: [650.0, 250.0, 450.0, 1350.0],
                cols.agg.transaction_id: [2, 2, 2, 6],
                cols.agg.customer_id: [1, 1, 1, 3],
                cols.calc.spend_per_cust: [650.0, 250.0, 450.0, 450.0],
                cols.calc.spend_per_trans: [325.0, 125.0, 225.0, 225.0],
                cols.calc.trans_per_cust: [2.0, 2.0, 2.0, 2.0],
            },
        )

        # Sort both dataframes by the segment columns for consistent comparison
        result_df = seg_stats.df.sort_values(["segment_name", "region"]).reset_index(drop=True)
        expected_output = expected_output.sort_values(["segment_name", "region"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(result_df[expected_output.columns], expected_output)

    def test_rollup_disabled(self):
        """Test that rollup rows are not included when calc_rollup is False."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Jeans", "Shirts", "Sneakers", "Boots", "Boots"],
            },
        )

        # Create SegTransactionStats with rollup disabled
        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory"],
            calc_rollup=False,
            calc_total=True,
        )

        result_df = seg_stats.df

        # Test constants
        expected_rows_without_rollup = 5  # 4 detail + 1 grand total

        # Should have:
        # - 4 detail rows (Clothing-Jeans, Clothing-Shirts, Footwear-Sneakers, Footwear-Boots)
        # - 1 grand total row (Total-Total)
        assert len(result_df) == expected_rows_without_rollup

        # Check for the absence of rollup rows
        rollup_rows = result_df[(result_df["subcategory"] == "Total") & (result_df["category"] != "Total")]
        assert len(rollup_rows) == 0

    def test_custom_rollup_value_string(self):
        """Test using a custom string value for rollup totals."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Jeans", "Shirts", "Sneakers", "Boots", "Boots"],
            },
        )

        custom_value = "ALL"

        # Create SegTransactionStats with a custom rollup value
        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory"],
            calc_rollup=True,
            rollup_value=custom_value,
        )

        result_df = seg_stats.df

        # Test constants
        expected_custom_rollup_rows = 3  # 2 category subtotals + 1 grand total

        # Check for the presence of rollup rows with custom value
        rollup_rows = result_df[result_df["subcategory"] == custom_value]
        assert len(rollup_rows) == expected_custom_rollup_rows

        # Verify grand total row uses custom value
        grand_total = result_df[(result_df["category"] == custom_value) & (result_df["subcategory"] == custom_value)]
        assert len(grand_total) == 1

    def test_rollup_with_different_value_types(self):
        """Test rollup with different value types for each column."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Jeans", "Shirts", "Sneakers", "Boots", "Boots"],
                cols.product_id: [10, 20, 30, 40, 50, 60],
            },
        )

        # Create SegTransactionStats with a list of different value types
        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory", cols.product_id],
            calc_rollup=True,
            rollup_value=["ALL", "Subtotal", 0],  # String, String, Integer
        )

        result_df = seg_stats.df

        # Test constants
        expected_level1_rollups = 2  # level-1 rollup rows (category only)

        # Check for level-1 rollup rows (category only)
        level1_rollups = result_df[
            (result_df["subcategory"] == "Subtotal")
            & (result_df[cols.product_id] == 0)
            & (result_df["category"] != "ALL")
        ]
        assert len(level1_rollups) == expected_level1_rollups

        # Verify grand total row uses the specified values
        grand_total = result_df[
            (result_df["category"] == "ALL")
            & (result_df["subcategory"] == "Subtotal")
            & (result_df[cols.product_id] == 0)
        ]
        assert len(grand_total) == 1

    def test_rollup_value_list_wrong_length(self):
        """Test that an error is raised when rollup_value list length doesn't match segment_col length."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.unit_spend: [100.0, 200.0, 300.0],
                cols.transaction_id: [101, 102, 103],
                "category": ["Clothing", "Footwear", "Electronics"],
                "subcategory": ["Jeans", "Sneakers", "Phones"],
            },
        )

        # Attempt to create SegTransactionStats with mismatched list length
        with pytest.raises(ValueError) as excinfo:
            SegTransactionStats(
                df,
                segment_col=["category", "subcategory"],
                calc_rollup=True,
                rollup_value=["Total"],  # Only one value for two columns
            )

        assert "must match the number of segment columns" in str(excinfo.value)

    def test_extra_aggs_functionality(self):
        """Test that the extra_aggs parameter works correctly."""
        # Constants for expected values
        segment_a_store_count = 3  # Segment A has stores 1, 2, 4
        segment_b_store_count = 2  # Segment B has stores 1, 3
        total_store_count = 4  # Total has stores 1, 2, 3, 4

        segment_a_product_count = 3  # Segment A has products 10, 20, 40
        segment_b_product_count = 2  # Segment B has products 10, 30
        total_product_count = 4  # Total has products 10, 20, 30, 40
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 3, 3],
                cols.unit_spend: [100.0, 150.0, 200.0, 250.0, 300.0, 350.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "segment_name": ["Premium", "Premium", "Standard", "Standard", "Premium", "Premium"],
                cols.store_id: [1, 2, 1, 3, 2, 4],
                cols.product_id: [10, 20, 10, 30, 20, 40],
            },
        )

        # Test with a single extra aggregation
        seg_stats = SegTransactionStats(
            df,
            "segment_name",
            extra_aggs={"distinct_stores": (cols.store_id, "nunique")},
        )

        # Sort by segment_name to ensure consistent order
        result_df = seg_stats.df.sort_values("segment_name").reset_index(drop=True)

        assert result_df["distinct_stores"].to_list() == [
            segment_a_store_count,
            segment_b_store_count,
            total_store_count,
        ]

        # Test with multiple extra aggregations
        seg_stats_multi = SegTransactionStats(
            df,
            "segment_name",
            extra_aggs={
                "distinct_stores": (cols.store_id, "nunique"),
                "distinct_products": (cols.product_id, "nunique"),
            },
        )

        # Sort by segment_name to ensure consistent order
        result_df_multi = seg_stats_multi.df.sort_values("segment_name").reset_index(drop=True)

        assert result_df_multi["distinct_products"].to_list() == [
            segment_a_product_count,
            segment_b_product_count,
            total_product_count,
        ]

    def test_extra_aggs_with_invalid_column(self):
        """Test that an error is raised when an invalid column is specified in extra_aggs."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.unit_spend: [100.0, 200.0, 300.0],
                cols.transaction_id: [101, 102, 103],
                "segment_name": ["Premium", "Standard", "Premium"],
            },
        )

        with pytest.raises(ValueError) as excinfo:
            SegTransactionStats(df, "segment_name", extra_aggs={"invalid_agg": ("nonexistent_column", "nunique")})

        assert "does not exist in the data" in str(excinfo.value)

    def test_extra_aggs_with_invalid_function(self):
        """Test that an error is raised when an invalid function is specified in extra_aggs."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.unit_spend: [100.0, 200.0, 300.0],
                cols.transaction_id: [101, 102, 103],
                "segment_name": ["Premium", "Standard", "Premium"],
            },
        )

        with pytest.raises(ValueError) as excinfo:
            SegTransactionStats(df, "segment_name", extra_aggs={"invalid_agg": (cols.customer_id, "invalid_function")})

        assert "not available for column" in str(excinfo.value)

    def test_with_custom_column_names(self):
        """Test SegTransactionStats with custom column names."""
        custom_columns = {
            "column.customer_id": "cust_id",
            "column.unit_spend": "revenue",
            "column.transaction_id": "trans_id",
            "column.unit_quantity": "quantity",
        }

        custom_df = pd.DataFrame(
            {
                "cust_id": [1, 1, 2, 2],
                "revenue": [100.0, 150.0, 200.0, 250.0],
                "trans_id": [101, 102, 103, 104],
                "segment_name": ["Premium", "Premium", "Standard", "Standard"],
                "quantity": [2, 3, 4, 5],
            },
        )

        with option_context(*[item for pair in custom_columns.items() for item in pair]):
            result = SegTransactionStats(custom_df, segment_col="segment_name").df

        expected_output = pd.DataFrame(
            {
                "segment_name": ["Premium", "Standard", "Total"],
                cols.agg.unit_spend: [250.0, 450.0, 700.0],
                cols.agg.transaction_id: [2, 2, 4],
                cols.agg.customer_id: [1, 1, 2],
                cols.agg.unit_qty: [5, 9, 14],
                cols.calc.spend_per_cust: [250.0, 450.0, 350.0],
                cols.calc.spend_per_trans: [125.0, 225.0, 175.0],
                cols.calc.trans_per_cust: [2.0, 2.0, 2.0],
                cols.calc.price_per_unit: [50.0, 50.0, 50.0],
                cols.calc.units_per_trans: [2.5, 4.5, 3.5],
            },
        )
        result = result.sort_values("segment_name").reset_index(drop=True)
        pd.testing.assert_frame_equal(result[expected_output.columns], expected_output)

    def test_complete_rollup_hierarchy_two_columns(self):
        """Expect prefix and suffix rollups plus grand total when calc_rollup and calc_total are True.

        Expected rows (with rollup_value defaulting to "Total"):
        - Detail: (Clothing, Jeans), (Clothing, Shirts), (Footwear, Jeans), (Footwear, Shirts)
        - Prefix rollups: (Clothing, Total), (Footwear, Total)
        - Suffix rollups: (Total, Jeans), (Total, Shirts)
        - Grand total: (Total, Total)
        Total expected rows = 9.
        """
        # Use a small in-memory dataset (self-contained, no external files)
        df_sample = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0],
                cols.transaction_id: [101, 102, 103, 104],
                "category_0_name": ["Clothing", "Clothing", "Footwear", "Footwear"],
                "category_1_name": ["Jeans", "Shirts", "Jeans", "Shirts"],
            },
        )

        segment_cols = ["category_0_name", "category_1_name"]
        measure_col = cols.unit_spend

        # Run SegTransactionStats with rollups
        seg_stats = SegTransactionStats(
            df_sample,
            segment_col=segment_cols,
            calc_rollup=True,
            calc_total=True,
        )

        result_df = seg_stats.df

        # Convert to dicts for order-insensitive comparison
        records = result_df.to_dict(orient="records")

        # Dynamically compute expected sums
        expected = {}

        # Detail rows
        for (cat0, cat1), group in df_sample.groupby(segment_cols):
            expected[(cat0, cat1)] = group[measure_col].sum()

        # Prefix rollups (category subtotal)
        for cat0, group in df_sample.groupby("category_0_name"):
            expected[(cat0, "Total")] = group[measure_col].sum()

        # Suffix rollups (subcategory subtotal)
        for cat1, group in df_sample.groupby("category_1_name"):
            expected[("Total", cat1)] = group[measure_col].sum()

        # Grand total
        expected[("Total", "Total")] = df_sample[measure_col].sum()

        # Validate each expected row exists and sums match
        for (cat0, cat1), expected_sum in expected.items():
            matches = [r for r in records if r["category_0_name"] == cat0 and r["category_1_name"] == cat1]
            assert len(matches) == 1, f"Missing row for ({cat0}, {cat1})"
            assert matches[0][cols.agg.unit_spend] == expected_sum

    def test_complete_rollup_hierarchy_three_columns(self):
        """Expect prefix + suffix rollups + grand total with 3 segment columns.

        Columns: region, category, subcategory
        Expected rows:
        - Detail: 2 regions x 2 categories x 2 subcategories = 8
        - Prefix rollups:
            (region, category, Total) → 4
            (region, Total, Total) → 2
        - Suffix rollups:
            (Total, category, subcategory) → 4
            (Total, Total, subcategory) → 2
        - Grand total: (Total, Total, Total) → 1

        Total expected rows = 21.
        """
        df = pd.DataFrame(
            {
                cols.customer_id: range(1, 9),
                cols.unit_spend: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                cols.transaction_id: range(101, 109),
                "region": ["North", "North", "North", "North", "South", "South", "South", "South"],
                "category": ["Clothing", "Clothing", "Footwear", "Footwear"] * 2,
                "subcategory": ["Jeans", "Shirts"] * 4,
            },
        )

        seg_stats = SegTransactionStats(
            df,
            segment_col=["region", "category", "subcategory"],
            calc_rollup=True,
            calc_total=True,
        )

        result_df = seg_stats.df

        # Row count check - catches duplicates!
        expected_rows = 21
        assert len(result_df) == expected_rows, f"Expected {expected_rows} rows, got {len(result_df)}"

        # Spot check: one prefix rollup
        north_clothing_total = result_df[
            (result_df["region"] == "North")
            & (result_df["category"] == "Clothing")
            & (result_df["subcategory"] == "Total")
        ]
        assert len(north_clothing_total) == 1
        assert north_clothing_total[cols.agg.unit_spend].to_numpy()[0] == 10.0 + 20.0

        # Spot check: one suffix rollup (Total, Total, Jeans)
        total_total_jeans = result_df[
            (result_df["region"] == "Total")
            & (result_df["category"] == "Total")
            & (result_df["subcategory"] == "Jeans")
        ]
        assert len(total_total_jeans) == 1
        assert total_total_jeans[cols.agg.unit_spend].to_numpy()[0] == 10.0 + 30.0 + 50.0 + 70.0

        # Grand total
        grand_total = result_df[
            (result_df["region"] == "Total")
            & (result_df["category"] == "Total")
            & (result_df["subcategory"] == "Total")
        ]
        assert len(grand_total) == 1
        assert grand_total[cols.agg.unit_spend].to_numpy()[0] == sum([10, 20, 30, 40, 50, 60, 70, 80])

    def test_rollup_enabled_total_disabled(self):
        """Test that rollup rows are included but grand total is excluded when calc_rollup=True, calc_total=False."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Jeans", "Shirts", "Sneakers", "Boots", "Boots"],
            },
        )

        # Create SegTransactionStats with rollup enabled but total disabled
        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory"],
            calc_rollup=True,
            calc_total=False,
        )

        result_df = seg_stats.df

        # Expected counts:
        # - 4 detail rows (Clothing-Jeans, Clothing-Shirts, Footwear-Sneakers, Footwear-Boots)
        # - 2 prefix rollup rows (Clothing-Total, Footwear-Total)
        # - NO suffix rollup rows (to avoid "Total" in category when calc_total=False)
        # - NO grand total row (Total-Total)
        expected_rows_with_rollup_no_total = 6

        assert len(result_df) == expected_rows_with_rollup_no_total

        # Test constants
        expected_prefix_rollups = 2  # Clothing-Total, Footwear-Total

        # Check for presence of prefix rollup rows
        prefix_rollups = result_df[(result_df["subcategory"] == "Total") & (result_df["category"] != "Total")]
        assert len(prefix_rollups) == expected_prefix_rollups

        # Check for absence of suffix rollup rows (should not exist when calc_total=False)
        suffix_rollups = result_df[(result_df["category"] == "Total") & (result_df["subcategory"] != "Total")]
        assert len(suffix_rollups) == 0

        # Check for absence of grand total row
        grand_total = result_df[(result_df["category"] == "Total") & (result_df["subcategory"] == "Total")]
        assert len(grand_total) == 0

    def test_rollup_disabled_total_disabled(self):
        """Test that only detail rows are included when calc_rollup=False, calc_total=False."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Jeans", "Shirts", "Sneakers", "Boots", "Boots"],
            },
        )

        # Create SegTransactionStats with both rollup and total disabled
        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory"],
            calc_rollup=False,
            calc_total=False,
        )

        result_df = seg_stats.df

        # Expected: only 4 detail rows (Clothing-Jeans, Clothing-Shirts, Footwear-Sneakers, Footwear-Boots)
        expected_rows_detail_only = 4

        assert len(result_df) == expected_rows_detail_only

        # Check for absence of any rollup rows
        rollup_rows = result_df[(result_df["subcategory"] == "Total") | (result_df["category"] == "Total")]
        assert len(rollup_rows) == 0

        # Verify all rows are detail rows (no "Total" values)
        assert "Total" not in result_df["category"].to_numpy()
        assert "Total" not in result_df["subcategory"].to_numpy()


@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestUnknownCustomerTracking:
    """Tests for unknown customer tracking functionality."""

    @pytest.mark.parametrize(
        ("unknown_value", "customer_ids"),
        [
            (-1, [1, 2, -1, 3]),  # int value
            ("UNKNOWN", ["C1", "C2", "UNKNOWN", "C3"]),  # string value
        ],
    )
    def test_unknown_customer_input_types(self, unknown_value, customer_ids):
        """Test unknown customer tracking with different input value types."""
        df = pd.DataFrame(
            {
                cols.customer_id: customer_ids,
                cols.unit_spend: [100.0, 200.0, 150.0, 300.0],
                cols.transaction_id: [101, 102, 103, 104],
                "segment_name": ["Premium", "Premium", "Premium", "Standard"],
            },
        )

        seg_stats = SegTransactionStats(df, "segment_name", unknown_customer_value=unknown_value)
        result_df = seg_stats.df.sort_values("segment_name").reset_index(drop=True)

        expected_output = pd.DataFrame(
            {
                "segment_name": ["Premium", "Standard", "Total"],
                cols.agg.unit_spend: [300.0, 300.0, 600.0],
                cols.agg.transaction_id: [2, 1, 3],
                cols.agg.customer_id: [2, 1, 3],
                cols.calc.spend_per_cust: [150.0, 300.0, 200.0],
                cols.calc.spend_per_trans: [150.0, 300.0, 200.0],
                cols.calc.trans_per_cust: [1.0, 1.0, 1.0],
                cols.agg.unit_spend_unknown: [150.0, 0.0, 150.0],
                cols.agg.transaction_id_unknown: [1, 0, 1],
                cols.calc.spend_per_trans_unknown: [150.0, np.nan, 150.0],
                cols.agg.unit_spend_total: [450.0, 300.0, 750.0],
                cols.agg.transaction_id_total: [3, 1, 4],
                cols.calc.spend_per_trans_total: [150.0, 300.0, 187.5],
            },
        )

        pd.testing.assert_frame_equal(result_df, expected_output)

    def test_unknown_customer_with_ibis_literal(self):
        """Test unknown customer tracking with ibis literal."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, -1, 3],
                cols.unit_spend: [100.0, 200.0, 150.0, 300.0],
                cols.transaction_id: [101, 102, 103, 104],
                "segment_name": ["Premium", "Premium", "Premium", "Standard"],
            },
        )

        seg_stats = SegTransactionStats(df, "segment_name", unknown_customer_value=ibis.literal(-1))
        result_df = seg_stats.df.sort_values("segment_name").reset_index(drop=True)

        expected_output = pd.DataFrame(
            {
                "segment_name": ["Premium", "Standard", "Total"],
                cols.agg.unit_spend: [300.0, 300.0, 600.0],
                cols.agg.transaction_id: [2, 1, 3],
                cols.agg.customer_id: [2, 1, 3],
                cols.calc.spend_per_cust: [150.0, 300.0, 200.0],
                cols.calc.spend_per_trans: [150.0, 300.0, 200.0],
                cols.calc.trans_per_cust: [1.0, 1.0, 1.0],
                cols.agg.unit_spend_unknown: [150.0, 0.0, 150.0],
                cols.agg.transaction_id_unknown: [1, 0, 1],
                cols.calc.spend_per_trans_unknown: [150.0, np.nan, 150.0],
                cols.agg.unit_spend_total: [450.0, 300.0, 750.0],
                cols.agg.transaction_id_total: [3, 1, 4],
                cols.calc.spend_per_trans_total: [150.0, 300.0, 187.5],
            },
        )

        pd.testing.assert_frame_equal(result_df, expected_output)

    def test_unknown_customer_with_boolean_expression(self):
        """Test unknown customer tracking with boolean expression."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, -1, -2, 3],
                cols.unit_spend: [100.0, 200.0, 150.0, 250.0, 300.0],
                cols.transaction_id: [101, 102, 103, 104, 105],
                "segment_name": ["Premium", "Premium", "Premium", "Premium", "Standard"],
            },
        )

        data_table = ibis.memtable(df)
        seg_stats = SegTransactionStats(
            data_table,
            "segment_name",
            unknown_customer_value=data_table[cols.customer_id] < 0,
        )
        result_df = seg_stats.df.sort_values("segment_name").reset_index(drop=True)

        expected_output = pd.DataFrame(
            {
                "segment_name": ["Premium", "Standard", "Total"],
                cols.agg.unit_spend: [300.0, 300.0, 600.0],
                cols.agg.transaction_id: [2, 1, 3],
                cols.agg.customer_id: [2, 1, 3],
                cols.calc.spend_per_cust: [150.0, 300.0, 200.0],
                cols.calc.spend_per_trans: [150.0, 300.0, 200.0],
                cols.calc.trans_per_cust: [1.0, 1.0, 1.0],
                cols.agg.unit_spend_unknown: [400.0, 0.0, 400.0],
                cols.agg.transaction_id_unknown: [2, 0, 2],
                cols.calc.spend_per_trans_unknown: [200.0, np.nan, 200.0],
                cols.agg.unit_spend_total: [700.0, 300.0, 1000.0],
                cols.agg.transaction_id_total: [4, 1, 5],
                cols.calc.spend_per_trans_total: [175.0, 300.0, 200.0],
            },
        )

        pd.testing.assert_frame_equal(result_df, expected_output)

    def test_unknown_customer_with_quantity(self):
        """Test unknown customer tracking with quantity columns."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, -1, 3],
                cols.unit_spend: [100.0, 200.0, 150.0, 300.0],
                cols.transaction_id: [101, 102, 103, 104],
                cols.unit_qty: [10, 20, 15, 30],
                "segment_name": ["Premium", "Premium", "Premium", "Standard"],
            },
        )

        seg_stats = SegTransactionStats(df, "segment_name", unknown_customer_value=-1)
        result_df = seg_stats.df.sort_values("segment_name").reset_index(drop=True)

        expected_output = pd.DataFrame(
            {
                "segment_name": ["Premium", "Standard", "Total"],
                cols.agg.unit_spend: [300.0, 300.0, 600.0],
                cols.agg.transaction_id: [2, 1, 3],
                cols.agg.customer_id: [2, 1, 3],
                cols.agg.unit_qty: [30, 30, 60],
                cols.calc.spend_per_cust: [150.0, 300.0, 200.0],
                cols.calc.spend_per_trans: [150.0, 300.0, 200.0],
                cols.calc.trans_per_cust: [1.0, 1.0, 1.0],
                cols.calc.price_per_unit: [10.0, 10.0, 10.0],
                cols.calc.units_per_trans: [15.0, 30.0, 20.0],
                cols.agg.unit_spend_unknown: [150.0, 0.0, 150.0],
                cols.agg.transaction_id_unknown: [1, 0, 1],
                cols.agg.unit_qty_unknown: [15, 0, 15],
                cols.calc.spend_per_trans_unknown: [150.0, np.nan, 150.0],
                cols.calc.price_per_unit_unknown: [10.0, np.nan, 10.0],
                cols.calc.units_per_trans_unknown: [15.0, np.nan, 15.0],
                cols.agg.unit_spend_total: [450.0, 300.0, 750.0],
                cols.agg.transaction_id_total: [3, 1, 4],
                cols.agg.unit_qty_total: [45, 30, 75],
                cols.calc.spend_per_trans_total: [150.0, 300.0, 187.5],
                cols.calc.price_per_unit_total: [10.0, 10.0, 10.0],
                cols.calc.units_per_trans_total: [15.0, 30.0, 18.75],
            },
        )

        pd.testing.assert_frame_equal(result_df, expected_output)

    def test_unknown_customer_error_when_customer_id_missing(self):
        """Test that error is raised when customer_id column is missing."""
        df = pd.DataFrame(
            {
                cols.unit_spend: [100.0, 200.0],
                cols.transaction_id: [101, 102],
                "segment_name": ["Premium", "Standard"],
            },
        )

        with pytest.raises(ValueError) as excinfo:
            SegTransactionStats(df, "segment_name", unknown_customer_value=-1)

        assert "required when unknown_customer_value parameter is specified" in str(excinfo.value)

    def test_unknown_customer_with_rollups(self):
        """Test unknown customer tracking with rollups enabled."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, -1, 2, -1],
                cols.unit_spend: [100.0, 150.0, 200.0, 250.0],
                cols.transaction_id: [101, 102, 103, 104],
                "category": ["Clothing", "Clothing", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Shirts", "Sneakers", "Boots"],
            },
        )

        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory"],
            calc_rollup=True,
            unknown_customer_value=-1,
        )
        result_df = seg_stats.df

        # Check that rollup rows include unknown and total columns
        total_row = result_df[(result_df["category"] == "Total") & (result_df["subcategory"] == "Total")]
        assert len(total_row) == 1
        assert cols.agg.unit_spend_unknown in total_row.columns
        assert cols.agg.unit_spend_total in total_row.columns
        expected_unknown_spend = 400.0
        expected_total_spend = 700.0
        assert total_row[cols.agg.unit_spend_unknown].iloc[0] == expected_unknown_spend
        assert total_row[cols.agg.unit_spend_total].iloc[0] == expected_total_spend

    def test_unknown_customer_with_extra_aggs(self):
        """Test unknown customer tracking with extra aggregations."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, -1, 2, -1],
                cols.unit_spend: [100.0, 150.0, 200.0, 250.0],
                cols.transaction_id: [101, 102, 103, 104],
                "segment_name": ["Premium", "Premium", "Standard", "Standard"],
                cols.store_id: [1, 2, 1, 3],
            },
        )

        stores_agg = get_option("column.agg.store_id")
        stores_unknown = ColumnHelper.join_options("column.agg.store_id", "column.suffix.unknown_customer")
        stores_total = ColumnHelper.join_options("column.agg.store_id", "column.suffix.total")

        seg_stats = SegTransactionStats(
            df,
            "segment_name",
            unknown_customer_value=-1,
            extra_aggs={stores_agg: (cols.store_id, "nunique")},
        )
        result_df = seg_stats.df.sort_values("segment_name").reset_index(drop=True)

        # Verify the three extra-agg variants for segment A
        expected_identified_stores = 1  # Segment A identified: store 1
        expected_unknown_stores = 1  # Segment A unknown: store 2
        expected_total_stores = 2  # Segment A total: stores 1, 2
        assert result_df.loc[0, stores_agg] == expected_identified_stores
        assert result_df.loc[0, stores_unknown] == expected_unknown_stores
        assert result_df.loc[0, stores_total] == expected_total_stores


class TestGenerateGroupingSets:
    """Test the _generate_grouping_sets helper method."""

    @pytest.mark.parametrize(
        ("segment_col", "calc_total", "calc_rollup", "expected"),
        [
            (["region", "store", "product"], False, False, [("region", "store", "product")]),
            (["region", "store", "product"], True, False, [("region", "store", "product"), ()]),
            (
                ["region", "store", "product"],
                False,
                True,
                [("region", "store", "product"), ("region",), ("region", "store")],
            ),
            (
                ["region", "store", "product"],
                True,
                True,
                [
                    ("region", "store", "product"),
                    ("region",),
                    ("region", "store"),
                    ("store", "product"),
                    ("product",),
                    (),
                ],
            ),
            (["region", "store"], True, True, [("region", "store"), ("region",), ("store",), ()]),
            (["region"], False, False, [("region",)]),
            (["region"], True, True, [("region",), ()]),
            (
                ["region", "store", "category", "product"],
                True,
                True,
                [
                    ("region", "store", "category", "product"),
                    ("region",),
                    ("region", "store"),
                    ("region", "store", "category"),
                    ("store", "category", "product"),
                    ("category", "product"),
                    ("product",),
                    (),
                ],
            ),
            (
                ["region", "store", "category", "product"],
                False,
                True,
                [
                    ("region", "store", "category", "product"),
                    ("region",),
                    ("region", "store"),
                    ("region", "store", "category"),
                ],
            ),
        ],
        ids=[
            "no_rollup_no_total",
            "no_rollup_with_total",
            "rollup_without_total",
            "rollup_with_total",
            "two_columns_rollup_with_total",
            "single_column_no_rollup_no_total",
            "single_column_with_total",
            "four_columns_rollup_with_total",
            "four_columns_rollup_without_total",
        ],
    )
    def test_generate_grouping_sets_legacy(self, segment_col, calc_total, calc_rollup, expected):
        """_generate_grouping_sets returns the expected sets across legacy calc_total/calc_rollup combinations."""
        result = SegTransactionStats._generate_grouping_sets(
            segment_col=segment_col,
            calc_total=calc_total,
            calc_rollup=calc_rollup,
        )
        assert result == expected

    def test_total_mode_generates_full_detail_plus_grand_total(self):
        """Test 'total' mode produces full segment detail plus grand total."""
        result = SegTransactionStats._generate_grouping_sets(
            segment_col=["region", "store", "product"],
            grouping_sets="total",
        )

        expected = [
            ("region", "store", "product"),  # Full detail
            (),  # Grand total
        ]

        assert set(result) == set(expected)
        assert len(result) == len(expected)

    def test_total_mode_integration(self):
        """Test 'total' mode produces correct aggregations with real data."""
        cols = ColumnHelper()
        data = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 3, 3],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "region": ["North", "North", "South", "South", "North", "South"],
                "store": ["Store_A", "Store_A", "Store_B", "Store_B", "Store_A", "Store_B"],
                cols.unit_spend: [100, 150, 200, 250, 300, 350],
            },
        )

        stats = SegTransactionStats(
            data=data,
            segment_col=["region", "store"],
            grouping_sets="total",
        )

        result = stats.df

        # Expected: North/Store_A detail + South/Store_B detail + grand total
        expected = pd.DataFrame(
            {
                "region": ["North", "South", "Total"],
                "store": ["Store_A", "Store_B", "Total"],
                cols.agg.unit_spend: [550, 800, 1350],
            },
        )

        # Sort and compare
        result_subset = (
            result[["region", "store", cols.agg.unit_spend]].sort_values(["region", "store"]).reset_index(drop=True)
        )
        expected_sorted = expected.sort_values(["region", "store"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(result_subset, expected_sorted)


class TestGroupingSetsRollupMode:
    """Test ROLLUP mode grouping_sets parameter."""

    @pytest.mark.parametrize(
        ("segment_col", "expected"),
        [
            (
                ["region", "store", "product"],
                [("region", "store", "product"), ("region", "store"), ("region",), ()],
            ),
            (
                ["category", "brand"],
                [("category", "brand"), ("category",), ()],
            ),
            (
                ["region"],
                [("region",), ()],
            ),
        ],
    )
    def test_generate_grouping_sets_rollup_mode(self, segment_col, expected):
        """Test ROLLUP mode generates hierarchical grouping sets."""
        result = SegTransactionStats._generate_grouping_sets(
            segment_col=segment_col,
            grouping_sets="rollup",
        )
        assert result == expected

    @pytest.mark.parametrize(
        ("grouping_sets", "calc_total", "calc_rollup", "should_raise"),
        [
            ("rollup", True, None, True),  # calc_total explicitly set -> error
            ("rollup", None, False, True),  # calc_rollup explicitly set -> error
            ("rollup", True, False, True),  # both explicitly set -> error
            ("rollup", None, None, False),  # both None -> valid
            ("cube", True, None, True),  # CUBE: calc_total explicitly set -> error
            ("cube", None, False, True),  # CUBE: calc_rollup explicitly set -> error
            ("cube", True, False, True),  # CUBE: both explicitly set -> error
            ("cube", None, None, False),  # CUBE: both None -> valid
            ([("region",)], True, None, True),  # Custom: calc_total explicitly set -> error
            ([("region",)], None, False, True),  # Custom: calc_rollup explicitly set -> error
            ([("region",)], True, False, True),  # Custom: both explicitly set -> error
            ([("region",)], None, None, False),  # Custom: both None -> valid
        ],
    )
    def test_grouping_sets_mutual_exclusivity(self, grouping_sets, calc_total, calc_rollup, should_raise):
        """Test that grouping_sets validates mutual exclusivity with calc_total/calc_rollup for all modes."""
        if should_raise:
            with pytest.raises(ValueError, match="Cannot use grouping_sets with calc_total or calc_rollup"):
                SegTransactionStats._validate_grouping_sets_params(
                    grouping_sets=grouping_sets,
                    calc_total=calc_total,
                    calc_rollup=calc_rollup,
                )
        else:
            # Should not raise - validation passes
            SegTransactionStats._validate_grouping_sets_params(
                grouping_sets=grouping_sets,
                calc_total=calc_total,
                calc_rollup=calc_rollup,
            )

    def test_grouping_sets_invalid_string_value(self):
        """Test that invalid string value raises error."""
        with pytest.raises(ValueError, match="grouping_sets must be 'rollup', 'cube'"):
            SegTransactionStats._validate_grouping_sets_params(
                grouping_sets="invalid",
                calc_total=None,
                calc_rollup=None,
            )

    def test_legacy_mode_warns_on_implicit_default(self):
        """A FutureWarning is raised in legacy mode when relying on the implicit calc_total default."""
        with pytest.warns(FutureWarning, match="calc_total parameter is deprecated"):
            SegTransactionStats._validate_grouping_sets_params(
                grouping_sets=None,
                calc_total=None,
                calc_rollup=None,
            )

    @pytest.mark.parametrize(
        ("calc_total", "calc_rollup"),
        [(False, True), (True, None), (False, None), (None, True), (None, False)],
    )
    def test_legacy_mode_no_warning_with_explicit_flags(self, calc_total, calc_rollup):
        """No deprecation warning when calc_total or calc_rollup is set explicitly."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # turn any warning into an error
            SegTransactionStats._validate_grouping_sets_params(
                grouping_sets=None,
                calc_total=calc_total,
                calc_rollup=calc_rollup,
            )

    def test_rollup_mode_integration(self):
        """Test ROLLUP mode produces correct aggregations."""
        # Create test data
        data = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 3, 3],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "region": ["North", "North", "South", "South", "North", "South"],
                "store": ["Store_A", "Store_A", "Store_B", "Store_B", "Store_C", "Store_C"],
                cols.unit_spend: [100, 150, 200, 250, 300, 350],
            },
        )

        # Create stats with ROLLUP mode
        stats = SegTransactionStats(
            data=data,
            segment_col=["region", "store"],
            grouping_sets="rollup",
        )

        result = stats.df

        # Create expected results for key aggregations
        # Note: Only checking unit_spend column for simplicity; full test would check all columns
        # 4 detail rows: North/Store_A, North/Store_C, South/Store_B, South/Store_C
        # 2 rollup rows: North/Total, South/Total
        # 1 grand total: Total/Total
        expected = pd.DataFrame(
            {
                "region": ["North", "North", "South", "South", "North", "South", "Total"],
                "store": ["Store_A", "Store_C", "Store_B", "Store_C", "Total", "Total", "Total"],
                cols.agg.unit_spend: [250, 300, 450, 350, 550, 800, 1350],
            },
        )

        # Verify result has correct number of rows (4 detail + 2 region rollups + 1 grand total = 7)
        expected_row_count = 7
        assert len(result) == expected_row_count

        # Sort both dataframes for consistent comparison
        result_subset = (
            result[["region", "store", cols.agg.unit_spend]].sort_values(["region", "store"]).reset_index(drop=True)
        )
        expected_sorted = expected.sort_values(["region", "store"]).reset_index(drop=True)

        # Compare using pandas assert_frame_equal
        pd.testing.assert_frame_equal(result_subset, expected_sorted)

    def test_validate_extra_aggs_invalid_column(self):
        """Test that _validate_extra_aggs raises ValueError for invalid column name."""
        data = pd.DataFrame({"region": ["North", "South"], cols.unit_spend: [100, 200]})
        table = ibis.memtable(data)
        extra_aggs = {"total_sales": ("invalid_column", "sum")}

        with pytest.raises(ValueError, match="Column 'invalid_column' specified in extra_aggs does not exist"):
            SegTransactionStats._validate_extra_aggs(table, extra_aggs)

    def test_validate_extra_aggs_invalid_function(self):
        """Test that _validate_extra_aggs raises ValueError for invalid aggregation function."""
        data = pd.DataFrame({"region": ["North", "South"], cols.unit_spend: [100, 200]})
        table = ibis.memtable(data)
        extra_aggs = {"total_sales": (cols.unit_spend, "invalid_func")}

        with pytest.raises(ValueError, match="Aggregation function 'invalid_func' not available"):
            SegTransactionStats._validate_extra_aggs(table, extra_aggs)

    def test_generate_grouping_sets_cube_two_columns(self):
        """Test CUBE mode generates all 2^n combinations for two columns."""
        result = SegTransactionStats._generate_grouping_sets(
            segment_col=["region", "store"],
            grouping_sets="cube",
        )
        expected = [
            ("region", "store"),  # full detail
            ("region",),  # region only
            ("store",),  # store only
            (),  # grand total
        ]
        expected_count_two_columns = 4  # 2^2 = 4
        # Convert to sets for order-independent comparison
        assert set(result) == set(expected)
        assert len(result) == expected_count_two_columns

    def test_generate_grouping_sets_cube_three_columns(self):
        """Test CUBE mode generates all 2^n combinations for three columns."""
        result = SegTransactionStats._generate_grouping_sets(
            segment_col=["region", "store", "product"],
            grouping_sets="cube",
        )
        expected = [
            ("region", "store", "product"),  # full detail
            ("region", "store"),  # region + store
            ("region", "product"),  # region + product
            ("region",),  # region only
            ("store", "product"),  # store + product
            ("store",),  # store only
            ("product",),  # product only
            (),  # grand total
        ]
        expected_count_three_columns = 8  # 2^3 = 8
        # Convert to sets for order-independent comparison
        assert set(result) == set(expected)
        assert len(result) == expected_count_three_columns

    def test_generate_grouping_sets_cube_single_column(self):
        """Test CUBE mode with single segment column."""
        result = SegTransactionStats._generate_grouping_sets(
            segment_col=["region"],
            grouping_sets="cube",
        )
        expected = [
            ("region",),  # detail
            (),  # grand total
        ]
        expected_count_single_column = 2  # 2^1 = 2
        assert set(result) == set(expected)
        assert len(result) == expected_count_single_column

    def test_cube_mode_integration(self):
        """Test CUBE mode produces correct aggregations across all dimension combinations."""
        # Create test data
        data = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 3, 3],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "region": ["North", "North", "South", "South", "North", "South"],
                "store": ["Store_A", "Store_A", "Store_B", "Store_B", "Store_C", "Store_C"],
                cols.unit_spend: [100, 150, 200, 250, 300, 350],
            },
        )

        # Create stats with CUBE mode
        stats = SegTransactionStats(
            data=data,
            segment_col=["region", "store"],
            grouping_sets="cube",
        )

        result = stats.df

        # CUBE should generate 4 grouping sets (2^2):
        # 1. (region, store) - 4 detail rows: North/Store_A, North/Store_C, South/Store_B, South/Store_C
        # 2. (region) - 2 region-only rows: North/Total, South/Total
        # 3. (store) - 3 store-only rows: Total/Store_A, Total/Store_B, Total/Store_C
        # 4. () - 1 grand total: Total/Total
        # Total: 10 rows
        expected = pd.DataFrame(
            {
                "region": ["North", "North", "South", "South", "North", "South", "Total", "Total", "Total", "Total"],
                "store": [
                    "Store_A",
                    "Store_C",
                    "Store_B",
                    "Store_C",
                    "Total",
                    "Total",
                    "Store_A",
                    "Store_B",
                    "Store_C",
                    "Total",
                ],
                cols.agg.unit_spend: [250, 300, 450, 350, 550, 800, 250, 450, 650, 1350],
            },
        )

        # Sort both dataframes for consistent comparison
        result_subset = (
            result[["region", "store", cols.agg.unit_spend]].sort_values(["region", "store"]).reset_index(drop=True)
        )
        expected_sorted = expected.sort_values(["region", "store"]).reset_index(drop=True)

        # Compare using pandas assert_frame_equal
        pd.testing.assert_frame_equal(result_subset, expected_sorted)

    def test_cube_mode_warns_on_many_dimensions(self):
        """Test CUBE mode warns when using more than 6 dimensions."""
        # Create test data with 7 dimensions (region, category, brand, channel, store_type, price_tier, promotion)
        data = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.transaction_id: [101, 102, 103],
                "region": ["North", "South", "East"],
                "category": ["Electronics", "Clothing", "Food"],
                "brand": ["BrandA", "BrandB", "BrandC"],
                "channel": ["Online", "Store", "Mobile"],
                "store_type": ["Flagship", "Outlet", "Express"],
                "price_tier": ["Premium", "Standard", "Budget"],
                "promotion": ["Sale", "Regular", "Clearance"],
                cols.unit_spend: [1000, 500, 250],
            },
        )

        # Should warn about 7 dimensions generating 128 grouping sets
        with pytest.warns(UserWarning, match="CUBE with 7 dimensions will generate 128 grouping sets"):
            SegTransactionStats(
                data=data,
                segment_col=["region", "category", "brand", "channel", "store_type", "price_tier", "promotion"],
                grouping_sets="cube",
            )


class TestGroupingSetsCustomMode:
    """Test custom grouping sets functionality."""

    @pytest.mark.parametrize(
        ("segment_col", "grouping_sets", "expected"),
        [
            (
                ["region", "product"],
                [("region", "product"), ("product",), ()],
                {("region", "product"), ("product",), ()},
            ),
            (
                ["region", "product"],
                [("region", "product"), ("product",), ("region", "product"), (), ()],  # two duplicates
                {("region", "product"), ("product",), ()},
            ),
            (["region"], [("region",)], {("region",)}),
        ],
        ids=["basic", "deduplicates", "single"],
    )
    def test_generate_custom_grouping_sets(self, segment_col, grouping_sets, expected):
        """Custom grouping sets return the expected sets, deduplicated (order not guaranteed)."""
        result = SegTransactionStats._generate_grouping_sets(segment_col=segment_col, grouping_sets=grouping_sets)
        assert set(result) == expected
        assert len(result) == len(expected)  # no duplicates remain

    def test_custom_grouping_sets_invalid_column(self):
        """Test custom grouping sets raises error for invalid column name."""
        with pytest.raises(ValueError, match=r"Columns .* in grouping_sets not found in segment_col"):
            SegTransactionStats._generate_grouping_sets(
                segment_col=["region", "store"],
                grouping_sets=[("region", "invalid_col")],
            )

    def test_custom_grouping_sets_unmentioned_column(self):
        """Test custom grouping sets raises error when segment_col column is never mentioned."""
        with pytest.raises(ValueError, match=r"Columns .* in segment_col are not mentioned in any grouping set"):
            SegTransactionStats._generate_grouping_sets(
                segment_col=["region", "store", "date"],
                grouping_sets=[
                    ("region", "store"),  # date never mentioned!
                    ("region",),
                    (),
                ],
            )

    @pytest.mark.parametrize(
        "grouping_sets",
        [
            ["region"],  # string instead of tuple
            [("region",), 123],  # integer instead of tuple
            [("region",), None],  # None instead of tuple
        ],
        ids=["string", "integer", "none"],
    )
    def test_custom_grouping_sets_rejects_non_tuple_elements(self, grouping_sets):
        """Custom grouping sets raise TypeError when any element is not a tuple."""
        with pytest.raises(TypeError, match="Each element must be a tuple"):
            SegTransactionStats._validate_grouping_sets_params(
                grouping_sets=grouping_sets,
                calc_total=None,
                calc_rollup=None,
            )

    def test_custom_grouping_sets_empty_list(self):
        """Test custom grouping sets raises error for empty list."""
        with pytest.raises(ValueError, match="grouping_sets list cannot be empty"):
            SegTransactionStats._validate_grouping_sets_params(
                grouping_sets=[],
                calc_total=None,
                calc_rollup=None,
            )

    def test_custom_grouping_sets_integration(self):
        """Test custom grouping sets produce correct aggregations."""
        # Create test data
        data = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 3, 3],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "region": ["North", "North", "South", "South", "North", "South"],
                "store": ["Store_A", "Store_A", "Store_B", "Store_B", "Store_C", "Store_C"],
                cols.unit_spend: [100, 150, 200, 250, 300, 350],
            },
        )

        # Create stats with custom grouping sets
        stats = SegTransactionStats(
            data=data,
            segment_col=["region", "store"],
            grouping_sets=[
                ("region", "store"),  # Detail level
                ("region",),  # Region-only
                (),  # Grand total
            ],
        )

        result = stats.df

        # Custom grouping sets should generate 3 grouping sets:
        # 1. (region, store) - 4 detail rows: North/Store_A, North/Store_C, South/Store_B, South/Store_C
        # 2. (region) - 2 region-only rows: North/Total, South/Total
        # 3. () - 1 grand total: Total/Total
        # Total: 7 rows
        expected = pd.DataFrame(
            {
                "region": ["North", "North", "South", "South", "North", "South", "Total"],
                "store": ["Store_A", "Store_C", "Store_B", "Store_C", "Total", "Total", "Total"],
                cols.agg.unit_spend: [250, 300, 450, 350, 550, 800, 1350],
            },
        )

        # Sort both dataframes for consistent comparison
        result_subset = (
            result[["region", "store", cols.agg.unit_spend]].sort_values(["region", "store"]).reset_index(drop=True)
        )
        expected_sorted = expected.sort_values(["region", "store"]).reset_index(drop=True)

        # Compare using pandas assert_frame_equal
        pd.testing.assert_frame_equal(result_subset, expected_sorted)


class TestComposableGroupingSets:
    """Test composable grouping sets with cube() and rollup() helper functions."""

    def test_cube_function_returns_list(self):
        """Test cube() helper returns list of tuples for all combinations."""
        result = cube("region", "store")
        expected_grouping_sets = {
            ("region", "store"),  # Full detail
            ("region",),  # Region totals
            ("store",),  # Store totals
            (),  # Grand total
        }
        assert set(result) == expected_grouping_sets
        # Catches duplicate emissions that set comparison would hide.
        assert len(result) == len(expected_grouping_sets)

    def test_cube_function_warns_on_many_dimensions(self):
        """Test cube() helper warns when using more than 6 dimensions."""
        with pytest.warns(UserWarning, match="CUBE with 7 dimensions will generate 128 grouping sets"):
            cube("region", "category", "brand", "channel", "store_type", "price_tier", "promotion")

    def test_rollup_function_returns_list(self):
        """Test rollup() helper returns list of tuples for hierarchical levels."""
        result = rollup("year", "quarter", "month")
        # ROLLUP(year, quarter, month) should generate hierarchical levels from right to left
        assert result == [
            ("year", "quarter", "month"),  # Full detail
            ("year", "quarter"),  # Monthly rollup
            ("year",),  # Quarterly rollup
            (),  # Grand total
        ]

    def test_cube_with_fixed_columns(self):
        """Test CUBE with fixed date column for time-consistent geographic analysis."""
        result = SegTransactionStats._generate_grouping_sets(
            segment_col=["region", "store", "date"],
            grouping_sets=[(cube("region", "store"), "date")],
        )

        expected = [
            ("region", "store", "date"),  # Full detail with date
            ("region", "date"),  # Regional totals with date
            ("store", "date"),  # Store totals with date
            ("date",),  # Date-only totals
        ]

        # CUBE(region, store) with 2 columns generates 2^2 = 4 grouping sets (each with "date" appended)
        assert set(result) == set(expected)
        assert len(result) == len(expected)

    def test_rollup_with_fixed_columns(self):
        """Test ROLLUP with fixed segment for time hierarchy analysis by customer type."""
        result = SegTransactionStats._generate_grouping_sets(
            segment_col=["year", "quarter", "customer_segment", "channel"],
            grouping_sets=[(rollup("year", "quarter"), "customer_segment", "channel")],
        )

        expected = [
            ("year", "quarter", "customer_segment", "channel"),  # Full detail
            ("year", "customer_segment", "channel"),  # Quarterly rollup
            ("customer_segment", "channel"),  # Yearly rollup
        ]

        # Use set comparison since deduplication doesn't preserve order
        assert set(result) == set(expected)
        assert len(result) == len(expected)

    def test_cube_direct_without_list(self):
        """Test CUBE generates all combinations for three retail dimensions."""
        result = SegTransactionStats._generate_grouping_sets(
            segment_col=["region", "category", "brand"],
            grouping_sets=cube("region", "category", "brand"),
        )

        expected_grouping_sets = {
            ("region", "category", "brand"),  # Full detail
            ("region", "category"),
            ("region", "brand"),
            ("category", "brand"),
            ("region",),
            ("category",),
            ("brand",),
            (),  # Grand total
        }
        assert set(result) == expected_grouping_sets
        # Catches duplicate emissions that set comparison would hide.
        assert len(result) == len(expected_grouping_sets)

    def test_multiple_cube_rollup_calls_rejected(self):
        """Test that mixing CUBE and ROLLUP in same specification raises error."""
        with pytest.raises(ValueError, match="Only one cube\\(\\)/rollup\\(\\) call allowed"):
            SegTransactionStats._generate_grouping_sets(
                segment_col=["region", "store", "product"],
                grouping_sets=[(cube("region"), rollup("store"), "product")],
            )

    def test_mix_composable_with_explicit_tuples(self):
        """Test mixing CUBE analysis with grand total."""
        result = SegTransactionStats._generate_grouping_sets(
            segment_col=["region", "store", "product"],
            grouping_sets=[
                (cube("region", "store"), "product"),  # Geographic CUBE with fixed product
                (),  # Grand total across all dimensions
            ],
        )

        expected = [
            ("region", "store", "product"),  # Full detail
            ("region", "product"),  # Regional product totals
            ("store", "product"),  # Store product totals
            ("product",),  # Product-only totals
            (),  # Grand total
        ]

        # CUBE(region, store) generates 2^2 = 4 sets, plus 1 grand total = 5 total
        cube_sets = 2**2  # 4 sets from CUBE
        grand_total_sets = 1
        expected_total = cube_sets + grand_total_sets  # 4 + 1 = 5

        assert set(result) == set(expected)
        assert len(result) == expected_total

    def test_specification_tuple_with_empty_list_rejected(self):
        """Test that specification tuples with empty lists are rejected."""
        with pytest.raises(
            ValueError,
            match="Specification tuple must contain non-empty cube\\(\\) or rollup\\(\\) result",
        ):
            SegTransactionStats._generate_grouping_sets(
                segment_col=["region", "store", "product"],
                grouping_sets=[([], "product")],  # Empty list in tuple - invalid
            )

    @pytest.mark.parametrize(
        ("func", "columns", "exc", "match"),
        [
            (cube, (), ValueError, r"cube\(\) requires at least one column"),
            (rollup, (), ValueError, r"rollup\(\) requires at least one column"),
            (cube, ("region", 123, "store"), TypeError, "All column names must be strings"),
            (rollup, ("year", "quarter", 456), TypeError, "All column names must be strings"),
        ],
        ids=["cube_empty", "rollup_empty", "cube_non_string", "rollup_non_string"],
    )
    def test_grouping_helper_input_validation(self, func, columns, exc, match):
        """cube()/rollup() validate inputs: no columns -> ValueError, non-string column -> TypeError."""
        with pytest.raises(exc, match=match):
            func(*columns)

    def test_flatten_item_invalid_type_error(self):
        """Test _flatten_item() raises TypeError for invalid types in specification tuple."""
        with pytest.raises(TypeError, match="Invalid type in specification tuple"):
            SegTransactionStats._flatten_item((cube("region"), 123))

    def test_multiple_composable_specifications(self):
        """Test combining geographic CUBE with time ROLLUP, both with fixed customer segment."""
        result = SegTransactionStats._generate_grouping_sets(
            segment_col=["region", "store", "quarter", "customer_segment"],
            grouping_sets=[
                (cube("region", "store"), "customer_segment"),  # Geographic CUBE by segment
                (rollup("quarter"), "customer_segment"),  # Time ROLLUP by segment
            ],
        )

        expected_cube = [
            ("region", "store", "customer_segment"),  # Full geographic detail
            ("region", "customer_segment"),  # Regional totals
            ("store", "customer_segment"),  # Store totals
            ("customer_segment",),  # Segment-only totals
        ]

        expected_rollup = [
            ("quarter", "customer_segment"),  # Quarterly detail
            ("customer_segment",),  # Segment-only totals (deduplicated)
        ]

        # Combine and dedupe (customer_segment appears in both)
        expected = list(set(expected_cube + expected_rollup))

        # CUBE(region, store) = 2^2 = 4 sets
        # ROLLUP(quarter) = 1+1 = 2 sets
        # Deduplication: ("customer_segment",) appears in both = -1
        # Total: 4 + 2 - 1 = 5 unique sets
        cube_sets = 2**2  # 4 sets from CUBE
        rollup_sets = 1 + 1  # n+1 sets from ROLLUP (n=1)
        deduplicated_sets = 1  # ("customer_segment",) appears in both
        expected_unique_sets = cube_sets + rollup_sets - deduplicated_sets  # 4 + 2 - 1 = 5

        assert set(result) == set(expected)
        assert len(result) == expected_unique_sets

    def test_composable_grouping_sets_integration(self):
        """Test composable grouping sets produce correct aggregations with fixed columns."""
        data = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 3, 3, 4, 4],
                cols.transaction_id: [101, 102, 103, 104, 105, 106, 107, 108],
                "store": ["Store_A", "Store_A", "Store_B", "Store_B", "Store_A", "Store_A", "Store_B", "Store_B"],
                "region": ["North", "North", "South", "South", "North", "North", "South", "South"],
                "date": ["2024-01", "2024-01", "2024-01", "2024-01", "2024-02", "2024-02", "2024-02", "2024-02"],
                cols.unit_spend: [100, 150, 200, 250, 300, 350, 400, 450],
            },
        )

        stats = SegTransactionStats(
            data=data,
            segment_col=["store", "region", "date"],
            grouping_sets=[(cube("store", "region"), "date")],
        )

        result = stats.df

        # CUBE(store, region) with "date" fixed generates 4 grouping sets, each for both dates:
        # 1. (store, region, date) - 4 detail rows (2 for 2024-01, 2 for 2024-02)
        # 2. (store, date) - 4 store totals (2 stores x 2 dates)
        # 3. (region, date) - 4 region totals (2 regions x 2 dates)
        # 4. (date) - 2 date-only totals (2 dates)
        # Total: 14 rows
        expected = pd.DataFrame(
            {
                "store": [
                    "Store_A",
                    "Store_B",
                    "Store_A",
                    "Store_B",  # Full detail for 2024-01 and 2024-02
                    "Store_A",
                    "Store_B",
                    "Store_A",
                    "Store_B",  # Store totals by date
                    "Total",
                    "Total",
                    "Total",
                    "Total",  # Region totals by date
                    "Total",
                    "Total",
                ],  # Date-only totals
                "region": [
                    "North",
                    "South",
                    "North",
                    "South",  # Full detail
                    "Total",
                    "Total",
                    "Total",
                    "Total",  # Store totals
                    "North",
                    "South",
                    "North",
                    "South",  # Region totals
                    "Total",
                    "Total",
                ],  # Date-only totals
                "date": [
                    "2024-01",
                    "2024-01",
                    "2024-02",
                    "2024-02",  # Full detail
                    "2024-01",
                    "2024-01",
                    "2024-02",
                    "2024-02",  # Store totals
                    "2024-01",
                    "2024-01",
                    "2024-02",
                    "2024-02",  # Region totals
                    "2024-01",
                    "2024-02",
                ],  # Date-only totals
                cols.agg.unit_spend: [
                    250,
                    450,
                    650,
                    850,  # Full detail: A/North (100+150), B/South (200+250), etc.
                    250,
                    450,
                    650,
                    850,  # Store totals (same values - only one region per store in data)
                    250,
                    450,
                    650,
                    850,  # Region totals (same values - only one store per region in data)
                    700,
                    1500,
                ],  # Date totals: 2024-01 (250+450), 2024-02 (650+850)
            },
        )

        # Sort both dataframes for consistent comparison
        result_subset = (
            result[["store", "region", "date", cols.agg.unit_spend]]
            .sort_values(["store", "region", "date"])
            .reset_index(drop=True)
        )
        expected_sorted = expected.sort_values(["store", "region", "date"]).reset_index(drop=True)

        # Compare using pandas assert_frame_equal
        pd.testing.assert_frame_equal(result_subset, expected_sorted)

        # Verify "date" appears in every row (never "Total") - key composability requirement
        assert (result["date"] != "Total").all()


class TestDivisionByZeroHandling:
    """Tests for division by zero handling in derived metrics.

    These tests verify that when denominators are zero (e.g., zero transactions,
    zero customers), the code returns NULL/NaN instead of raising division by zero errors.
    This is particularly important for SQL Server which throws errors on division by zero.
    """

    def test_zero_customers_returns_nan_for_spend_per_cust(self):
        """Test that zero customers in a segment produce NaN for spend_per_cust metrics."""
        # Create data where after rollup, some segments may have zero identified customers
        df = pd.DataFrame(
            {
                cols.customer_id: [-1, -1, -1, -1],  # All unknown customers
                cols.unit_spend: [100.0, 200.0, 150.0, 250.0],
                cols.transaction_id: [101, 102, 103, 104],
                cols.store_id: ["S1", "S1", "S2", "S2"],
                "week": ["W1", "W2", "W1", "W2"],
                cols.unit_qty: [10, 20, 15, 25],
            },
        )

        # Using grouping_sets="rollup" with multi-dimensional segments and unknown_customer_value
        # This is the exact scenario that caused the production error
        result = SegTransactionStats(
            df,
            segment_col=[cols.store_id, "week"],
            grouping_sets="rollup",
            unknown_customer_value=-1,
        ).df

        expected_segments = {
            ("S1", "W1"),
            ("S1", "W2"),
            ("S2", "W1"),
            ("S2", "W2"),  # leaf cells
            ("S1", "Total"),
            ("S2", "Total"),  # per-store rollup
            ("Total", "Total"),  # grand total
        }
        assert set(zip(result[cols.store_id], result["week"], strict=True)) == expected_segments

        # Verify that spend_per_cust is NaN when customer_id count is 0
        # (all customers are unknown, so identified customer count is 0)
        assert result[cols.calc.spend_per_cust].isna().all()
        assert result[cols.calc.trans_per_cust].isna().all()

    def test_zero_unknown_transactions_returns_nan(self):
        """Test that zero unknown transactions produce NaN for unknown-specific metrics."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1004],  # All known customers
                cols.unit_spend: [100.0, 200.0, 150.0, 250.0],
                cols.transaction_id: [101, 102, 103, 104],
                "category": ["Electronics", "Electronics", "Apparel", "Apparel"],
                cols.unit_qty: [10, 20, 15, 25],
            },
        )

        result = (
            SegTransactionStats(
                df,
                segment_col="category",
                unknown_customer_value=-1,  # No customer has this value
            )
            .df.sort_values("category")
            .reset_index(drop=True)
        )

        # Verify expected structure: 2 detail rows + 1 total = 3 rows
        expected_row_count = 3
        assert len(result) == expected_row_count

        # Verify precondition: no unknown transactions exist (all customers are known)
        assert (result[cols.agg.transaction_id_unknown] == 0).all()

        # spend_per_trans_unknown should be NaN (not an error) when transaction_id_unknown is 0
        assert result[cols.calc.spend_per_trans_unknown].isna().all()

    def test_rollup_with_all_unknown_customers_no_division_error(self):
        """Test rollup with all unknown customers doesn't cause division by zero errors.

        This is a regression test for the production error where SQL Server
        raised 'Divide by zero error encountered' when using:
        - segment_col with multiple dimensions
        - grouping_sets="rollup"
        - unknown_customer_value=-1
        """
        df = pd.DataFrame(
            {
                cols.customer_id: [-1, -1, -1, -1, -1, -1],
                cols.unit_spend: [100.0, 200.0, 150.0, 250.0, 300.0, 350.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                cols.store_id: ["S1", "S1", "S2", "S2", "S3", "S3"],
                "week": ["W1", "W2", "W1", "W2", "W1", "W2"],
                cols.unit_qty: [10, 20, 15, 25, 30, 35],
            },
        )

        # This exact combination caused the production error
        result = SegTransactionStats(
            df,
            segment_col=[cols.store_id, "week"],
            grouping_sets="rollup",
            unknown_customer_value=-1,
        ).df

        # ROLLUP(store_id, week) generates:
        # - 6 detail rows (S1/W1, S1/W2, S2/W1, S2/W2, S3/W1, S3/W2)
        # - 3 store rollups (S1/Total, S2/Total, S3/Total)
        # - 1 grand total
        # Total: 10 rows
        expected_row_count = 10
        assert len(result) == expected_row_count

        # Verify derived metrics that divide by transaction/customer counts are NaN
        # when those counts are zero (which they are for identified customers)
        assert result[cols.calc.spend_per_cust].isna().all()
        assert result[cols.calc.trans_per_cust].isna().all()

        # But unknown customer metrics should have valid values
        assert result[cols.calc.spend_per_trans_unknown].notna().all()

    def test_mixed_zero_and_nonzero_segments(self):
        """Test that segments with zero values return NaN while others compute correctly."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, -1, -1],  # Mix of known and unknown
                cols.unit_spend: [100.0, 200.0, 150.0, 250.0],
                cols.transaction_id: [101, 102, 103, 104],
                # Loyalty has known customers, Walk-In has only unknown
                "customer_type": ["Loyalty", "Loyalty", "Walk-In", "Walk-In"],
                cols.unit_qty: [10, 20, 15, 25],
            },
        )

        result = (
            SegTransactionStats(
                df,
                segment_col="customer_type",
                unknown_customer_value=-1,
                calc_total=False,
            )
            .df.sort_values("customer_type")
            .reset_index(drop=True)
        )

        # Loyalty segment (known customers) should have valid spend_per_cust
        loyalty_segment = result[result["customer_type"] == "Loyalty"]
        assert loyalty_segment[cols.calc.spend_per_cust].notna().all()
        # Verify the actual computed value: (100 + 200) / 2 customers = 150
        expected_loyalty_spend_per_cust = 150.0
        assert loyalty_segment[cols.calc.spend_per_cust].iloc[0] == pytest.approx(expected_loyalty_spend_per_cust)

        # Walk-In segment (only unknown customers) should have NaN for spend_per_cust
        walkin_segment = result[result["customer_type"] == "Walk-In"]
        assert walkin_segment[cols.calc.spend_per_cust].isna().all()

        # But Walk-In segment should have valid spend_per_trans_unknown
        assert walkin_segment[cols.calc.spend_per_trans_unknown].notna().all()
        # Verify the actual computed value: (150 + 250) / 2 unknown transactions = 200
        expected_walkin_spend_per_trans_unknown = 200.0
        assert walkin_segment[cols.calc.spend_per_trans_unknown].iloc[0] == pytest.approx(
            expected_walkin_spend_per_trans_unknown,
        )


_NATIVE_SEGMENTS = ["region", "category"]


def _native_sample_df():
    """Realistic multi-segment retail data (incl. unknown -1 customers) for native grouping-sets tests."""
    return pd.DataFrame(
        {
            cols.customer_id: [1, 2, 3, 4, -1, 6, 7, -1] * 4,
            cols.transaction_id: list(range(32)),
            cols.unit_spend: [10.0, 20.5, 30.0, 40.25, 15.0, 25.0, 35.5, 45.0] * 4,
            cols.unit_qty: [1, 2, 3, 4, 1, 2, 3, 4] * 4,
            cols.store_id: [1, 2, 3, 4] * 8,
            "region": ["North", "South"] * 16,
            "category": ["Dairy", "Bakery", "Produce", "Meat"] * 8,
        },
    )


class TestNativeGroupingSets:
    """Tests for the backend-native GROUP BY GROUPING SETS path and its fallback gating."""

    @pytest.mark.parametrize(
        ("native_opt", "use_memtable", "expect_grouping_sets"),
        [
            (True, False, True),  # real table + option on -> native single scan
            (False, False, False),  # real table + option off -> union fallback
            (True, True, False),  # in-memory table is gated out -> union fallback
        ],
    )
    def test_native_path_selection(self, native_opt, use_memtable, expect_grouping_sets):
        """The native path is used only for a real table with the option on; otherwise union fallback."""
        df = _native_sample_df()
        data = df if use_memtable else ibis.duckdb.connect().create_table("transactions", df)
        with option_context("optimization.use_native_sql", native_opt):
            sql = ibis.to_sql(SegTransactionStats(data, _NATIVE_SEGMENTS, grouping_sets="rollup").table).upper()
        assert ("GROUPING SETS" in sql) == expect_grouping_sets
        # The fallback unions one aggregation per grouping set; the native path does not.
        assert ("UNION" in sql) != expect_grouping_sets

    @pytest.mark.parametrize(
        ("df", "segments", "kwargs"),
        [
            (_native_sample_df(), _NATIVE_SEGMENTS, {"grouping_sets": "rollup"}),
            (_native_sample_df(), _NATIVE_SEGMENTS, {"grouping_sets": "cube"}),
            (_native_sample_df(), _NATIVE_SEGMENTS, {"grouping_sets": "total"}),
            (_native_sample_df(), _NATIVE_SEGMENTS, {"grouping_sets": [("region", "category"), ("region",), ()]}),
            (_native_sample_df(), _NATIVE_SEGMENTS, {"grouping_sets": "rollup", "unknown_customer_value": -1}),
            (
                _native_sample_df(),
                _NATIVE_SEGMENTS,
                {"grouping_sets": "cube", "extra_aggs": {"unique_stores": (cols.store_id, "nunique")}},
            ),
            (
                _native_sample_df(),
                _NATIVE_SEGMENTS,
                {
                    "grouping_sets": "rollup",
                    "unknown_customer_value": -1,
                    "extra_aggs": {"unique_stores": (cols.store_id, "nunique")},
                },
            ),
            # A segment column literally named like the internal GROUPING() flag must not break the rewrite.
            (
                _native_sample_df().rename(columns={"region": "grouping_flag_0"}),
                ["grouping_flag_0", "category"],
                {"grouping_sets": "rollup"},
            ),
        ],
        ids=[
            "rollup",
            "cube",
            "total",
            "custom",
            "unknown_customer",
            "extra_aggs",
            "unknown_with_extra_aggs",
            "flag_name_collision",
        ],
    )
    def test_native_output_matches_union_fallback(self, df, segments, kwargs):
        """The native single-scan output is identical to the portable union fallback across configurations."""
        table = ibis.duckdb.connect().create_table("transactions", df)
        with option_context("optimization.use_native_sql", True):
            native_stats = SegTransactionStats(table, segments, **kwargs)
            native = native_stats.df
        # Guard against a vacuous pass: confirm the native path actually engaged. Without this, a regression
        # that silently routed the "native" run through the fallback would compare the fallback to itself.
        assert "GROUPING SETS" in ibis.to_sql(native_stats.table).upper()
        with option_context("optimization.use_native_sql", False):
            fallback = SegTransactionStats(table, segments, **kwargs).df
        native = native.sort_values(segments).reset_index(drop=True)
        fallback = fallback.sort_values(segments).reset_index(drop=True)
        pd.testing.assert_frame_equal(native, fallback)

    @pytest.mark.parametrize(
        ("key", "expected_sql"),
        [
            (exp.Literal.number(1), "region"),  # positional ordinal -> 1st SELECT column (duckdb/snowflake/bigquery)
            (exp.Literal.number(2), "store"),  # positional ordinal -> 2nd SELECT column
            (exp.column("store"), "store"),  # named key -> returned unchanged (the mssql/oracle path)
        ],
        ids=["ordinal_first", "ordinal_second", "named"],
    )
    def test_resolve_group_key_handles_positional_and_named(self, key, expected_sql):
        """Ordinal GROUP BY keys resolve to their SELECT column (alias stripped); named keys pass through."""
        selects = [exp.column("region").as_("region"), exp.column("store").as_("store")]
        assert _resolve_group_key(key, selects).sql() == expected_sql

    def test_native_path_skipped_for_single_grouping_set(self):
        """A lone grouping set is one scan regardless, so the native rewrite is skipped for a plain aggregation."""
        table = ibis.duckdb.connect().create_table("transactions", _native_sample_df())
        with option_context("optimization.use_native_sql", True):
            sql = ibis.to_sql(
                SegTransactionStats(table, _NATIVE_SEGMENTS, grouping_sets=[tuple(_NATIVE_SEGMENTS)]).table,
            ).upper()
        assert "GROUPING SETS" not in sql
        assert "UNION" not in sql
