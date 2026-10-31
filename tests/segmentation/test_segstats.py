"""Tests for the SegTransactionStats class."""

import ibis
import numpy as np
import pandas as pd
import pytest
from sqlglot import exp

from openretailscience.options import ColumnHelper, get_option, option_context
from openretailscience.segmentation.segstats import SegTransactionStats, _resolve_group_key, cube, rollup

cols = ColumnHelper()


class TestMetrics:
    """Spend, transaction, customer and unit metrics, their ratios, and extra_aggs."""

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

    @pytest.mark.parametrize(
        ("prepare", "expected"),
        [
            pytest.param(
                lambda df: df,
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
                id="item_level",
            ),
            pytest.param(
                lambda df: df.drop(columns=[cols.unit_qty]),
                {
                    "segment_name": ["Premium", "Standard", "Total"],
                    cols.agg.unit_spend: [500.0, 500.0, 1000.0],
                    cols.agg.transaction_id: [3, 2, 5],
                    cols.agg.customer_id: [3, 2, 5],
                    cols.calc.spend_per_cust: [166.666667, 250.0, 200.0],
                    cols.calc.spend_per_trans: [166.666667, 250.0, 200.0],
                    cols.calc.trans_per_cust: [1.0, 1.0, 1.0],
                },
                id="without_quantity",
            ),
            pytest.param(
                lambda df: df.drop(columns=[cols.customer_id]),
                {
                    "segment_name": ["Premium", "Standard", "Total"],
                    cols.agg.unit_spend: [500.0, 500.0, 1000.0],
                    cols.agg.transaction_id: [3, 2, 5],
                    cols.agg.unit_qty: [50, 50, 100],
                    cols.calc.spend_per_trans: [166.666667, 250.0, 200.0],
                    cols.calc.price_per_unit: [10.0, 10.0, 10.0],
                    cols.calc.units_per_trans: [16.666667, 25.0, 20.0],
                },
                id="without_customer",
            ),
            pytest.param(
                lambda df: df.assign(segment_name="Premium"),
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
                id="one_segment",
            ),
            pytest.param(
                lambda df: df.assign(**{cols.unit_qty: [10, 20, 15, 30, -25]}),
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
                id="zero_net_units",
            ),
        ],
    )
    def test_calculates_metrics_per_segment_and_total(self, base_df, prepare, expected):
        """Each metric and ratio is computed per segment and for the total, skipping absent customer/unit columns."""
        segment_stats = (
            SegTransactionStats(prepare(base_df), segment_col="segment_name", grouping_sets="total")
            .df.sort_values("segment_name")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(segment_stats, pd.DataFrame(expected))

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
        seg_stats = SegTransactionStats(df, segment_col=["segment_name", "region"], grouping_sets="total")

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
            result = SegTransactionStats(custom_df, segment_col="segment_name", grouping_sets="total").df

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

    def test_extra_aggs_functionality(self):
        """Test that the extra_aggs parameter works correctly."""
        # Constants for expected values
        premium_store_count = 3  # Premium: stores 1, 2, 4
        standard_store_count = 2  # Standard: stores 1, 3
        total_store_count = 4  # Total: stores 1, 2, 3, 4

        premium_product_count = 3  # Premium: products 10, 20, 40
        standard_product_count = 2  # Standard: products 10, 30
        total_product_count = 4  # Total: products 10, 20, 30, 40
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
            segment_col="segment_name",
            extra_aggs={"distinct_stores": (cols.store_id, "nunique")},
            grouping_sets="total",
        )

        # Sort by segment_name to ensure consistent order
        result_df = seg_stats.df.sort_values("segment_name").reset_index(drop=True)

        assert result_df["distinct_stores"].to_list() == [
            premium_store_count,
            standard_store_count,
            total_store_count,
        ]

        # Test with multiple extra aggregations
        seg_stats_multi = SegTransactionStats(
            df,
            segment_col="segment_name",
            extra_aggs={
                "distinct_stores": (cols.store_id, "nunique"),
                "distinct_products": (cols.product_id, "nunique"),
            },
            grouping_sets="total",
        )

        # Sort by segment_name to ensure consistent order
        result_df_multi = seg_stats_multi.df.sort_values("segment_name").reset_index(drop=True)

        assert result_df_multi["distinct_products"].to_list() == [
            premium_product_count,
            standard_product_count,
            total_product_count,
        ]


class TestInputValidation:
    """Constructor errors for missing, mistyped or invalid inputs."""

    @pytest.fixture
    def transactions(self) -> pd.DataFrame:
        """Return two transactions at two stores, one from an unknown (-1) customer."""
        return pd.DataFrame(
            {
                cols.store_id: ["Camden", "Leeds"],
                cols.unit_spend: [24.99, 61.50],
                cols.unit_qty: [1, 3],
                cols.transaction_id: [9001, 9002],
                cols.customer_id: [101, -1],
                "brand": ["Coke", "Pepsi"],
            },
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param(
                {"segment_col": "segment_name"},
                r"segment_col references columns not present in the data: \['segment_name'\]",
                id="missing_segment_col",
            ),
            pytest.param({"segment_col": []}, "segment_col must not be an empty list", id="empty_segment_col"),
            # Exhaustive extra_aggs cases live in tests/core/test_validation.py; this pins the constructor wiring.
            pytest.param(
                {"segment_col": cols.store_id, "extra_aggs": {"stores": ("nonexistent_column", "nunique")}},
                r"Column 'nonexistent_column' specified in extra_aggs does not exist in the data",
                id="extra_agg_column_missing",
            ),
            # unknown_customer_value coalesces each extra_aggs result to 0, so a text result is rejected up front.
            pytest.param(
                {
                    "segment_col": cols.store_id,
                    "grouping_sets": "total",
                    "extra_aggs": {"top_brand": ("brand", "max")},
                    "unknown_customer_value": -1,
                },
                r"extra_aggs\['top_brand'\] 'max' is not an aggregation .* with a numeric result",
                id="text_extra_agg_with_unknown_customers",
            ),
        ],
    )
    def test_rejects_invalid_input(self, transactions, kwargs, match):
        """A bad segment_col or extra_aggs raises a ValueError that names the problem."""
        with pytest.raises(ValueError, match=match):
            SegTransactionStats(transactions, **kwargs)

    @pytest.mark.parametrize("measure_col", [cols.unit_spend, cols.unit_qty])
    @pytest.mark.parametrize("sum_in_extra_aggs", [False, True], ids=["plain", "summed_in_extra_aggs"])
    def test_rejects_non_numeric_measure(self, transactions, measure_col, sum_in_extra_aggs):
        """Spend or units read as text (e.g. straight from a CSV) is rejected naming the column, ahead of extra_aggs."""
        transactions[measure_col] = transactions[measure_col].astype(str)
        extra_aggs = {"measure_total": (measure_col, "sum")} if sum_in_extra_aggs else None
        with pytest.raises(TypeError, match=rf"^Column '{measure_col}' must be numeric\. Got string\.$"):
            SegTransactionStats(transactions, segment_col=cols.store_id, grouping_sets="total", extra_aggs=extra_aggs)


_REGION_STORE_ROWS = [
    ("North", "Camden", 250),
    ("North", "Reading", 300),
    ("South", "Leeds", 450),
    ("South", "Reading", 350),
]
_STORE_TOTAL_ROWS = [("Total", "Camden", 250), ("Total", "Leeds", 450), ("Total", "Reading", 650)]
_REGION_TOTAL_ROWS = [("North", "Total", 550), ("South", "Total", 800)]
_GRAND_TOTAL_ROW = [("Total", "Total", 1350)]


class TestGroupingSets:
    """Tests for grouping_sets: the sets each mode generates, the rows they produce, and invalid input."""

    @pytest.fixture
    def transactions(self) -> pd.DataFrame:
        """Return transactions across two regions and three stores."""
        return pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 3, 3],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "region": ["North", "North", "South", "South", "North", "South"],
                "store": ["Camden", "Camden", "Leeds", "Leeds", "Reading", "Reading"],
                "product": ["Jeans", "Shirts", "Jeans", "Boots", "Shirts", "Boots"],
                cols.unit_spend: [100, 150, 200, 250, 300, 350],
            },
        )

    @pytest.fixture
    def id_keyed_transactions(self) -> pd.DataFrame:
        """Return transactions with integer customer and store ids."""
        return pd.DataFrame(
            {
                cols.customer_id: [101, 101, 102, 103],
                cols.transaction_id: [5001, 5002, 5003, 5004],
                cols.store_id: [7, 7, 9, 9],
                "category_0_name": ["Grocery", "Grocery", "Beauty", "Grocery"],
                cols.unit_spend: [12.50, 30.00, 8.75, 4.25],
            },
        )

    @pytest.mark.parametrize(
        ("segment_col", "expected_segments", "expected_spend"),
        [
            pytest.param("category_0_name", ["Beauty", "Grocery"], [8.75, 46.75], id="text_column"),
            pytest.param(cols.customer_id, [101, 102, 103], [42.50, 8.75, 4.25], id="integer_customer"),
            pytest.param(cols.store_id, [7, 9], [42.50, 13.00], id="integer_store"),
        ],
    )
    def test_default_returns_one_row_per_segment(
        self, id_keyed_transactions, segment_col, expected_segments, expected_spend
    ):
        """One row per segment with no total, on text and integer segment columns."""
        result = (
            SegTransactionStats(id_keyed_transactions, segment_col=segment_col)
            .df[[segment_col, cols.agg.unit_spend]]
            .sort_values(segment_col)
            .reset_index(drop=True)
        )
        expected = pd.DataFrame({segment_col: expected_segments, cols.agg.unit_spend: expected_spend})
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ("segment_col", "grouping_sets", "expected"),
        [
            pytest.param(["region", "store", "product"], None, [("region", "store", "product")], id="none"),
            pytest.param(["region"], None, [("region",)], id="none_single_column"),
            pytest.param(["region", "store", "product"], "total", [("region", "store", "product"), ()], id="total"),
            pytest.param(
                ["region", "store", "product"],
                "rollup",
                [("region", "store", "product"), ("region", "store"), ("region",), ()],
                id="rollup",
            ),
            pytest.param(["region"], "rollup", [("region",), ()], id="rollup_single_column"),
            pytest.param(
                ["region", "store", "product"],
                "cube",
                [
                    ("region", "store", "product"),
                    ("region", "store"),
                    ("region", "product"),
                    ("store", "product"),
                    ("region",),
                    ("store",),
                    ("product",),
                    (),
                ],
                id="cube",
            ),
            pytest.param(["region"], "cube", [("region",), ()], id="cube_single_column"),
            pytest.param(
                ["region", "product"],
                [("region", "product"), ("product",), ("region", "product"), (), ()],
                [("region", "product"), ("product",), ()],
                id="custom_deduplicates",
            ),
            pytest.param(
                ["region", "store", "date"],
                [(cube("region", "store"), "date")],
                [("region", "store", "date"), ("region", "date"), ("store", "date"), ("date",)],
                id="cube_with_fixed_column",
            ),
            pytest.param(
                ["year", "quarter", "customer_segment", "channel"],
                [(rollup("year", "quarter"), "customer_segment", "channel")],
                [
                    ("year", "quarter", "customer_segment", "channel"),
                    ("year", "customer_segment", "channel"),
                    ("customer_segment", "channel"),
                ],
                id="rollup_with_fixed_columns",
            ),
            pytest.param(
                ["region", "category", "brand"],
                cube("region", "category", "brand"),
                [
                    ("region", "category", "brand"),
                    ("region", "category"),
                    ("region", "brand"),
                    ("category", "brand"),
                    ("region",),
                    ("category",),
                    ("brand",),
                    (),
                ],
                id="cube_result_as_list",
            ),
            pytest.param(
                ["region", "store", "product"],
                [(cube("region", "store"), "product"), ()],
                [("region", "store", "product"), ("region", "product"), ("store", "product"), ("product",), ()],
                id="spec_with_explicit_set",
            ),
            pytest.param(
                ["region", "store", "quarter", "customer_segment"],
                [(cube("region", "store"), "customer_segment"), (rollup("quarter"), "customer_segment")],
                [
                    ("region", "store", "customer_segment"),
                    ("region", "customer_segment"),
                    ("store", "customer_segment"),
                    ("customer_segment",),
                    ("quarter", "customer_segment"),
                ],
                id="overlapping_specs_deduplicate",
            ),
        ],
    )
    def test_generate_grouping_sets(self, segment_col, grouping_sets, expected):
        """Each mode yields its grouping sets exactly once, in first-occurrence order so the SQL is stable."""
        assert SegTransactionStats._generate_grouping_sets(segment_col, grouping_sets) == expected

    def test_generate_grouping_sets_raises_on_unhandled_mode(self):
        """A mode no branch handles raises instead of returning None."""
        with pytest.raises(TypeError, match=r"Unhandled grouping_sets: \('region',\)"):
            SegTransactionStats._generate_grouping_sets(["region"], ("region",))

    @pytest.mark.parametrize(
        ("grouping_sets", "expected_rows"),
        [
            pytest.param(None, _REGION_STORE_ROWS, id="none"),
            pytest.param("total", _REGION_STORE_ROWS + _GRAND_TOTAL_ROW, id="total"),
            pytest.param("rollup", _REGION_STORE_ROWS + _REGION_TOTAL_ROWS + _GRAND_TOTAL_ROW, id="rollup"),
            pytest.param(
                "cube",
                _REGION_STORE_ROWS + _REGION_TOTAL_ROWS + _STORE_TOTAL_ROWS + _GRAND_TOTAL_ROW,
                id="cube",
            ),
            pytest.param([("region", "store"), ("store",)], _REGION_STORE_ROWS + _STORE_TOTAL_ROWS, id="custom"),
        ],
    )
    def test_mode_aggregates_each_grouping_set(self, transactions, grouping_sets, expected_rows):
        """Each grouping set adds its rows, with rolled-up columns holding rollup_value."""
        segments = ["region", "store"]
        result = SegTransactionStats(transactions, segment_col=segments, grouping_sets=grouping_sets).df
        result = result[[*segments, cols.agg.unit_spend]].sort_values(segments).reset_index(drop=True)
        expected = (
            pd.DataFrame(expected_rows, columns=[*segments, cols.agg.unit_spend])
            .sort_values(segments)
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_spec_keeps_fixed_column_on_every_row(self):
        """A cube() spec with a fixed column aggregates every cube set within each value of that column."""
        data = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 3, 3, 4, 4],
                cols.transaction_id: [101, 102, 103, 104, 105, 106, 107, 108],
                "store": ["Camden", "Camden", "Leeds", "Leeds", "Camden", "Camden", "Leeds", "Leeds"],
                "region": ["North", "North", "South", "South", "North", "North", "South", "South"],
                "date": ["2024-01", "2024-01", "2024-01", "2024-01", "2024-02", "2024-02", "2024-02", "2024-02"],
                cols.unit_spend: [100, 150, 200, 250, 300, 350, 400, 450],
            },
        )
        segments = ["store", "region", "date"]
        result = SegTransactionStats(data, segment_col=segments, grouping_sets=[(cube("store", "region"), "date")]).df

        expected = pd.DataFrame(
            [
                ("Camden", "North", "2024-01", 250),
                ("Leeds", "South", "2024-01", 450),
                ("Camden", "North", "2024-02", 650),
                ("Leeds", "South", "2024-02", 850),
                ("Camden", "Total", "2024-01", 250),
                ("Leeds", "Total", "2024-01", 450),
                ("Camden", "Total", "2024-02", 650),
                ("Leeds", "Total", "2024-02", 850),
                ("Total", "North", "2024-01", 250),
                ("Total", "South", "2024-01", 450),
                ("Total", "North", "2024-02", 650),
                ("Total", "South", "2024-02", 850),
                ("Total", "Total", "2024-01", 700),
                ("Total", "Total", "2024-02", 1500),
            ],
            columns=[*segments, cols.agg.unit_spend],
        )
        pd.testing.assert_frame_equal(
            result[[*segments, cols.agg.unit_spend]].sort_values(segments).reset_index(drop=True),
            expected.sort_values(segments).reset_index(drop=True),
        )

    @pytest.mark.parametrize(
        ("segment_col", "expected_rows"),
        [
            pytest.param(["store"], [("Camden",), ("Leeds",), ("Reading",), ("Total",)], id="one_column"),
            pytest.param(
                ["region", "store"],
                [
                    ("North", "Camden"),
                    ("North", "Reading"),
                    ("South", "Leeds"),
                    ("South", "Reading"),
                    ("Total", "Total"),
                ],
                id="two_columns",
            ),
        ],
    )
    def test_sort_by_orders_rows_ascending(self, transactions, segment_col, expected_rows):
        """sort_by=True orders rows ascending by each segment column in turn, total rows included."""
        result = SegTransactionStats(transactions, segment_col=segment_col, grouping_sets="total", sort_by=True).df
        assert list(result[segment_col].itertuples(index=False, name=None)) == expected_rows

    @pytest.mark.parametrize(
        ("segment_col", "grouping_sets", "exc", "match"),
        [
            pytest.param(["region"], "invalid", ValueError, "grouping_sets must be 'rollup', 'cube'", id="bad_string"),
            pytest.param(["region"], [], ValueError, "grouping_sets list cannot be empty", id="empty_list"),
            pytest.param(
                ["region", "store"],
                ("region",),
                TypeError,
                r"^grouping_sets must be 'rollup', 'cube', 'total', a list of tuples, or None\. Got tuple\.",
                id="bare_tuple",
            ),
            pytest.param(
                ["region", "store"],
                True,
                TypeError,
                r"^grouping_sets must be 'rollup', 'cube', 'total', a list of tuples, or None\. Got bool\.",
                id="bool",
            ),
            pytest.param(
                ["region"],
                ["region"],
                TypeError,
                r"Each element must be a tuple of segment_col names.*e\.g\. \[\('category_1_name',\), \(\)\]",
                id="string_element",
            ),
            pytest.param(["region"], [("region",), 123], TypeError, "Each element must be a tuple", id="int_element"),
            pytest.param(["region"], [("region",), None], TypeError, "Each element must be a tuple", id="none_element"),
            pytest.param(
                ["region", "store"],
                [("region", "product")],
                ValueError,
                r"Columns \['product'\] in grouping_sets not found in segment_col.*e\.g\. \[\('region',\), \(\)\]",
                id="column_not_in_segment_col",
            ),
            pytest.param(
                ["region", "store"],
                [("region", 7)],
                TypeError,
                r"Grouping set columns must be segment_col names \(str\)\. Got int in \('region', 7\)",
                id="int_column_in_set",
            ),
            pytest.param(
                ["region", "store"],
                [("region", {"store"})],
                TypeError,
                r"Grouping set columns must be segment_col names \(str\)\. Got set in",
                id="unhashable_column_in_set",
            ),
            pytest.param(
                ["region", "store", "product"],
                [("region", "store"), ("region",), ()],
                ValueError,
                r"Columns \['product'\] in segment_col are not mentioned in any grouping set",
                id="segment_col_not_in_any_set",
            ),
            pytest.param(
                ["region", "store", "product"],
                [(cube("region"), rollup("store"), "product")],
                ValueError,
                r"Only one cube\(\)/rollup\(\) call allowed",
                id="two_helpers_in_spec",
            ),
            pytest.param(
                ["region", "product"],
                [([], "product")],
                ValueError,
                r"Specification tuple must contain non-empty cube\(\) or rollup\(\) result",
                id="empty_helper_result",
            ),
            pytest.param(
                ["region"],
                [(cube("region"), 123)],
                TypeError,
                r"Grouping set columns must be segment_col names \(str\)\. Got int in",
                id="non_str_fixed_column",
            ),
            pytest.param(
                ["region", "store", "product"],
                [(["region", "store"], "product")],
                TypeError,
                r"A list in a specification tuple must be a cube\(\)/rollup\(\) result \(a list of tuples\)",
                id="column_list_in_spec",
            ),
        ],
    )
    def test_rejects_invalid_grouping_sets(self, transactions, segment_col, grouping_sets, exc, match):
        """Invalid grouping_sets raise a specific error that says what to pass instead."""
        with pytest.raises(exc, match=match):
            SegTransactionStats(transactions, segment_col=segment_col, grouping_sets=grouping_sets)

    @pytest.mark.parametrize(
        ("func", "columns", "expected"),
        [
            (cube, ("region", "store"), [("region", "store"), ("region",), ("store",), ()]),
            (
                rollup,
                ("year", "quarter", "month"),
                [("year", "quarter", "month"), ("year", "quarter"), ("year",), ()],
            ),
        ],
        ids=["cube", "rollup"],
    )
    def test_grouping_helper_output(self, func, columns, expected):
        """cube() returns every subset of its columns; rollup() returns each prefix, longest first."""
        assert func(*columns) == expected

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda data, dims: cube(*dims), id="cube_helper"),
            pytest.param(
                lambda data, dims: SegTransactionStats(data, segment_col=dims, grouping_sets="cube"),
                id="grouping_sets_cube",
            ),
        ],
    )
    def test_cube_warns_on_many_dimensions(self, build):
        """cube(), directly or via grouping_sets="cube", warns when more than 6 dimensions would be cubed."""
        dimensions = {
            "region": ["North", "South"],
            "category": ["Grocery", "Beauty"],
            "brand": ["Heinz", "Dove"],
            "channel": ["Online", "Store"],
            "store_type": ["Flagship", "Express"],
            "price_tier": ["Premium", "Value"],
            "promotion": ["Sale", "Regular"],
        }
        data = pd.DataFrame(
            {
                cols.customer_id: [101, 102],
                cols.transaction_id: [5001, 5002],
                cols.unit_spend: [24.99, 61.50],
                **dimensions,
            },
        )
        with pytest.warns(UserWarning, match="CUBE with 7 dimensions will generate 128 grouping sets"):
            build(data, list(dimensions))

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


class TestRollupValue:
    """Labels on rolled-up rows: custom values, per-column values and invalid values."""

    @pytest.fixture
    def transactions(self) -> pd.DataFrame:
        """Return six single-transaction customers across two categories, four subcategories and two stores."""
        return pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                cols.store_id: [7, 7, 9, 9, 9, 7],
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Jeans", "Shirts", "Sneakers", "Boots", "Boots"],
            },
        )

    @pytest.mark.parametrize(
        ("segment_col", "rollup_value", "expected"),
        [
            pytest.param(
                ["category", "subcategory"],
                "ALL",
                {
                    "category": ["ALL", "Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                    "subcategory": ["ALL", "ALL", "Jeans", "Shirts", "ALL", "Boots", "Sneakers"],
                    cols.agg.unit_spend: [2100.0, 600.0, 300.0, 300.0, 1500.0, 1100.0, 400.0],
                    cols.agg.transaction_id: [6, 3, 2, 1, 3, 2, 1],
                    cols.agg.customer_id: [6, 3, 2, 1, 3, 2, 1],
                    cols.calc.spend_per_cust: [350.0, 200.0, 150.0, 300.0, 500.0, 550.0, 400.0],
                    cols.calc.spend_per_trans: [350.0, 200.0, 150.0, 300.0, 500.0, 550.0, 400.0],
                    cols.calc.trans_per_cust: [1.0] * 7,
                },
                id="one_value_for_all_columns",
            ),
            pytest.param(
                ["category", cols.store_id],
                ["All categories", 0],
                {
                    "category": ["All categories"] + ["Clothing"] * 3 + ["Footwear"] * 3,
                    cols.store_id: [0, 0, 7, 9, 0, 7, 9],
                    cols.agg.unit_spend: [2100.0, 600.0, 300.0, 300.0, 1500.0, 600.0, 900.0],
                    cols.agg.transaction_id: [6, 3, 2, 1, 3, 1, 2],
                    cols.agg.customer_id: [6, 3, 2, 1, 3, 1, 2],
                    cols.calc.spend_per_cust: [350.0, 200.0, 150.0, 300.0, 500.0, 600.0, 450.0],
                    cols.calc.spend_per_trans: [350.0, 200.0, 150.0, 300.0, 500.0, 600.0, 450.0],
                    cols.calc.trans_per_cust: [1.0] * 7,
                },
                id="value_per_column_with_integer_store",
            ),
        ],
    )
    def test_labels_rolled_up_rows(self, transactions, segment_col, rollup_value, expected):
        """Each rolled-up column holds its rollup_value, cast to the column's type."""
        result = SegTransactionStats(
            transactions,
            segment_col=segment_col,
            grouping_sets="rollup",
            rollup_value=rollup_value,
            sort_by=True,
        ).df
        pd.testing.assert_frame_equal(result, pd.DataFrame(expected))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param(
                {"segment_col": ["category", "subcategory"], "grouping_sets": "rollup", "rollup_value": ["Total"]},
                "must match the number of segment columns",
                id="list_wrong_length",
            ),
            pytest.param(
                {"segment_col": "category", "rollup_value": "All categories"},
                r"rollup_value labels rolled-up rows.*Pass grouping_sets, e\.g\. 'total'",
                id="without_grouping_sets",
            ),
        ],
    )
    def test_rejects_invalid_rollup_value(self, transactions, kwargs, match):
        """A rollup_value that can't label the rolled-up rows raises a ValueError saying what to pass."""
        with pytest.raises(ValueError, match=match):
            SegTransactionStats(transactions, **kwargs)

    @pytest.mark.parametrize(
        ("segment_col", "segment_values", "col_type"),
        [
            (cols.store_id, [11, 11, 24], "int64"),
            # ibis coerces "Total" to True here, so an unchecked total row would pass as the True segment
            ("is_loyalty_member", [True, True, False], "boolean"),
            # a warehouse uuid key: ibis rejects "Total" with a ValueError rather than a TypeError
            (
                "store_key",
                [
                    "6f1c2a4e-5b7d-4c3e-9a8f-0d1e2f3a4b5c",
                    "6f1c2a4e-5b7d-4c3e-9a8f-0d1e2f3a4b5c",
                    "0a9b8c7d-6e5f-4a3b-8c2d-1e0f9a8b7c6d",
                ],
                "uuid",
            ),
        ],
        ids=["integer_store_id", "boolean_flag", "uuid_store_key"],
    )
    def test_string_rollup_value_on_non_string_segment_col_names_rollup_value(
        self,
        segment_col,
        segment_values,
        col_type,
    ):
        """The default rollup_value "Total" can't label a non-string segment's total row; the error says so."""
        data = ibis.memtable(
            pd.DataFrame(
                {
                    segment_col: segment_values,
                    cols.unit_spend: [24.99, 61.50, 12.75],
                    cols.transaction_id: [9001, 9002, 9003],
                },
            ),
        )
        data = data.mutate(**{segment_col: data[segment_col].cast(col_type)})
        with pytest.raises(
            ValueError, match=rf"rollup_value 'Total' can't be stored in column '{segment_col}' \({col_type}\)"
        ):
            SegTransactionStats(data, segment_col=segment_col, grouping_sets="total")


class TestUnknownCustomerTracking:
    """Tests for unknown customer tracking functionality."""

    @pytest.mark.parametrize(
        ("unknown_value", "customer_ids"),
        [
            (-1, [1, 2, -1, 3]),  # int value
            ("UNKNOWN", ["C1", "C2", "UNKNOWN", "C3"]),  # string value
            (ibis.literal(-1), [1, 2, -1, 3]),  # ibis literal expression
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

        seg_stats = SegTransactionStats(
            df, segment_col="segment_name", unknown_customer_value=unknown_value, grouping_sets="total"
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
            segment_col="segment_name",
            unknown_customer_value=data_table[cols.customer_id] < 0,
            grouping_sets="total",
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

        seg_stats = SegTransactionStats(
            df, segment_col="segment_name", unknown_customer_value=-1, grouping_sets="total"
        )
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
            SegTransactionStats(df, segment_col="segment_name", unknown_customer_value=-1)

        assert "required when unknown_customer_value parameter is specified" in str(excinfo.value)

    def test_unknown_customer_with_rollups(self):
        """Subtotal and total rows split spend and transactions into identified, unknown and total."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, -1, 2, -1],
                cols.unit_spend: [100.0, 150.0, 200.0, 250.0],
                cols.transaction_id: [101, 102, 103, 104],
                "category": ["Clothing", "Clothing", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Shirts", "Sneakers", "Boots"],
            },
        )

        result = SegTransactionStats(
            df,
            segment_col=["category", "subcategory"],
            grouping_sets="rollup",
            unknown_customer_value=-1,
            sort_by=True,
        ).df

        expected = pd.DataFrame(
            {
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear", "Total"],
                "subcategory": ["Jeans", "Shirts", "Total", "Boots", "Sneakers", "Total", "Total"],
                cols.agg.unit_spend: [100.0, 0.0, 100.0, 0.0, 200.0, 200.0, 300.0],
                cols.agg.transaction_id: [1, 0, 1, 0, 1, 1, 2],
                cols.agg.customer_id: [1, 0, 1, 0, 1, 1, 2],
                cols.calc.spend_per_cust: [100.0, np.nan, 100.0, np.nan, 200.0, 200.0, 150.0],
                cols.calc.spend_per_trans: [100.0, np.nan, 100.0, np.nan, 200.0, 200.0, 150.0],
                cols.calc.trans_per_cust: [1.0, np.nan, 1.0, np.nan, 1.0, 1.0, 1.0],
                cols.agg.unit_spend_unknown: [0.0, 150.0, 150.0, 250.0, 0.0, 250.0, 400.0],
                cols.agg.transaction_id_unknown: [0, 1, 1, 1, 0, 1, 2],
                cols.calc.spend_per_trans_unknown: [np.nan, 150.0, 150.0, 250.0, np.nan, 250.0, 200.0],
                cols.agg.unit_spend_total: [100.0, 150.0, 250.0, 250.0, 200.0, 450.0, 700.0],
                cols.agg.transaction_id_total: [1, 1, 2, 1, 1, 2, 4],
                cols.calc.spend_per_trans_total: [100.0, 150.0, 125.0, 250.0, 200.0, 225.0, 175.0],
            },
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_unknown_customer_with_extra_aggs(self):
        """Each extra_aggs column gets identified, unknown and total variants."""
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

        result = SegTransactionStats(
            df,
            segment_col="segment_name",
            unknown_customer_value=-1,
            extra_aggs={stores_agg: (cols.store_id, "nunique")},
            sort_by=True,
        ).df

        expected = pd.DataFrame(
            {
                "segment_name": ["Premium", "Standard"],
                cols.agg.unit_spend: [100.0, 200.0],
                cols.agg.transaction_id: [1, 1],
                cols.agg.customer_id: [1, 1],
                cols.calc.spend_per_cust: [100.0, 200.0],
                cols.calc.spend_per_trans: [100.0, 200.0],
                cols.calc.trans_per_cust: [1.0, 1.0],
                cols.agg.unit_spend_unknown: [150.0, 250.0],
                cols.agg.transaction_id_unknown: [1, 1],
                cols.calc.spend_per_trans_unknown: [150.0, 250.0],
                cols.agg.unit_spend_total: [250.0, 450.0],
                cols.agg.transaction_id_total: [2, 2],
                cols.calc.spend_per_trans_total: [125.0, 225.0],
                stores_agg: [1, 1],
                stores_unknown: [1, 1],
                stores_total: [2, 2],
            },
        )
        pd.testing.assert_frame_equal(result, expected)


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
                grouping_sets="total",
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
            sql = ibis.to_sql(
                SegTransactionStats(data, segment_col=_NATIVE_SEGMENTS, grouping_sets="rollup").table
            ).upper()
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
            # A non-string segment column that every set keeps is never labelled, so "Total" needn't fit its type.
            (
                _native_sample_df(),
                ["region", cols.store_id],
                {"grouping_sets": [("region", cols.store_id), (cols.store_id,)]},
            ),
            (
                _native_sample_df().assign(is_loyalty_member=lambda df: df[cols.customer_id] > 0),
                ["region", "is_loyalty_member"],
                {"grouping_sets": [("region", "is_loyalty_member"), ("is_loyalty_member",)]},
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
            "integer_segment_never_rolled_up",
            "boolean_segment_never_rolled_up",
        ],
    )
    def test_native_output_matches_union_fallback(self, df, segments, kwargs):
        """The native single-scan output is identical to the portable union fallback across configurations."""
        table = ibis.duckdb.connect().create_table("transactions", df)
        with option_context("optimization.use_native_sql", True):
            native_stats = SegTransactionStats(table, segment_col=segments, **kwargs)
            native = native_stats.df
        # Guard against a vacuous pass: confirm the native path actually engaged. Without this, a regression
        # that silently routed the "native" run through the fallback would compare the fallback to itself.
        assert "GROUPING SETS" in ibis.to_sql(native_stats.table).upper()
        with option_context("optimization.use_native_sql", False):
            fallback = SegTransactionStats(table, segment_col=segments, **kwargs).df
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
                SegTransactionStats(table, segment_col=_NATIVE_SEGMENTS, grouping_sets=[tuple(_NATIVE_SEGMENTS)]).table,
            ).upper()
        assert "GROUPING SETS" not in sql
        assert "UNION" not in sql
