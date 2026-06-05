"""Tests for openretailscience.experimental.metrics.distribution.pct_of_stores."""

import ibis
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from openretailscience.experimental.metrics.distribution.pct_of_stores import PctOfStores
from openretailscience.options import ColumnHelper, get_option

cols = ColumnHelper()
stores_col = get_option("column.agg.store_id")
pct_stores_col = ColumnHelper.join_options("column.agg.store_id", "column.suffix.percent")


class TestPctOfStores:
    """Tests for the PctOfStores metric class."""

    def test_basic_calculation(self):
        """Test basic % of stores with multiple products across four stores."""
        df = pd.DataFrame(
            {
                cols.store_id: [10, 20, 20, 30, 40],
                cols.product_id: [501, 501, 502, 502, 503],
                cols.unit_spend: [5.99, 3.49, 4.00, 6.00, 2.50],
            }
        )
        result = PctOfStores(df).df.sort_values(cols.product_id).reset_index(drop=True)
        expected = pd.DataFrame(
            {
                cols.product_id: [501, 502, 503],
                stores_col: [2, 2, 1],
                pct_stores_col: [50.0, 50.0, 25.0],
            }
        )
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ("store_ids", "product_ids"),
        [
            ([10, 20, 30, 40], [501, 501, 501, 501]),
            ([10], [501]),
        ],
        ids=["all_stores", "single_store"],
    )
    def test_product_in_every_store_returns_100(self, store_ids, product_ids):
        """Test that a product sold in every store returns 100%, including single-store datasets."""
        df = pd.DataFrame(
            {
                cols.store_id: store_ids,
                cols.product_id: product_ids,
                cols.unit_spend: [5.99] * len(store_ids),
            }
        )
        result = PctOfStores(df).df
        expected = pd.DataFrame(
            {
                cols.product_id: [501],
                stores_col: [len(set(store_ids))],
                pct_stores_col: [100.0],
            }
        )
        assert_frame_equal(result, expected)

    def test_empty_dataframe(self):
        """Test with an empty DataFrame returns empty result with correct columns."""
        df = pd.DataFrame(
            {
                cols.store_id: pd.Series([], dtype="int64"),
                cols.product_id: pd.Series([], dtype="int64"),
            }
        )
        result = PctOfStores(df).df
        expected = pd.DataFrame(
            {
                cols.product_id: pd.Series([], dtype="int64"),
                stores_col: pd.Series([], dtype="int64"),
                pct_stores_col: pd.Series([], dtype="float64"),
            }
        )
        assert_frame_equal(result, expected)

    def test_missing_column_raises(self):
        """Test that missing store_id column raises ValueError."""
        df = pd.DataFrame(
            {
                cols.product_id: [501, 502],
                cols.unit_spend: [5.99, 3.49],
            }
        )
        with pytest.raises(ValueError, match="Input data is missing required columns"):
            PctOfStores(df)

    def test_invalid_type_raises(self):
        """Test that passing a non-DataFrame/Table raises TypeError."""
        with pytest.raises(TypeError, match="pandas DataFrame or an Ibis Table"):
            PctOfStores({cols.store_id: [10], cols.product_id: [501]})

    @pytest.mark.parametrize(
        "kwargs",
        [{"group_col": "region"}, {"group_col": "region", "within_group": False}],
        ids=["default", "explicit_false"],
    )
    def test_group_col_uses_global_denominator(self, kwargs):
        """Test % of stores with group_col uses global denominator by default and with explicit within_group=False."""
        df = pd.DataFrame(
            {
                cols.store_id: [10, 20, 30, 40, 10],
                cols.product_id: [501, 501, 502, 502, 502],
                "region": ["North", "North", "South", "South", "North"],
                cols.unit_spend: [5.99, 3.49, 4.00, 6.00, 2.50],
            }
        )
        # Total stores = 4 (10, 20, 30, 40)
        # (501, North): stores {10, 20} → 50%
        # (502, North): stores {10} → 25%
        # (502, South): stores {30, 40} → 50%
        result = PctOfStores(df, **kwargs).df.sort_values([cols.product_id, "region"]).reset_index(drop=True)
        expected = pd.DataFrame(
            {
                cols.product_id: [501, 502, 502],
                "region": ["North", "North", "South"],
                stores_col: [2, 1, 2],
                pct_stores_col: [50.0, 25.0, 50.0],
            }
        )
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize("input_type", ["pandas", "ibis"])
    def test_accepts_pandas_and_ibis(self, input_type):
        """Test that both pandas and ibis inputs produce the same result."""
        pdf = pd.DataFrame(
            {
                cols.store_id: [10, 20, 20, 30],
                cols.product_id: [501, 501, 502, 502],
                cols.unit_spend: [5.99, 3.49, 4.00, 6.00],
            }
        )
        df = ibis.memtable(pdf) if input_type == "ibis" else pdf
        result = PctOfStores(df).df.sort_values(cols.product_id).reset_index(drop=True)
        expected = pd.DataFrame(
            {
                cols.product_id: [501, 502],
                stores_col: [2, 2],
                pct_stores_col: [2 / 3 * 100, 2 / 3 * 100],
            }
        )
        assert_frame_equal(result, expected)

    def test_custom_product_col_string(self):
        """Test with a custom product_col as a string."""
        df = pd.DataFrame(
            {
                cols.store_id: [10, 20, 30],
                "brand_name": ["Coca-Cola", "Coca-Cola", "Pepsi"],
                cols.unit_spend: [5.99, 3.49, 4.00],
            }
        )
        result = PctOfStores(df, product_col="brand_name").df.sort_values("brand_name").reset_index(drop=True)
        expected = pd.DataFrame(
            {
                "brand_name": ["Coca-Cola", "Pepsi"],
                stores_col: [2, 1],
                pct_stores_col: [2 / 3 * 100, 1 / 3 * 100],
            }
        )
        assert_frame_equal(result, expected)

    def test_custom_product_col_list(self):
        """Test with product_col as a list of columns."""
        df = pd.DataFrame(
            {
                cols.store_id: [10, 20, 30, 10],
                "brand": ["Coca-Cola", "Coca-Cola", "Pepsi", "Pepsi"],
                "size": ["330ml", "330ml", "500ml", "500ml"],
                cols.unit_spend: [1.50, 1.50, 2.00, 2.00],
            }
        )
        # Total stores = 3 (10, 20, 30)
        # (Coca-Cola, 330ml): stores {10, 20} → 66.67%
        # (Pepsi, 500ml): stores {10, 30} → 66.67%
        result = PctOfStores(df, product_col=["brand", "size"]).df.sort_values("brand").reset_index(drop=True)
        expected = pd.DataFrame(
            {
                "brand": ["Coca-Cola", "Pepsi"],
                "size": ["330ml", "500ml"],
                stores_col: [2, 2],
                pct_stores_col: [2 / 3 * 100, 2 / 3 * 100],
            }
        )
        assert_frame_equal(result, expected)

    def test_overlapping_product_col_and_group_col_raises(self):
        """Test that overlapping product_col and group_col raises ValueError."""
        df = pd.DataFrame(
            {
                cols.store_id: [10, 20, 30],
                "category": ["Beverages", "Snacks", "Beverages"],
                cols.unit_spend: [5.99, 3.49, 4.00],
            }
        )
        with pytest.raises(ValueError, match="category"):
            PctOfStores(df, product_col="category", group_col="category")

    def test_within_group_true_uses_per_group_denominator(self):
        """Test that within_group=True computes pct relative to stores in each group."""
        df = pd.DataFrame(
            {
                cols.store_id: [10, 20, 30, 40, 10],
                cols.product_id: [501, 501, 502, 502, 502],
                "region": ["North", "North", "South", "South", "North"],
                cols.unit_spend: [5.99, 3.49, 4.00, 6.00, 2.50],
            }
        )
        # North stores: {10, 20} = 2, South stores: {30, 40} = 2
        # (501, North): 2 selling / 2 in North → 100%
        # (502, North): 1 selling / 2 in North → 50%
        # (502, South): 2 selling / 2 in South → 100%
        result = (
            PctOfStores(df, group_col="region", within_group=True)
            .df.sort_values([cols.product_id, "region"])
            .reset_index(drop=True)
        )
        expected = pd.DataFrame(
            {
                cols.product_id: [501, 502, 502],
                "region": ["North", "North", "South"],
                stores_col: [2, 1, 2],
                pct_stores_col: [100.0, 50.0, 100.0],
            }
        )
        assert_frame_equal(result, expected)

    def test_within_group_ignored_without_group_col(self):
        """Test that within_group=True has no effect when group_col is None."""
        df = pd.DataFrame(
            {
                cols.store_id: [10, 20, 30, 40],
                cols.product_id: [501, 501, 502, 502],
                cols.unit_spend: [5.99, 3.49, 4.00, 6.00],
            }
        )
        result_with = PctOfStores(df, within_group=True).df.sort_values(cols.product_id).reset_index(drop=True)
        result_without = PctOfStores(df, within_group=False).df.sort_values(cols.product_id).reset_index(drop=True)
        assert_frame_equal(result_with, result_without)

    def test_duplicate_store_product_not_double_counted(self):
        """Test that duplicate store-product rows don't inflate the store count."""
        df = pd.DataFrame(
            {
                cols.store_id: [10, 10, 10, 20],
                cols.product_id: [501, 501, 501, 502],
                cols.unit_spend: [5.99, 3.49, 2.00, 6.00],
            }
        )
        # Total stores = 2 (10, 20)
        # Product 501: stores {10} → 50%
        # Product 502: stores {20} → 50%
        result = PctOfStores(df).df.sort_values(cols.product_id).reset_index(drop=True)
        expected = pd.DataFrame(
            {
                cols.product_id: [501, 502],
                stores_col: [1, 1],
                pct_stores_col: [50.0, 50.0],
            }
        )
        assert_frame_equal(result, expected)
