"""Tests for openretailscience.metrics.distribution.acv."""

import ibis
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from openretailscience.metrics.distribution.acv import Acv
from openretailscience.options import ColumnHelper

cols = ColumnHelper()


class TestAcv:
    """Tests for the Acv metric class."""

    def test_acv_total_no_grouping(self):
        """Test total ACV across all transactions without grouping."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 1, 2],
                cols.store_id: [101, 101, 102, 102, 103],
                cols.product_id: [10, 20, 30, 40, 50],
                cols.unit_spend: [500_000.0, 750_000.0, 300_000.0, 600_000.0, 350_000.0],
            }
        )
        result = Acv(df).df
        expected = pd.DataFrame({"acv": [2.5]})
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize("input_type", ["pandas", "ibis"])
    def test_acv_grouped_by_store(self, input_type):
        """Test ACV grouped by store returns correct per-store values for both input types."""
        pdf = pd.DataFrame(
            {
                cols.store_id: [101, 101, 102, 102, 103],
                cols.unit_spend: [400_000.0, 600_000.0, 300_000.0, 200_000.0, 500_000.0],
            }
        )
        df = ibis.memtable(pdf) if input_type == "ibis" else pdf
        result = Acv(df, group_col=cols.store_id).df.sort_values(cols.store_id).reset_index(drop=True)
        expected = pd.DataFrame(
            {
                cols.store_id: [101, 102, 103],
                "acv": [1.0, 0.5, 0.5],
            }
        )
        assert_frame_equal(result, expected)

    def test_acv_group_col_list(self):
        """Test ACV grouped by multiple columns."""
        df = pd.DataFrame(
            {
                cols.store_id: [101, 101, 102],
                "region": ["North", "North", "South"],
                cols.unit_spend: [1_000_000.0, 500_000.0, 2_000_000.0],
            }
        )
        result = Acv(df, group_col=[cols.store_id, "region"]).df.sort_values(cols.store_id).reset_index(drop=True)
        expected = pd.DataFrame(
            {
                cols.store_id: [101, 102],
                "region": ["North", "South"],
                "acv": [1.5, 2.0],
            }
        )
        assert_frame_equal(result, expected)

    def test_acv_with_nan_values(self):
        """Test that NaN values are excluded from the ACV sum."""
        df = pd.DataFrame(
            {
                cols.store_id: [101, 101, 102],
                cols.unit_spend: [1_000_000.0, np.nan, 500_000.0],
            }
        )
        result = Acv(df, group_col=cols.store_id).df.sort_values(cols.store_id).reset_index(drop=True)
        expected = pd.DataFrame(
            {
                cols.store_id: [101, 102],
                "acv": [1.0, 0.5],
            }
        )
        assert_frame_equal(result, expected)

    def test_acv_missing_column_raises(self):
        """Test that missing unit_spend column raises ValueError naming required_cols."""
        df = pd.DataFrame({cols.customer_id: [1, 2], cols.store_id: [101, 102]})
        with pytest.raises(ValueError, match="required_cols references columns not present in the DataFrame"):
            Acv(df)

    def test_acv_missing_group_col_raises(self):
        """Test that missing group_col column raises ValueError naming group_col."""
        df = pd.DataFrame({cols.unit_spend: [100.0, 200.0]})
        with pytest.raises(ValueError, match="group_col references columns not present in the DataFrame"):
            Acv(df, group_col=cols.store_id)

    def test_acv_custom_scale_factor(self):
        """Test ACV with a custom scale factor."""
        df = pd.DataFrame(
            {
                cols.store_id: [101, 102],
                cols.unit_spend: [5_000.0, 10_000.0],
            }
        )
        result = Acv(df, acv_scale_factor=1_000).df
        expected = pd.DataFrame({"acv": [15.0]})
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize("scale_factor", [0, -1_000])
    def test_acv_non_positive_scale_factor_raises(self, scale_factor):
        """Test that zero or negative acv_scale_factor raises ValueError."""
        df = pd.DataFrame({cols.unit_spend: [500_000.0, 1_000_000.0]})
        with pytest.raises(ValueError, match="acv_scale_factor must be positive"):
            Acv(df, acv_scale_factor=scale_factor)

    def test_acv_invalid_type_raises(self):
        """Test that passing a non-DataFrame/Table raises TypeError."""
        with pytest.raises(TypeError, match="pandas DataFrame or an Ibis Table"):
            Acv({cols.unit_spend: [100.0]})
