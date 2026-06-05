"""Tests for openretailscience.experimental.metrics.base."""

import ibis
import numpy as np
import pandas as pd
import pytest

from openretailscience.experimental.metrics.base import PERCENTAGE_SCALE, ratio_metric


class TestRatioMetric:
    """Tests for the ratio_metric helper function."""

    @pytest.mark.parametrize(
        ("num", "denom", "scale", "expected"),
        [
            pytest.param(10, 4, 100, 250.0, id="division_before_scale"),
            pytest.param(1, 2, PERCENTAGE_SCALE, 50.0, id="percentage_scale"),
            pytest.param(3, 4, 1000, 750.0, id="custom_scale"),
        ],
    )
    def test_ratio_calculation(self, num, denom, scale, expected):
        """Test ratio_metric computes (numerator / denominator) * scale correctly."""
        result = ratio_metric(ibis.literal(num), ibis.literal(denom), scale=scale).execute()
        assert result == expected

    def test_zero_denominator_returns_nan(self):
        """Test that zero denominator returns NaN (NULL in ibis)."""
        numerator = ibis.literal(10)
        denominator = ibis.literal(0)
        result = ratio_metric(numerator, denominator).execute()
        assert np.isnan(result)

    def test_ratio_with_table_columns(self):
        """Test ratio_metric with actual table columns from realistic data."""
        df = pd.DataFrame(
            {
                "stores_selling": [2, 1, 4],
                "total_stores": [4, 4, 4],
            }
        )
        table = ibis.memtable(df)
        result_table = table.mutate(pct=ratio_metric(table.stores_selling, table.total_stores))
        result = result_table.execute()
        expected_pct = [50.0, 25.0, 100.0]
        pd.testing.assert_series_equal(
            result["pct"],
            pd.Series(expected_pct, name="pct"),
        )

    def test_multiple_zero_denominators_in_column(self):
        """Test handling of mixed zero and non-zero denominators in a column."""
        df = pd.DataFrame(
            {
                "numerator": [10, 20, 30],
                "denominator": [2, 0, 5],
            }
        )
        table = ibis.memtable(df)
        result_table = table.mutate(ratio=ratio_metric(table.numerator, table.denominator, scale=1))
        result = result_table.execute()
        expected = pd.Series([5.0, float("nan"), 6.0], name="ratio")
        pd.testing.assert_series_equal(result["ratio"], expected)
