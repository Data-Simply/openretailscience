"""Unified integration tests for Threshold Segmentation with multiple database backends."""

import pytest

from openretailscience.options import ColumnHelper
from openretailscience.segmentation.threshold import ThresholdSegmentation

cols = ColumnHelper()


@pytest.mark.parametrize(
    "zero_value_handling",
    ["separate_segment", "exclude", "include_with_light"],
)
def test_threshold_segmentation_integration(
    transactions_table,
    zero_value_handling,
):
    """Integration test for ThresholdSegmentation using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
        zero_value_handling: Parameter for handling zero-value customers
    """
    limited_table = transactions_table.limit(1000)

    threshold_segmentation = ThresholdSegmentation(
        df=limited_table,
        thresholds=[0.5, 1.0],
        segments=["Low", "High"],
        value_col=cols.unit_spend,
        agg_func="mean",
        zero_value_customers=zero_value_handling,
    )

    assert threshold_segmentation is not None
    result = threshold_segmentation.df
    assert result is not None
