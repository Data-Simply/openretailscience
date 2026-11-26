"""Integration tests for GainLoss across database backends."""

import pandas as pd
import pytest

from openretailscience.analysis.gain_loss import GainLoss
from openretailscience.options import ColumnHelper
from openretailscience.utils import filter_and_label_by_periods

cols = ColumnHelper()

PERIODS = {"H1 2023": ("2023-01-01", "2023-06-30"), "H2 2023": ("2023-07-01", "2023-12-31")}
SELECTION = {
    "period_col": "period",
    "p1_value": "H1 2023",
    "p2_value": "H2 2023",
    "compare_col": "brand_name",
    "focus_value": "Apple",
    "comparison_value": "Samsung",
}


@pytest.mark.parametrize("group_col", [None, cols.store_id])
def test_gain_loss_matches_duckdb(transactions_table, duckdb_transactions_table, group_col):
    """The backend's summary matches DuckDB's on the same transactions."""
    labelled_locally = filter_and_label_by_periods(duckdb_transactions_table, PERIODS, "period")
    expected = GainLoss(labelled_locally, **SELECTION, group_col=group_col).df

    labelled = filter_and_label_by_periods(transactions_table, PERIODS, "period")
    result = GainLoss(labelled, **SELECTION, group_col=group_col).df

    pd.testing.assert_frame_equal(result, expected)
