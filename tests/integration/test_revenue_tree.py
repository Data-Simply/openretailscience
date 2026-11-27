"""Integration tests for RevenueTree across database backends."""

import ibis
import pandas as pd
import pytest

from openretailscience.analysis.revenue_tree import RevenueTree
from openretailscience.options import ColumnHelper
from openretailscience.utils import filter_and_label_by_periods

cols = ColumnHelper()

PERIODS = {"H1 2023": ("2023-01-01", "2023-06-30"), "H2 2023": ("2023-07-01", "2023-12-31")}
SELECTION = {"period_col": "period", "p1_value": "H1 2023", "p2_value": "H2 2023"}


def _tree(transactions: ibis.Table, group_col: str | None, include_qty: bool) -> pd.DataFrame:
    labelled = filter_and_label_by_periods(transactions, PERIODS, "period")
    if not include_qty:
        labelled = labelled.drop(cols.unit_qty)
    return RevenueTree(labelled, **SELECTION, group_col=group_col).df


@pytest.mark.parametrize("include_qty", [True, False])
@pytest.mark.parametrize("group_col", [None, "category_0_name"])
def test_revenue_tree_matches_duckdb(transactions_table, duckdb_transactions_table, group_col, include_qty):
    """The backend's tree matches DuckDB's on the same transactions."""
    expected = _tree(duckdb_transactions_table, group_col, include_qty)

    pd.testing.assert_frame_equal(_tree(transactions_table, group_col, include_qty), expected)
