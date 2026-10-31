"""Unified integration tests for segmentation statistics with multiple database backends."""

import pytest

from openretailscience.segmentation.segstats import SegTransactionStats


@pytest.mark.parametrize(
    ("grouping_sets", "extra_aggs"),
    [
        ("total", None),
        ("total", {"unique_products": ("product_id", "nunique")}),
        (None, None),
        (None, {"unique_products": ("product_id", "nunique")}),
    ],
)
def test_seg_transaction_stats_integration(
    transactions_table,
    grouping_sets,
    extra_aggs,
):
    """Integration test for SegTransactionStats using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
        grouping_sets: Grouping sets mode passed to SegTransactionStats
        extra_aggs: Extra aggregations to include
    """
    limited_table = transactions_table.limit(10000)

    seg_transaction_stats = SegTransactionStats(
        data=limited_table,
        segment_col=["category_0_name", "category_1_name"],
        grouping_sets=grouping_sets,
        extra_aggs=extra_aggs,
    )

    result = seg_transaction_stats.df
    grand_total_rows = (result["category_0_name"] == "Total") & (result["category_1_name"] == "Total")
    assert grand_total_rows.sum() == (1 if grouping_sets == "total" else 0)


def test_seg_transaction_stats_with_unknown_customer_integration(transactions_table):
    """Integration test for SegTransactionStats with unknown_customer_value using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
    """
    limited_table = transactions_table.limit(10000)

    seg_transaction_stats = SegTransactionStats(
        data=limited_table,
        segment_col=["category_0_name", "category_1_name"],
        grouping_sets="total",
        unknown_customer_value=-1,
    )

    result = seg_transaction_stats.df
    grand_total_rows = (result["category_0_name"] == "Total") & (result["category_1_name"] == "Total")
    assert grand_total_rows.sum() == 1

    assert "spend_unknown" in result.columns
    assert "transactions_unknown" in result.columns
    assert "spend_total" in result.columns
    assert "transactions_total" in result.columns
