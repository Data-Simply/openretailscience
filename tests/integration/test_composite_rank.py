"""Unified integration tests for Composite Rank with multiple database backends."""

from openretailscience.analysis.composite_rank import CompositeRank


def test_composite_rank_integration(transactions_table):
    """Integration test for CompositeRank using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
    """
    limited_table = transactions_table.limit(1000)

    composite_rank = CompositeRank(
        df=limited_table,
        rank_cols=[
            ("unit_spend", "desc"),
            ("unit_quantity", "desc"),
        ],
        agg_func="mean",
        ignore_ties=False,
    )

    result = composite_rank.df
    assert result is not None
    assert "composite_rank" in result.columns
