"""Tests for the CompositeRank class."""

import ibis
import pandas as pd
import pytest

from openretailscience.analysis.composite_rank import CompositeRank


class TestCompositeRank:
    """Tests for the CompositeRank class."""

    @pytest.fixture
    def simple_df(self):
        """Create a simple DataFrame fixture for basic tests."""
        return pd.DataFrame(
            {
                "product_id": [1, 2, 3],
                "spend": [100, 150, 75],
                "customers": [20, 30, 15],
            },
        )

    def test_basic_functionality(self):
        """Test the basic functionality of the CompositeRank class."""
        # Create test data
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5],
                "spend": [100, 150, 75, 200, 125],
                "customers": [20, 30, 15, 40, 25],
                "spend_per_customer": [5.0, 5.0, 5.0, 5.0, 5.0],
            },
        )

        # Test with ascending and descending columns
        rank_cols = [
            ("spend", "desc"),  # Higher spend is better
            ("customers", "desc"),  # Higher customer count is better
            ("spend_per_customer", "desc"),  # Higher spend per customer is better
        ]

        # Create expected dataframe with ranks
        expected_df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5],
                "spend": [100, 150, 75, 200, 125],
                "customers": [20, 30, 15, 40, 25],
                "spend_per_customer": [5.0, 5.0, 5.0, 5.0, 5.0],
                "spend_rank": [4, 2, 5, 1, 3],
                "customers_rank": [4, 2, 5, 1, 3],
                "spend_per_customer_rank": [1, 1, 1, 1, 1],
                "composite_rank": [3.0, 1.6666667, 3.6666667, 1.0, 2.3333333],
            },
        )

        # Create instance with mean aggregation
        cr = CompositeRank(df=df, rank_cols=rank_cols, agg_func="mean", ignore_ties=False)

        # Extract only the relevant columns for comparison
        result_subset = cr.df.sort_values("product_id").reset_index(drop=True)

        pd.testing.assert_frame_equal(result_subset, expected_df)

    def test_string_column_specs(self, simple_df):
        """Test that the class accepts string column specifications."""
        # Test with just column names (should default to ascending)
        cr = CompositeRank(df=simple_df, rank_cols=["spend", "customers"], agg_func="mean", ignore_ties=False)

        result = cr.df.sort_values("product_id").reset_index(drop=True)

        expected = pd.DataFrame(
            {
                "product_id": [1, 2, 3],
                "spend": [100, 150, 75],
                "customers": [20, 30, 15],
                "spend_rank": [2, 3, 1],  # Ascending: lowest values (75) get rank 1
                "customers_rank": [2, 3, 1],  # Ascending: lowest values (15) get rank 1
                "composite_rank": [2.0, 3.0, 1.0],
            },
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_ignore_ties(self):
        """Test the behavior with the ignore_ties parameter."""
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4],
                "spend": [100, 100, 150, 150],
            },
        )

        # Without ignore_ties (should have ties)
        cr_with_ties = CompositeRank(df=df, rank_cols=[("spend", "desc")], agg_func="mean", ignore_ties=False)

        result_with_ties = cr_with_ties.df.sort_values("product_id").reset_index(drop=True)

        expected_df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4],
                "spend": [100, 100, 150, 150],
                "spend_rank": [3, 3, 1, 1],  # Descending: [100, 100] gets rank 3, [150, 150] gets rank 1
                "composite_rank": [3.0, 3.0, 1.0, 1.0],
            },
        )

        pd.testing.assert_frame_equal(result_with_ties, expected_df)

        # With ignore_ties (should have unique ranks)
        cr_no_ties = CompositeRank(df=df, rank_cols=[("spend", "desc")], agg_func="mean", ignore_ties=True)

        # Verify all ranks are unique and span the expected range
        expected_unique_count = 4
        assert len(cr_no_ties.df["spend_rank"].unique()) == expected_unique_count
        assert set(cr_no_ties.df["spend_rank"]) == {1, 2, 3, 4}

        # Verify ordering: products with higher spend should have lower (better) ranks
        result = cr_no_ties.df
        high_spend = 150
        low_spend = 100
        high_spend_ranks = result.loc[result["spend"] == high_spend, "spend_rank"]
        low_spend_ranks = result.loc[result["spend"] == low_spend, "spend_rank"]
        assert high_spend_ranks.max() < low_spend_ranks.min()

    @pytest.mark.parametrize(
        ("agg_func", "expected_composite_ranks"),
        [
            ("mean", [3.0, 3.6667, 3.3333, 2.3333, 2.6667]),
            ("sum", [9, 11, 10, 7, 8]),
            ("min", [2, 2, 1, 1, 2]),
            ("max", [4, 5, 5, 5, 3]),
        ],
    )
    def test_different_agg_functions(self, agg_func, expected_composite_ranks):
        """Test different aggregation functions using parametrization."""
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5],
                "spend": [100, 150, 75, 200, 125],  # Ranks descending: 4, 2, 5, 1, 3
                "customers": [30, 10, 20, 40, 25],  # Ranks descending: 2, 5, 4, 1, 3
                "profit": [50, 30, 70, 10, 60],  # Ranks descending: 3, 4, 1, 5, 2
            },
        )
        rank_cols = [("spend", "desc"), ("customers", "desc"), ("profit", "desc")]

        # Create instance with specified aggregation function
        cr = CompositeRank(df=df, rank_cols=rank_cols, agg_func=agg_func, ignore_ties=False)
        result = cr.df.sort_values("product_id").reset_index(drop=True)

        # Create expected results DataFrame
        expected = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5],
                "spend": [100, 150, 75, 200, 125],
                "customers": [30, 10, 20, 40, 25],
                "profit": [50, 30, 70, 10, 60],
                "spend_rank": [4, 2, 5, 1, 3],
                "customers_rank": [2, 5, 4, 1, 3],
                "profit_rank": [3, 4, 1, 5, 2],
                "composite_rank": expected_composite_ranks,
            },
        )

        pd.testing.assert_frame_equal(
            result,
            expected.sort_values("product_id").reset_index(drop=True),
            rtol=1e-3,
        )

    @pytest.mark.parametrize(
        ("rank_cols", "agg_func", "expected_error_pattern"),
        [
            (
                [],
                "mean",
                "rank_cols must contain at least one column specification",
            ),
            (
                [("spend", "desc"), ("non_existent_column", "asc")],
                "mean",
                r"\['non_existent_column'\]",
            ),
            (
                [("spend", "invalid_sort_order")],
                "mean",
                "sort_order must be one of",
            ),
            (
                [("spend", "desc", "extra_value")],
                "mean",
                "Column specification must be a string or a tuple of",
            ),
            (
                [("spend", "desc")],
                "invalid_func",
                "agg_func must be one of",
            ),
        ],
    )
    def test_invalid_inputs(self, simple_df, rank_cols, agg_func, expected_error_pattern):
        """Test behavior with various invalid inputs."""
        with pytest.raises(ValueError, match=expected_error_pattern):
            CompositeRank(df=simple_df, rank_cols=rank_cols, agg_func=agg_func, ignore_ties=False)

    @pytest.mark.parametrize("sort_order", ["ASC", "Asc", "ASCENDING"])
    def test_sort_order_case_insensitive_ascending(self, simple_df, sort_order):
        """Uppercase ascending sort_order values produce true ascending ranks (regression for #429 bug)."""
        cr = CompositeRank(df=simple_df, rank_cols=[("spend", sort_order)], agg_func="mean")
        result = cr.df.sort_values("product_id").reset_index(drop=True)
        # spend = [100, 150, 75] → ascending ranks are [2, 3, 1]
        assert result["spend_rank"].tolist() == [2, 3, 1]

    @pytest.mark.parametrize("agg_func", ["MEAN", "Mean", "SUM", "Min", "MAX"])
    def test_agg_func_case_insensitive(self, simple_df, agg_func):
        """Mixed-case agg_func produces identical output to the lowercase form (regression for #429 bug)."""
        rank_cols = [("spend", "desc"), ("customers", "desc")]
        upper = CompositeRank(df=simple_df, rank_cols=rank_cols, agg_func=agg_func).df
        lower = CompositeRank(df=simple_df, rank_cols=rank_cols, agg_func=agg_func.lower()).df
        pd.testing.assert_frame_equal(upper, lower)

    def test_ibis_table_input(self, simple_df):
        """Test that the class works with Ibis table input."""
        # Convert to Ibis table
        ibis_table = ibis.memtable(simple_df)

        # Create CompositeRank instance
        cr = CompositeRank(
            df=ibis_table,
            rank_cols=[("spend", "desc"), ("customers", "desc")],
            agg_func="mean",
            ignore_ties=False,
        )

        assert all(col in cr.df.columns for col in ["spend_rank", "customers_rank", "composite_rank"])

    def test_df_cache(self, simple_df):
        """Test that the df property caches results."""
        # Create instance
        cr = CompositeRank(df=simple_df, rank_cols=[("spend", "desc")], agg_func="mean", ignore_ties=False)

        # Access df property to populate cache
        result1 = cr.df

        # Verify _df is populated
        assert cr._df is not None

        # Access df property again to use cache
        result2 = cr.df

        # Verify both results are the same object (cached)
        assert result1 is result2

    def test_group_vs_global_ranking_difference(self):
        """Test that group-based ranking produces different results than global ranking."""
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4],
                "category": ["Electronics", "Electronics", "Apparel", "Apparel"],
                "sales": [100, 80, 90, 70],  # Global ranks: 1, 3, 2, 4
            },
        )

        # Global ranking
        global_ranker = CompositeRank(
            df=df,
            rank_cols=[("sales", "desc")],
            agg_func="mean",
            group_col=None,
        )
        global_result = global_ranker.df.sort_values("product_id").reset_index(drop=True)

        # Group-based ranking
        group_ranker = CompositeRank(
            df=df,
            rank_cols=[("sales", "desc")],
            agg_func="mean",
            group_col="category",
        )
        group_result = group_ranker.df.sort_values("product_id").reset_index(drop=True)

        # Expected global rankings (across all products)
        expected_global = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4],
                "category": ["Electronics", "Electronics", "Apparel", "Apparel"],
                "sales": [100, 80, 90, 70],
                "sales_rank": [1, 3, 2, 4],  # Global ranks based on sales values
                "composite_rank": [1.0, 3.0, 2.0, 4.0],
            },
        )

        # Expected group rankings (within each category)
        expected_group = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4],
                "category": ["Electronics", "Electronics", "Apparel", "Apparel"],
                "sales": [100, 80, 90, 70],
                "sales_rank": [1, 2, 1, 2],  # Within-group ranks
                "composite_rank": [1.0, 2.0, 1.0, 2.0],
            },
        )

        # Compare global results
        pd.testing.assert_frame_equal(
            global_result,
            expected_global.sort_values("product_id").reset_index(drop=True),
        )

        # Compare group results
        pd.testing.assert_frame_equal(
            group_result,
            expected_group.sort_values("product_id").reset_index(drop=True),
        )

    def test_invalid_group_col(self, simple_df):
        """Test behavior with an invalid group column."""
        with pytest.raises(ValueError, match=r"\['non_existent_group'\]"):
            CompositeRank(
                df=simple_df,
                rank_cols=[("spend", "desc")],
                agg_func="mean",
                group_col="non_existent_group",
            )

    @pytest.mark.parametrize(
        "invalid_group_col",
        [123, 12.5, {"key": "value"}],
    )
    def test_invalid_group_col_type(self, simple_df, invalid_group_col):
        """Test behavior with invalid group_col type."""
        with pytest.raises(TypeError, match="columns must be a string or list of strings"):
            CompositeRank(
                df=simple_df,
                rank_cols=[("spend", "desc")],
                agg_func="mean",
                group_col=invalid_group_col,
            )

    @pytest.mark.parametrize(
        ("invalid_group_col", "expected_error_message"),
        [
            ([], "columns must not be an empty list"),
            (["invalid"], r"\['invalid'\]"),
        ],
    )
    def test_invalid_group_col_value(self, simple_df, invalid_group_col, expected_error_message):
        """Test behavior with invalid group_col values."""
        with pytest.raises(ValueError, match=expected_error_message):
            CompositeRank(
                df=simple_df,
                rank_cols=[("spend", "desc")],
                agg_func="mean",
                group_col=invalid_group_col,
            )

    def test_group_col_with_null_values(self):
        """Test behavior when group column contains NULL values."""
        # Test data with None values in group column (common in retail data)
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5],
                "category": ["Electronics", None, "Apparel", None, "Electronics"],
                "sales": [100, 80, 90, 70, 60],
                "margin": [20, 15, 25, 10, 18],
            },
        )

        ranker = CompositeRank(
            df=df,
            rank_cols=[("sales", "desc"), ("margin", "desc")],
            agg_func="mean",
            group_col="category",
        )

        result = ranker.df.sort_values("product_id").reset_index(drop=True)

        # Create expected results DataFrame
        expected = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5],
                "category": ["Electronics", None, "Apparel", None, "Electronics"],
                "sales": [100, 80, 90, 70, 60],
                "margin": [20, 15, 25, 10, 18],
                "sales_rank": [1, 1, 1, 2, 2],  # Within-group ranks: Electronics [1,2], NULL [1,2], Apparel [1]
                "margin_rank": [1, 1, 1, 2, 2],  # Within-group ranks: Electronics [1,2], NULL [1,2], Apparel [1]
                "composite_rank": [1.0, 1.0, 1.0, 2.0, 2.0],  # Mean of sales_rank and margin_rank
            },
        )

        pd.testing.assert_frame_equal(
            result,
            expected.sort_values("product_id").reset_index(drop=True),
        )

    def test_group_with_ignore_ties(self):
        """Test group-based ranking with ignore_ties=True produces unique ranks within each group."""
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4],
                "category": ["Electronics", "Electronics", "Apparel", "Apparel"],
                "sales": [100, 100, 50, 50],  # Ties within each group
            },
        )

        ranker = CompositeRank(
            df=df,
            rank_cols=[("sales", "desc")],
            agg_func="mean",
            group_col="category",
            ignore_ties=True,
        )

        result = ranker.df
        expected_group_size = 2
        expected_ranks = {1, 2}

        # Verify ranks are unique within each group and in the expected range
        for group_name in ["Electronics", "Apparel"]:
            group_data = result.loc[result["category"] == group_name]
            group_ranks = group_data["sales_rank"]
            assert len(group_ranks.unique()) == expected_group_size
            assert set(group_ranks) == expected_ranks

            # composite_rank equals sales_rank when there's only one rank column with mean agg
            group_composite = group_data["composite_rank"]
            assert len(group_composite.unique()) == expected_group_size
            assert set(group_composite) == {1.0, 2.0}

    @pytest.mark.parametrize(
        ("agg_func", "expected_composite_ranks"),
        [
            ("mean", [1.5, 1.5, 3.0, 1.5, 1.5, 3.0]),
            ("sum", [3, 3, 6, 3, 3, 6]),
            ("min", [1, 1, 3, 1, 1, 3]),
            ("max", [2, 2, 3, 2, 2, 3]),
        ],
    )
    def test_group_agg_functions(self, agg_func, expected_composite_ranks):
        """Test different aggregation functions with group-based ranking."""
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5, 6],
                "category": ["Electronics", "Electronics", "Electronics", "Apparel", "Apparel", "Apparel"],
                "sales": [1000, 800, 600, 500, 400, 300],  # Within group ranks: E=[1,2,3], A=[1,2,3]
                "margin": [20, 25, 15, 30, 35, 25],  # Within group ranks: E=[2,1,3], A=[2,1,3]
            },
        )

        ranker = CompositeRank(
            df=df,
            rank_cols=[("sales", "desc"), ("margin", "desc")],
            agg_func=agg_func,
            group_col="category",
        )

        result = ranker.df.sort_values("product_id").reset_index(drop=True)

        # Create expected results DataFrame
        expected = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5, 6],
                "category": ["Electronics", "Electronics", "Electronics", "Apparel", "Apparel", "Apparel"],
                "sales": [1000, 800, 600, 500, 400, 300],
                "margin": [20, 25, 15, 30, 35, 25],
                "sales_rank": [1, 2, 3, 1, 2, 3],  # Within-group ranks for sales
                "margin_rank": [2, 1, 3, 2, 1, 3],  # Within-group ranks for margin
                "composite_rank": expected_composite_ranks,
            },
        )

        pd.testing.assert_frame_equal(
            result,
            expected.sort_values("product_id").reset_index(drop=True),
            rtol=1e-3,
        )

    def test_single_group(self):
        """Test behavior when all data belongs to a single group."""
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3],
                "category": ["Electronics", "Electronics", "Electronics"],  # All same group
                "sales": [100, 200, 150],
            },
        )

        ranker = CompositeRank(
            df=df,
            rank_cols=[("sales", "desc")],
            agg_func="mean",
            group_col="category",
        )

        result = ranker.df.sort_values("product_id").reset_index(drop=True)

        expected = pd.DataFrame(
            {
                "product_id": [1, 2, 3],
                "category": ["Electronics", "Electronics", "Electronics"],
                "sales": [100, 200, 150],
                "sales_rank": [3, 1, 2],  # Values [100, 200, 150] descending = ranks [3, 1, 2]
                "composite_rank": [3.0, 1.0, 2.0],
            },
        )

        pd.testing.assert_frame_equal(
            result,
            expected.sort_values("product_id").reset_index(drop=True),
        )

    def test_group_with_ibis_table(self):
        """Test that group-based ranking works with Ibis table input."""
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4],
                "category": ["Electronics", "Electronics", "Apparel", "Apparel"],
                "sales": [100, 80, 90, 70],
            },
        )

        ibis_table = ibis.memtable(df)

        ranker = CompositeRank(
            df=ibis_table,
            rank_cols=[("sales", "desc")],
            agg_func="mean",
            group_col="category",
        )

        result = ranker.df.sort_values("product_id").reset_index(drop=True)

        expected = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4],
                "category": ["Electronics", "Electronics", "Apparel", "Apparel"],
                "sales": [100, 80, 90, 70],
                "sales_rank": [1, 2, 1, 2],  # Within-group ranks: Electronics [1,2], Apparel [1,2]
                "composite_rank": [1.0, 2.0, 1.0, 2.0],
            },
        )

        pd.testing.assert_frame_equal(
            result,
            expected.sort_values("product_id").reset_index(drop=True),
        )

    def test_group_col_with_list(self):
        """Test group-based ranking with list of group columns."""
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5, 6, 7, 8],
                "region": ["North", "North", "South", "South", "North", "North", "South", "South"],
                "category": [
                    "Electronics",
                    "Apparel",
                    "Electronics",
                    "Apparel",
                    "Electronics",
                    "Apparel",
                    "Electronics",
                    "Apparel",
                ],
                "sales": [100, 80, 90, 70, 95, 85, 88, 75],
            },
        )

        ranker = CompositeRank(
            df=df,
            rank_cols=[("sales", "desc")],
            agg_func="mean",
            group_col=["region", "category"],  # Group by both region and category
        )

        result = ranker.df.sort_values("product_id").reset_index(drop=True)

        expected = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5, 6, 7, 8],
                "region": ["North", "North", "South", "South", "North", "North", "South", "South"],
                "category": [
                    "Electronics",
                    "Apparel",
                    "Electronics",
                    "Apparel",
                    "Electronics",
                    "Apparel",
                    "Electronics",
                    "Apparel",
                ],
                "sales": [100, 80, 90, 70, 95, 85, 88, 75],
                "sales_rank": [1, 2, 1, 2, 2, 1, 2, 1],
                "composite_rank": [1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0],
            },
        )

        pd.testing.assert_frame_equal(
            result,
            expected.sort_values("product_id").reset_index(drop=True),
        )
