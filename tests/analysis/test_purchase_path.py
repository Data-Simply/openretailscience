"""Tests for the purchase_path_analysis function."""

import ibis
import numpy as np
import pandas as pd
import pytest

from pyretailscience.analysis.purchase_path import purchase_path_analysis


class TestPurchasePathAnalysis:
    """Test the purchase_path_analysis function."""

    @pytest.fixture
    def sample_transactions_df(self):
        """Return a sample transactions DataFrame for testing."""
        return pd.DataFrame(
            {
                "customer_id": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
                "transaction_id": [101, 102, 103, 104, 201, 202, 203, 301, 302, 303, 304, 305],
                "transaction_date": [
                    "2023-01-01",
                    "2023-01-15",
                    "2023-02-01",
                    "2023-02-15",
                    "2023-01-05",
                    "2023-01-20",
                    "2023-02-05",
                    "2023-01-10",
                    "2023-01-25",
                    "2023-02-10",
                    "2023-02-25",
                    "2023-03-10",
                ],
                "product_id": ["A1", "B1", "A2", "C1", "B2", "A3", "C2", "A4", "B3", "C3", "A5", "D1"],
                "revenue": [50.0, 75.0, 100.0, 25.0, 80.0, 60.0, 90.0, 70.0, 85.0, 95.0, 110.0, 120.0],
                "product_category": [
                    "Electronics",
                    "Books",
                    "Electronics",
                    "Clothing",
                    "Books",
                    "Electronics",
                    "Clothing",
                    "Electronics",
                    "Books",
                    "Clothing",
                    "Electronics",
                    "Home",
                ],
            },
        )

    @pytest.fixture
    def multi_category_transactions_df(self):
        """Return transactions with multiple categories per basket."""
        return pd.DataFrame(
            {
                "customer_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                "transaction_id": [101, 101, 102, 102, 103, 103, 201, 201, 202, 202],
                "transaction_date": [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-15",
                    "2023-01-15",
                    "2023-02-01",
                    "2023-02-01",
                    "2023-01-05",
                    "2023-01-05",
                    "2023-01-20",
                    "2023-01-20",
                ],
                "product_id": ["A1", "B1", "A2", "C1", "A3", "B2", "B3", "C2", "A4", "C3"],
                "revenue": [50.0, 75.0, 100.0, 25.0, 80.0, 60.0, 85.0, 90.0, 70.0, 95.0],
                "product_category": [
                    "Electronics",
                    "Books",
                    "Electronics",
                    "Clothing",
                    "Electronics",
                    "Books",
                    "Books",
                    "Clothing",
                    "Electronics",
                    "Clothing",
                ],
            },
        )

    def test_basic_functionality(self, sample_transactions_df):
        """Test basic functionality with default parameters."""
        result = purchase_path_analysis(sample_transactions_df)

        assert isinstance(result, pd.DataFrame)
        assert "customer_count" in result.columns
        assert "transition_probability" in result.columns

    def test_missing_required_columns(self):
        """Test that function raises ValueError for missing required columns."""
        incomplete_df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "transaction_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                # Missing transaction_id, product_category, etc.
            },
        )

        with pytest.raises(ValueError, match="The following columns are required but missing:"):
            purchase_path_analysis(incomplete_df)

    @pytest.mark.parametrize(
        ("filter_param", "high_value", "low_value"),
        [
            ("min_transactions", 10, 1),
            ("min_basket_size", 5, 1),
            ("min_basket_value", 200.0, 1.0),
            ("min_customers", 10, 1),
        ],
    )
    def test_filtering_parameters(self, sample_transactions_df, filter_param, high_value, low_value):
        """Test that various filtering parameters work correctly."""
        high_kwargs = {filter_param: high_value}
        low_kwargs = {filter_param: low_value}

        result_high = purchase_path_analysis(sample_transactions_df, **high_kwargs)
        result_low = purchase_path_analysis(sample_transactions_df, **low_kwargs)

        assert len(result_high) <= len(result_low)

        # Check min_customers constraint specifically
        if filter_param == "min_customers" and len(result_high) > 0:
            assert all(result_high["customer_count"] >= high_value)

    def test_exclude_negative_revenue(self):
        """Test exclusion of negative revenue transactions."""
        df_with_negative = pd.DataFrame(
            {
                "customer_id": [1, 1, 1, 2, 2, 2],
                "transaction_id": [101, 102, 103, 201, 202, 203],
                "transaction_date": [
                    "2023-01-01",
                    "2023-01-15",
                    "2023-02-01",
                    "2023-01-05",
                    "2023-01-20",
                    "2023-02-05",
                ],
                "product_id": ["A1", "B1", "A2", "B2", "A3", "C1"],
                "revenue": [50.0, -25.0, 100.0, 80.0, -30.0, 90.0],
                "product_category": ["Electronics", "Books", "Electronics", "Books", "Electronics", "Clothing"],
            },
        )

        result_exclude = purchase_path_analysis(df_with_negative, exclude_negative_revenue=True)
        result_include = purchase_path_analysis(df_with_negative, exclude_negative_revenue=False)

        # Results should be different when including/excluding negative revenue
        if len(result_exclude) > 0 or len(result_include) > 0:
            assert not result_exclude.equals(result_include), (
                "Results should differ when negative revenue handling changes"
            )

    @pytest.mark.parametrize(
        ("multi_category_handling", "sort_by", "aggregation_function"),
        [
            ("concatenate", "alphabetical", None),
            ("concatenate", "aggregation", "sum"),
            ("concatenate", "aggregation", "max"),
            ("concatenate", "aggregation", "min"),
            ("concatenate", "aggregation", "avg"),
            ("separate_rows", "alphabetical", None),
        ],
    )
    def test_multi_category_configurations(
        self,
        multi_category_transactions_df,
        multi_category_handling,
        sort_by,
        aggregation_function,
    ):
        """Test different multi-category handling configurations."""
        kwargs = {
            "multi_category_handling": multi_category_handling,
            "sort_by": sort_by,
            "min_customers": 1,
            "min_transactions": 1,
        }

        if aggregation_function:
            kwargs.update(
                {
                    "aggregation_column": "revenue",
                    "aggregation_function": aggregation_function,
                },
            )

        result = purchase_path_analysis(multi_category_transactions_df, **kwargs)
        assert isinstance(result, pd.DataFrame)

    def test_custom_category_column(self, sample_transactions_df):
        """Test custom category column name."""
        df_custom = sample_transactions_df.copy()
        df_custom["custom_category"] = df_custom["product_category"]
        df_custom = df_custom.drop("product_category", axis=1)

        result = purchase_path_analysis(df_custom, category_column="custom_category")
        assert isinstance(result, pd.DataFrame)

    def test_ibis_table_input(self, sample_transactions_df):
        """Test that function works with Ibis table input."""
        ibis_table = ibis.memtable(sample_transactions_df)
        result = purchase_path_analysis(ibis_table)
        assert isinstance(result, pd.DataFrame)

    def test_result_structure_and_sorting(self, sample_transactions_df):
        """Test result structure, column naming, and sorting."""
        result = purchase_path_analysis(sample_transactions_df, min_customers=1)

        # Check basket column naming (should be basket_1, basket_2, etc.)
        basket_columns = [col for col in result.columns if col.startswith("basket_")]
        if basket_columns:
            expected_columns = [f"basket_{i}" for i in range(1, len(basket_columns) + 1)]
            actual_basket_cols = sorted(basket_columns, key=lambda x: int(x.split("_")[1]))
            assert actual_basket_cols == expected_columns

        # Check sorting by customer count (descending)
        if len(result) > 1:
            customer_counts = result["customer_count"].tolist()
            assert customer_counts == sorted(customer_counts, reverse=True)

        # Check transition probabilities
        if len(result) > 0:
            assert all(0 <= prob <= 1 for prob in result["transition_probability"])
            prob_sum = result["transition_probability"].sum()
            value = 0.001
            assert abs(prob_sum - 1.0) < value

    @pytest.mark.parametrize(
        ("scenario", "expected_empty"),
        [
            # Very restrictive filters should return empty results
            ({"min_transactions": 100, "min_basket_size": 50, "min_basket_value": 10000}, True),
            # Normal filters should return results
            ({"min_transactions": 1, "min_customers": 1}, False),
        ],
    )
    def test_edge_cases_and_empty_results(self, sample_transactions_df, scenario, expected_empty):
        """Test edge cases that may result in empty DataFrames."""
        result = purchase_path_analysis(sample_transactions_df, **scenario)

        assert isinstance(result, pd.DataFrame)
        assert "customer_count" in result.columns
        assert "transition_probability" in result.columns

        if expected_empty:
            assert len(result) == 0

    def test_minimal_data_single_customer(self):
        """Test with minimal data - single customer, single transaction."""
        minimal_df = pd.DataFrame(
            {
                "customer_id": [1],
                "transaction_id": [101],
                "transaction_date": ["2023-01-01"],
                "product_id": ["A1"],
                "revenue": [50.0],
                "product_category": ["Electronics"],
            },
        )

        result = purchase_path_analysis(
            minimal_df,
            min_transactions=1,
            min_customers=1,
        )

        assert isinstance(result, pd.DataFrame)

    def test_same_category_all_transactions(self):
        """Test edge case where all transactions are in the same category."""
        same_category_df = pd.DataFrame(
            {
                "customer_id": [1, 1, 2, 2, 3, 3],
                "transaction_id": [101, 102, 201, 202, 301, 302],
                "transaction_date": [
                    "2023-01-01",
                    "2023-01-15",
                    "2023-01-05",
                    "2023-01-20",
                    "2023-01-10",
                    "2023-01-25",
                ],
                "product_id": ["A1", "A2", "A3", "A4", "A5", "A6"],
                "revenue": [50.0, 75.0, 80.0, 60.0, 70.0, 85.0],
                "product_category": ["Electronics"] * 6,
            },
        )

        result = purchase_path_analysis(same_category_df)
        assert isinstance(result, pd.DataFrame)

    def test_performance_with_larger_dataset(self):
        """Test performance characteristics with a larger synthetic dataset."""
        rng = np.random.default_rng(42)

        n_customers = 50  # Reduced from 100 for faster testing
        n_transactions_per_customer = 5  # Reduced from 10
        categories = ["Electronics", "Books", "Clothing", "Home", "Sports"]

        data = []
        for customer_id in range(1, n_customers + 1):
            for trans_num in range(1, n_transactions_per_customer + 1):
                transaction_id = customer_id * 1000 + trans_num
                data.append(
                    {
                        "customer_id": customer_id,
                        "transaction_id": transaction_id,
                        "transaction_date": f"2023-{trans_num:02d}-01",
                        "product_id": f"P{transaction_id}",
                        "revenue": rng.uniform(10, 200),
                        "product_category": rng.choice(categories),
                    },
                )

        large_df = pd.DataFrame(data)
        result = purchase_path_analysis(large_df)

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.parametrize("max_depth", [2, 5, 10])
    def test_max_depth_parameter(self, sample_transactions_df, max_depth):
        """Test maximum depth parameter with different values."""
        result = purchase_path_analysis(sample_transactions_df, max_depth=max_depth)
        assert isinstance(result, pd.DataFrame)

        # Check that basket columns don't exceed max_depth
        basket_cols = [col for col in result.columns if col.startswith("basket_")]
        if basket_cols:
            max_basket_num = max(int(col.split("_")[1]) for col in basket_cols)
            assert max_basket_num <= max_depth
