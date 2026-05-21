"""Tests for the ProductAssociation module."""

import pandas as pd
import pytest

from openretailscience.analysis.product_association import ProductAssociation
from openretailscience.options import ColumnHelper, option_context

cols = ColumnHelper()


class TestProductAssociations:
    """Tests for the ProductAssociations class."""

    @pytest.fixture
    def transactions_df(self) -> pd.DataFrame:
        """Return a sample DataFrame for testing."""
        # fmt: off
        return pd.DataFrame({
            cols.transaction_id: [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5],
            "product": ["milk", "bread", "fruit", "butter", "eggs", "fruit", "beer", "diapers",
                        "milk", "bread", "butter", "eggs", "fruit", "bread"],
        })
        # fmt: on

    @pytest.fixture
    def expected_results_single_items_df(self) -> pd.DataFrame:
        """Return the expected results for the single items association analysis."""
        # fmt: off
        return pd.DataFrame(
                {
                    "product_1": [
                        "beer", "bread", "bread", "bread", "bread", "butter", "butter", "butter", "butter", "diapers",
                        "eggs", "eggs", "eggs", "eggs", "fruit", "fruit", "fruit", "fruit", "milk", "milk", "milk",
                        "milk",
                    ],
                    "product_2": [
                        "diapers", "butter", "eggs", "fruit", "milk", "bread", "eggs", "fruit", "milk", "beer", "bread",
                        "butter", "fruit", "milk", "bread", "butter", "eggs", "milk", "bread", "butter", "eggs",
                        "fruit",
                    ],
                    "occurrences_1": [1, 3, 3, 3, 3, 2, 2, 2, 2, 1, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2],
                    "occurrences_2": [1, 2, 2, 3, 2, 3, 2, 3, 2, 1, 3, 2, 3, 2, 3, 2, 2, 2, 3, 2, 2, 3],
                    "cooccurrences": [1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2],
                    "support": [
                        0.2, 0.2, 0.2, 0.4, 0.4, 0.2, 0.4, 0.4, 0.2, 0.2, 0.2, 0.4, 0.4, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4,
                        0.2, 0.2, 0.4,
                    ],
                    "confidence": [
                        1.0, 0.333333, 0.333333, 0.666667, 0.666667, 0.5, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5,
                        0.666667, 0.666667, 0.666667, 0.666667, 1.0, 0.5, 0.5, 1.0,
                    ],
                    "uplift": [
                        5.0, 0.833333, 0.833333, 1.111111, 1.666667, 0.833333, 2.5, 1.666667, 1.25, 5.0, 0.833333, 2.5,
                        1.666667, 1.25, 1.111111, 1.666667, 1.666667, 1.666667, 1.666667, 1.25, 1.25, 1.666667,
                    ],
                },
        )
        # fmt: on

    @pytest.fixture
    def expected_results_pair_items_df(self) -> pd.DataFrame:
        """Return the expected results for the pair items association analysis."""
        # fmt: off
        return pd.DataFrame(
            {
                "product_1": [
                    ("bread", "butter"), ("bread", "butter"), ("bread", "butter"), ("bread", "eggs"), ("bread", "eggs"),
                    ("bread", "eggs"), ("bread", "fruit"), ("bread", "fruit"), ("bread", "fruit"), ("bread", "milk"),
                    ("bread", "milk"), ("bread", "milk"), ("butter", "eggs"), ("butter", "eggs"), ("butter", "eggs"),
                    ("butter", "fruit"), ("butter", "fruit"), ("butter", "fruit"), ("butter", "milk"),
                    ("butter", "milk"), ("butter", "milk"), ("eggs", "fruit"), ("eggs", "fruit"), ("eggs", "fruit"),
                    ("eggs", "milk"), ("eggs", "milk"), ("eggs", "milk"), ("fruit", "milk"), ("fruit", "milk"),
                    ("fruit", "milk"),
                ],
                "product_2": [
                    "eggs", "fruit", "milk", "butter", "fruit", "milk", "butter", "eggs", "milk", "butter", "eggs",
                    "fruit", "bread", "fruit", "milk", "bread", "eggs", "milk", "bread", "eggs", "fruit", "bread",
                    "butter", "milk", "bread", "butter", "fruit", "bread", "butter", "eggs",
                ],
                "occurrences_1": [
                    1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                ],
                "occurrences_2": [
                    2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 3, 2, 2, 3, 2, 3, 3, 2, 2, 3, 2, 3, 3, 2, 2,
                ],
                "cooccurrences": [
                    1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1,
                ],
                "support": [
                    0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.4, 0.2, 0.4, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2,
                    0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2,
                ],
                "confidence": [
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0,
                    1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5,
                ],
                "uplift": [
                    2.5, 1.666667, 2.5, 2.5, 1.666667, 2.5, 1.25, 1.25, 2.5, 1.25, 1.25, 1.666667, 0.833333, 1.666667,
                    1.25, 0.833333, 2.5, 1.25, 1.666667, 2.5, 1.666667, 0.833333, 2.5, 1.25, 1.666667, 2.5, 1.666667,
                    1.666667, 1.25,1.25,
                ],
            },
        )
        # fmt: on

    def test_calc_association_all_single_items(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules for a single item versus another of item for all items."""
        associations_df = ProductAssociation(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
        )
        result = associations_df.df
        pd.testing.assert_frame_equal(result, expected_results_single_items_df)

    def test_calc_association_target_single_items(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules for target single item versus another of item."""
        target_item = "bread"

        calc_df = ProductAssociation(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            target_item=target_item,
        )
        result = calc_df.df
        pd.testing.assert_frame_equal(
            result,
            expected_results_single_items_df[expected_results_single_items_df["product_1"] == target_item].reset_index(
                drop=True,
            ),
        )

    def test_calc_association_min_occurrences(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules with a min occurrences level."""
        min_occurrences = 2

        calc_df = ProductAssociation(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            min_occurrences=min_occurrences,
        )

        result = calc_df.df
        pd.testing.assert_frame_equal(
            result,
            expected_results_single_items_df[
                (expected_results_single_items_df["occurrences_1"] >= min_occurrences)
                & (expected_results_single_items_df["occurrences_2"] >= min_occurrences)
            ].reset_index(drop=True),
        )

    def test_calc_association_min_cooccurrences(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules with a min occurrences level."""
        min_cooccurrences = 2

        calc_df = ProductAssociation(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            min_cooccurrences=min_cooccurrences,
        )

        result = calc_df.df
        pd.testing.assert_frame_equal(
            result,
            expected_results_single_items_df[
                (expected_results_single_items_df["cooccurrences"] >= min_cooccurrences)
            ].reset_index(drop=True),
        )

    def test_calc_association_min_support(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules with a min occurrences level."""
        min_support = 0.25

        calc_df = ProductAssociation(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            min_support=min_support,
        )

        result = calc_df.df
        pd.testing.assert_frame_equal(
            result,
            expected_results_single_items_df[(expected_results_single_items_df["support"] >= min_support)].reset_index(
                drop=True,
            ),
        )

    def test_calc_association_min_confidence(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules with a min occurrences level."""
        min_confidence = 0.25

        calc_df = ProductAssociation(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            min_confidence=min_confidence,
        )

        result = calc_df.df
        pd.testing.assert_frame_equal(
            result,
            expected_results_single_items_df[
                (expected_results_single_items_df["confidence"] >= min_confidence)
            ].reset_index(drop=True),
        )

    def test_calc_association_min_uplift(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules with a min occurrences level."""
        min_uplift = 1

        calc_df = ProductAssociation(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            min_uplift=min_uplift,
        )

        result = calc_df.df
        pd.testing.assert_frame_equal(
            result,
            expected_results_single_items_df[(expected_results_single_items_df["uplift"] >= min_uplift)].reset_index(
                drop=True,
            ),
        )

    def test_calc_association_invalid_min_occurrences(self, transactions_df):
        """Test calculating association rules with an invalid minimum occurrences value."""
        with pytest.raises(ValueError, match="Minimum occurrences must be at least 1"):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_occurrences=0,
            )

    def test_calc_association_invalid_min_cooccurrences(self, transactions_df):
        """Test calculating association rules with an invalid minimum cooccurrences value."""
        with pytest.raises(ValueError, match="Minimum cooccurrences must be at least 1"):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_cooccurrences=0,
            )

    def test_calc_association_min_support_invalid_range(self, transactions_df):
        """Test calculating association rules with an invalid minimum support range."""
        with pytest.raises(ValueError, match="Minimum support must be between 0 and 1"):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_support=-0.1,
            )
        with pytest.raises(ValueError, match="Minimum support must be between 0 and 1"):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_support=1.1,
            )

    def test_calc_association_min_confidence_invalid_range(self, transactions_df):
        """Test calculating association rules with an invalid minimum confidence range."""
        with pytest.raises(ValueError, match="Minimum confidence must be between 0 and 1"):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_confidence=-0.1,
            )
        with pytest.raises(ValueError, match="Minimum confidence must be between 0 and 1"):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_confidence=1.1,
            )

    def test_calc_association_min_uplift_invalid_range(self, transactions_df):
        """Test calculating association rules with an invalid minimum uplift range."""
        with pytest.raises(ValueError, match="Minimum uplift must be greater or equal to 0"):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_uplift=-0.1,
            )

    def test_calc_association_target_item_list(self, transactions_df):
        """Test calculating association rules with a list of target items."""
        target_items = ["milk", "bread"]

        calc_df = ProductAssociation(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            target_item=target_items,
        )

        result = calc_df.df

        # Verify that we get exactly the target items we expect
        assert set(target_items) == set(result["product_1"].unique())

    def test_calc_association_target_item_single_vs_list(self, transactions_df):
        """Test that single target item and list with single item produce same results."""
        single_target = "milk"
        list_target = ["milk"]

        calc_single = ProductAssociation(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            target_item=single_target,
        )

        calc_list = ProductAssociation(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            target_item=list_target,
        )

        # Results should be identical
        pd.testing.assert_frame_equal(calc_single.df, calc_list.df)

    def test_calc_association_target_item_empty_list(self, transactions_df):
        """Test that empty target item list raises ValueError."""
        with pytest.raises(ValueError, match="target_item cannot be an empty list"):
            ProductAssociation(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                target_item=[],
            )

    def test_calc_association_target_item_invalid_type(self, transactions_df):
        """Test that invalid types in target item list raise TypeError."""
        with pytest.raises(TypeError, match="target_item must contain only str or float values"):
            ProductAssociation(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                target_item=["milk", {"invalid": "dict"}],
            )

    def test_with_custom_column_names(self, transactions_df):
        """Test ProductAssociation with custom column names."""
        custom_df = transactions_df.rename(
            columns={
                "transaction_id": "custom_customer_id",
            },
        )

        with option_context("column.customer_id", "custom_customer_id"):
            pa = ProductAssociation(
                df=custom_df,
                value_col="product",
                group_col="custom_customer_id",
            )

            result = pa.df
            assert isinstance(result, pd.DataFrame)
            assert not result.empty, "Should produce results with custom column names"
