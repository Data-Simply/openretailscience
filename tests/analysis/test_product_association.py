"""Tests for the ProductAssociation module."""

import uuid

import ibis
import numpy as np
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

    @pytest.mark.parametrize(
        ("param", "value", "match"),
        [
            ("min_occurrences", 0, "min_occurrences must be at least 1. Got 0."),
            ("min_cooccurrences", 0, "min_cooccurrences must be at least 1. Got 0."),
            ("min_support", -0.1, r"min_support must be between 0 and 1 \(inclusive\). Got -0.1."),
            ("min_support", 1.1, r"min_support must be between 0 and 1 \(inclusive\). Got 1.1."),
            ("min_confidence", -0.1, r"min_confidence must be between 0 and 1 \(inclusive\). Got -0.1."),
            ("min_confidence", 1.1, r"min_confidence must be between 0 and 1 \(inclusive\). Got 1.1."),
            ("min_uplift", -0.1, "min_uplift must be greater than or equal to 0. Got -0.1."),
        ],
    )
    def test_out_of_range_minimum_names_param_and_value(self, transactions_df, param, value, match):
        """Each out-of-range min_* threshold is rejected naming the parameter and the value passed."""
        with pytest.raises(ValueError, match=match):
            ProductAssociation(df=transactions_df, value_col="product", group_col=cols.transaction_id, **{param: value})

    @pytest.mark.parametrize(
        ("param", "value", "match"),
        [
            ("min_occurrences", "5", "min_occurrences must be an integer. Got str."),
            ("min_cooccurrences", 1.5, "min_cooccurrences must be an integer. Got float."),
            ("min_uplift", "1.2", "min_uplift must be a number. Got str."),
        ],
    )
    def test_wrong_type_minimum_names_param(self, transactions_df, param, value, match):
        """A min_* threshold of the wrong type is rejected naming the parameter, not with a bare comparison error."""
        with pytest.raises(TypeError, match=match):
            ProductAssociation(df=transactions_df, value_col="product", group_col=cols.transaction_id, **{param: value})

    def test_numpy_minimums_match_python_numbers(self, transactions_df):
        """Thresholds read from a DataFrame as numpy scalars filter the same as their Python equivalents."""
        numpy_result = ProductAssociation(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            min_occurrences=np.int64(2),
            min_support=np.float32(0.25),
            min_uplift=np.float64(1.0),
        ).df
        python_result = ProductAssociation(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            min_occurrences=2,
            min_support=0.25,
            min_uplift=1.0,
        ).df
        pd.testing.assert_frame_equal(numpy_result, python_result)

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

    @pytest.mark.parametrize("target_item", [1001, [1001], np.int64(1001)])
    def test_integer_product_ids_accepted_as_target_item(self, transactions_df, target_item):
        """Integer SKUs work as target_item and give the same rules as the matching product names."""
        skus = {"milk": 1001, "bread": 1002, "fruit": 1003, "butter": 1004, "eggs": 1005, "beer": 1006, "diapers": 1007}
        by_name = ProductAssociation(
            df=transactions_df, value_col="product", group_col=cols.transaction_id, target_item="milk"
        ).df
        by_sku = ProductAssociation(
            df=transactions_df.assign(product=transactions_df["product"].map(skus)),
            value_col="product",
            group_col=cols.transaction_id,
            target_item=target_item,
        ).df

        expected = by_name.assign(product_1=by_name["product_1"].map(skus), product_2=by_name["product_2"].map(skus))
        pd.testing.assert_frame_equal(
            by_sku.sort_values("product_2", ignore_index=True), expected.sort_values("product_2", ignore_index=True)
        )

    def test_string_target_item_accepted_for_uuid_product_ids(self, transactions_df):
        """A warehouse uuid SKU column takes its IDs as strings and gives the same rules as the product names."""
        skus = {name: str(uuid.uuid5(uuid.NAMESPACE_URL, name)) for name in transactions_df["product"].unique()}
        by_name = ProductAssociation(
            df=transactions_df, value_col="product", group_col=cols.transaction_id, target_item="milk"
        ).df
        uuid_table = ibis.memtable(transactions_df.assign(product=transactions_df["product"].map(skus)))
        by_sku = ProductAssociation(
            df=uuid_table.mutate(product=uuid_table["product"].cast("uuid")),
            value_col="product",
            group_col=cols.transaction_id,
            target_item=skus["milk"],
        ).df

        by_sku = by_sku.assign(product_1=by_sku["product_1"].astype(str), product_2=by_sku["product_2"].astype(str))
        expected = by_name.assign(product_1=by_name["product_1"].map(skus), product_2=by_name["product_2"].map(skus))
        pd.testing.assert_frame_equal(
            by_sku.sort_values("product_2", ignore_index=True), expected.sort_values("product_2", ignore_index=True)
        )

    @pytest.mark.parametrize(
        ("value_col", "target_item", "match"),
        [
            ("sku", "La Roche-Posay", "target_item 'La Roche-Posay' doesn't match value_col 'sku' \\(int64\\)"),
            ("product", 1001, "target_item 1001 doesn't match value_col 'product' \\(string\\)"),
        ],
        ids=["name_for_integer_sku", "integer_for_product_name"],
    )
    def test_target_item_of_wrong_type_for_value_col_raises(self, transactions_df, value_col, target_item, match):
        """A product name looked up in an integer SKU column (or the reverse) is reported, not sent to the backend."""
        df = transactions_df.assign(sku=transactions_df["product"].factorize()[0] + 1001)
        with pytest.raises(ValueError, match=match):
            ProductAssociation(df=df, value_col=value_col, group_col=cols.transaction_id, target_item=target_item)

    @pytest.mark.parametrize(
        "target_item",
        [["milk", {"invalid": "dict"}], 1001.0, [np.float32(1001)], True],
    )
    def test_calc_association_target_item_invalid_type(self, transactions_df, target_item):
        """Non-str, non-integer target items raise; floats are rejected because float IDs compare inexactly."""
        with pytest.raises(TypeError, match="target_item must contain only str or integer values"):
            ProductAssociation(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                target_item=target_item,
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
