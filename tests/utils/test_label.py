"""Tests for the label module."""

import ibis
import pandas as pd
import pytest

from openretailscience.options import ColumnHelper
from openretailscience.utils.label import label_by_condition


class TestLabelByCondition:
    """Test class for label_by_condition function."""

    def test_binary_strategy_basic(self):
        """Test binary labeling strategy with basic data."""
        df = pd.DataFrame(
            {
                "customer_id": [1, 1, 1, 2, 2, 3, 3],
                "product_category": ["toys", "books", "toys", "books", "clothes", "clothes", "clothes"],
            },
        )
        table = ibis.memtable(df)

        result = label_by_condition(
            table=table,
            condition=table["product_category"] == "toys",
            labeling_strategy="binary",
        )

        expected_df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "label_name": ["contains", "not_contains", "not_contains"],
            },
        )

        result_df = result.execute().sort_values("customer_id").reset_index(drop=True)
        expected_df = expected_df.sort_values("customer_id")

        assert result_df.equals(expected_df)

    def test_extended_strategy_all_scenarios(self):
        """Test extended labeling strategy with all possible scenarios."""
        df = pd.DataFrame(
            {
                "transaction_id": [1, 1, 1, 2, 2, 3, 3, 4],
                "on_promotion": [True, True, True, True, False, False, False, True],
            },
        )
        table = ibis.memtable(df)

        result = label_by_condition(
            table=table,
            label_col="transaction_id",
            condition=table["on_promotion"],
            labeling_strategy="extended",
        )

        expected_df = pd.DataFrame(
            {
                "transaction_id": [1, 2, 3, 4],
                "label_name": ["contains", "mixed", "not_contains", "contains"],
            },
        )

        result_df = result.execute().sort_values("transaction_id").reset_index(drop=True)
        expected_df = expected_df.sort_values("transaction_id").reset_index(drop=True)

        assert result_df.equals(expected_df)

    @pytest.mark.parametrize(
        ("strategy", "expected_labels"),
        [
            ("binary", ["contains", "contains", "not_contains"]),
            ("extended", ["contains", "mixed", "not_contains"]),
        ],
    )
    def test_labeling_strategies_comparison(self, strategy, expected_labels):
        """Test both labeling strategies with the same data."""
        df = pd.DataFrame(
            {
                "group_id": [1, 1, 2, 2, 3, 3],
                "has_condition": [True, True, True, False, False, False],
            },
        )
        table = ibis.memtable(df)

        result = label_by_condition(
            table=table,
            label_col="group_id",
            condition=table["has_condition"],
            labeling_strategy=strategy,
        )

        expected_df = pd.DataFrame(
            {
                "group_id": [1, 2, 3],
                "label_name": expected_labels,
            },
        )

        result_df = result.execute().sort_values("group_id").reset_index(drop=True)
        expected_df = expected_df.sort_values("group_id").reset_index(drop=True)

        assert result_df.equals(expected_df)

    def test_unknown_labeling_strategy_raises(self):
        """A misspelt strategy raises instead of silently running the extended strategy."""
        table = ibis.memtable(
            pd.DataFrame({ColumnHelper().customer_id: [1, 1, 2], "on_promotion": [True, False, False]})
        )

        with pytest.raises(ValueError, match="labeling_strategy must be one of"):
            label_by_condition(table=table, condition=table["on_promotion"], labeling_strategy="binray")

    def test_custom_labels_and_column_names(self):
        """Test custom label names and return column name."""
        df = pd.DataFrame(
            {
                "store_id": [1, 1, 2, 2],
                "high_value": [True, False, False, False],
            },
        )
        table = ibis.memtable(df)

        result = label_by_condition(
            table=table,
            label_col="store_id",
            condition=table["high_value"],
            return_col="value_segment",
            labeling_strategy="extended",
            contains_label="all_high_value",
            not_contains_label="no_high_value",
            mixed_label="some_high_value",
        )

        expected_df = pd.DataFrame(
            {
                "store_id": [1, 2],
                "value_segment": ["some_high_value", "no_high_value"],
            },
        )

        result_df = result.execute().sort_values("store_id").reset_index(drop=True)
        expected_df = expected_df.sort_values("store_id").reset_index(drop=True)

        assert result_df.equals(expected_df)

    @pytest.mark.parametrize(
        ("condition_values", "expected_label"),
        [
            ([True, True, True], "contains"),
            ([False, False, False], "not_contains"),
        ],
    )
    def test_single_group_uniform_conditions(self, condition_values, expected_label):
        """Test single group where all items have the same condition value."""
        df = pd.DataFrame(
            {
                "customer_id": [1, 1, 1],
                "premium_product": condition_values,
            },
        )
        table = ibis.memtable(df)

        result = label_by_condition(
            table=table,
            label_col="customer_id",
            condition=table["premium_product"],
            labeling_strategy="extended",
        )

        expected_df = pd.DataFrame(
            {
                "customer_id": [1],
                "label_name": [expected_label],
            },
        )

        result_df = result.execute().sort_values("customer_id").reset_index(drop=True)
        expected_df = expected_df.sort_values("customer_id").reset_index(drop=True)

        assert result_df.equals(expected_df)
