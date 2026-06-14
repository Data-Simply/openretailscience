"""Tests for the filter_and_label module."""

from typing import cast

import ibis
import pandas as pd

from openretailscience.utils.filter_and_label import filter_and_label_by_condition


def test_filter_and_label_by_condition():
    """Test cases for filtering and labeling by condition with default label column."""
    df = pd.DataFrame(
        {
            "product_id": [1, 2, 3, 4],
            "category": ["toys", "shoes", "toys", "books"],
        },
    )
    table = ibis.memtable(df)

    result = filter_and_label_by_condition(
        table,
        {
            "toys": table["category"] == "toys",
            "shoes": table["category"] == "shoes",
        },
    )

    expected_df = pd.DataFrame(
        {
            "product_id": [1, 2, 3],
            "category": ["toys", "shoes", "toys"],
            "label": ["toys", "shoes", "toys"],
        },
    )

    pd.testing.assert_frame_equal(
        cast("pd.DataFrame", result.execute()).sort_values(by="product_id").reset_index(drop=True),
        expected_df.sort_values("product_id").reset_index(drop=True),
    )


def test_filter_and_label_by_condition_custom_column():
    """Test filtering and labeling with a custom label column name."""
    df = pd.DataFrame(
        {
            "product_id": [1, 2, 3, 4],
            "category": ["toys", "shoes", "toys", "books"],
        },
    )
    table = ibis.memtable(df)

    result = filter_and_label_by_condition(
        table,
        {
            "toys": table["category"] == "toys",
            "shoes": table["category"] == "shoes",
        },
        label_col="product_type",
    )

    expected_df = pd.DataFrame(
        {
            "product_id": [1, 2, 3],
            "category": ["toys", "shoes", "toys"],
            "product_type": ["toys", "shoes", "toys"],
        },
    )

    pd.testing.assert_frame_equal(
        cast("pd.DataFrame", result.execute()).sort_values(by="product_id").reset_index(drop=True),
        expected_df.sort_values("product_id").reset_index(drop=True),
    )
