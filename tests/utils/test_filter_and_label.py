"""Tests for the filter_and_label module."""

import ibis
import pandas as pd
import pytest

from openretailscience.utils.filter_and_label import filter_and_label_by_condition


@pytest.mark.parametrize(
    ("kwargs", "label_col"),
    [({}, "label"), ({"label_col": "product_type"}, "product_type")],
    ids=["default_label_col", "custom_label_col"],
)
def test_filter_and_label_by_condition(kwargs, label_col):
    """Rows matching a condition are kept and tagged with its label; the rest are dropped."""
    table = ibis.memtable(
        pd.DataFrame({"product_id": [1, 2, 3, 4], "category": ["toys", "shoes", "toys", "books"]}),
    )

    result = filter_and_label_by_condition(
        table,
        {"toys": table["category"] == "toys", "shoes": table["category"] == "shoes"},
        **kwargs,
    )

    pd.testing.assert_frame_equal(
        result.order_by("product_id").to_pandas(),
        pd.DataFrame(
            {"product_id": [1, 2, 3], "category": ["toys", "shoes", "toys"], label_col: ["toys", "shoes", "toys"]},
        ),
    )
