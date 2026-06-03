"""Tests for the Cross Shop module."""

import re

import pandas as pd
import pytest

from openretailscience.analysis.cross_shop import CrossShop
from openretailscience.options import ColumnHelper, option_context

# Test constants
TWO_GROUPS = 2
THREE_GROUPS = 3
FLOATING_POINT_TOLERANCE = 0.001
PERCENT_ONE_DECIMAL = re.compile(r"^\d+\.\d%$")

cols = ColumnHelper()


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return pd.DataFrame(
        {
            cols.customer_id: [1, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10],
            "category_1_name": [
                "Jeans",
                "Shoes",
                "Dresses",
                "Hats",
                "Shoes",
                "Jeans",
                "Jeans",
                "Shoes",
                "Dresses",
                "Jeans",
                "Shoes",
                "Jeans",
            ],
            cols.unit_spend: [10, 20, 30, 40, 20, 50, 10, 20, 30, 15, 40, 50],
        },
    )


def test_calc_cross_shop_two_groups(sample_data):
    """Test the _calc_cross_shop method with two groups."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
    )
    ret_df = pd.DataFrame(
        {
            cols.customer_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "group_1": pd.Series([1, 0, 0, 0, 1, 1, 0, 1, 0, 1], dtype="int32"),
            "group_2": pd.Series([0, 1, 0, 0, 1, 0, 1, 0, 1, 0], dtype="int32"),
            "groups": [(1, 0), (0, 1), (0, 0), (0, 0), (1, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0)],
            "group_labels": ["A", "B", "No Groups", "No Groups", "A, B", "A", "B", "A", "B", "A"],
            cols.unit_spend: [10, 20, 30, 40, 70, 10, 20, 45, 40, 50],
        },
    ).set_index(cols.customer_id)

    assert cross_shop_df.equals(ret_df)


def test_calc_cross_shop_three_groups(sample_data):
    """Test the _calc_cross_shop method with three groups."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        group_3_col="category_1_name",
        group_3_val="Dresses",
    )
    ret_df = pd.DataFrame(
        {
            cols.customer_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "group_1": pd.Series([1, 0, 0, 0, 1, 1, 0, 1, 0, 1], dtype="int32"),
            "group_2": pd.Series([0, 1, 0, 0, 1, 0, 1, 0, 1, 0], dtype="int32"),
            "group_3": pd.Series([0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype="int32"),
            "groups": [
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (0, 0, 0),
                (1, 1, 0),
                (1, 0, 0),
                (0, 1, 0),
                (1, 0, 1),
                (0, 1, 0),
                (1, 0, 0),
            ],
            "group_labels": ["A", "B", "C", "No Groups", "A, B", "A", "B", "A, C", "B", "A"],
            cols.unit_spend: [10, 20, 30, 40, 70, 10, 20, 45, 40, 50],
        },
    ).set_index(cols.customer_id)

    pd.testing.assert_frame_equal(cross_shop_df, ret_df)


def test_calc_cross_shop_three_groups_customer_id_nunique(sample_data):
    """Test the _calc_cross_shop method with three groups and customer_id as the value column."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        group_3_col="category_1_name",
        group_3_val="Dresses",
        value_col=cols.customer_id,
        agg_func="nunique",
    )
    ret_df = pd.DataFrame(
        {
            "group_1": [1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
            "group_2": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            "group_3": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            "groups": [
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (0, 0, 0),
                (1, 1, 0),
                (1, 0, 0),
                (0, 1, 0),
                (1, 0, 1),
                (0, 1, 0),
                (1, 0, 0),
            ],
            "group_labels": ["A", "B", "C", "No Groups", "A, B", "A", "B", "A, C", "B", "A"],
            cols.customer_id: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    ret_df.index.name = cols.customer_id
    ret_df = ret_df.astype({"group_1": "int32", "group_2": "int32", "group_3": "int32"})
    assert cross_shop_df.equals(ret_df)


def test_calc_cross_shop_table(sample_data):
    """Test the _calc_cross_shop_table method."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        group_3_col="category_1_name",
        group_3_val="Dresses",
        value_col=cols.unit_spend,
    )
    cross_shop_table = CrossShop._calc_cross_shop_table(
        cross_shop_df,
        value_col=cols.unit_spend,
    )
    ret_df = pd.DataFrame(
        {
            "groups": [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
            ],
            "group_labels": ["No Groups", "C", "B", "A", "A, C", "A, B"],
            cols.unit_spend: [40, 30, 80, 70, 45, 70],
            "percent": [0.119402985, 0.089552239, 0.23880597, 0.208955224, 0.134328358, 0.208955224],
        },
    )

    # Equals should be using allclose for float columns but it needs
    ret_df["percent"] = ret_df["percent"].round(6)
    cross_shop_table["percent"] = cross_shop_table["percent"].round(6)

    assert cross_shop_table.equals(ret_df)

    # Test with labels
    cross_shop_df_with_labels = CrossShop._calc_cross_shop(
        sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        group_3_col="category_1_name",
        group_3_val="Dresses",
        value_col=cols.unit_spend,
        labels=["Denim", "Footwear", "Clothing"],
    )
    cross_shop_table_with_labels = CrossShop._calc_cross_shop_table(
        cross_shop_df_with_labels,
        value_col=cols.unit_spend,
    )

    assert "group_labels" in cross_shop_table_with_labels.columns
    assert "groups" in cross_shop_table_with_labels.columns

    expected_group_labels = ["Clothing", "Denim", "Denim, Clothing", "Denim, Footwear", "Footwear", "No Groups"]
    assert set(cross_shop_table_with_labels["group_labels"]) == set(expected_group_labels)
    assert cross_shop_table_with_labels["percent"].sum() == pytest.approx(1.0, rel=1e-6)


def test_calc_cross_shop_table_customer_id_nunique(sample_data):
    """Test the _calc_cross_shop_table method with customer_id as the value column."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        group_3_col="category_1_name",
        group_3_val="Dresses",
        value_col=cols.customer_id,
        agg_func="nunique",
    )
    cross_shop_table = CrossShop._calc_cross_shop_table(
        cross_shop_df,
        value_col=cols.customer_id,
    )
    ret_df = pd.DataFrame(
        {
            "groups": [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)],
            "group_labels": ["No Groups", "C", "B", "A", "A, C", "A, B"],
            cols.customer_id: [1, 1, 3, 3, 1, 1],
            "percent": [0.1, 0.1, 0.3, 0.3, 0.1, 0.1],
        },
    )

    assert cross_shop_table.equals(ret_df)


def test_calc_cross_shop_invalid_group_3(sample_data):
    """Test that _calc_cross_shop raises ValueError if only one of group_3_col or group_3_val is provided."""
    with pytest.raises(ValueError, match="If group_3_col or group_3_val is populated, then the other must be as well"):
        CrossShop._calc_cross_shop(
            sample_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
            group_3_col="category_1_name",
        )

    with pytest.raises(ValueError, match="If group_3_col or group_3_val is populated, then the other must be as well"):
        CrossShop._calc_cross_shop(
            sample_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
            group_3_val="T-Shirts",
        )


def test_missing_columns_raises(sample_data):
    """Test that ValueError is raised when required columns are missing in df."""
    bad_data = sample_data.drop(columns=[cols.unit_spend])
    with pytest.raises(ValueError, match="Input data is missing required columns"):
        CrossShop(
            df=bad_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
        )


def test_labels_length_mismatch(sample_data):
    """Test that ValueError is raised when labels length does not match group count."""
    with pytest.raises(ValueError, match="The number of labels must be equal to the number of group indexes given"):
        CrossShop(
            df=sample_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
            group_3_col="category_1_name",
            group_3_val="Dresses",
            labels=["A", "B"],
        )


def test_group_3_only_one_side_provided(sample_data):
    """Test that ValueError is raised when only group_3_col or group_3_val is given."""
    with pytest.raises(ValueError, match="If group_3_col or group_3_val is populated"):
        CrossShop._calc_cross_shop(
            sample_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
            group_3_col="category_1_name",
        )


def test_plot_default_labels_two_groups_render_as_set_labels(sample_data):
    """Without explicit labels, two groups render as set labels "A" and "B"."""
    cross_shop = CrossShop(
        df=sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
    )
    ax = cross_shop.plot()
    text_values = {t.get_text() for t in ax.texts}
    assert {"A", "B"}.issubset(text_values)


def test_plot_default_labels_three_groups_render_as_set_labels(sample_data):
    """Without explicit labels, three groups render as set labels "A", "B", "C"."""
    cross_shop = CrossShop(
        df=sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        group_3_col="category_1_name",
        group_3_val="Dresses",
    )
    ax = cross_shop.plot()
    text_values = {t.get_text() for t in ax.texts}
    assert {"A", "B", "C"}.issubset(text_values)


def test_plot_custom_labels_render_as_set_labels(sample_data):
    """Custom labels passed at construction render verbatim on the diagram."""
    cross_shop = CrossShop(
        df=sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        labels=["Denim", "Footwear"],
    )
    ax = cross_shop.plot()
    text_values = {t.get_text() for t in ax.texts}
    assert {"Denim", "Footwear"}.issubset(text_values)


def test_plot_renders_custom_title(sample_data):
    """A title passed to plot() renders as a figure text element."""
    title = "Cross-shop overlap"
    cross_shop = CrossShop(
        df=sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        labels=["Jeans", "Shoes"],
    )
    ax = cross_shop.plot(title=title)
    title_texts = [t for t in ax.figure.texts if t.get_text() == title]
    assert len(title_texts) == 1


def test_cross_shop_with_non_empty_dataframe():
    """Test CrossShop initialization with a minimal non-empty DataFrame."""
    group_count = 2
    minimal_df = pd.DataFrame(
        {
            cols.customer_id: [1, 2],
            cols.unit_spend: [10, 20],
            "category": ["Jeans", "Shoes"],
        },
    )

    cross_shop = CrossShop(
        df=minimal_df,
        group_1_col="category",
        group_1_val="Jeans",
        group_2_col="category",
        group_2_val="Shoes",
    )

    assert cross_shop.group_count == group_count
    assert not cross_shop.cross_shop_df.empty
    assert not cross_shop.cross_shop_table_df.empty
    assert "group_labels" in cross_shop.cross_shop_df.columns
    assert "group_labels" in cross_shop.cross_shop_table_df.columns

    # Should have default alphabetical labels when none provided
    expected_group_labels = ["A", "B"]
    assert set(cross_shop.cross_shop_table_df["group_labels"]) == set(expected_group_labels)


def test_cross_shop_with_custom_value_col_and_agg_func(sample_data):
    """Test CrossShop initialization with custom value_col and agg_func."""
    customer_value = 70.0
    sample_data["custom_value"] = sample_data[cols.unit_spend] * 2

    cross_shop = CrossShop(
        df=sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        value_col="custom_value",
        agg_func="mean",
    )

    assert "custom_value" in cross_shop.cross_shop_df.columns

    assert cross_shop.cross_shop_df.loc[5, "custom_value"] == customer_value


def test_plot_default_formatter_renders_subset_labels_as_percent_with_one_decimal(sample_data):
    """Without a custom subset_label_formatter, subset percentages render as ``N.N%``."""
    cross_shop = CrossShop(
        df=sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        labels=["Denim", "Footwear"],
    )
    ax = cross_shop.plot()
    set_labels = {"Denim", "Footwear"}
    subset_labels = [t.get_text() for t in ax.texts if t.get_text() not in set_labels]
    expected_subset_count = (cross_shop.cross_shop_table_df["groups"] != (0, 0)).sum()
    assert len(subset_labels) == expected_subset_count
    assert all(PERCENT_ONE_DECIMAL.match(text) for text in subset_labels)


def test_plot_custom_subset_label_formatter_replaces_default(sample_data):
    """A custom subset_label_formatter overrides the default percentage format."""
    cross_shop = CrossShop(
        df=sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        labels=["Denim", "Footwear"],
    )
    ax = cross_shop.plot(subset_label_formatter=lambda x: f"CUSTOM-{x:.2f}")
    set_labels = {"Denim", "Footwear"}
    subset_labels = [t.get_text() for t in ax.texts if t.get_text() not in set_labels]
    expected_subset_count = (cross_shop.cross_shop_table_df["groups"] != (0, 0)).sum()
    assert len(subset_labels) == expected_subset_count
    assert all(text.startswith("CUSTOM-") for text in subset_labels)


def test_with_custom_column_names(sample_data):
    """Test CrossShop with custom column names to ensure column overrides work correctly."""
    custom_data = sample_data.rename(
        columns={
            "customer_id": "my_customer_identifier",
            "unit_spend": "total_amount_spent",
            "category_1_name": "product_category",
        },
    )

    with option_context("column.customer_id", "my_customer_identifier", "column.unit_spend", "total_amount_spent"):
        cross_shop = CrossShop(
            df=custom_data,
            group_1_col="product_category",
            group_1_val="Jeans",
            group_2_col="product_category",
            group_2_val="Shoes",
        )
        cross_shop_df = cross_shop.cross_shop_df
        assert isinstance(cross_shop_df, pd.DataFrame)
        assert not cross_shop_df.empty
        assert "my_customer_identifier" in cross_shop_df.index.name or cross_shop_df.index.name is None, (
            "Should handle custom customer_id column name"
        )
        assert "total_amount_spent" in cross_shop_df.columns, "Should handle custom unit_spend column name"


def test_generate_default_labels():
    """Test the _generate_default_labels static method."""
    assert CrossShop._generate_default_labels(2) == ["A", "B"]
    assert CrossShop._generate_default_labels(3) == ["A", "B", "C"]


@pytest.mark.parametrize(
    ("test_case", "kwargs", "expected_group_count"),
    [
        (
            "group_2_col_defaults",
            {
                "group_1_col": "category_1_name",
                "group_1_val": "Jeans",
                "group_2_col": None,  # Should default to group_1_col
                "group_2_val": "Shoes",
            },
            TWO_GROUPS,
        ),
        (
            "group_3_col_defaults",
            {
                "group_1_col": "category_1_name",
                "group_1_val": "Jeans",
                "group_2_col": "category_1_name",
                "group_2_val": "Shoes",
                "group_3_col": None,  # Should default to group_1_col
                "group_3_val": "Dresses",
            },
            THREE_GROUPS,
        ),
    ],
)
def test_group_col_defaults_to_group_1_col(sample_data, test_case, kwargs, expected_group_count):
    """Test that group_2_col and group_3_col default to group_1_col when None is provided."""
    # This test should fail initially until the feature is implemented
    cross_shop = CrossShop(df=sample_data, **kwargs)

    # Verify the analysis worked correctly
    assert cross_shop.group_count == expected_group_count
    assert len(cross_shop.labels) == expected_group_count
    assert isinstance(cross_shop.cross_shop_df, pd.DataFrame)
    assert not cross_shop.cross_shop_df.empty


@pytest.mark.parametrize(
    ("test_name", "custom_col", "use_option_context", "pass_group_col"),
    [
        ("explicit_group_col", "user_id", False, True),
        ("option_context_default", "account_id", True, False),
    ],
)
def test_custom_group_col_handling(sample_data, test_name, custom_col, use_option_context, pass_group_col):
    """Test custom customer column handling through explicit parameter or option context."""
    # Create test data with custom customer identifier
    custom_data = sample_data.rename(columns={cols.customer_id: custom_col})

    # Build kwargs for CrossShop
    kwargs = {
        "df": custom_data,
        "group_1_col": "category_1_name",
        "group_1_val": "Jeans",
        "group_2_col": "category_1_name",
        "group_2_val": "Shoes",
    }

    if pass_group_col:
        kwargs["group_col"] = custom_col  # Explicit parameter

    # This test should fail initially until the feature is implemented
    if use_option_context:
        with option_context("column.customer_id", custom_col):
            cross_shop = CrossShop(**kwargs)
    else:
        cross_shop = CrossShop(**kwargs)

    # Verify the analysis worked correctly
    assert cross_shop.group_count == TWO_GROUPS
    assert len(cross_shop.labels) == TWO_GROUPS
    assert isinstance(cross_shop.cross_shop_df, pd.DataFrame)
    assert not cross_shop.cross_shop_df.empty
    # Verify it used the custom customer column
    assert cross_shop.cross_shop_df.index.name == custom_col


def test_backward_compatibility_explicit_params(sample_data):
    """Test that existing code with explicit parameters continues to work exactly as before."""
    # This test should pass both before and after implementation
    # It ensures that existing explicit parameter usage is not broken
    cross_shop = CrossShop(
        df=sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",  # Explicitly specified
        group_2_val="Shoes",
        group_3_col="category_1_name",  # Explicitly specified
        group_3_val="Dresses",
        labels=["Jeans", "Shoes", "Dresses"],
        value_col=cols.unit_spend,
        agg_func="sum",
    )

    # Verify the analysis worked correctly
    assert cross_shop.group_count == THREE_GROUPS
    assert cross_shop.labels == ["Jeans", "Shoes", "Dresses"]
    assert isinstance(cross_shop.cross_shop_df, pd.DataFrame)
    assert not cross_shop.cross_shop_df.empty
    assert isinstance(cross_shop.cross_shop_table_df, pd.DataFrame)
    assert not cross_shop.cross_shop_table_df.empty


@pytest.mark.parametrize(
    ("test_scenario", "groups", "expected_count", "expected_segments"),
    [
        (
            "two_group_simplified",
            {
                "group_1_col": "category_1_name",
                "group_1_val": "Jeans",
                # group_2_col omitted - should use group_1_col
                "group_2_val": "Shoes",
            },
            TWO_GROUPS,
            ["(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"],  # Possible segment combinations
        ),
        (
            "three_group_simplified",
            {
                "group_1_col": "category_1_name",
                "group_1_val": "Jeans",
                # group_2_col omitted - should use group_1_col
                "group_2_val": "Shoes",
                # group_3_col omitted - should use group_1_col
                "group_3_val": "Dresses",
            },
            THREE_GROUPS,
            [
                "(0, 0, 0)",
                "(0, 0, 1)",
                "(0, 1, 0)",
                "(0, 1, 1)",
                "(1, 0, 0)",
                "(1, 0, 1)",
                "(1, 1, 0)",
                "(1, 1, 1)",
            ],  # All possible combinations
        ),
    ],
)
def test_simplified_interface_integration(sample_data, test_scenario, groups, expected_count, expected_segments):
    """Integration test for simplified interface with real data scenarios."""
    # This test should fail initially until the feature is implemented
    cross_shop = CrossShop(df=sample_data, **groups)

    # Verify the analysis worked correctly
    assert cross_shop.group_count == expected_count
    assert len(cross_shop.labels) == expected_count

    # Check that the DataFrame has the expected structure
    assert isinstance(cross_shop.cross_shop_df, pd.DataFrame)
    assert not cross_shop.cross_shop_df.empty
    assert "groups" in cross_shop.cross_shop_df.columns
    assert "group_labels" in cross_shop.cross_shop_df.columns

    # Verify that observed segments are valid combinations from the documented universe.
    # expected_segments lists every possible bitstring combination; the fixture data only
    # exercises a subset, so equality would be incorrect.
    unique_groups = cross_shop.cross_shop_df["groups"].unique()
    assert {str(group) for group in unique_groups} <= set(expected_segments)

    # Check the aggregated table
    assert isinstance(cross_shop.cross_shop_table_df, pd.DataFrame)
    assert not cross_shop.cross_shop_table_df.empty
    assert "percent" in cross_shop.cross_shop_table_df.columns

    # Verify percentages sum to 100%
    total_percent = cross_shop.cross_shop_table_df["percent"].sum()
    assert abs(total_percent - 1.0) < FLOATING_POINT_TOLERANCE  # Allow for small floating point errors


def test_custom_customer_columns_integration(sample_data):
    """Integration test for custom customer columns with various scenarios."""
    # Test scenario 1: Explicit group_col with simplified interface
    custom_data = sample_data.rename(columns={cols.customer_id: "shopper_id"})

    # This test should fail initially until the feature is implemented
    cross_shop = CrossShop(
        df=custom_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        # group_2_col omitted - should use group_1_col
        group_2_val="Shoes",
        group_col="shopper_id",  # Explicit customer column
    )

    # Verify it worked with custom column
    assert cross_shop.cross_shop_df.index.name == "shopper_id"
    assert cross_shop.group_count == TWO_GROUPS

    # Test scenario 2: Option context with simplified interface
    custom_data2 = sample_data.rename(columns={cols.customer_id: "member_id"})

    with option_context("column.customer_id", "member_id"):
        cross_shop2 = CrossShop(
            df=custom_data2,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            # Both group_2_col and group_col omitted
            group_2_val="Shoes",
            # group_3_col omitted
            group_3_val="Dresses",
        )

        # Verify it used option context for customer column
        assert cross_shop2.cross_shop_df.index.name == "member_id"
        assert cross_shop2.group_count == THREE_GROUPS

    # Test scenario 3: Explicit group_col overrides option context
    custom_data3 = sample_data.copy()
    custom_data3["buyer_id"] = custom_data3[cols.customer_id]

    with option_context("column.customer_id", "wrong_column"):
        cross_shop3 = CrossShop(
            df=custom_data3,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_val="Shoes",  # group_2_col defaults to group_1_col
            group_col="buyer_id",  # Should override option context
        )

        # Verify explicit parameter overrides option context
        assert cross_shop3.cross_shop_df.index.name == "buyer_id"
        assert cross_shop3.group_count == TWO_GROUPS
