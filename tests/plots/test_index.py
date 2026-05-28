"""Tests for the index plot module."""

import ibis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from openretailscience.plots.index import (
    BASELINE_INDEX,
    filter_by_groups,
    filter_by_value_thresholds,
    filter_top_bottom_n,
    get_indexes,
    plot,
)
from openretailscience.plots.styles.colors import get_named_color

OFFSET_THRESHOLD = 5


def test_get_indexes_basic():
    """Test get_indexes function with basic input to ensure it returns a valid DataFrame."""
    df = pd.DataFrame(
        {
            "category": ["Bakery", "Bakery", "Dairy", "Dairy", "Produce", "Produce"],
            "value": [10, 20, 30, 40, 50, 60],
        },
    )

    result = get_indexes(df, value_to_index="Bakery", index_col="category", value_col="value", group_col="category")
    assert isinstance(result, pd.DataFrame)
    assert "category" in result.columns
    assert "index" in result.columns
    assert not result.empty


def test_get_indexes_with_subgroup():
    """Test get_indexes function when a subgroup column is provided."""
    df = pd.DataFrame(
        {
            "subgroup": ["Loyalty", "Loyalty", "Loyalty", "Regular", "Regular", "Regular"],
            "category": ["Bakery", "Bakery", "Dairy", "Dairy", "Produce", "Produce"],
            "value": [10, 20, 30, 40, 50, 60],
        },
    )

    result = get_indexes(
        df,
        value_to_index="Bakery",
        index_col="category",
        value_col="value",
        group_col="category",
        index_subgroup_col="subgroup",
    )
    assert isinstance(result, pd.DataFrame)
    assert "category" in result.columns
    assert "index" in result.columns
    assert not result.empty


def test_get_indexes_invalid_agg_func():
    """Test get_indexes function with an invalid aggregation function."""
    df = pd.DataFrame(
        {
            "category": ["Bakery", "Dairy", "Produce"],
            "value": [10, 20, 30],
        },
    )

    with pytest.raises(ValueError, match="agg_func must be one of"):
        get_indexes(
            df,
            value_to_index="Bakery",
            index_col="category",
            value_col="value",
            group_col="category",
            agg_func="invalid_func",
        )


@pytest.mark.parametrize(
    ("agg", "expected_dairy_index"),
    [
        ("sum", 500.0),
        ("mean", 700.0),
        ("max", 600.0),
        ("min", 900.0),
        ("nunique", 233.33333333333334),
    ],
)
def test_get_indexes_with_different_aggregations(agg, expected_dairy_index):
    """Test get_indexes computes aggregation-specific index values for the baseline category."""
    # Uneven row counts per department ensure each agg_func produces a distinct index
    # (sum and mean would otherwise coincide on equal-count groups).
    df = pd.DataFrame(
        {
            "department": ["Dairy", "Dairy", "Dairy", "Bakery", "Bakery", "Meat", "Meat"],
            "spend": [100, 200, 150, 300, 400, 500, 600],
        },
    )

    result = get_indexes(
        df,
        value_to_index="Dairy",
        index_col="department",
        value_col="spend",
        group_col="department",
        agg_func=agg,
    )

    expected = pd.DataFrame({"department": ["Dairy"], "index": [expected_dairy_index]})
    pd.testing.assert_frame_equal(result, expected)


def test_get_indexes_with_offset():
    """Test get_indexes function with an offset value."""
    df = pd.DataFrame(
        {
            "category": ["Bakery", "Dairy", "Produce"],
            "value": [10, 20, 30],
        },
    )
    result = get_indexes(
        df,
        value_to_index="Bakery",
        index_col="category",
        value_col="value",
        group_col="category",
        offset=OFFSET_THRESHOLD,
    )

    assert isinstance(result, pd.DataFrame)
    assert "category" in result.columns
    assert "index" in result.columns
    assert not result.empty
    assert all(result["index"] >= -OFFSET_THRESHOLD)


def test_get_indexes_single_column():
    """Test that the function works with a single column index."""
    df = pd.DataFrame(
        {
            "group_col": ["Bakery", "Bakery", "Dairy", "Dairy", "Produce", "Produce"],
            "filter_col": ["Loyalty", "Regular", "Loyalty", "Regular", "Loyalty", "Regular"],
            "value_col": [1, 2, 3, 4, 5, 6],
        },
    )
    expected_output = pd.DataFrame(
        {"group_col": ["Bakery", "Dairy", "Produce"], "index": [77.77777778, 100, 106.0606]},
    )
    output = get_indexes(
        df=df,
        value_to_index="Loyalty",
        index_col="filter_col",
        value_col="value_col",
        group_col="group_col",
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_two_columns():
    """Test that the function works with two columns as the index."""
    df = pd.DataFrame(
        {
            "group_col1": ["Bakery", "Bakery", "Dairy", "Dairy", "Produce", "Produce"] * 2,
            "group_col2": ["Mall"] * 6 + ["Outlet"] * 6,
            "filter_col": ["Loyalty", "Regular"] * 6,
            "value_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        },
    )
    expected_output = pd.DataFrame(
        {
            "group_col2": ["Mall", "Mall", "Mall", "Outlet", "Outlet", "Outlet"],
            "group_col1": ["Bakery", "Dairy", "Produce", "Bakery", "Dairy", "Produce"],
            "index": [77.77777778, 100, 106.0606, 98.51851852, 100, 100.9661836],
        },
    )

    output = get_indexes(
        df=df,
        value_to_index="Loyalty",
        index_col="filter_col",
        value_col="value_col",
        group_col="group_col1",
        index_subgroup_col="group_col2",
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_with_ibis_table_input():
    """Test that the get_indexes function works with an ibis Table."""
    df = pd.DataFrame(
        {
            "category": ["Bakery", "Dairy", "Produce"],
            "value": [10, 20, 30],
        },
    )
    table = ibis.memtable(df)

    result = get_indexes(
        table,
        value_to_index="Bakery",
        index_col="category",
        value_col="value",
        group_col="category",
    )
    assert isinstance(result, pd.DataFrame)
    assert "category" in result.columns
    assert "index" in result.columns
    assert not result.empty


class TestGetIndexesZeroDivision:
    """Tests for get_indexes division by zero edge cases.

    Verifies that when denominators are zero (overall total, subset total,
    or group proportion), the code returns NaN instead of raising errors.
    """

    @pytest.mark.parametrize(
        "df_data",
        [
            pytest.param(
                {
                    "region": ["North", "South"],
                    "category": ["Electronics", "Electronics"],
                    "sales": [0, 0],
                },
                id="zero_overall_total",
            ),
            pytest.param(
                {
                    "region": ["North", "South", "North", "South"],
                    "category": ["Electronics", "Electronics", "Grocery", "Grocery"],
                    "sales": [0, 0, 100, 200],
                },
                id="zero_subset_total",
            ),
        ],
    )
    def test_returns_all_nan_when_totals_are_zero(self, df_data):
        """Test that get_indexes produces all NaN index values when totals are zero."""
        df = pd.DataFrame(df_data)

        result = get_indexes(
            df,
            value_to_index="Electronics",
            index_col="category",
            value_col="sales",
            group_col="region",
        )
        assert result["index"].isna().all(), "Expected all NaN index values when totals are zero"

    def test_zero_overall_total_with_subgroup_returns_nan(self):
        """Test that get_indexes produces NaN index when a subgroup's overall total is zero."""
        df = pd.DataFrame(
            {
                "store": ["Mall", "Mall", "Mall", "Mall", "Outlet", "Outlet", "Outlet", "Outlet"],
                "region": ["North", "South", "North", "South", "North", "South", "North", "South"],
                "category": [
                    "Electronics",
                    "Electronics",
                    "Grocery",
                    "Grocery",
                    "Electronics",
                    "Electronics",
                    "Grocery",
                    "Grocery",
                ],
                "sales": [10, 20, 30, 40, 0, 0, 0, 0],
            },
        )

        result = get_indexes(
            df,
            value_to_index="Electronics",
            index_col="category",
            value_col="sales",
            group_col="region",
            index_subgroup_col="store",
        )
        mall_rows = result[result["store"] == "Mall"]
        assert not mall_rows["index"].isna().any(), "Expected valid index for subgroup with non-zero overall total"

        outlet_rows = result[result["store"] == "Outlet"]
        assert outlet_rows["index"].isna().all(), "Expected NaN index for subgroup with zero overall total"

    def test_zero_group_proportion_returns_nan(self):
        """Test that get_indexes produces NaN for a group with zero overall proportion."""
        df = pd.DataFrame(
            {
                "region": ["North", "South", "East"],
                "category": ["Electronics", "Electronics", "Electronics"],
                "sales": [100, 200, 0],
            },
        )

        result = get_indexes(
            df,
            value_to_index="Electronics",
            index_col="category",
            value_col="sales",
            group_col="region",
        )
        east_row = result[result["region"] == "East"]
        assert east_row["index"].isna().all(), "Expected NaN index for group with zero proportion_overall"

        non_zero_rows = result[result["region"] != "East"]
        assert not non_zero_rows["index"].isna().any(), "Expected valid index for groups with non-zero proportions"


class TestIndexPlot:
    """Tests for the index_plot function."""

    def teardown_method(self):
        """Clean up after each test method."""
        plt.close("all")

    @pytest.fixture
    def test_data(self):
        """Return a sample dataframe for plotting."""
        rng = np.random.default_rng(42)
        data = {
            "category": ["Bakery", "Dairy", "Produce", "Snacks", "Beverages"] * 2,
            "sales": rng.integers(100, 500, size=10),
            "region": ["North", "South", "East", "West", "Central"] * 2,
        }
        return pd.DataFrame(data)

    def test_generates_index_plot_with_default_parameters(
        self,
        test_data,
    ):
        """Test that the function generates an index plot with default parameters."""
        df = test_data
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="Bakery",
        )

        assert isinstance(result_ax, plt.Axes)
        # barh adds a single BarContainer holding one Rectangle per y-axis row.
        assert len(result_ax.containers) == 1
        assert len(result_ax.containers[0]) == len(result_ax.get_yticklabels())
        assert result_ax.get_xlabel() == ""
        assert result_ax.get_ylabel() == ""

    def test_generates_index_plot_with_custom_title(self, test_data):
        """Test that the function generates an index plot with a custom title."""
        df = test_data
        custom_title = "Sales Performance by Category"
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="Bakery",
            title=custom_title,
        )

        assert isinstance(result_ax, plt.Axes)
        # Title is rendered as figure-level text by the chrome layout, not via ax.set_title.
        title_texts = [t for t in result_ax.figure.texts if t.get_text() == custom_title]
        assert len(title_texts) == 1

    def test_generates_index_plot_with_highlight_range(self, test_data):
        """Test that the function generates an index plot with a highlighted range."""
        df = test_data
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="Bakery",
            highlight_range=(80, 120),
        )

        assert isinstance(result_ax, plt.Axes)
        assert result_ax.get_xlim()[0] < BASELINE_INDEX < result_ax.get_xlim()[1]

    def test_generates_index_plot_with_group_filter(self, test_data):
        """Test that the function generates an index plot with a group filter applied."""
        df = test_data
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="region",
            value_to_index="North",
            include_only_groups=["Bakery", "Dairy"],
        )

        assert isinstance(result_ax, plt.Axes)

        # Verify that only the filtered groups appear in the plot
        y_labels = [label.get_text() for label in result_ax.get_yticklabels()]
        plotted_groups = set(y_labels)
        expected_groups = {"Bakery"}

        assert plotted_groups == expected_groups, (
            f"Found groups {plotted_groups - expected_groups} that should have been filtered out. "
            f"Expected only groups from {expected_groups}."
        )

    @pytest.mark.parametrize("kwarg", ["sort_by", "sort_order"])
    def test_raises_value_error_for_invalid_sort_kwarg(self, test_data, kwarg):
        """Test that the function raises a ValueError for an invalid sort_by or sort_order parameter."""
        df = test_data

        with pytest.raises(ValueError):
            plot(
                df,
                value_col="sales",
                group_col="category",
                index_col="category",
                value_to_index="Bakery",
                **{kwarg: "invalid"},
            )

    def test_generates_index_plot_with_source_text(self, test_data):
        """Test that the function generates an index plot with source text."""
        df = test_data
        source_text = "Data source: Company XYZ"
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="Bakery",
            source_text=source_text,
        )

        assert isinstance(result_ax, plt.Axes)
        source_texts = [text for text in result_ax.figure.texts if text.get_text() == source_text]
        assert len(source_texts) == 1

    def test_generates_index_plot_with_custom_labels(self, test_data):
        """Test that the function generates an index plot with custom x and y labels."""
        df = test_data
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="Bakery",
            x_label="Sales Value",
            y_label="Category Group",
        )

        assert isinstance(result_ax, plt.Axes)
        assert result_ax.get_xlabel() == "Sales Value"
        assert result_ax.get_ylabel() == "Category Group"

    def test_drop_na_index_values(self, test_data):
        """Test that the function can drop NA index values."""
        df = test_data.copy()
        # Introduce NA value by making one group have the same proportion
        df.loc[0, "sales"] = df.loc[5, "sales"]

        # This should work without error
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="Bakery",
            drop_na=True,
        )
        assert isinstance(result_ax, plt.Axes)
        ytick_labels = [label.get_text() for label in result_ax.get_yticklabels()]
        xtick_labels = [label.get_text() for label in result_ax.get_xticklabels()]
        assert not any(
            label == "" or label is None or str(label).lower() == "nan" or str(label).lower() == "na"
            for label in ytick_labels
        )
        assert not any(
            label == "" or label is None or str(label).lower() == "nan" or str(label).lower() == "na"
            for label in xtick_labels
        )
        bar_values = [patch.get_width() for patch in result_ax.patches]
        assert not any(pd.isna(val) for val in bar_values)

    def test_nan_index_values_present_in_data(self, test_data):
        """Test that NaN index values are present in the index DataFrame when expected."""
        df = test_data.copy()
        df.loc[df["category"] == "Bakery", "sales"] = np.nan

        index_df = get_indexes(
            df,
            value_to_index="Bakery",
            index_col="category",
            value_col="sales",
            group_col="category",
        )
        assert index_df["index"].isna().any(), "Expected at least one NaN value in the 'index' column"

    def test_filter_by_index_values(self):
        """Test that filter_above drops groups whose raw index does not exceed the threshold."""
        df = pd.DataFrame(
            {
                "department": ["Dairy", "Bakery", "Meat", "Produce", "Snacks"] * 2,
                "cust_type": ["Loyalty"] * 5 + ["Regular"] * 5,
                "spend": [100, 200, 150, 300, 50, 120, 80, 200, 100, 90],
            },
        )

        index_df = get_indexes(
            df=df,
            value_to_index="Loyalty",
            index_col="cust_type",
            value_col="spend",
            group_col="department",
            offset=BASELINE_INDEX,
        )
        index_df["raw_index"] = index_df["index"] + BASELINE_INDEX

        # Place the threshold between the 2nd and 3rd sorted raw indexes so a strict subset
        # of departments is dropped, exercising the actual filter path rather than a no-op.
        sorted_raw = sorted(index_df["raw_index"].tolist())
        threshold = (sorted_raw[1] + sorted_raw[2]) / 2
        expected_kept = set(index_df.loc[index_df["raw_index"] > threshold, "department"])

        result_ax = plot(
            df,
            value_col="spend",
            group_col="department",
            index_col="cust_type",
            value_to_index="Loyalty",
            filter_above=threshold,
        )

        kept_labels = {t.get_text() for t in result_ax.get_yticklabels()}

        assert kept_labels == expected_kept
        assert len(kept_labels) < len(index_df)

    def test_empty_dataset_after_filtering(self, test_data):
        """Test that filtering that results in an empty dataset raises ValueError."""
        df = test_data

        # Use an extremely high filter value that will result in an empty dataset
        with pytest.raises(ValueError, match="Filtering resulted in an empty dataset"):
            plot(
                df,
                value_col="sales",
                group_col="category",
                index_col="category",
                value_to_index="Bakery",
                filter_above=100000,  # Using an extremely high value that should cause all data to be filtered out
            )

    @pytest.mark.parametrize(
        ("sort_by", "sort_order", "top_n", "bottom_n", "expected_y_labels"),
        [
            pytest.param(
                "value",
                "ascending",
                3,
                2,
                ["Snacks", "Meat", "Dairy", "Bakery", "Produce"],
                id="value_asc_top3_bot2",
            ),
            pytest.param(
                "value",
                "descending",
                3,
                2,
                ["Produce", "Bakery", "Dairy", "Meat", "Snacks"],
                id="value_desc_top3_bot2",
            ),
            pytest.param("value", "ascending", 3, None, ["Dairy", "Bakery", "Produce"], id="value_asc_top3_only"),
            pytest.param("value", "ascending", None, 2, ["Snacks", "Meat"], id="value_asc_bot2_only"),
            pytest.param(
                "value",
                "ascending",
                None,
                None,
                ["Snacks", "Meat", "Dairy", "Bakery", "Produce"],
                id="value_asc_no_filter",
            ),
            pytest.param(
                "group",
                "ascending",
                3,
                2,
                ["Bakery", "Dairy", "Meat", "Produce", "Snacks"],
                id="group_asc_top3_bot2",
            ),
            pytest.param(
                "group",
                "descending",
                None,
                None,
                ["Snacks", "Produce", "Meat", "Dairy", "Bakery"],
                id="group_desc_no_filter",
            ),
            pytest.param(
                None,
                "ascending",
                None,
                None,
                ["Bakery", "Dairy", "Meat", "Produce", "Snacks"],
                id="no_sort_asc",
            ),
            pytest.param(
                None,
                "descending",
                None,
                None,
                ["Bakery", "Dairy", "Meat", "Produce", "Snacks"],
                id="no_sort_desc",
            ),
            pytest.param(None, "ascending", 2, 2, ["Produce", "Bakery", "Meat", "Snacks"], id="no_sort_top2_bot2"),
        ],
    )
    def test_plot_sort_order(self, sort_by, sort_order, top_n, bottom_n, expected_y_labels):
        """Test that y-axis labels reflect the correct sort order with optional top_n/bottom_n filtering."""
        # Index values (ascending): Snacks(-37.9), Meat(-25.5), Dairy(-21.0), Bakery(24.1), Produce(30.3)
        df = pd.DataFrame(
            {
                "department": ["Dairy", "Bakery", "Meat", "Produce", "Snacks"] * 2,
                "cust_type": ["Loyalty"] * 5 + ["Regular"] * 5,
                "spend": [100, 200, 150, 300, 50, 120, 80, 200, 100, 90],
            },
        )

        ax = plot(
            df,
            value_col="spend",
            group_col="department",
            index_col="cust_type",
            value_to_index="Loyalty",
            sort_by=sort_by,
            sort_order=sort_order,
            top_n=top_n,
            bottom_n=bottom_n,
        )

        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert y_labels == expected_y_labels

    @pytest.mark.parametrize("sort_order", ["ASC", "Asc", "ASCENDING"])
    def test_plot_sort_order_case_insensitive_ascending(self, test_data, sort_order):
        """Mixed-case ascending sort_order values produce the same y-axis ordering as 'ascending'."""
        lower = plot(
            test_data,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="Bakery",
            sort_by="value",
            sort_order="ascending",
        )
        upper = plot(
            test_data,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="Bakery",
            sort_by="value",
            sort_order=sort_order,
        )
        assert [t.get_text() for t in upper.get_yticklabels()] == [t.get_text() for t in lower.get_yticklabels()]

    @pytest.mark.parametrize("sort_by", ["GROUP", "Group", "VALUE"])
    def test_plot_sort_by_case_insensitive(self, test_data, sort_by):
        """Mixed-case sort_by values produce the same y-axis ordering as the lowercase form."""
        lower = plot(
            test_data,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="Bakery",
            sort_by=sort_by.lower(),
            sort_order="ascending",
        )
        upper = plot(
            test_data,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="Bakery",
            sort_by=sort_by,
            sort_order="ascending",
        )
        assert [t.get_text() for t in upper.get_yticklabels()] == [t.get_text() for t in lower.get_yticklabels()]

    def test_error_with_series_and_filtering(self, test_data):
        """Test that appropriate error is raised when using filtering with series_col."""
        df = test_data

        with pytest.raises(
            ValueError,
            match="top_n, bottom_n, filter_above, and filter_below cannot be used when series_col is provided",
        ):
            plot(
                df,
                value_col="sales",
                group_col="category",
                index_col="category",
                value_to_index="Bakery",
                series_col="region",
                top_n=2,
            )

    @pytest.mark.parametrize(
        ("sort_order", "expected_pairs", "expected_y_labels"),
        [
            (
                "ascending",
                [
                    ("Bakery", "North"),
                    ("Bakery", "South"),
                    ("Dairy", "North"),
                    ("Dairy", "South"),
                    ("Produce", "North"),
                    ("Produce", "South"),
                ],
                ["Bakery", "Dairy", "Produce"],
            ),
            (
                "descending",
                [
                    ("Produce", "South"),
                    ("Produce", "North"),
                    ("Dairy", "South"),
                    ("Dairy", "North"),
                    ("Bakery", "South"),
                    ("Bakery", "North"),
                ],
                ["Produce", "Dairy", "Bakery"],
            ),
        ],
    )
    def test_sort_and_plot_with_series_col(
        self,
        sort_order,
        expected_pairs,
        expected_y_labels,
    ):
        """Combined test: validates sorting of dataframe and sorting in plot output."""
        test_df = pd.DataFrame(
            {
                "category": ["Bakery", "Dairy", "Produce", "Bakery", "Dairy", "Produce"],
                "region": ["North", "North", "North", "South", "South", "South"],
                "sales": [100, 200, 150, 120, 180, 160],
                "baseline_category": ["Bakery"] * 6,
            },
        )
        ascending_flag = sort_order == "ascending"

        sorted_df = test_df.sort_values(by=["category", "region"], ascending=ascending_flag)
        actual_pairs = list(zip(sorted_df["category"], sorted_df["region"], strict=False))
        assert actual_pairs == expected_pairs, (
            f"{sort_order=} sort mismatch: expected {expected_pairs}, got {actual_pairs}"
        )

        ax = plot(
            test_df,
            value_col="sales",
            group_col="category",
            index_col="baseline_category",
            value_to_index="Bakery",  # Compare all categories against the Bakery baseline
            series_col="region",
            sort_by="group",
            sort_order=sort_order,
        )

        assert isinstance(ax, plt.Axes)

        # Verify y-axis labels reflect sorted categories
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert y_labels == expected_y_labels, (
            f"{sort_order=} y-ticks mismatch: expected {expected_y_labels}, got {y_labels}"
        )

        # Verify legend contains series_col values
        legend = ax.get_legend()
        assert legend is not None
        legend_labels = [t.get_text() for t in legend.get_texts()]
        expected_legend_labels = ["North", "South"]
        assert set(legend_labels) == set(expected_legend_labels), (
            f"Legend mismatch: expected {expected_legend_labels}, got {legend_labels}"
        )

    def test_error_with_top_n_exceeding_available_groups(self, test_data):
        """Test that appropriate error is raised when top_n exceeds available groups."""
        df = test_data
        total_count = len(df["category"].unique())

        with pytest.raises(ValueError, match=r"top_n .* cannot exceed the number of available groups"):
            plot(
                df,
                value_col="sales",
                group_col="category",
                index_col="category",
                value_to_index="Bakery",
                top_n=total_count + 1,
            )

    def test_error_with_bottom_n_exceeding_available_groups(self, test_data):
        """Test that appropriate error is raised when bottom_n exceeds available groups."""
        df = test_data
        total_count = len(df["category"].unique())

        with pytest.raises(ValueError, match=r"bottom_n .* cannot exceed the number of available groups"):
            plot(
                df,
                value_col="sales",
                group_col="category",
                index_col="category",
                value_to_index="Bakery",
                bottom_n=total_count + 1,
            )

    def test_error_with_sum_of_top_and_bottom_n(self):
        """Test that appropriate error is raised when top_n + bottom_n exceeds available groups."""
        # Create a test index dataframe with the same groups
        test_df = pd.DataFrame(
            {
                "category": ["Bakery", "Dairy", "Produce"],
                "index": [90, 110, 120],
            },
        )

        # Test that sum of top_n and bottom_n validation works
        with pytest.raises(
            ValueError,
            match=r"The sum of top_n .* and bottom_n .* cannot exceed the total number of groups",
        ):
            filter_top_bottom_n(
                df=test_df,
                top_n=2,
                bottom_n=2,
            )


def test_filter_by_groups_exclude_groups():
    """Test that filter_by_groups correctly excludes specified groups."""
    # Create test dataframe with multiple categories
    test_df = pd.DataFrame(
        {
            "category": ["Bakery", "Dairy", "Produce", "Snacks", "Beverages"],
            "value": [10, 20, 30, 40, 50],
        },
    )

    # Test excluding specific groups
    exclude_list = ["Dairy", "Snacks"]
    result_df = filter_by_groups(
        df=test_df,
        group_col="category",
        exclude_groups=exclude_list,
    )

    # Check that excluded groups are not in the result
    assert all(value not in result_df["category"].to_numpy() for value in ["Dairy", "Snacks"])

    # Check that other groups are still in the result
    assert all(value in result_df["category"].to_numpy() for value in ["Bakery", "Produce", "Beverages"])

    # Check that the result has the expected number of rows
    expected_row_count = len(test_df) - len(exclude_list)
    assert len(result_df) == expected_row_count


def test_filter_by_groups_validation_error():
    """Test that filter_by_groups raises ValueError when both exclude and include params are provided."""
    test_df = pd.DataFrame(
        {
            "category": ["Bakery", "Dairy", "Produce", "Snacks", "Beverages"],
            "value": [10, 20, 30, 40, 50],
        },
    )

    exclude_list = ["Dairy", "Snacks"]
    include_list = ["Bakery", "Produce"]

    # Test with both exclude_groups and include_only_groups
    with pytest.raises(ValueError, match="exclude_groups and include_only_groups cannot be used together"):
        plot(
            df=test_df,
            value_col="value",
            group_col="category",
            index_col="category",
            value_to_index="Bakery",
            exclude_groups=exclude_list,
            include_only_groups=include_list,
        )


def test_series_col_with_sort_by_value_validation_error():
    """Test that providing series_col with sort_by='value' raises ValueError."""
    test_df = pd.DataFrame(
        {
            "category": ["Bakery", "Dairy", "Produce"] * 2,
            "series": ["Loyalty", "Loyalty", "Loyalty", "Regular", "Regular", "Regular"],
            "value": [10, 20, 30, 40, 50, 60],
        },
    )

    with pytest.raises(ValueError, match="sort_by cannot be 'value' when series_col is provided"):
        plot(
            df=test_df,
            value_col="value",
            group_col="category",
            index_col="category",
            value_to_index="Bakery",
            series_col="series",
            sort_by="value",
        )


class TestFilterByValueThresholds:
    """Tests for filter_by_value_thresholds.

    The ``index`` column is delta-from-baseline (raw index minus ``BASELINE_INDEX``), matching what
    ``get_indexes(offset=BASELINE_INDEX)`` produces. The fixture below covers raw indexes 99, 100,
    and 101 so the strict-inequality boundary at the threshold can be pinned down precisely:
    departments indexing just under, at, or just over the baseline.
    """

    @pytest.fixture
    def departments_around_baseline(self):
        """Three departments with raw indexes 99, 100, 101 (i.e. just under/at/over baseline)."""
        return pd.DataFrame(
            {
                "department": ["Bakery", "Dairy", "Produce"],
                "index": [-1, 0, 1],
            },
        )

    @pytest.mark.parametrize(
        ("kwargs", "expected_departments"),
        [
            # filter_above is strict (>): raw=100 is dropped at the boundary, raw=101 kept.
            pytest.param({"filter_above": 100}, ["Produce"], id="filter_above_strict_boundary"),
            # filter_below is strict (<): raw=100 is dropped at the boundary, raw=99 kept.
            pytest.param({"filter_below": 100}, ["Bakery"], id="filter_below_strict_boundary"),
            # Combined thresholds form an open interval (filter_above, filter_below).
            pytest.param(
                {"filter_above": 99, "filter_below": 101},
                ["Dairy"],
                id="open_interval_keeps_only_baseline",
            ),
        ],
    )
    def test_thresholds_compare_against_raw_index_with_strict_inequality(
        self,
        departments_around_baseline,
        kwargs,
        expected_departments,
    ):
        """Test thresholds are interpreted as raw indexes with strict inequality at the boundary."""
        # Passing filter_above=100 must keep only Produce (raw=101) under the current contract;
        # the previous delta-based contract would have kept every row in this fixture.
        result_df = filter_by_value_thresholds(df=departments_around_baseline, **kwargs)

        assert sorted(result_df["department"].tolist()) == expected_departments

    @pytest.mark.parametrize(
        ("filter_above", "filter_below"),
        [
            pytest.param(120, 80, id="inverted_thresholds"),
            pytest.param(100, 100, id="equal_thresholds"),
        ],
    )
    def test_rejects_overlapping_thresholds(self, departments_around_baseline, filter_above, filter_below):
        """Test that filter_above >= filter_below raises a clear ValueError before filtering runs."""
        with pytest.raises(ValueError, match=r"filter_above .* must be < filter_below"):
            filter_by_value_thresholds(
                df=departments_around_baseline,
                filter_above=filter_above,
                filter_below=filter_below,
            )


class TestColorByThreshold:
    """Tests for color_by_threshold functionality in the index plot."""

    def teardown_method(self):
        """Clean up after each test method."""
        plt.close("all")

    @pytest.fixture
    def threshold_data(self):
        """Return data with index values that fall above, below, and between default thresholds.

        Computed visual index values:
            Dairy ≈ 99 (neutral), Bakery ≈ 120 (neutral/positive boundary),
            Meat ≈ 72 (negative), Produce ≈ 126 (positive), Snacks ≈ 60 (negative).
        This ensures all three color branches (positive, negative, neutral) are exercised
        for both the default (80, 120) and custom (90, 110) highlight ranges.
        """
        return pd.DataFrame(
            {
                "department": ["Dairy", "Bakery", "Meat", "Produce", "Snacks"] * 2,
                "cust_type": ["Loyalty"] * 5 + ["Regular"] * 5,
                "spend": [130, 200, 150, 300, 50, 90, 80, 200, 100, 90],
            },
        )

    @pytest.mark.parametrize(
        ("expected_color_names", "plot_kwargs"),
        [
            pytest.param(
                # Default range (80, 120), sorted by value ascending:
                # Snacks≈60 (neg), Meat≈72 (neg), Dairy≈99 (neutral), Bakery≈120 (neutral), Produce≈126 (pos)
                ["negative", "negative", "neutral", "neutral", "positive"],
                {"sort_by": "value", "sort_order": "ascending"},
                id="default_range_value_sort",
            ),
            pytest.param(
                # Custom range (90, 110), sorted by value ascending:
                # Snacks≈60 (neg), Meat≈72 (neg), Dairy≈99 (neutral), Bakery≈120 (pos), Produce≈126 (pos)
                ["negative", "negative", "neutral", "positive", "positive"],
                {"highlight_range": (90, 110), "sort_by": "value", "sort_order": "ascending"},
                id="custom_range_value_sort",
            ),
            pytest.param(
                # Default range (80, 120), sorted by group ascending:
                # Bakery≈120 (neutral), Dairy≈99 (neutral), Meat≈72 (neg), Produce≈126 (pos), Snacks≈60 (neg)
                ["neutral", "neutral", "negative", "positive", "negative"],
                {},
                id="default_range_group_sort",
            ),
        ],
    )
    def test_bars_colored_by_threshold(self, threshold_data, expected_color_names, plot_kwargs):
        """Test that bars are colored positive/negative/neutral based on highlight range thresholds."""
        ax = plot(
            threshold_data,
            value_col="spend",
            group_col="department",
            index_col="cust_type",
            value_to_index="Loyalty",
            color_by_threshold=True,
            **plot_kwargs,
        )

        expected_colors = [get_named_color(name) for name in expected_color_names]

        # Filter to only bar patches (alpha=1.0), excluding axvspan highlight (alpha=0.1)
        bar_patches = [p for p in ax.patches if p.get_alpha() is None or p.get_alpha() == 1.0]
        actual_colors = [to_hex(p.get_facecolor()) for p in bar_patches]
        assert actual_colors == expected_colors

    def test_raises_error_when_highlight_range_is_none(self, threshold_data):
        """Test that ValueError is raised when color_by_threshold=True but highlight_range is None."""
        with pytest.raises(ValueError, match="color_by_threshold requires highlight_range to be set"):
            plot(
                threshold_data,
                value_col="spend",
                group_col="department",
                index_col="cust_type",
                value_to_index="Loyalty",
                highlight_range=None,
                color_by_threshold=True,
            )

    def test_raises_error_when_series_col_provided(self, threshold_data):
        """Test that ValueError is raised when color_by_threshold=True with series_col."""
        with pytest.raises(ValueError, match="color_by_threshold cannot be used when series_col is provided"):
            plot(
                threshold_data,
                value_col="spend",
                group_col="department",
                index_col="cust_type",
                value_to_index="Loyalty",
                series_col="cust_type",
                color_by_threshold=True,
            )
