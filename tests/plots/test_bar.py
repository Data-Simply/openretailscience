"""Tests for the bar plot module."""

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from openretailscience.plots import bar


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_dataframe():
    """A sample retail product-sales dataframe for testing."""
    data = {
        "product": ["Apples", "Bananas", "Cereal", "Donuts"],
        "sales_q1": [1000, 1500, 2000, 2500],
        "sales_q2": [1100, 1600, 2100, 2600],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_series():
    """A sample retail product-sales series for testing."""
    return pd.Series([1000, 1500, 2000, 2500], index=["Apples", "Bananas", "Cereal", "Donuts"])


def test_plot_with_empty_dataframe():
    """Test bar plot with an empty DataFrame."""
    empty_df = pd.DataFrame({"sales_q1": [], "sales_q2": []})

    with pytest.raises(ValueError, match="Cannot plot with empty DataFrame"):
        bar.plot(
            df=empty_df,
            value_col="sales_q1",
            x_col="product",
            title="Test Plot with Empty DataFrame",
        )


@pytest.mark.parametrize(
    ("plot_kwargs", "expected_num_patches"),
    [
        ({"value_col": "sales_q1", "x_col": "product"}, 4),
        ({"value_col": "sales_q1"}, 4),
        ({"value_col": ["sales_q1", "sales_q2"]}, 8),
    ],
    ids=["single_value_col_with_x", "single_value_col_no_x", "list_value_col_no_x"],
)
def test_plot_produces_expected_patch_count(sample_dataframe, plot_kwargs, expected_num_patches):
    """bar.plot returns one patch per (x value, value_col) combination."""
    result_ax = bar.plot(df=sample_dataframe, title="Test Bar Patch Count", **plot_kwargs)

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches


def test_missing_x_col_in_dataframe(sample_dataframe):
    """Test bar plot when the provided x_col does not exist in the DataFrame."""
    with pytest.raises(ValueError, match=r"\['missing_col'\]"):
        bar.plot(
            df=sample_dataframe,
            value_col="sales_q1",
            x_col="missing_col",
            title="Test Plot with Missing x_col",
        )


def test_x_col_with_series_input_raises_clear_error(sample_series):
    """When df is a pd.Series, passing x_col raises a clear ValueError rather than an opaque AttributeError."""
    with pytest.raises(ValueError, match=r"x_col cannot be provided when df is a pd\.Series"):
        bar.plot(df=sample_series, x_col="anything")


def test_plot_with_none_value_col(sample_dataframe):
    """Test bar plot with None for value_col (default 'Value' column)."""
    result_ax = bar.plot(
        df=sample_dataframe["sales_q1"],
        title="Test Plot with None Value Col",
    )

    expected_heights = sample_dataframe["sales_q1"].tolist()
    actual_heights = [p.get_height() for p in result_ax.patches if isinstance(p, Rectangle)]

    expected_num_patches = 4

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches
    assert actual_heights == pytest.approx(expected_heights)


def test_plot_with_nan_values():
    """Test bar plot with NaN values in the data."""
    nan_dataframe = pd.DataFrame(
        {
            "product": ["Apples", "Bananas", "Cereal", "Donuts"],
            "sales_q1": [1000, 1500, None, 2500],
            "sales_q2": [1100, 1600, 2100, None],
        },
    )

    result_ax = bar.plot(
        df=nan_dataframe,
        value_col=["sales_q1", "sales_q2"],
        x_col="product",
        data_label_format="absolute",
        title="Test Plot with NaN Values",
    )

    expected_heights = [1000.0, 1500.0, 0.0, 2500.0, 1100.0, 1600.0, 2100.0, 0.0]
    actual_heights = [p.get_height() for p in result_ax.patches if isinstance(p, Rectangle)]

    expected_num_patches = 8

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches
    assert actual_heights == pytest.approx(expected_heights)


def test_plot_horizontal_orientation(sample_dataframe):
    """Test bar plot with horizontal orientation."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        orientation="horizontal",
        title="Test Horizontal Bar Plot",
    )

    expected_height = 0.8

    assert isinstance(result_ax, Axes)
    # Check that the bars are oriented horizontally by checking their width, not height
    bars = [p for p in result_ax.patches if isinstance(p, Rectangle)]
    assert all(p.get_width() > 0 for p in bars)
    assert all(p.get_height() == expected_height for p in bars)  # Default bar height in horizontal bars


def test_plot_vertical_orientation(sample_dataframe):
    """Test bar plot with vertical orientation."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        orientation="vertical",
        title="Test Vertical Bar Plot",
    )

    expected_width = 0.8

    assert isinstance(result_ax, Axes)
    # Check that the bars are oriented vertically by checking their height, not width
    bars = [p for p in result_ax.patches if isinstance(p, Rectangle)]
    assert all(p.get_height() > 0 for p in bars)
    assert all(p.get_width() == expected_width for p in bars)  # Default bar width in vertical bars


def test_plot_with_custom_bar_width(sample_dataframe):
    """Test bar plot with a custom width for bars."""
    custom_width = 0.5
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        width=custom_width,
        title="Test Plot with Custom Bar Width",
    )

    assert isinstance(result_ax, Axes)
    # Check the width of individual bars
    assert all(p.get_width() == custom_width for p in result_ax.patches if isinstance(p, Rectangle))


def test_plot_with_hatch(sample_dataframe):
    """use_hatch=True applies a hatch pattern to every bar patch."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        use_hatch=True,
        title="Test Bar Plot with Hatch",
    )

    hatches = [p.get_hatch() for p in result_ax.patches]
    assert len(hatches) == len(sample_dataframe)
    assert all(isinstance(h, str) and len(h) > 0 for h in hatches)


def test_plot_multiple_bars(sample_dataframe):
    """Test the bar plot function with multiple value columns."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col=["sales_q1", "sales_q2"],
        x_col="product",
        title="Test Multiple Bar Plot",
    )

    expected_heights = sample_dataframe["sales_q1"].tolist() + sample_dataframe["sales_q2"].tolist()
    actual_heights = [p.get_height() for p in result_ax.patches if isinstance(p, Rectangle)]

    expected_num_patches = 8

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches
    assert actual_heights == pytest.approx(expected_heights)


@pytest.mark.parametrize("sort_order", ["ascending", "asc", "ASC", "Ascending"])
def test_plot_with_ascending_sorting(sample_dataframe, sort_order):
    """Test that all ascending sort_order aliases produce identical sorted bar plots."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        sort_order=sort_order,
        title="Test Sorted Bar Plot",
    )

    expected_order = ["Apples", "Bananas", "Cereal", "Donuts"]
    actual_order = [label.get_text() for label in result_ax.get_xticklabels()]
    bar_heights = [patch.get_height() for patch in result_ax.patches if isinstance(patch, Rectangle)]
    expected_num_patches = 4

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches
    assert result_ax.get_xticklabels()[0].get_text() == "Apples"
    assert actual_order == expected_order
    assert bar_heights == sorted(bar_heights)


@pytest.mark.parametrize("sort_order", ["descending", "desc", "DESC", "Descending"])
def test_plot_with_descending_sorting(sample_dataframe, sort_order):
    """Test that all descending sort_order aliases produce identical reverse-sorted bar plots."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        sort_order=sort_order,
        title="Test Sorted Bar Plot",
    )

    expected_order = ["Donuts", "Cereal", "Bananas", "Apples"]
    actual_order = [label.get_text() for label in result_ax.get_xticklabels()]
    bar_heights = [patch.get_height() for patch in result_ax.patches if isinstance(patch, Rectangle)]

    assert isinstance(result_ax, Axes)
    assert actual_order == expected_order
    assert bar_heights == sorted(bar_heights, reverse=True)


def test_plot_with_data_labels(sample_dataframe):
    """Test the bar plot function with data labels in absolute format."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        data_label_format="absolute",
        title="Test Bar Plot with Data Labels",
    )

    expected_labels = ["1K", "1.5K", "2K", "2.5K"]
    actual_labels = [text.get_text() for text in result_ax.texts]

    assert isinstance(result_ax, Axes)
    assert len(result_ax.containers) == 1
    assert actual_labels == expected_labels


def test_plot_with_percentage_by_bar_group_labels(sample_dataframe):
    """Test the bar plot function with data labels in 'percentage_by_bar_group' format and verify percentages."""
    # Plot the bars with percentage labels
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col=["sales_q1", "sales_q2"],
        x_col="product",
        data_label_format="percentage_by_bar_group",
        title="Test Bar Plot with Percentage Within Bar Groups Labels",
    )

    # Calculate expected percentages within each bar group (sum within each row)
    total_sales_per_product = sample_dataframe["sales_q1"] + sample_dataframe["sales_q2"]
    expected_percentages_q1 = (sample_dataframe["sales_q1"] / total_sales_per_product) * 100
    expected_percentages_q2 = (sample_dataframe["sales_q2"] / total_sales_per_product) * 100

    # Combine the expected percentages for both Q1 and Q2
    expected_percentages = list(expected_percentages_q1) + list(expected_percentages_q2)

    # Retrieve all text labels applied to the bars
    labels = [t.get_text() for t in result_ax.texts]

    # Check that the retrieved labels match the expected percentages
    for label, expected_percentage in zip(labels, expected_percentages, strict=False):
        assert float(label.strip("%")) == pytest.approx(expected_percentage, 0.01)


def test_plot_with_percentage_by_series_labels(sample_dataframe):
    """Test the bar plot function with data labels in 'percentage_by_series' format and verify percentages."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col=["sales_q1", "sales_q2"],
        x_col="product",
        data_label_format="percentage_by_series",
        title="Test Bar Plot with Percentage Across Bar Groups Labels",
    )

    # Calculate expected percentages across bar groups for each value column
    total_sum_q1 = sample_dataframe["sales_q1"].sum()
    total_sum_q2 = sample_dataframe["sales_q2"].sum()

    expected_percentages_q1 = (sample_dataframe["sales_q1"] / total_sum_q1) * 100
    expected_percentages_q2 = (sample_dataframe["sales_q2"] / total_sum_q2) * 100

    # Combine the expected percentages for both Q1 and Q2
    expected_percentages = list(expected_percentages_q1) + list(expected_percentages_q2)

    # Retrieve all text labels applied to the bars
    labels = [t.get_text() for t in result_ax.texts]

    # Check that the retrieved labels match the expected percentages
    for label, expected_percentage in zip(labels, expected_percentages, strict=False):
        assert float(label.strip("%")) == pytest.approx(expected_percentage, 0.29)


def test_plot_adds_source_text(sample_dataframe):
    """The bar plot renders source_text as a figure-level text element."""
    source_text = "Source: Test Data"
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        title="Test Bar Plot with Source Text",
        source_text=source_text,
    )

    rendered = [t.get_text() for t in result_ax.figure.texts]
    assert source_text in rendered


def test_default_bar_styling(sample_dataframe):
    """Single value_col bar plot has the documented default width (0.8) and no legend."""
    default_width = 0.8

    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        title="Test Bar Plot",
    )

    rectangles = [p for p in result_ax.patches if isinstance(p, Rectangle)]
    assert len(rectangles) == len(sample_dataframe)
    assert all(p.get_width() == default_width for p in rectangles)
    assert result_ax.get_legend() is None
    assert [label.get_text() for label in result_ax.get_xticklabels()] == sample_dataframe["product"].tolist()


@pytest.mark.parametrize(
    ("kwarg", "value", "match"),
    [
        ("orientation", "invalid_orientation", r"orientation must be one of .*'invalid_orientation'"),
        ("sort_order", "invalid_sort_order", r"sort_order must be one of .*'invalid_sort_order'"),
        ("data_label_format", "invalid_format", r"data_label_format must be one of .*'invalid_format'"),
    ],
)
def test_invalid_kwarg_raises_value_error(sample_dataframe, kwarg, value, match):
    """Test that invalid values for enum-like kwargs raise a ValueError with a descriptive message."""
    with pytest.raises(ValueError, match=match):
        bar.plot(
            df=sample_dataframe,
            value_col="sales",
            x_col="product",
            **{kwarg: value},
        )


def test_default_value_col_handling(sample_series):
    """Test that a default 'Value' column is created when no value_col is passed."""
    result_ax = bar.plot(
        df=sample_series,
        title="Test Default Value Column Handling",
    )

    expected_num_patches = 4

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches
    _, legend_labels = result_ax.get_legend_handles_labels()
    assert legend_labels, "Expected legend labels but found none"
    assert legend_labels == ["Value"], f"Expected legend label to be 'Value', got {legend_labels}"


def test_plot_with_series(sample_series):
    """Test the bar plot function works with a pandas Series."""
    result_ax = bar.plot(
        df=sample_series,
        title="Test Bar Plot with Series",
    )

    expected_num_patches = 4

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches


def test_percentage_by_bar_group_with_negative_values():
    """Test percentage_by_bar_group with negative values, triggering warning."""
    df = pd.DataFrame(
        {
            "product": ["Apples", "Bananas", "Cereal", "Donuts"],
            "sales_q1": [1000, -1500, 2000, -2500],
            "sales_q2": [-1100, 1600, -2100, 2600],
        },
    )
    with pytest.warns(UserWarning, match="Negative values detected"):
        bar.plot(
            df=df,
            value_col=["sales_q1", "sales_q2"],
            x_col="product",
            data_label_format="percentage_by_bar_group",
        )


def test_percentage_by_bar_group_with_zero_group_total():
    """Test percentage_by_bar_group with zero group totals and verify warning is emitted."""
    df = pd.DataFrame({"product": ["Apples", "Bananas"], "sales": [0, 0]})

    with pytest.warns(UserWarning, match="Division by zero detected"):
        result_ax = bar.plot(
            df=df,
            value_col="sales",
            x_col="product",
            data_label_format="percentage_by_bar_group",
        )

    labels = [t.get_text() for t in result_ax.texts]
    assert all(label == "" for label in labels)  # Should all be empty due to division by zero
