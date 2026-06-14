"""Tests for the price architecture bubble plot module."""

from typing import cast

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

from openretailscience.plots import price


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_price_dataframe():
    """A sample dataframe for testing price architecture plots."""
    rng = np.random.default_rng(42)
    data = {
        "product_id": range(1, 101),
        "unit_price": [
            # Walmart - mix of low and medium prices
            *rng.uniform(1, 3, 25),
            # Target - mostly medium prices
            *rng.uniform(2, 5, 25),
            # Amazon - higher prices
            *rng.uniform(4, 8, 25),
            # Best Buy - wide range
            *rng.uniform(1, 10, 25),
        ],
        "retailer": (["Walmart"] * 25 + ["Target"] * 25 + ["Amazon"] * 25 + ["Best Buy"] * 25),
        "country": (["US"] * 50 + ["UK"] * 50),
    }
    return pd.DataFrame(data)


@pytest.fixture
def simple_price_dataframe():
    """A simple dataframe for predictable testing."""
    data = {
        "unit_price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "retailer": ["Walmart", "Walmart", "Target", "Target", "Amazon", "Amazon"],
    }
    return pd.DataFrame(data)


def test_plot_with_empty_dataframe():
    """Test price architecture plot with an empty DataFrame."""
    # pandas-stubs lacks an overload for a bare list of column names; wrap in pd.Index to disambiguate.
    empty_df = pd.DataFrame(columns=pd.Index(["unit_price", "retailer"]))

    with pytest.raises(ValueError, match="Cannot plot with empty DataFrame"):
        price.plot(
            df=empty_df,
            value_col="unit_price",
            group_col="retailer",
            bins=5,
        )


def test_plot_missing_value_col(simple_price_dataframe):
    """Test price architecture plot when value_col doesn't exist."""
    # Remove the unit_price column to test missing value_col
    df = simple_price_dataframe.drop(columns=["unit_price"])

    with pytest.raises(KeyError, match="value_col 'unit_price' not found in DataFrame"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=5,
        )


def test_plot_missing_group_col(simple_price_dataframe):
    """Test price architecture plot when group_col doesn't exist."""
    # Remove the retailer column to test missing group_col
    df = simple_price_dataframe.drop(columns=["retailer"])

    with pytest.raises(KeyError, match="group_col 'retailer' not found in DataFrame"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=5,
        )


def test_plot_non_numeric_value_col(simple_price_dataframe):
    """Test price architecture plot with non-numeric value column."""
    # Replace unit_price with non-numeric values to test validation
    df = simple_price_dataframe.copy()
    df["unit_price"] = ["low", "medium", "high", "low", "medium", "high"]

    with pytest.raises(ValueError, match="value_col 'unit_price' must be numeric for binning"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=5,
        )


@pytest.mark.parametrize(
    ("bins", "match"),
    [
        (0, "bins must be a positive integer"),
        (-5, "bins must be a positive integer"),
        ([1], "bins list must contain at least 2 values"),
        ([1, "invalid", 3], "All values in bins list must be numeric"),
    ],
)
def test_plot_invalid_bins_raises_value_error(bins, match, simple_price_dataframe):
    """Test price architecture plot raises ValueError for invalid bins values."""
    with pytest.raises(ValueError, match=match):
        price.plot(
            df=simple_price_dataframe,
            value_col="unit_price",
            group_col="retailer",
            bins=bins,
        )


def test_plot_with_unsorted_bins_list(simple_price_dataframe):
    """Unsorted bin edges are sorted before binning so the y-tick labels read low → high."""
    result_ax = price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=[3, 1, 2],  # Unsorted on input.
    )

    # If bins were not sorted, the resulting y-tick labels would be jumbled
    # (e.g. "3.0 - 1.0"). Sorted bins produce ascending "1.0 - 2.0", "2.0 - 3.0".
    assert [label.get_text() for label in result_ax.get_yticklabels()] == ["1.0 - 2.0", "2.0 - 3.0"]


def test_plot_invalid_bins_type(simple_price_dataframe):
    """Test price architecture plot with invalid bins type."""
    # Deliberately pass a wrong-typed bins value to exercise the runtime TypeError guard.
    invalid_bins = cast("int", "invalid")
    with pytest.raises(TypeError, match="bins must be either an integer or a list of numeric values"):
        price.plot(
            df=simple_price_dataframe,
            value_col="unit_price",
            group_col="retailer",
            bins=invalid_bins,
        )


def test_plot_with_missing_values():
    """Rows with NaN in value_col or group_col are dropped, leaving only valid retailers on the x-axis."""
    df = pd.DataFrame(
        {
            "unit_price": [1.0, 2.0, None, 4.0, 5.0],
            "retailer": ["Walmart", "Walmart", "Target", "Target", None],
        },
    )

    result_ax = price.plot(
        df=df,
        value_col="unit_price",
        group_col="retailer",
        bins=3,
    )

    # After dropping NaN rows: Walmart has prices [1.0, 2.0], Target has [4.0]; the row
    # with retailer=None is gone. Both retailers should appear on the x-axis; "None"
    # must not.
    rendered_retailers = {label.get_text() for label in result_ax.get_xticklabels()}
    assert rendered_retailers == {"Walmart", "Target"}


def test_plot_all_missing_values():
    """Test price architecture plot with all missing values."""
    df = pd.DataFrame(
        {
            "unit_price": [None, None, None],
            "retailer": [None, None, None],
        },
    )

    with pytest.raises(ValueError, match="value_col 'unit_price' must be numeric for binning"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=3,
        )


@pytest.mark.parametrize(
    ("bins"),
    [
        (3),
        ([1, 3, 5, 7]),
    ],
)
def test_plot_with_bins(simple_price_dataframe, bins):
    """Test price architecture plot with integer bins and custom bin boundaries."""
    result_ax = price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=bins,
    )

    assert isinstance(result_ax, Axes)

    expected_retailers = 3  # Walmart, Target, Amazon retailers
    assert len(result_ax.get_xticks()) == expected_retailers

    expected_bins = 3  # 3 price bins/boundaries
    assert len(result_ax.get_yticks()) == expected_bins


def test_plot_basic_functionality(sample_price_dataframe):
    """Title and axis labels are rendered, and the x-axis has one tick per retailer."""
    title = "Price Distribution Analysis"
    x_label = "Retailers"
    y_label = "Price Bands"

    result_ax = price.plot(
        df=sample_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=5,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )

    title_texts = [t for t in result_ax.figure.texts if t.get_text() == title]
    assert len(title_texts) == 1
    assert result_ax.get_xlabel() == x_label
    assert result_ax.get_ylabel() == y_label

    expected_retailers = 4  # Walmart, Target, Amazon, Best Buy
    assert len(result_ax.get_xticks()) == expected_retailers

    expected_bins = 5
    assert len(result_ax.get_yticks()) == expected_bins


def test_plot_with_country_grouping(sample_price_dataframe):
    """Test price architecture plot with country grouping."""
    result_ax = price.plot(
        df=sample_price_dataframe,
        value_col="unit_price",
        group_col="country",
        bins=4,
        title="Price Distribution by Country",
    )

    assert isinstance(result_ax, Axes)

    expected_countries = 2  # US, UK
    assert len(result_ax.get_xticks()) == expected_countries


def test_plot_strips_negative_zero_in_y_tick_labels():
    """Y-tick labels never display "-0.0" even when the lowest bin's left edge is a near-zero negative.

    pd.cut(..., include_lowest=True) extends the lowest bin's left edge by a small epsilon below
    zero so the minimum is included. With one-decimal formatting the resulting "-0.0" leaks into
    the y-tick labels; the plot should display "0.0" instead.
    """
    df = pd.DataFrame(
        {
            "unit_price": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 0.5, 12.0, 18.0],
            "retailer": ["Walmart"] * 5 + ["Target"] * 5,
        },
    )

    result_ax = price.plot(df=df, value_col="unit_price", group_col="retailer", bins=4)

    y_tick_labels = [t.get_text() for t in result_ax.get_yticklabels()]
    assert all("-0.0" not in label for label in y_tick_labels), (
        f"Expected no '-0.0' in y-tick labels, got: {y_tick_labels}"
    )
    # Lowest band should start at the data minimum (zero), formatted positively.
    assert y_tick_labels[0].startswith("0.0 - "), f"First label should start with '0.0 - ', got: {y_tick_labels[0]}"


def test_plot_calls_standard_styling(simple_price_dataframe):
    """Title, axis labels and the supplied legend_title are rendered on the axes."""
    title = "Test Title"
    x_label = "Test X Label"
    y_label = "Test Y Label"
    legend_title = "Test Legend"

    result_ax = price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=3,
        title=title,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        move_legend_outside=True,
    )

    title_texts = [t for t in result_ax.figure.texts if t.get_text() == title]
    assert len(title_texts) == 1
    assert result_ax.get_xlabel() == x_label
    assert result_ax.get_ylabel() == y_label
    legend = result_ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == legend_title


def test_plot_legend_outside_reflows_axes_to_reserve_room(simple_price_dataframe):
    """move_legend_outside=True must shrink the axes so the outside legend fits in the figure.

    Without a reflow, the legend anchored at bbox=(1.02, 1.0) extends past the
    figure's right edge and is clipped. tight_layout only reserves room for an
    outside legend when the legend is present on the axes *before* chrome runs.
    """
    _, ax_inside = plt.subplots()
    price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=3,
        ax=ax_inside,
        move_legend_outside=False,
    )
    inside_right = ax_inside.get_position().x1

    _, ax_outside = plt.subplots()
    price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=3,
        ax=ax_outside,
        move_legend_outside=True,
    )
    outside_right = ax_outside.get_position().x1

    legend = ax_outside.get_legend()
    assert legend is not None
    expected_groups = 3
    assert len(legend.get_texts()) == expected_groups

    min_reserved_room = 0.05
    assert outside_right < inside_right - min_reserved_room, (
        f"Expected axes to be reflowed narrower with move_legend_outside=True "
        f"(inside_right={inside_right:.3f}, outside_right={outside_right:.3f})"
    )


def test_plot_adds_source_text(simple_price_dataframe):
    """The price plot renders source_text as a figure-level text element."""
    source_text = "Source: Test Data"

    result_ax = price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=3,
        source_text=source_text,
    )

    rendered = [t.get_text() for t in result_ax.figure.texts]
    assert source_text in rendered


def test_plot_with_kwargs(simple_price_dataframe):
    """Additional kwargs (e.g. alpha) are forwarded to the underlying scatter collection."""
    custom_alpha = 0.8

    result_ax = price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=3,
        alpha=custom_alpha,
    )

    collection = result_ax.collections[0]
    assert collection.get_alpha() == custom_alpha


def test_plot_raises_error_when_no_data_in_bins(simple_price_dataframe):
    """Test that plot raises error when no data falls within the specified bins."""
    # simple_price_dataframe has prices [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # so bins [7.0, 8.0, 9.0, 10.0] will exclude all data
    with pytest.raises(ValueError, match="All proportions are zero - no data falls within the specified bins"):
        price.plot(
            df=simple_price_dataframe,
            value_col="unit_price",
            group_col="retailer",
            bins=[7.0, 8.0, 9.0, 10.0],  # All data is below these bins
        )


def test_percentages_sum_to_100_for_each_group():
    """Per-group bubble sizes (proportions times scale_factor) sum to one full scale per retailer."""
    df = pd.DataFrame(
        {
            "unit_price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "retailer": ["Walmart", "Walmart", "Walmart", "Walmart", "Target", "Target", "Target", "Target"],
        },
    )
    scale_factor = 1000
    floating_point_tolerance = 0.001

    result_ax = price.plot(
        df=df,
        value_col="unit_price",
        group_col="retailer",
        bins=[1.0, 3.0, 5.0, 7.0, 9.0],
        s=scale_factor,
    )

    # All bubbles share a single PathCollection; offsets give x (retailer index) and
    # y (bin index), and sizes are proportions scaled by `scale_factor`.
    collection = result_ax.collections[0]
    assert isinstance(collection, PathCollection)
    # get_offsets() returns a wide ArrayLike union in matplotlib stubs; np.asarray pins it to an ndarray.
    offsets = np.asarray(collection.get_offsets())
    sizes = collection.get_sizes()
    x_positions = offsets[:, 0]

    rendered = pd.DataFrame({"x": x_positions, "size": sizes})
    # pandas-stubs widens the grouped sum to DataFrame | Series; selecting one column yields a Series.
    group_sums = cast("pd.Series", rendered.groupby("x")["size"].sum())
    per_group_totals = group_sums / scale_factor

    assert all(abs(total - 1.0) < floating_point_tolerance for total in per_group_totals)
    assert (sizes > 0).all()


def test_individual_percentage_calculations_are_correct():
    """Each (retailer, bin) bubble has the exact proportion expected from the input distribution.

    Includes the edge case of a third retailer with 100% in one band and 0% in the others.
    """
    df = pd.DataFrame(
        {
            "unit_price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.5],
            "retailer": ["Walmart"] * 4 + ["Target"] * 4 + ["Amazon"],
        },
    )
    scale_factor = 800

    result_ax = price.plot(
        df=df,
        value_col="unit_price",
        group_col="retailer",
        bins=[0.0, 2.5, 6.5, 9.0, 10.0],
        s=scale_factor,
    )

    # Pandas groupby orders retailers alphabetically: Amazon=0, Target=1, Walmart=2.
    # Distribution per retailer over bins (1-indexed for display, 0-indexed here):
    #   Walmart [1,2,3,4]:  bin0=2/4, bin1=2/4, bin2=0,   bin3=0
    #   Target  [5,6,7,8]:  bin0=0,   bin1=2/4, bin2=2/4, bin3=0
    #   Amazon  [9.5]:      bin0=0,   bin1=0,   bin2=0,   bin3=1/1
    expected = (
        pd.DataFrame(
            {
                "x": np.array([2, 2, 1, 1, 0], dtype=float),
                "y": np.array([0, 1, 1, 2, 3], dtype=float),
                "size": np.array([0.5, 0.5, 0.5, 0.5, 1.0]) * scale_factor,
            },
        )
        .sort_values(["x", "y"])
        .reset_index(drop=True)
    )

    collection = result_ax.collections[0]
    assert isinstance(collection, PathCollection)
    # get_offsets() returns a wide ArrayLike union in matplotlib stubs; np.asarray pins it to an ndarray.
    offsets = np.asarray(collection.get_offsets())
    actual = (
        pd.DataFrame({"x": offsets[:, 0], "y": offsets[:, 1], "size": collection.get_sizes()})
        .sort_values(["x", "y"])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(actual, expected)
