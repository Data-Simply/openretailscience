"""Tests for the histograms plot module."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes

from openretailscience.plots import histogram


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_dataframe():
    """A sample dataframe for testing."""
    data = {
        "value_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "value_2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "group": ["Loyalty"] * 5 + ["Guest"] * 5,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_series():
    """A sample series for testing."""
    return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test_plot_single_histogram(sample_dataframe):
    """Test the plot function with a single histogram."""
    bins = 7
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        bins=bins,
        title="Test Single Histogram",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == bins


def test_plot_grouped_histogram(sample_dataframe):
    """Test the plot function with grouped histograms."""
    bins = 7
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        group_col="group",
        bins=bins,
        title="Test Grouped Histogram",
    )

    num_groups = sample_dataframe["group"].nunique()
    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == bins * num_groups


def test_clip_range_piles_outliers_into_boundary_bins():
    """clip_range=(lo, hi) clamps out-of-range values so they pile up at the edge bins."""
    # 5 in-range values + 3 below-lower outliers + 2 above-upper outliers.
    in_range = [10, 20, 30, 40, 50]
    below_lower = [-5, -10, -20]
    above_upper = [200, 300]
    df = pd.DataFrame({"basket_amount": in_range + below_lower + above_upper})

    result_ax = histogram.plot(
        df=df,
        value_col="basket_amount",
        clip_range=(0, 60),
        bins=6,
    )

    heights = [p.get_height() for p in result_ax.patches]
    # Bin edges [0, 10, 20, 30, 40, 50, 60]; last bin is right-closed.
    # All below-lower outliers clamp to 0 → first bin.
    # All above-upper outliers + the in-range 50 land in the last bin (60 is closed).
    expected_first_bin = len(below_lower)
    expected_last_bin = len(above_upper) + 1  # +1 for the in-range value 50
    assert heights[0] == expected_first_bin
    assert heights[-1] == expected_last_bin
    assert sum(heights) == len(df)


@pytest.mark.parametrize(
    ("data", "clip_range", "expected_leftmost", "expected_rightmost"),
    [
        ([10, 20, 30, 40, 50, -5, -10, 200, 300], (0, 60), 0, 60),
        ([0, 60, 200, 300], (None, 60), None, 60),
        ([-50, -20, 10, 20, 30], (0, None), 0, None),
    ],
)
def test_clip_range_bin_edges_respect_bounds(data, clip_range, expected_leftmost, expected_rightmost):
    """clip_range bin edges hug the clipped bound on whichever side is set."""
    df = pd.DataFrame({"basket_amount": data})

    result_ax = histogram.plot(
        df=df,
        value_col="basket_amount",
        clip_range=clip_range,
        bins=6,
    )

    if expected_leftmost is not None:
        leftmost = min(p.get_x() for p in result_ax.patches)
        assert leftmost == pytest.approx(expected_leftmost)
    if expected_rightmost is not None:
        rightmost = max(p.get_x() + p.get_width() for p in result_ax.patches)
        assert rightmost == pytest.approx(expected_rightmost)


def test_passing_range_and_clip_range_raises(sample_dataframe):
    """Specifying both matplotlib's `range` and ORS's `clip_range` raises ValueError."""
    with pytest.raises(ValueError, match="Cannot specify both `range` and `clip_range`"):
        histogram.plot(
            df=sample_dataframe,
            value_col="value_1",
            clip_range=(2, 8),
            range=(2, 8),
        )


@pytest.mark.parametrize("clip_range", [(50,), (0, 50, 100)])
def test_clip_range_wrong_length_raises(sample_dataframe, clip_range):
    """clip_range must be a 2-tuple; other lengths raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="clip_range must be a 2-tuple"):
        histogram.plot(
            df=sample_dataframe,
            value_col="value_1",
            clip_range=clip_range,
        )


def test_clip_range_lower_greater_than_upper_raises():
    """clip_range with lower > upper raises rather than silently collapsing values to the bound."""
    df = pd.DataFrame({"basket_amount": [10, 20, 30, 40, 50]})
    with pytest.raises(ValueError, match=r"clip_range lower \(60\) must be <= upper \(0\)"):
        histogram.plot(
            df=df,
            value_col="basket_amount",
            clip_range=(60, 0),
        )


def test_plot_single_histogram_series(sample_series):
    """Test the plot function with a pandas series."""
    bins = 7
    result_ax = histogram.plot(
        df=sample_series,
        bins=bins,
        title="Test Single Histogram (Series)",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == bins


def test_plot_histogram_with_hatch(sample_dataframe):
    """use_hatch=True applies a hatch pattern to every histogram patch."""
    bins = 7
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        bins=bins,
        title="Test Histogram with Hatch",
        use_hatch=True,
    )

    hatches = [p.get_hatch() for p in result_ax.patches]
    assert len(hatches) == bins
    assert all(isinstance(h, str) and len(h) > 0 for h in hatches)


def test_plot_invalid_value_col_with_group_col(sample_dataframe):
    """Test the plot function raises an error when both `value_col` is a list and `group_col` is provided."""
    with pytest.raises(ValueError, match="`value_col` cannot be a list when `group_col` is provided"):
        histogram.plot(
            df=sample_dataframe,
            value_col=["value_1", "value_2"],
            group_col="group",
            title="Test Invalid Value Col with Group Col",
        )


def test_plot_adds_source_text(sample_dataframe):
    """The histogram renders source_text as a figure-level text element."""
    source_text = "Source: Test Data"
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        title="Test with Source Text",
        source_text=source_text,
    )

    rendered = [t.get_text() for t in result_ax.figure.texts]
    assert source_text in rendered


def test_plot_multiple_histograms(sample_dataframe):
    """Test the plot function with multiple histograms."""
    bins = 7
    value_cols = ["value_1", "value_2"]
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col=value_cols,
        bins=bins,
        title="Test Multiple Histograms",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == bins * len(value_cols)


@pytest.mark.parametrize(
    ("group_col", "expected_legend", "expected_alpha"),
    [
        (None, False, None),
        ("group", True, 0.7),
    ],
)
def test_histogram_legend_and_alpha_by_grouping(sample_dataframe, group_col, expected_legend, expected_alpha):
    """Single histograms render without a legend/alpha; grouped histograms render with both."""
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        group_col=group_col,
        title="Histogram grouping",
    )

    assert (result_ax.get_legend() is not None) is expected_legend

    patch_alphas = {p.get_alpha() for p in result_ax.patches}
    assert patch_alphas == {expected_alpha}


def test_single_histogram_with_legend_title_does_not_render_legend(sample_dataframe):
    """A single-series histogram must not render a one-entry legend even if `legend_title` is set."""
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        legend_title="Spend distribution",
        title="Histogram with legend title but single series",
    )

    assert result_ax.get_legend() is None
