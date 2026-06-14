"""Tests for the period_on_period overlapping_periods function."""

from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from openretailscience.options import ColumnHelper
from openretailscience.plots.period_on_period import plot

# plot()'s periods parameter is invariant list[tuple[str | datetime, str | datetime]]; annotate the
# string-only literals used in these tests so they match without widening at every call site.
Periods = list[tuple[str | datetime, str | datetime]]


def _period_label(period: tuple[str | datetime, str | datetime]) -> str:
    """Render a (start, end) period tuple as the legend label period_on_period uses."""
    return f"{pd.to_datetime(period[0]).date()} to {pd.to_datetime(period[1]).date()}"


cols = ColumnHelper()


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_dataframe():
    """Daily store revenue across a 20-day window for period-on-period comparison."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    daily_revenue = 50000 + rng.normal(0, 5000, len(dates))
    return pd.DataFrame(
        {
            cols.transaction_date: dates,
            cols.agg.unit_spend: daily_revenue,
        },
    )


def test_overlapping_periods_basic(sample_dataframe):
    """Test basic overlapping periods plot."""
    periods: Periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]
    ax = plot(
        df=sample_dataframe,
        x_col=cols.transaction_date,
        value_col=cols.agg.unit_spend,
        periods=periods,
    )

    assert isinstance(ax, Axes)
    expected_lines_count = 2
    assert len(ax.get_lines()) == expected_lines_count


def test_overlapping_periods_with_labels_and_title(sample_dataframe):
    """The plot renders the supplied title and axis labels on the axes."""
    periods: Periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]
    title = "Overlapping Periods Test"
    ax = plot(
        df=sample_dataframe,
        x_col=cols.transaction_date,
        value_col=cols.agg.unit_spend,
        periods=periods,
        x_label="Time",
        y_label="Sales",
        title=title,
    )

    title_texts = [t for t in ax.figure.texts if t.get_text() == title]
    assert len(title_texts) == 1
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Sales"


def test_overlapping_periods_with_source_text(sample_dataframe):
    """The plot renders source_text as a figure-level text element."""
    periods: Periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]
    source_text = "Source: Sales Data"

    ax = plot(
        df=sample_dataframe,
        x_col=cols.transaction_date,
        value_col=cols.agg.unit_spend,
        periods=periods,
        source_text=source_text,
    )

    rendered = [t.get_text() for t in ax.figure.texts]
    assert source_text in rendered


def test_default_renders_legend_with_one_entry_per_period(sample_dataframe):
    """Default plot must label every period so readers can identify each line."""
    periods: Periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]

    ax = plot(
        df=sample_dataframe,
        x_col=cols.transaction_date,
        value_col=cols.agg.unit_spend,
        periods=periods,
    )

    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == [_period_label(p) for p in periods]


def test_overlapping_periods_with_legend_title_and_outside(sample_dataframe):
    """legend_title is rendered as the legend's title when move_legend_outside=True."""
    periods: Periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]
    legend_title = "Periods"

    ax = plot(
        df=sample_dataframe,
        x_col=cols.transaction_date,
        value_col=cols.agg.unit_spend,
        periods=periods,
        move_legend_outside=True,
        legend_title=legend_title,
    )

    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == legend_title


def test_periods_sampled_from_sequential_cmap_newest_darkest(sample_dataframe):
    """Periods read as ordered: newest is darker than middle, middle is darker than oldest.

    Caller passes periods in reverse-chronological order to confirm the function sorts internally
    rather than relying on input order.
    """
    periods: Periods = [
        ("2023-01-15", "2023-01-19"),  # newest
        ("2023-01-08", "2023-01-12"),
        ("2023-01-01", "2023-01-05"),  # oldest
    ]
    ax = plot(
        df=sample_dataframe,
        x_col=cols.transaction_date,
        value_col=cols.agg.unit_spend,
        periods=periods,
    )

    line_by_label = {ln.get_label(): mcolors.to_rgb(ln.get_color()) for ln in ax.get_lines()}
    newest_brightness = sum(line_by_label[_period_label(periods[0])]) / 3
    middle_brightness = sum(line_by_label[_period_label(periods[1])]) / 3
    oldest_brightness = sum(line_by_label[_period_label(periods[-1])]) / 3

    assert newest_brightness < middle_brightness < oldest_brightness


def test_newer_periods_drawn_above_older_periods(sample_dataframe):
    """Newer periods draw on top of older ones regardless of caller-supplied period order.

    Caller deliberately passes periods in non-chronological order (oldest, newest, middle) to
    prove the zorder is derived from the period dates, not from list position.
    """
    oldest = ("2023-01-01", "2023-01-05")
    middle = ("2023-01-08", "2023-01-12")
    newest = ("2023-01-15", "2023-01-19")
    periods = [oldest, newest, middle]

    ax = plot(
        df=sample_dataframe,
        x_col=cols.transaction_date,
        value_col=cols.agg.unit_spend,
        periods=periods,
    )

    zorder_by_label = {ln.get_label(): ln.get_zorder() for ln in ax.get_lines()}

    assert zorder_by_label[_period_label(newest)] > zorder_by_label[_period_label(middle)]
    assert zorder_by_label[_period_label(middle)] > zorder_by_label[_period_label(oldest)]


def test_newest_period_drawn_thicker_than_oldest(sample_dataframe):
    """Linewidth ramp reinforces the color/linestyle ordering: newest thickest, oldest thinnest."""
    periods: Periods = [
        ("2023-01-15", "2023-01-19"),  # newest
        ("2023-01-08", "2023-01-12"),
        ("2023-01-01", "2023-01-05"),  # oldest
    ]
    ax = plot(
        df=sample_dataframe,
        x_col=cols.transaction_date,
        value_col=cols.agg.unit_spend,
        periods=periods,
    )

    line_by_label = {ln.get_label(): ln for ln in ax.get_lines()}

    assert (
        line_by_label[_period_label(periods[0])].get_linewidth()
        > line_by_label[_period_label(periods[1])].get_linewidth()
        > line_by_label[_period_label(periods[-1])].get_linewidth()
    )


def test_user_supplied_linewidth_overrides_gradient(sample_dataframe):
    """A caller-supplied linewidth applies uniformly to every period instead of the gradient."""
    periods: Periods = [
        ("2023-01-15", "2023-01-19"),
        ("2023-01-01", "2023-01-05"),
    ]
    user_linewidth = 4.0

    ax = plot(
        df=sample_dataframe,
        x_col=cols.transaction_date,
        value_col=cols.agg.unit_spend,
        periods=periods,
        linewidth=user_linewidth,
    )

    widths = [ln.get_linewidth() for ln in ax.get_lines()]
    assert widths == [pytest.approx(user_linewidth)] * len(periods)


def test_plot_does_not_mutate_caller_dataframe_when_x_col_is_string():
    """plot() must not coerce the caller's x_col dtype when it is passed as strings.

    Without a defensive copy, ``pd.to_datetime`` reassigns the column on the caller's
    DataFrame, permanently changing its dtype from object to datetime64.
    """
    rng = np.random.default_rng(42)
    dates_as_strings = pd.date_range("2023-01-01", periods=20, freq="D").strftime("%Y-%m-%d").tolist()
    daily_revenue = 50000 + rng.normal(0, 5000, len(dates_as_strings))
    df = pd.DataFrame(
        {
            cols.transaction_date: dates_as_strings,
            cols.agg.unit_spend: daily_revenue,
        },
    )
    df_before = df.copy(deep=True)
    periods: Periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]

    plot(
        df=df,
        x_col=cols.transaction_date,
        value_col=cols.agg.unit_spend,
        periods=periods,
    )

    pd.testing.assert_frame_equal(df, df_before)


def test_overlapping_periods_raises_on_empty_periods(sample_dataframe):
    """Test overlapping periods raises a ValueError when an empty list is passed."""
    with pytest.raises(
        ValueError,
        match=r"The 'periods' list must contain at least two \(start, end\) tuples for comparison",
    ):
        plot(
            df=sample_dataframe,
            x_col=cols.transaction_date,
            value_col=cols.agg.unit_spend,
            periods=[],
        )


@pytest.mark.parametrize("invalid_value", ["endofline", "box ", "", "BOX"])
def test_plot_rejects_invalid_legend_style(sample_dataframe, invalid_value):
    """period_on_period.plot raises ValueError for legend_style values outside the documented set."""
    periods: Periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]
    with pytest.raises(ValueError, match="legend_style"):
        plot(
            df=sample_dataframe,
            x_col=cols.transaction_date,
            value_col=cols.agg.unit_spend,
            periods=periods,
            legend_style=invalid_value,
        )
