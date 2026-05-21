"""Tests for the plots.area module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from openretailscience.plots import area


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


PERIODS = 6


@pytest.fixture
def sample_dataframe():
    """A sample dataframe for Jeans sales data."""
    rng = np.random.default_rng(42)
    data = {
        "transaction_date": np.repeat(pd.date_range("2023-01-01", periods=PERIODS, freq="ME"), 3),
        "unit_spend": rng.integers(1, 6, size=3 * PERIODS),
        "category": ["Jeans", "Shoes", "Dresses"] * PERIODS,
    }
    return pd.DataFrame(data)


def test_plot_single_column(sample_dataframe):
    """Test the plot function with a single value column."""
    result_ax = area.plot(
        df=sample_dataframe,
        value_col="unit_spend",
        x_label="Transaction Date",
        y_label="Sales",
        title="Jeans Sales (Single Column)",
        x_col="transaction_date",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.collections) == 1


def test_plot_with_group_col(sample_dataframe):
    """Test the plot function with a group column (stacked area chart)."""
    result_ax = area.plot(
        df=sample_dataframe,
        value_col="unit_spend",
        x_label="Transaction Date",
        y_label="Sales",
        title="Sales by Category",
        x_col="transaction_date",
        group_col="category",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.collections) == sample_dataframe["category"].nunique()


def test_plot_multiple_columns(sample_dataframe):
    """Test the plot function with multiple columns as a stacked area chart."""
    rng = np.random.default_rng(42)
    sample_dataframe["additional_spend"] = rng.integers(1, 6, size=3 * PERIODS)
    value_cols = ["unit_spend", "additional_spend"]
    result_ax = area.plot(
        df=sample_dataframe,
        value_col=value_cols,
        x_label="Transaction Date",
        y_label="Sales",
        title="Sales by Product Category",
        x_col="transaction_date",
        alpha=0.9,
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.collections) == len(value_cols)


def test_plot_multiple_columns_with_group_col(sample_dataframe):
    """Test the plot function when using multiple columns along with a group column."""
    rng = np.random.default_rng(42)
    sample_dataframe["additional_spend"] = rng.integers(1, 6, size=3 * PERIODS)
    with pytest.raises(ValueError, match=r"Cannot use both a list for `value_col` and a `group_col`. Choose one."):
        area.plot(
            df=sample_dataframe,
            value_col=["unit_spend", "additional_spend", "Shoes"],
            x_label="Transaction Date",
            y_label="Sales",
            title="Sales by Product Category",
            x_col="transaction_date",
            group_col="category",
        )


def test_plot_adds_source_text(sample_dataframe):
    """The area plot renders source_text as a figure-level text element."""
    source_text = "Source: Test Data"

    result_ax = area.plot(
        df=sample_dataframe,
        value_col="unit_spend",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Source Text",
        x_col="transaction_date",
        source_text=source_text,
    )

    rendered = [t.get_text() for t in result_ax.figure.texts]
    assert source_text in rendered


def test_plot_single_column_series(sample_dataframe):
    """Test the plot function with a single value as a Pandas Series."""
    result_ax = area.plot(
        df=sample_dataframe["unit_spend"],
        value_col="unit_spend",
        x_label="Transaction Date",
        y_label="Sales",
        title="Product Sales (Series)",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.collections) == 1


@pytest.mark.parametrize("invalid_value", ["endofline", "box ", "", "BOX"])
def test_plot_rejects_invalid_legend_style(sample_dataframe, invalid_value):
    """area.plot raises ValueError for legend_style values outside the documented set."""
    with pytest.raises(ValueError, match="legend_style"):
        area.plot(
            df=sample_dataframe,
            value_col="unit_spend",
            x_col="transaction_date",
            legend_style=invalid_value,
        )
