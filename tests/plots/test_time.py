"""Tests for the time_plot function in the openretailscience.plots.time module.

This module contains unit tests that validate the behavior of the time_plot
function. The tests cover different scenarios, such as plotting with
default parameters, custom titles, group columns, and handling invalid
periods. Additionally, the tests ensure proper handling of source text and
legend visibility.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from openretailscience.plots.time import plot


class TestTimePlot:
    """Tests for the time_plot function."""

    def teardown_method(self):
        """Clean up after each test method."""
        plt.close("all")

    @pytest.fixture
    def test_data(self):
        """Return a sample dataframe for plotting."""
        rng = np.random.default_rng(42)
        data = {
            "transaction_date": pd.date_range(start="2022-01-01", periods=10, freq="D"),
            "sales": rng.integers(100, 500, size=10),
            "category": ["Bakery", "Dairy"] * 5,
        }
        return pd.DataFrame(data)

    def test_generates_time_plot_with_default_parameters(self, test_data):
        """Test that the function generates a time plot with default parameters."""
        df = test_data
        result_ax = plot(df, value_col="sales")
        assert isinstance(result_ax, Axes)
        assert len(result_ax.lines) == 1  # Single time series line

    def test_generates_time_plot_with_group_col(self, test_data):
        """Test that the function generates a time plot with a group column."""
        df = test_data
        result_ax = plot(df, value_col="sales", group_col="category")

        assert isinstance(result_ax, Axes)
        expected_lines_count = 2
        assert len(result_ax.lines) == expected_lines_count

    def test_generates_time_plot_with_custom_title(self, test_data):
        """Test that the function generates a time plot with a custom title."""
        df = test_data
        custom_title = "Sales Over Time"
        result_ax = plot(df, value_col="sales", title=custom_title)

        assert isinstance(result_ax, Axes)
        title_texts = [t for t in result_ax.figure.texts if t.get_text() == custom_title]
        assert len(title_texts) == 1

    def test_raises_value_error_for_invalid_period(self, test_data):
        """Test that the function raises a ValueError for an invalid period."""
        df = test_data

        with pytest.raises(ValueError):
            plot(df, value_col="sales", period="invalid_period")

    def test_generates_time_plot_with_source_text(self, test_data):
        """Test that the function generates a time plot with source text."""
        df = test_data
        source_text = "Data source: Company XYZ"
        result_ax = plot(df, value_col="sales", source_text=source_text)

        assert isinstance(result_ax, Axes)
        assert len(result_ax.lines) == 1  # Still just one line for the data
        source_texts = [text for text in result_ax.figure.texts if text.get_text() == source_text]
        assert len(source_texts) == 1

    def test_plot_with_custom_x_and_y_labels(self, test_data):
        """Test that the function generates a time plot with custom x and y labels."""
        df = test_data
        result_ax = plot(df, value_col="sales", x_label="Date", y_label="Sales Amount")

        assert isinstance(result_ax, Axes)
        assert result_ax.get_xlabel() == "Date"
        assert result_ax.get_ylabel() == "Sales Amount"

    def test_legend_visibility(self, test_data):
        """Test that the function correctly shows or hides the legend based on the group_col."""
        df = test_data

        result_ax = plot(df, value_col="sales", group_col="category")
        legend_visible = result_ax.get_legend() is not None
        assert legend_visible, "Legend should be visible when group_col is provided."

    @pytest.mark.parametrize("invalid_value", ["endofline", "box ", "", "BOX"])
    def test_plot_rejects_invalid_legend_style(self, test_data, invalid_value):
        """time.plot raises ValueError for legend_style values outside the documented set."""
        with pytest.raises(ValueError, match="legend_style"):
            plot(test_data, value_col="sales", legend_style=invalid_value)
