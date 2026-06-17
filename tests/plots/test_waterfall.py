"""Tests for the waterfall plot module."""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from openretailscience.options import option_context
from openretailscience.plots.waterfall import format_data_labels, plot


class TestWaterfallPlot:
    """Tests for the waterfall_plot function."""

    def teardown_method(self):
        """Clean up after each test method."""
        plt.close("all")

    @pytest.fixture
    def test_data(self):
        """Return a list of amounts."""
        return [100, -50, 30, -10], ["Start", "Decrease", "Increase", "End"]

    def test_generates_waterfall_plot_with_default_parameters(self, test_data):
        """Test that the function generates a waterfall plot with default parameters."""
        amounts, labels = test_data
        result_ax = plot(amounts, labels)

        assert isinstance(result_ax, plt.Axes)
        assert len(result_ax.patches) == len(labels)

    def test_raises_value_error_for_mismatched_lengths(self, test_data):
        """Test that the function raises a ValueError when the lengths of amounts and labels are mismatched."""
        amounts, labels = test_data
        # Remove a value from amounts
        amounts.pop()

        with pytest.raises(
            ValueError,
            match="The lengths of amounts and labels must be the same",
        ):
            plot(amounts, labels)

    def test_generates_waterfall_plot_with_source_text(self, test_data):
        """Test that the function generates a waterfall plot with source text."""
        amounts, labels = test_data
        source_text = "Data source: Company XYZ"

        result_ax = plot(amounts, labels, source_text=source_text)

        assert isinstance(result_ax, plt.Axes)
        assert len(result_ax.patches) == len(labels)

        # Check if the source text is added correctly
        source_texts = [text for text in result_ax.figure.texts if text.get_text() == source_text]
        assert len(source_texts) == 1

    def test_plot_colors_assigned_correctly(self, test_data):
        """Configured positive/negative colors should flow through to bar facecolors."""
        amounts, labels = test_data
        positive_color = "#1f77b4"
        negative_color = "#d62728"

        with option_context("plot.color.positive", positive_color, "plot.color.negative", negative_color):
            result_ax = plot(amounts, labels)

        colors = [to_hex(patch.get_facecolor()) for patch in result_ax.patches]

        assert len(result_ax.patches) == len(labels)
        assert colors == [
            positive_color,
            negative_color,
            positive_color,
            negative_color,
        ]

    def test_net_bar_uses_configured_difference_color(self, test_data):
        """Configured difference color should flow through to the net bar's facecolor."""
        amounts, labels = test_data
        difference_color = "#9467bd"

        with option_context("plot.color.difference", difference_color):
            result_ax = plot(amounts, labels, display_net_bar=True)

        last_bar_color = to_hex(result_ax.patches[-1].get_facecolor())

        assert len(result_ax.patches) == len(labels) + 1
        assert last_bar_color == difference_color

    def test_generates_waterfall_plot_with_zero_amounts_removed(self, test_data):
        """Test that the function generates a waterfall plot with zero amounts removed."""
        amounts, labels = test_data
        # Set the first amount to zero
        amounts[0] = 0

        result_ax = plot(amounts, labels)

        non_zero_amounts = len([amount for amount in amounts if amount != 0])

        assert isinstance(result_ax, plt.Axes)
        assert len(result_ax.patches) == non_zero_amounts

    # Raises a ValueError for an invalid data label format
    def test_raises_value_error_for_invalid_data_label_format(self, test_data):
        """Test that the function raises a ValueError for an invalid data label format."""
        amounts, labels = test_data

        with pytest.raises(ValueError):
            plot(amounts, labels, data_label_format="invalid_format")

    def test_bar_labels_use_data_label_size(self, test_data):
        """Bar-end data labels should track plot.font.data_label_size, not label_size."""
        amounts, labels = test_data
        custom_data_label_size = 17.0

        with option_context("plot.font.data_label_size", custom_data_label_size):
            result_ax = plot(amounts, labels, data_label_format="absolute")

        bar_label_texts = [t for t in result_ax.texts if len(t.get_text()) > 0]
        assert len(bar_label_texts) == len(amounts)
        for text in bar_label_texts:
            assert text.get_fontsize() == custom_data_label_size

    def test_data_labels_stay_within_axes_when_a_bar_dominates(self):
        """Edge labels on extreme bars must not spill out of the plot area onto the tick/header bands.

        matplotlib bars carry sticky edges that pin the y-view to the bar extents, so a dominant
        negative bar leaves its edge label overlapping the x-axis tick labels (and the topmost
        positive label overlapping the header) unless the y-limits are grown to make room.
        """
        amounts = [66444, -284585, 26890]
        labels = ["New", "Lapsed", "Repeating"]

        result_ax = plot(amounts, labels, data_label_format="both", display_net_bar=True)
        fig = result_ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        axes_box = result_ax.get_window_extent(renderer=renderer)
        data_labels = [text for text in result_ax.texts if len(text.get_text()) > 0]
        assert len(data_labels) == len(amounts) + 1  # one per input bar plus the net bar

        for text in data_labels:
            label_box = text.get_window_extent(renderer=renderer)
            assert label_box.y0 >= axes_box.y0
            assert label_box.y1 <= axes_box.y1

    @pytest.mark.parametrize("label_format", ["percentage", "both"])
    def test_zero_total_change_does_not_raise(self, label_format):
        """Test that offsetting amounts (zero total) do not raise ZeroDivisionError."""
        amounts = [500.0, -300.0, -200.0]
        labels = ["Revenue", "Cost of Goods", "Operating Expenses"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result_ax = plot(amounts, labels, data_label_format=label_format)

        assert isinstance(result_ax, plt.Axes)
        assert any("Total change is zero" in str(w.message) for w in caught)


class TestFormatDataLabels:
    """Tests for the format_data_labels function."""

    @pytest.fixture
    def sample_amounts(self):
        """Return a sample Series of retail transaction amounts."""
        return pd.Series([500.0, -300.0, 200.0])

    def test_percentage_format_with_zero_total_returns_empty_strings(self):
        """Test that percentage format returns empty strings when total change is zero."""
        amounts = pd.Series([500.0, -300.0, -200.0])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = format_data_labels(amounts, total_change=0, label_format="percentage", decimals=0)

        assert result == ["", "", ""]
        assert len(caught) == 1
        assert "Total change is zero" in str(caught[0].message)

    def test_both_format_with_zero_total_returns_absolute_values_only(self):
        """Test that 'both' format returns only absolute values when total change is zero."""
        amounts = pd.Series([500.0, -300.0, -200.0])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = format_data_labels(amounts, total_change=0, label_format="both", decimals=0)

        assert result == ["500", "-300", "-200"]
        assert len(caught) == 1
        assert "Total change is zero" in str(caught[0].message)

    def test_absolute_format_unaffected_by_zero_total(self):
        """Test that absolute format works normally regardless of total change."""
        amounts = pd.Series([500.0, -300.0, -200.0])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = format_data_labels(amounts, total_change=0, label_format="absolute", decimals=0)

        assert result == ["500", "-300", "-200"]
        # No warning should be emitted for absolute format
        assert not any("Total change is zero" in str(w.message) for w in caught)

    def test_percentage_format_with_nonzero_total(self, sample_amounts):
        """Test that percentage format calculates correctly with a valid total."""
        total = sample_amounts.sum()  # 400.0
        result = format_data_labels(sample_amounts, total_change=total, label_format="percentage", decimals=0)

        assert result[0] == "125%"
        assert result[1] == "-75%"
        assert result[2] == "50%"

    def test_both_format_with_nonzero_total(self, sample_amounts):
        """Test that 'both' format includes absolute value and percentage with a valid total."""
        total = sample_amounts.sum()  # 400.0
        result = format_data_labels(sample_amounts, total_change=total, label_format="both", decimals=0)

        assert list(result) == ["500 (125%)", "-300 (-75%)", "200 (50%)"]
