"""Tests for the color module in the style package."""

import pytest

from openretailscience.options import get_option, option_context
from openretailscience.plots.styles.colors import (
    get_named_color,
    get_plot_colors,
    get_sequential_cmap,
)


class TestGetPlotColors:
    """Test get_plot_colors() function behavior."""

    def test_get_plot_colors_single_series_uses_primary(self):
        """Single-series plots use the brand primary color, not the mono palette's first stop."""
        colors = get_plot_colors(1)
        assert len(colors) == 1
        assert colors[0] == get_option("plot.color.primary")

    @pytest.mark.parametrize("num_series", [2, 3, 4, 10])
    def test_get_plot_colors_multi_series_uses_multi_palette(self, num_series):
        """Multi-series plots draw from the multi-color palette so each series gets a distinguishable hue."""
        colors = get_plot_colors(num_series)
        assert len(colors) == num_series
        multi_palette = get_option("plot.color.multi_color_palette")
        assert all(c in multi_palette for c in colors)

    def test_get_plot_colors_cycling_behavior(self):
        """get_plot_colors() cycles through the multi-color palette when num_series exceeds its length."""
        five_series = 5
        two_color_palette = ["#1e40af", "#dc2626"]  # blue-700, red-600
        with option_context("plot.color.multi_color_palette", two_color_palette):
            colors = get_plot_colors(five_series)

        assert len(colors) == five_series
        expected = [two_color_palette[i % len(two_color_palette)] for i in range(five_series)]
        assert colors == expected


class TestGetNamedColor:
    """Test get_named_color() function behavior."""

    def test_get_named_color_custom_values(self):
        """Test get_named_color() uses custom configured values."""
        custom_positive = "#16a34a"  # green-600
        custom_negative = "#dc2626"  # red-600

        with option_context(
            "plot.color.positive",
            custom_positive,
            "plot.color.negative",
            custom_negative,
        ):
            assert get_named_color("positive") == custom_positive
            assert get_named_color("negative") == custom_negative

    def test_get_named_color_invalid_type(self):
        """Test get_named_color() raises error for invalid color type."""
        with pytest.raises(ValueError):
            get_named_color("invalid_color_type")


class TestGetSequentialCmap:
    """Test get_sequential_cmap() function behavior."""

    @pytest.mark.parametrize(
        ("cmap_config", "expected_type"),
        [
            ("green", "ListedColormap"),  # Tailwind color name
            ("blue", "ListedColormap"),  # Tailwind color name
            ("viridis", "ListedColormap"),  # Matplotlib colormap (ListedColormap in this version)
            ("Greens", "LinearSegmentedColormap"),  # Matplotlib colormap
        ],
    )
    def test_get_sequential_cmap_supports_both_tailwind_and_matplotlib(self, cmap_config, expected_type):
        """Test get_sequential_cmap() handles both Tailwind and matplotlib colormap names."""
        with option_context("plot.color.sequential", cmap_config):
            cmap = get_sequential_cmap()
            assert type(cmap).__name__ == expected_type
