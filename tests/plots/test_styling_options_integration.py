"""Integration tests for styling options with actual plots."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.text import Text

from openretailscience.options import option_context
from openretailscience.plots import scatter

TEST_TITLE_SIZE = 24.0
TEST_LABEL_SIZE = 14.0
TEST_FONT_SIZE_SMALL = 13.0
TEST_FONT_SIZE_MEDIUM = 22.0
TEST_DATA_LABEL_SIZE = 12.0
TEST_LEGEND_SIZE = 7.0
TEST_X_LABEL_PAD = 20
TEST_Y_LABEL_PAD = 15
WHITESMOKE_RGBA = mcolors.to_rgba("whitesmoke")
LIGHTGRAY_RGBA = mcolors.to_rgba("lightgray")
TEST_GRID_ALPHA = 0.8


class TestStylingOptionsIntegration:
    """Tests for integration of styling options with actual plot functions."""

    @pytest.fixture(autouse=True)
    def cleanup_figures(self):
        """Clean up matplotlib figures after each test."""
        yield
        plt.close("all")

    @pytest.fixture
    def sample_dataframe(self):
        """Sample data for testing plots."""
        return pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 5, 3, 8, 7],
                "category": ["Bakery", "Bakery", "Dairy", "Dairy", "Bakery"],
                "labels": ["Point1", "Point2", "Point3", "Point4", "Point5"],
            },
        )

    def test_font_customization_with_context(self, sample_dataframe):
        """Test that font options affect plot appearance via context manager."""
        # Test with custom font settings using available bundled fonts
        with option_context(
            "plot.font.title_font",
            "poppins_bold",
            "plot.font.title_size",
            TEST_TITLE_SIZE,
            "plot.font.label_size",
            TEST_LABEL_SIZE,
        ):
            ax = scatter.plot(
                df=sample_dataframe,
                value_col="y",
                x_col="x",
                title="Custom Font Test",
                x_label="X Axis",
                y_label="Y Axis",
            )

            # Title is now a figure-level Text rendered by the chrome layout.
            title_texts = [t for t in ax.figure.texts if t.get_text() == "Custom Font Test"]
            assert len(title_texts) == 1
            assert title_texts[0].get_fontsize() == TEST_TITLE_SIZE
            title_font_file = title_texts[0].get_fontproperties().get_file()
            assert isinstance(title_font_file, str)
            assert "Poppins-Bold.ttf" in title_font_file

            # Verify label properties
            assert ax.get_xlabel() == "X Axis"
            assert ax.get_ylabel() == "Y Axis"
            assert ax.xaxis.label.get_fontsize() == TEST_LABEL_SIZE
            assert ax.yaxis.label.get_fontsize() == TEST_LABEL_SIZE

    def test_spacing_customization_with_context(self, sample_dataframe):
        """Test that spacing options affect plot layout via context manager."""
        with option_context(
            "plot.spacing.x_label_pad",
            TEST_X_LABEL_PAD,
            "plot.spacing.y_label_pad",
            TEST_Y_LABEL_PAD,
        ):
            ax = scatter.plot(
                df=sample_dataframe,
                value_col="y",
                x_col="x",
                title="Custom Spacing Test",
                x_label="X Axis",
                y_label="Y Axis",
            )

            assert ax.xaxis.labelpad == TEST_X_LABEL_PAD
            assert ax.yaxis.labelpad == TEST_Y_LABEL_PAD

    def test_style_customization_with_context(self, sample_dataframe):
        """Test that style options affect plot appearance via context manager."""
        with option_context(
            "plot.style.background_color",
            "lightgray",
            "plot.style.grid_alpha",
            TEST_GRID_ALPHA,
            "plot.style.show_top_spine",
            True,
            "plot.style.show_right_spine",
            True,
        ):
            ax = scatter.plot(df=sample_dataframe, value_col="y", x_col="x", title="Custom Style Test")

            # Verify background color
            assert ax.get_facecolor() == LIGHTGRAY_RGBA

            # Verify grid alpha
            for line in ax.xaxis.get_gridlines():
                assert line.get_alpha() == TEST_GRID_ALPHA

            # Verify spine visibility
            assert ax.spines["top"].get_visible() is True
            assert ax.spines["right"].get_visible() is True
            assert ax.spines["bottom"].get_visible() is True
            assert ax.spines["left"].get_visible() is True

    def test_legend_size_customization(self, sample_dataframe):
        """Test that plot.font.legend_size controls legend entry font size."""
        with option_context("plot.font.legend_size", TEST_LEGEND_SIZE):
            ax = scatter.plot(
                df=sample_dataframe,
                value_col="y",
                x_col="x",
                group_col="category",
                title="Custom Legend Size Test",
                legend_title="Category",
            )

            legend = ax.get_legend()
            assert legend is not None, "Expected a legend on grouped scatter plot"
            legend_texts = legend.get_texts()
            expected_categories = sample_dataframe["category"].nunique()
            assert len(legend_texts) == expected_categories
            for text in legend_texts:
                assert text.get_fontsize() == TEST_LEGEND_SIZE

    def test_legend_font_customization(self, sample_dataframe):
        """Test that plot.font.legend_font controls legend title and entry font family."""
        with option_context("plot.font.legend_font", "poppins_bold"):
            ax = scatter.plot(
                df=sample_dataframe,
                value_col="y",
                x_col="x",
                group_col="category",
                title="Custom Legend Font Test",
                legend_title="Category",
            )

            legend = ax.get_legend()
            assert legend is not None, "Expected a legend on grouped scatter plot"

            legend_title_font_file = legend.get_title().get_fontproperties().get_file()
            assert isinstance(legend_title_font_file, str)
            assert "Poppins-Bold.ttf" in legend_title_font_file

            legend_texts = legend.get_texts()
            expected_categories = sample_dataframe["category"].nunique()
            assert len(legend_texts) == expected_categories
            for text in legend_texts:
                text_font_file = text.get_fontproperties().get_file()
                assert isinstance(text_font_file, str)
                assert "Poppins-Bold.ttf" in text_font_file

    def test_label_font_customization(self, sample_dataframe):
        """Test that data label fonts can be customized."""
        with option_context("plot.font.data_label_size", TEST_DATA_LABEL_SIZE):
            ax = scatter.plot(
                df=sample_dataframe,
                value_col="y",
                x_col="x",
                label_col="labels",
                title="Custom Label Font Test",
            )

            # Find data label text objects and verify their font size
            label_values = set(sample_dataframe["labels"].to_numpy())
            label_texts = [
                child for child in ax.get_children() if isinstance(child, Text) and child.get_text() in label_values
            ]

            assert len(label_texts) == len(label_values), "Not all data labels were found on the plot"
            for text in label_texts:
                assert text.get_fontsize() == TEST_DATA_LABEL_SIZE

    def test_multiple_option_changes(self, sample_dataframe):
        """Test that multiple styling options work together."""
        with option_context(
            # Font options
            "plot.font.title_size",
            TEST_FONT_SIZE_MEDIUM,
            "plot.font.label_size",
            TEST_FONT_SIZE_SMALL,
            # Style options
            "plot.style.show_top_spine",
            True,
            "plot.style.background_color",
            "whitesmoke",
        ):
            ax = scatter.plot(
                df=sample_dataframe,
                value_col="y",
                x_col="x",
                group_col="category",
                title="Multi-Option Test",
                x_label="X Values",
                y_label="Y Values",
            )

            # Verify various properties
            title_texts = [t for t in ax.figure.texts if t.get_text() == "Multi-Option Test"]
            assert len(title_texts) == 1
            assert title_texts[0].get_fontsize() == TEST_FONT_SIZE_MEDIUM
            assert ax.get_xlabel() == "X Values"
            assert ax.get_ylabel() == "Y Values"
            assert ax.xaxis.label.get_fontsize() == TEST_FONT_SIZE_SMALL
            assert ax.yaxis.label.get_fontsize() == TEST_FONT_SIZE_SMALL

            # Verify style changes
            assert ax.spines["top"].get_visible() is True
            assert ax.get_facecolor() == WHITESMOKE_RGBA

    def test_default_behavior_unchanged(self, sample_dataframe):
        """Test that default behavior works when no options are changed."""
        ax = scatter.plot(
            df=sample_dataframe,
            value_col="y",
            x_col="x",
            title="Default Behavior Test",
            x_label="X Axis",
            y_label="Y Axis",
        )

        # Default behavior should still work
        title_texts = [t for t in ax.figure.texts if t.get_text() == "Default Behavior Test"]
        assert len(title_texts) == 1
        assert ax.get_xlabel() == "X Axis"
        assert ax.get_ylabel() == "Y Axis"

        # Default spines configuration
        assert ax.spines["top"].get_visible() is False
        assert ax.spines["right"].get_visible() is False
        assert ax.spines["bottom"].get_visible() is True
        assert ax.spines["left"].get_visible() is True

        # Default background should be white
        assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)  # white in RGBA
