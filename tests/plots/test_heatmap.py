"""Tests for the heatmap plot module."""

import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.patches import FancyBboxPatch

from openretailscience.options import option_context
from openretailscience.plots import heatmap


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_heatmap_dataframe():
    """Generates a sample retail foot-traffic DataFrame for heatmap testing (day-of-week by store)."""
    rng = np.random.default_rng(42)
    data = np.round(rng.uniform(0, 1, size=(4, 4)), 2)
    return pd.DataFrame(
        data,
        columns=["Store_North", "Store_South", "Store_East", "Store_West"],
        index=["Mon", "Tue", "Wed", "Thu"],
    )


def test_plot_basic_heatmap(sample_heatmap_dataframe):
    """Test basic heatmap creation with required parameters."""
    result_ax = heatmap.plot(
        df=sample_heatmap_dataframe,
        cbar_label="Test Values",
        title="Basic Heatmap Test",
    )

    assert isinstance(result_ax, Axes)
    # Verify colorbar exists (plot + colorbar = 2 axes)
    expected_axis_count = 2
    assert len(result_ax.figure.axes) == expected_axis_count
    # Verify text elements for each cell
    assert len(result_ax.texts) == sample_heatmap_dataframe.size


def test_plot_with_source_text(sample_heatmap_dataframe):
    """The heatmap renders source_text as a figure-level text element."""
    source_text = "Source: Test Data"
    result_ax = heatmap.plot(
        df=sample_heatmap_dataframe,
        cbar_label="Test Values",
        source_text=source_text,
    )

    rendered = [t.get_text() for t in result_ax.figure.texts]
    assert source_text in rendered


def test_plot_with_figsize(sample_heatmap_dataframe):
    """Test heatmap with custom figure size."""
    width, height = 12, 8
    result_ax = heatmap.plot(
        df=sample_heatmap_dataframe,
        cbar_label="Test Values",
        figsize=(width, height),
    )

    assert result_ax.figure.get_size_inches()[0] == width
    assert result_ax.figure.get_size_inches()[1] == height


@pytest.mark.parametrize("shape", [(1, 5), (5, 1), (1, 1)])
def test_plot_edge_case_dimensions(shape):
    """Test heatmap with minimal DataFrame dimensions: single row, single column, and single cell."""
    rng = np.random.default_rng(42)
    rows, cols = shape
    data = np.round(rng.uniform(0, 1, size=shape), 2)
    df = pd.DataFrame(
        data,
        columns=[f"Store_{i}" for i in range(cols)],
        index=[f"Week_{i}" for i in range(rows)],
    )

    result_ax = heatmap.plot(df=df, cbar_label="Value")

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_xticks()) == cols
    assert len(result_ax.get_yticks()) == rows
    assert len(result_ax.texts) == df.size


def test_plot_text_values_accuracy(sample_heatmap_dataframe):
    """Test that displayed text values match DataFrame values."""
    result_ax = heatmap.plot(df=sample_heatmap_dataframe, cbar_label="Test Values")

    texts = result_ax.texts
    for i, text in enumerate(texts):
        row, col = divmod(i, sample_heatmap_dataframe.shape[1])
        expected_value = sample_heatmap_dataframe.iloc[row, col]
        displayed_value = float(text.get_text())
        value_tolerance = 0.01
        assert abs(displayed_value - expected_value) < value_tolerance


@pytest.mark.parametrize("data_range", [(0, 1), (-1, 1), (100, 200)])
def test_plot_different_value_ranges(data_range):
    """Test heatmap with different data value ranges."""
    rng = np.random.default_rng(42)
    min_val, max_val = data_range
    data = np.round(rng.uniform(min_val, max_val, size=(3, 3)), 2)
    df = pd.DataFrame(
        data,
        columns=[f"Store_{i}" for i in range(3)],
        index=[f"Week_{i}" for i in range(3)],
    )

    result_ax = heatmap.plot(df=df, cbar_label="Value")

    # Verify all displayed values are within expected range
    texts = result_ax.texts
    text_values = [float(text.get_text()) for text in texts]
    assert all(min_val <= val <= max_val for val in text_values)


def test_plot_skips_nan_cells():
    """NaN cells render as empty (no patch, no text)."""
    df = pd.DataFrame(
        [[1.0, 2.0], [np.nan, 4.0]],
        columns=["Morning", "Afternoon"],
        index=["Mon", "Tue"],
    )

    result_ax = heatmap.plot(df=df, cbar_label="Foot traffic")

    rounded_cells = [patch for patch in result_ax.patches if isinstance(patch, FancyBboxPatch)]
    non_nan_count = int(df.notna().to_numpy().sum())
    assert len(rounded_cells) == non_nan_count
    assert all("nan" not in text.get_text().lower() for text in result_ax.texts)


@pytest.mark.parametrize("label_length", ["short", "very_long_column_name_that_exceeds_threshold"])
def test_plot_label_rotation(label_length):
    """Test automatic label rotation based on label length."""
    cols = [label_length] * 3
    data = np.ones((2, 3))
    df = pd.DataFrame(data, columns=cols, index=["Mon", "Tue"])

    result_ax = heatmap.plot(df=df, cbar_label="Value")

    x_tick_labels = result_ax.get_xticklabels()
    assert len(x_tick_labels) == len(cols)
    rotation = x_tick_labels[0].get_rotation()
    threshold = 10
    expected_rotation = 45
    if len(label_length) > threshold:
        assert rotation == expected_rotation, "Long labels should be rotated"
    else:
        assert rotation == 0, "Short labels should not be rotated"


def test_plot_label_alignment():
    """Test horizontal alignment of x-axis labels based on rotation."""
    short_cols = ["Mon", "Tue", "Wed"]
    data = np.ones((2, 3))
    df_short = pd.DataFrame(data, columns=short_cols, index=["Week_1", "Week_2"])

    result_ax = heatmap.plot(df=df_short, cbar_label="Value")
    x_tick_labels = result_ax.get_xticklabels()

    assert len(x_tick_labels) == len(short_cols)
    alignment = x_tick_labels[0].get_horizontalalignment()
    assert alignment == "center", "Short labels should be center-aligned"

    # Test with long labels (rotated)
    long_cols = ["very_long_column_name_1", "very_long_column_name_2", "very_long_column_name_3"]
    df_long = pd.DataFrame(data, columns=long_cols, index=["Week_1", "Week_2"])

    result_ax = heatmap.plot(df=df_long, cbar_label="Value")
    x_tick_labels = result_ax.get_xticklabels()

    assert len(x_tick_labels) == len(long_cols)
    alignment = x_tick_labels[0].get_horizontalalignment()
    assert alignment == "right", "Long rotated labels should be right-aligned"


def test_colorbar_label_set(sample_heatmap_dataframe):
    """Verify colorbar label is set correctly."""
    label = "Test Colorbar Label"
    result_ax = heatmap.plot(df=sample_heatmap_dataframe, cbar_label=label)

    # Get colorbar axes (should be the last axes in figure)
    cbar_ax = result_ax.figure.axes[-1]
    # Check ylabel
    ylabel = cbar_ax.get_ylabel()
    assert ylabel == label, f"Expected colorbar label '{label}', got '{ylabel}'"


def test_colorbar_label_uses_label_font_family(sample_heatmap_dataframe):
    """Colorbar label tracks plot.font.label_font, not matplotlib's default."""
    with option_context("plot.font.label_font", "poppins_bold"):
        result_ax = heatmap.plot(df=sample_heatmap_dataframe, cbar_label="Foot traffic")

        cbar_ax = result_ax.figure.axes[-1]
        font_file = cbar_ax.yaxis.label.get_fontproperties().get_file()
        assert font_file is not None
        assert "Poppins-Bold.ttf" in font_file


def test_axis_labels_applied(sample_heatmap_dataframe):
    """Verify axis labels and title are applied correctly."""
    result_ax = heatmap.plot(
        df=sample_heatmap_dataframe,
        cbar_label="Value",
        x_label="X Axis Label",
        y_label="Y Axis Label",
        title="Test Title",
    )

    assert result_ax.get_xlabel() == "X Axis Label"
    assert result_ax.get_ylabel() == "Y Axis Label"
    title_texts = [t for t in result_ax.figure.texts if t.get_text() == "Test Title"]
    assert len(title_texts) == 1


def test_plot_cells_are_rounded_patches(sample_heatmap_dataframe):
    """Each cell renders as a rounded FancyBboxPatch."""
    result_ax = heatmap.plot(df=sample_heatmap_dataframe, cbar_label="Value")

    rounded_cells = [patch for patch in result_ax.patches if isinstance(patch, FancyBboxPatch)]
    assert len(rounded_cells) == sample_heatmap_dataframe.size


def test_plot_hides_axis_spines(sample_heatmap_dataframe):
    """Heatmap cells and white separators define the boundaries; axis spines must be hidden."""
    result_ax = heatmap.plot(df=sample_heatmap_dataframe, cbar_label="Value")

    visible_spines = [name for name, spine in result_ax.spines.items() if spine.get_visible()]
    assert visible_spines == [], f"Heatmap should not show any axis spines, got: {visible_spines}"


def test_discrete_colorbar_separator_uses_background_color(sample_heatmap_dataframe):
    """The discrete-colorbar swatch separator must track plot.style.background_color.

    Hardcoding "white" produces visible bands when users configure a non-white
    background; the separator's purpose is to mimic the surrounding canvas.
    """
    custom_background = "lightgray"
    with option_context("plot.style.background_color", custom_background):
        result_ax = heatmap.plot(
            df=sample_heatmap_dataframe,
            cbar_label="Value",
            colormap_style="discrete",
        )

    cbar_ax = result_ax.figure.axes[-1]
    quad_meshes = [c for c in cbar_ax.collections if isinstance(c, QuadMesh)]
    assert len(quad_meshes) == 1, "Expected exactly one QuadMesh (colorbar solids) on the cbar axes"
    edge_rgba = mcolors.to_rgba(custom_background)
    actual_edges = quad_meshes[0].get_edgecolor()
    assert len(actual_edges) == 1, "Expected a single shared colorbar swatch edge color"
    for edge in actual_edges:
        assert tuple(edge) == edge_rgba


def test_discrete_colorbar_ignores_cbar_format(sample_heatmap_dataframe):
    """Discrete mode must show fixed Low/High anchors regardless of cbar_format (docstring contract).

    The colorbar's tick labels are the documented signal; passing a distinctive format like
    ``"{x:.2%}"`` would yield "0.00%"/"100.00%" if applied, so this guards against any future
    ordering change that lets ``format=`` leak past ``set_ticklabels``.
    """
    result_ax = heatmap.plot(
        df=sample_heatmap_dataframe,
        cbar_label="Foot traffic",
        colormap_style="discrete",
        cbar_format="{x:.2%}",
    )

    cbar_ax = result_ax.figure.axes[-1]
    tick_labels = [t.get_text() for t in cbar_ax.get_yticklabels()]
    assert tick_labels == ["Low", "High"]


def test_text_color_contrast():
    """Verify text color switches based on cell background intensity."""
    # Create data with known light and dark cells
    data = np.array([[0.0, 1.0]])  # Dark cell, light cell
    df = pd.DataFrame(data, columns=["Morning", "Afternoon"], index=["Mon"])

    result_ax = heatmap.plot(df=df, cbar_label="Value")

    texts = result_ax.texts
    expected_text_count = 2
    assert len(texts) == expected_text_count

    # Get text colors. texts[0] is the low-value (0.0) cell, texts[1] is the
    # high-value (1.0) cell, matching the row-major iteration in heatmap.plot.
    text_0_color = texts[0].get_color()
    text_1_color = texts[1].get_color()

    # Low-value cells get black text, high-value cells get white text so the
    # data labels remain readable against the sequential colormap's two ends.
    assert text_0_color == "black", f"Low-value cell should have black text, got {text_0_color}"
    assert text_1_color == "white", f"High-value cell should have white text, got {text_1_color}"


def test_plot_raises_on_empty_dataframe():
    """Empty input should raise a descriptive ValueError at the public boundary."""
    with pytest.raises(ValueError, match="empty DataFrame"):
        heatmap.plot(df=pd.DataFrame(), cbar_label="Foot traffic")


def test_plot_raises_on_all_nan_data():
    """An all-NaN frame has no finite vmin/vmax to normalise against; raise rather than leak NaN."""
    df = pd.DataFrame(
        np.full((3, 3), np.nan),
        columns=["Store_North", "Store_South", "Store_East"],
        index=["Mon", "Tue", "Wed"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="no finite values"):
            heatmap.plot(df=df, cbar_label="Foot traffic")
    nan_slice_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning) and "All-NaN" in str(w.message)]
    assert nan_slice_warnings == [], (
        f"Expected no 'All-NaN slice' RuntimeWarning before the ValueError; got {nan_slice_warnings}"
    )


def test_plot_handles_constant_value_dataframe():
    """Constant-value frames must render at the midpoint colour with no warnings.

    A constant-value frame has no magnitude to colour-encode; every cell and every discrete
    cbar bin must render at the same midpoint colour, with no matplotlib UserWarning.
    """
    df = pd.DataFrame(
        [[100.0, 100.0], [100.0, 100.0]],
        columns=["Store_North", "Store_South"],
        index=["Mon", "Tue"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result_ax = heatmap.plot(df=df, cbar_label="Retention", colormap_style="discrete")

    rounded_cells = [patch for patch in result_ax.patches if isinstance(patch, FancyBboxPatch)]
    assert len(rounded_cells) == df.size

    cell_facecolors = [tuple(patch.get_facecolor()) for patch in rounded_cells]
    assert len(set(cell_facecolors)) == 1, f"Expected uniform cell colour, got {set(cell_facecolors)}"

    cbar_ax = result_ax.figure.axes[-1]
    quad_meshes = [c for c in cbar_ax.collections if isinstance(c, QuadMesh)]
    assert len(quad_meshes) == 1
    # QuadMesh swatches are coloured via cmap(norm(array)), not get_facecolor() — the latter
    # returns the QuadMesh's unset background, not the rendered bin colours.
    cbar_swatches = quad_meshes[0].to_rgba(quad_meshes[0].get_array()).reshape(-1, 4)
    cbar_facecolors = {tuple(rgba) for rgba in cbar_swatches}
    assert len(cbar_facecolors) == 1, f"Expected uniform cbar swatch colour, got {cbar_facecolors}"
    assert cbar_facecolors == {cell_facecolors[0]}, "Cbar swatch colour must match cell colour (cmap midpoint)"
