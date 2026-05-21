"""Tests for the chart chrome layout engine."""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from openretailscience.options import PlotStyleHelper, get_option, option_context
from openretailscience.plots import heatmap, line
from openretailscience.plots.styles.styling_helpers import apply_chart_chrome, apply_legend, standard_graph_styles


@pytest.fixture
def fig_ax():
    """Provide a fresh figure and axes for each test."""
    fig, ax = plt.subplots(figsize=(10, 5))
    yield fig, ax
    plt.close(fig)


def _texts_with(fig, text: str) -> list:
    return [t for t in fig.texts if t.get_text() == text]


def _measure_chrome_spacing(figheight: float) -> dict[str, float]:
    """Render eyebrow/title/subtitle/source on a figure of the given height and return inter-element pixel gaps."""
    fig, ax = plt.subplots(figsize=(10, figheight))
    eyebrow = "Category index"
    title = "Premium categories over-index"
    subtitle = "Vs. mainstream during Q4"
    source = "Source: 2024 transactions"
    apply_chart_chrome(
        ax,
        eyebrow=eyebrow,
        title=title,
        subtitle=subtitle,
        source_text=source,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    eye = _texts_with(fig, eyebrow.upper())[0].get_window_extent(renderer=renderer)
    ttl = _texts_with(fig, title)[0].get_window_extent(renderer=renderer)
    sub = _texts_with(fig, subtitle)[0].get_window_extent(renderer=renderer)
    src = _texts_with(fig, source)[0].get_window_extent(renderer=renderer)
    axes_pos = ax.get_position()
    measurements = {
        "top_margin_px": fig.bbox.height - eye.y1,
        "eye_to_title_px": eye.y0 - ttl.y1,
        "title_to_sub_px": ttl.y0 - sub.y1,
        "sub_to_axes_px": sub.y0 - axes_pos.y1 * fig.bbox.height,
        "source_to_axes_px": axes_pos.y0 * fig.bbox.height - src.y1,
        "bottom_margin_px": src.y0,
    }
    plt.close(fig)
    return measurements


_CHROME_TEST_FIGHEIGHT_IN = 5.0


def _topmost_axes_content_fig_y(*, labeltop: bool) -> float:
    """Render a chart and return the figure-fraction y of the topmost axes content.

    With labels at the top this is the top edge of the highest visible top-tick-label;
    otherwise it is the spine top. tight_layout pins this position to the chrome's
    reserved header_bottom, so comparing it between the two cases isolates the extra
    headroom the chrome layout reserves for top-rendered tick labels.
    """
    fig, ax = plt.subplots(figsize=(10, _CHROME_TEST_FIGHEIGHT_IN))
    ax.tick_params(top=labeltop, bottom=not labeltop, labeltop=labeltop, labelbottom=not labeltop)
    apply_chart_chrome(ax, title="Title", subtitle="Subtitle")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    spine_top_fig = ax.get_position().y1
    top_label_tops_fig = [
        t.label2.get_window_extent(renderer=renderer).y1 / fig.bbox.height
        for t in [*ax.xaxis.get_major_ticks(), *ax.yaxis.get_major_ticks()]
        if t.label2.get_visible() and t.label2.get_text() != ""
    ]
    topmost_fig = max([spine_top_fig, *top_label_tops_fig])
    plt.close(fig)
    return topmost_fig


class TestApplyChartChrome:
    """Verify the chrome layout engine renders only the present elements."""

    def test_renders_all_chrome_elements(self, fig_ax):
        """All four chrome elements (eyebrow, title, subtitle, source) render as figure text."""
        fig, ax = fig_ax
        apply_chart_chrome(
            ax,
            eyebrow="Category index",
            title="Premium categories over-index",
            subtitle="Supporting copy",
            source_text="Source: demo",
        )

        # Eyebrow is rendered uppercase
        assert len(_texts_with(fig, "CATEGORY INDEX")) == 1
        assert len(_texts_with(fig, "Premium categories over-index")) == 1
        assert len(_texts_with(fig, "Supporting copy")) == 1
        assert len(_texts_with(fig, "Source: demo")) == 1

    def test_omits_absent_elements(self, fig_ax):
        """Only the chrome elements explicitly passed in are rendered."""
        fig, ax = fig_ax
        apply_chart_chrome(ax, title="Just a title")

        assert len(_texts_with(fig, "Just a title")) == 1
        # No eyebrow, subtitle, or source; figure should have only the title text
        assert len(fig.texts) == 1

    def test_reflows_axes_even_when_all_chrome_absent(self, fig_ax):
        """With no chrome elements, axes still reflow via tight_layout to reserve tick-label room."""
        fig, ax = fig_ax
        pos_before = ax.get_position().bounds
        apply_chart_chrome(ax)

        assert len(fig.texts) == 0
        assert len(fig.patches) == 0
        # Even with no chrome elements, apply_chart_chrome must still reflow the
        # axes via tight_layout to reserve room for tick labels; otherwise
        # long y-tick labels (e.g. horizontal bar categories) clip off the
        # left edge of the figure. The chrome rect's horizontal margins are
        # narrower than matplotlib's defaults, so the reflowed axes occupy a
        # wider band: left edge moves left, width grows.
        bounds_after = ax.get_position().bounds
        assert bounds_after[0] < pos_before[0], "Left edge should have moved left into the chrome rect"
        assert bounds_after[2] > pos_before[2], "Axes width should have grown to fill the chrome rect"

    def test_tab_drawn_when_header_present(self, fig_ax):
        """A header element triggers rendering of the tab patch in the configured color."""
        fig, ax = fig_ax
        apply_chart_chrome(ax, title="With tab")

        assert len(fig.patches) == 1
        assert to_hex(fig.patches[0].get_facecolor()) == get_option("plot.color.tab")

    def test_show_tab_option_false_suppresses_tab(self, fig_ax):
        """plot.style.show_tab=False prevents the tab patch from being drawn."""
        fig, ax = fig_ax
        with option_context("plot.style.show_tab", False):
            apply_chart_chrome(ax, title="No tab")

        assert len(fig.patches) == 0

    def test_tab_skipped_when_no_header(self, fig_ax):
        """The tab is only drawn when there's a header element to anchor against."""
        fig, ax = fig_ax
        apply_chart_chrome(ax, source_text="Source only")

        # Tab requires a header (eyebrow/title/subtitle) to anchor against.
        assert len(fig.patches) == 0
        assert len(_texts_with(fig, "Source only")) == 1

    def test_long_title_pushes_subtitle_below(self, fig_ax):
        """Subtitle stays below the title without overlap even when the title wraps to multiple lines."""
        fig, ax = fig_ax
        long_title = " ".join(["A very long title"] * 8)
        apply_chart_chrome(
            ax,
            title=long_title,
            subtitle="Subtitle below",
        )

        title_text = fig.texts[0]
        subtitle_text = fig.texts[1]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title_bottom = title_text.get_window_extent(renderer=renderer).y0
        subtitle_top = subtitle_text.get_window_extent(renderer=renderer).y1
        # Subtitle's top must sit below the title's bottom (no overlap).
        assert subtitle_top <= title_bottom

    def test_wrap_is_frozen_so_redraws_do_not_change_line_count(self):
        """Chrome must bake matplotlib's wrap into the text content with explicit newlines.

        matplotlib's ``wrap=True`` recomputes line breaks on every draw based on the
        current figure bbox. ``bbox_inches='tight'`` (used by jupyter notebook saves)
        triggers an extra draw cycle with a different bbox, which can change the
        line count. Chrome positions sibling elements relative to the title's
        measured bottom, so any post-layout re-wrap leaves an oversized gap.
        Freezing the wrap ensures the layout stays correct across redraws.
        """
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        try:
            long_title = "High-AOV stores convert fewer visits but earn the most revenue per day"
            apply_chart_chrome(ax, title=long_title)

            title_artist = fig.texts[0]
            # Wrap must be off so the text doesn't re-wrap on subsequent draws.
            assert title_artist.get_wrap() is False
            # The wrapped form is baked into the text content as explicit newlines.
            assert "\n" in title_artist.get_text()
            # All original words preserved, only whitespace replaced with newlines.
            assert title_artist.get_text().replace("\n", " ") == long_title
        finally:
            plt.close(fig)

    def test_chrome_spacing_is_absolute_across_figure_heights(self):
        """Vertical chrome spacing must stay at the same pixel distance regardless of figure height.

        Eyebrow / title / subtitle / source are sized in points, so the whitespace around
        them must also be absolute. A figure-relative gap stretches with figheight and breaks
        the visual relationship to the (fixed-size) text.
        """
        short = _measure_chrome_spacing(figheight=4)
        tall = _measure_chrome_spacing(figheight=8)

        tolerance_px = 1.0
        for key in short:
            assert abs(short[key] - tall[key]) <= tolerance_px, (
                f"{key} differs by {short[key] - tall[key]:.2f}px between figheights "
                f"(short={short[key]:.2f}, tall={tall[key]:.2f}); spacing is figure-relative, not absolute"
            )

    def test_chrome_uses_configured_colors(self, fig_ax):
        """Each chrome element pulls its color from its corresponding plot.color.* option."""
        fig, ax = fig_ax
        apply_chart_chrome(
            ax,
            eyebrow="Eyebrow text",
            title="Title text",
            subtitle="Subtitle text",
            source_text="Source text",
        )

        eyebrow_text = _texts_with(fig, "EYEBROW TEXT")[0]
        title_text = _texts_with(fig, "Title text")[0]
        subtitle_text = _texts_with(fig, "Subtitle text")[0]
        source_text = _texts_with(fig, "Source text")[0]
        # Color comes through from options, not hardcoded.
        assert eyebrow_text.get_color() == get_option("plot.color.eyebrow")
        assert title_text.get_color() == get_option("plot.color.title")
        assert subtitle_text.get_color() == get_option("plot.color.subtitle")
        assert source_text.get_color() == get_option("plot.color.source")

    def test_source_color_decoupled_from_eyebrow(self, fig_ax):
        """plot.color.source and plot.color.eyebrow are independent options."""
        fig, ax = fig_ax
        # Override the eyebrow color only. The source text must keep using
        # plot.color.source, proving the two are independent knobs.
        with option_context("plot.color.eyebrow", "#ff0000"):
            apply_chart_chrome(ax, eyebrow="Eyebrow", source_text="Source text")

            eyebrow_text = _texts_with(fig, "EYEBROW")[0]
            source_text = _texts_with(fig, "Source text")[0]
            assert eyebrow_text.get_color() == "#ff0000"
            assert source_text.get_color() == get_option("plot.color.source")
            assert source_text.get_color() != "#ff0000"

    def test_repeated_calls_replace_chrome_on_same_axes(self, fig_ax):
        """Re-rendering chrome on the same axes must not stack duplicate texts or tabs.

        Typical Jupyter flow: call plot() with one title, tweak, call plot() again with
        a revised title on the same axes. The second pass must leave a single set of
        chrome artists, not two overlapping copies.
        """
        fig, ax = fig_ax
        apply_chart_chrome(
            ax,
            eyebrow="Draft eyebrow",
            title="Draft title",
            subtitle="Draft subtitle",
            source_text="Source v1",
        )
        apply_chart_chrome(
            ax,
            eyebrow="Final eyebrow",
            title="Final title",
            subtitle="Final subtitle",
            source_text="Source v2",
        )

        assert len(_texts_with(fig, "DRAFT EYEBROW")) == 0
        assert len(_texts_with(fig, "Draft title")) == 0
        assert len(_texts_with(fig, "Draft subtitle")) == 0
        assert len(_texts_with(fig, "Source v1")) == 0

        assert len(_texts_with(fig, "FINAL EYEBROW")) == 1
        assert len(_texts_with(fig, "Final title")) == 1
        assert len(_texts_with(fig, "Final subtitle")) == 1
        assert len(_texts_with(fig, "Source v2")) == 1
        assert len(fig.patches) == 1

    def test_multi_axes_chrome_warns_once_on_second_axes(self):
        """Applying chrome to a second axes on the same figure warns once; same-axes repeats do not warn."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        try:
            with warnings.catch_warnings(record=True) as first_call:
                warnings.simplefilter("always")
                apply_chart_chrome(ax1, title="First axes")
            assert [w for w in first_call if "subplot-aware" in str(w.message)] == []

            with warnings.catch_warnings(record=True) as second_call:
                warnings.simplefilter("always")
                apply_chart_chrome(ax2, title="Second axes")
            second_warnings = [w for w in second_call if "subplot-aware" in str(w.message)]
            assert len(second_warnings) == 1

            with warnings.catch_warnings(record=True) as repeat_call:
                warnings.simplefilter("always")
                apply_chart_chrome(ax1, title="First axes again")
            assert [w for w in repeat_call if "subplot-aware" in str(w.message)] == []
        finally:
            plt.close(fig)

    def test_top_tick_labels_reserve_extra_header_gap(self):
        """Top-rendered tick labels (heatmap/cohort) reserve one tick-label line-height of extra headroom.

        The chrome layout pushes the topmost axes content down by ``tick_size * 1.2 / 72``
        inches when top tick labels are present, so subtitle-to-content whitespace doesn't
        feel cramped relative to the bottom-labels case.
        """
        with_top = _topmost_axes_content_fig_y(labeltop=True)
        without_top = _topmost_axes_content_fig_y(labeltop=False)

        tick_size = get_option("plot.font.tick_size")
        expected_drop_fig = tick_size * 1.2 / 72.0 / _CHROME_TEST_FIGHEIGHT_IN
        actual_drop_fig = without_top - with_top

        # Tolerance absorbs tight_layout's asymmetric internal padding: the top-labels
        # case pins labels directly to the rect top, whereas the bottom-labels case
        # leaves a few pixels of default padding above the spine.
        tolerance_fig = 0.015
        assert abs(actual_drop_fig - expected_drop_fig) <= tolerance_fig, (
            f"Top-labels case should push topmost content down by ~{expected_drop_fig:.4f} "
            f"figure-fraction (tick_size * 1.2 / 72 / fig_h); measured {actual_drop_fig:.4f}"
        )


class TestLegendStyleEndOfLine:
    """End-of-line legend behavior + warning when conflicting params are set."""

    @pytest.fixture
    def line_data(self):
        """Two-series line-chart data used by the end-of-line legend tests."""
        return pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "Whole Bean Coffee": [10, 12, 14, 16, 18],
                "Loose Leaf Tea": [5, 6, 7, 8, 9],
            }
        )

    @pytest.mark.parametrize(
        ("conflicting_kwargs"),
        [
            {"move_legend_outside": True},
            {"legend_title": "Products"},
        ],
        ids=["move_legend_outside", "legend_title"],
    )
    def test_end_of_line_warns_on_conflicting_legend_params(self, line_data, conflicting_kwargs):
        """legend_style='end_of_line' emits a UserWarning when boxed-legend params are also set."""
        with pytest.warns(UserWarning, match="end_of_line"):
            line.plot(
                line_data,
                value_col=["Whole Bean Coffee", "Loose Leaf Tea"],
                x_col="x",
                title="t",
                legend_style="end_of_line",
                **conflicting_kwargs,
            )

    def test_box_legend_style_does_not_warn(self, line_data):
        """The default boxed legend does not warn when move_legend_outside is set."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ax = line.plot(
                line_data,
                value_col=["Whole Bean Coffee", "Loose Leaf Tea"],
                x_col="x",
                title="t",
                move_legend_outside=True,
            )
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend_labels == ["Whole Bean Coffee", "Loose Leaf Tea"]

    def test_end_of_line_renders_no_boxed_legend(self, line_data):
        """legend_style='end_of_line' removes the boxed legend entirely."""
        ax = line.plot(
            line_data,
            value_col=["Whole Bean Coffee", "Loose Leaf Tea"],
            x_col="x",
            title="t",
            legend_style="end_of_line",
        )
        assert ax.get_legend() is None

    @pytest.mark.parametrize(
        ("value_col", "expected_labels"),
        [
            (["Whole Bean Coffee", "Loose Leaf Tea"], {"Whole Bean Coffee", "Loose Leaf Tea"}),
            ("Whole Bean Coffee", set()),
        ],
        ids=["multi_series_one_label_each", "single_series_no_op"],
    )
    def test_end_of_line_labels_match_series_count(self, line_data, value_col, expected_labels):
        """legend_style='end_of_line' places exactly one annotation per series; no-op for one series."""
        ax = line.plot(
            line_data,
            value_col=value_col,
            x_col="x",
            title="t",
            legend_style="end_of_line",
        )
        annotation_texts = [a.get_text().strip() for a in ax.texts]
        assert set(annotation_texts) == expected_labels
        assert len(annotation_texts) == len(expected_labels)

    def test_standard_graph_styles_resolves_end_of_line_internally(self, fig_ax):
        """standard_graph_styles accepts legend_style and applies end-of-line resolution itself."""
        _, ax = fig_ax
        ax.plot([1, 2, 3], [10, 12, 14], label="Whole Bean Coffee")
        ax.plot([1, 2, 3], [5, 6, 7], label="Loose Leaf Tea")
        ax.legend()

        result = standard_graph_styles(
            ax=ax,
            title="t",
            legend_style="end_of_line",
            show_legend=True,
        )

        assert result.get_legend() is None
        annotation_texts = [a.get_text().strip() for a in result.texts]
        expected_labels = {"Whole Bean Coffee", "Loose Leaf Tea"}
        assert set(annotation_texts) == expected_labels
        # Extra assertion catches duplicate emissions (set comparison alone would hide them).
        assert len(annotation_texts) == len(expected_labels)


class TestHeatmapColormapStyle:
    """colormap_style param controls the heatmap colorbar rendering."""

    @pytest.fixture
    def heat_df(self):
        """Small day-by-time heatmap dataframe used by the colormap_style tests."""
        return pd.DataFrame(
            [[10, 50, 100], [20, 60, 110], [30, 70, 120]],
            index=["Mon", "Tue", "Wed"],
            columns=["Morning", "Noon", "Evening"],
        )

    def test_discrete_default_uses_low_high_labels(self, heat_df):
        """Default colormap_style='discrete' labels the colorbar with Low and High."""
        ax = heatmap.plot(heat_df, cbar_label="Traffic")
        # Find the colorbar axes; it's the last axes in the figure.
        cbar_ax = ax.figure.axes[-1]
        labels = [t.get_text() for t in cbar_ax.get_yticklabels()]
        assert "Low" in labels
        assert "High" in labels

    def test_continuous_uses_numeric_labels(self, heat_df):
        """colormap_style='continuous' shows numeric tick labels instead of Low and High."""
        ax = heatmap.plot(heat_df, cbar_label="Traffic", colormap_style="continuous")
        cbar_ax = ax.figure.axes[-1]
        labels = [t.get_text() for t in cbar_ax.get_yticklabels()]
        # Continuous colorbar shows numeric tick labels; none should be "Low"/"High".
        assert "Low" not in labels
        assert "High" not in labels


class TestApplyLegend:
    """apply_legend wires bbox anchoring based on the outside flag.

    This is the single integration test for the shared legend-positioning helper.
    Per-plot tests only need to confirm they pass move_legend_outside through; they
    do not need to re-verify the bbox math, since every plot routes through this
    one helper.
    """

    @pytest.mark.parametrize("outside", [True, False])
    def test_outside_flag_controls_bbox_anchor(self, fig_ax, outside):
        """outside=True anchors at the configured bbox; outside=False falls back to the axes bbox (0, 0, 1, 1)."""
        _, ax = fig_ax
        ax.plot([0, 1], [0, 1], label="series")
        apply_legend(ax, outside=outside)

        legend = ax.get_legend()
        assert legend is not None
        anchor = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
        if outside:
            outside_x, outside_y = PlotStyleHelper().legend_bbox_to_anchor
            assert anchor.x0 == pytest.approx(outside_x)
            assert anchor.y0 == pytest.approx(outside_y)
        else:
            assert anchor.x0 == pytest.approx(0.0)
            assert anchor.y0 == pytest.approx(0.0)
            assert anchor.x1 == pytest.approx(1.0)
            assert anchor.y1 == pytest.approx(1.0)
