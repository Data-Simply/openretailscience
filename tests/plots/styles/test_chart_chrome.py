"""Tests for the chart chrome layout engine."""

import io
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from openretailscience.options import PlotStyleHelper, get_option, option_context
from openretailscience.plots import heatmap, line
from openretailscience.plots.styles.styling_helpers import (
    _CHROME_TAB_WIDTH_IN,
    _CHROME_TOP_LABEL_HEADROOM_FACTOR,
    apply_chart_chrome,
    apply_legend,
    standard_graph_styles,
)


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def fig_ax():
    """Provide a fresh figure and axes for each test."""
    return plt.subplots(figsize=(10, 5))


def _texts_with(fig, text: str) -> list:
    return [t for t in fig.texts if t.get_text() == text]


def _measure_chrome_spacing(figheight: float, build_height: float | None = None) -> dict[str, float]:
    """Render eyebrow/title/subtitle/source and return inter-element pixel gaps at ``figheight``.

    When ``build_height`` is given and differs from ``figheight``, chrome is applied at
    ``build_height`` and the figure is then resized to ``figheight`` before measuring. This
    reproduces the common export flow (build on a default-sized figure, resize for the final
    render) and proves the layout tracks the figure's *current* size rather than the size it
    was first laid out at.
    """
    build_height = figheight if build_height is None else build_height
    fig, ax = plt.subplots(figsize=(10, build_height))
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
    if build_height != figheight:
        fig.set_size_inches(10, figheight)
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

_TITLE_WRAP_RESIZE_TITLE = "Two regions are carrying the entire quarter while the rest of the portfolio lags"
_TITLE_WRAP_RESIZE_SUBTITLE = "Sales index by region versus the company-wide average baseline of one hundred"

# A real chrome figure rendered to SVG/PDF is tens of KB; a failed/empty render is far smaller.
_MIN_RENDERED_VECTOR_BYTES = 1000


def _render_title_wrap(build_w: float, final_w: float) -> tuple[int, float]:
    """Build chrome at ``build_w`` wide, resize to ``final_w``, and return the title's line count and width fraction.

    The width fraction is the title's rendered width as a fraction of the figure width. Resizing
    after the build reproduces the export flow that froze the wrap at the build-time width.
    """
    fig, ax = plt.subplots(figsize=(build_w, 5))
    ax.barh(["North", "South"], [120, 80])
    apply_chart_chrome(ax, title=_TITLE_WRAP_RESIZE_TITLE, subtitle=_TITLE_WRAP_RESIZE_SUBTITLE)
    if build_w != final_w:
        fig.set_size_inches(final_w, 5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_artist = next(t for t in fig.texts if t.get_text().replace("\n", " ") == _TITLE_WRAP_RESIZE_TITLE)
    line_count = title_artist.get_text().count("\n") + 1
    width_frac = title_artist.get_window_extent(renderer=renderer).width / fig.bbox.width
    plt.close(fig)
    return line_count, width_frac


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

    def test_title_wrap_is_baked_into_explicit_newlines(self):
        """Chrome bakes the title's wrap into explicit newlines (wrap off), not matplotlib soft-wrap.

        ``get_window_extent`` ignores ``wrap=True``, so a soft-wrapped title would measure a
        single-line height and collapse the layout. Chrome instead bakes the wrapped lines into the
        text content with wrap off, giving a stable multi-line height to position siblings against;
        the engine re-bakes to the current width on each draw.
        """
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        long_title = "High-AOV stores convert fewer visits but earn the most revenue per day"
        apply_chart_chrome(ax, title=long_title)

        title_artist = fig.texts[0]
        # Wrap is baked into explicit newlines with soft-wrap off, so a redraw measures a stable height.
        assert title_artist.get_wrap() is False
        assert "\n" in title_artist.get_text()
        # All original words preserved, only whitespace replaced with newlines.
        assert title_artist.get_text().replace("\n", " ") == long_title

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

    def test_chrome_spacing_survives_figure_resize(self):
        """Resizing the figure after layout must not change the inter-element gaps.

        The text is sized in points but the gaps were once stored as a fraction of the
        layout-time height, so a later resize ballooned (or, when shrunk, overlapped) them.
        """
        reference = _measure_chrome_spacing(figheight=8)
        resized_up = _measure_chrome_spacing(figheight=8, build_height=4)
        resized_down = _measure_chrome_spacing(figheight=4, build_height=8)
        reference_down = _measure_chrome_spacing(figheight=4)

        tolerance_px = 1.0
        for key in reference:
            assert abs(resized_up[key] - reference[key]) <= tolerance_px, (
                f"{key}: building at 4in then resizing to 8in gives {resized_up[key]:.2f}px, "
                f"but building directly at 8in gives {reference[key]:.2f}px; layout did not "
                f"track the resized figure height"
            )
            assert abs(resized_down[key] - reference_down[key]) <= tolerance_px, (
                f"{key}: building at 8in then resizing to 4in gives {resized_down[key]:.2f}px, "
                f"but building directly at 4in gives {reference_down[key]:.2f}px; layout did not "
                f"track the resized figure height"
            )

    def test_source_rewraps_but_axes_follow_on_width_resize(self):
        """Narrowing the figure re-wraps the source to more lines, and the axes bottom follows it.

        The source re-wraps to the current width like the header; the engine then pushes the axes
        bottom up so the reserved source-to-axes gap is preserved, instead of the taller source
        crowding into the data area.
        """
        long_source = (
            "Source: 2024 loyalty-program transactions across all banners, excluding staff "
            "purchases and returns; index baseline is the chain average of 100."
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(["Whole Milk", "Free-Range Eggs", "Sourdough"], [120, 88, 145])
        apply_chart_chrome(ax, title="Premium categories over-index", source_text=long_source)
        source_artist = next(t for t in fig.texts if t.get_text().startswith("Source:"))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        wide_lines = source_artist.get_text().count("\n") + 1
        gap_wide = ax.get_position().y0 * fig.bbox.height - source_artist.get_window_extent(renderer=renderer).y1

        fig.set_size_inches(4, 5)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        narrow_lines = source_artist.get_text().count("\n") + 1
        gap_narrow = ax.get_position().y0 * fig.bbox.height - source_artist.get_window_extent(renderer=renderer).y1

        assert narrow_lines > wide_lines, (
            f"source stayed at {wide_lines} lines when the figure narrowed; its wrap did not track the current width"
        )
        assert abs(gap_narrow - gap_wide) <= 1.0, (
            f"source-to-axes gap changed from {gap_wide:.1f}px to {gap_narrow:.1f}px when the figure "
            "narrowed; the axes bottom did not follow the re-wrapped source"
        )

    # The build-time width is narrow enough that the title wraps there; each target is wide enough
    # that a direct build wraps to fewer lines.
    _NARROW_BUILD_WIDTH_IN = 6.0
    # A title still frozen on the narrow figure's breaks spans only ~40-55% of the widened figure;
    # a re-wrapped one spans ~80-90%. 0.7 sits cleanly between the two regimes.
    _TITLE_SPANS_WIDTH_FRAC = 0.7

    @pytest.mark.parametrize("final_w", [10.0, 12.0, 14.0])
    def test_title_rewraps_to_match_width_after_resize(self, final_w):
        """A title built narrow then widened must re-wrap to match the line count of a direct wide build.

        Chrome baked matplotlib's wrap at the build-time width, so widening the figure afterwards
        left the title broken onto the narrow figure's line breaks, spanning only a fraction of the
        wider figure. The wrap must re-flow to the figure's current width.
        """
        lines_resized, width_frac_resized = _render_title_wrap(build_w=self._NARROW_BUILD_WIDTH_IN, final_w=final_w)
        lines_direct, _ = _render_title_wrap(build_w=final_w, final_w=final_w)

        assert lines_resized == lines_direct, (
            f"title wrapped to {lines_resized} lines after widening to {final_w}in but {lines_direct} "
            "when built wide directly; the wrap stayed frozen at the build-time width"
        )
        assert width_frac_resized > self._TITLE_SPANS_WIDTH_FRAC, (
            f"title spans only {width_frac_resized:.0%} of the widened figure; it stayed collapsed on "
            "the narrow figure's line breaks instead of re-wrapping to the current width"
        )

    @pytest.mark.parametrize(("fmt", "signature"), [("svg", b"<svg"), ("pdf", b"%PDF")])
    def test_chrome_saves_to_vector_backend_after_resize(self, fmt, signature):
        """Saving a chromed figure to a vector backend (SVG/PDF) must work, including after a resize.

        The layout engine re-measures on every draw via the figure renderer. Saving to a vector
        format swaps in a canvas with no Agg ``get_renderer``, so looking the renderer up off the
        canvas crashed the save; the export is the bug report's own repro and was untested.
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(["North", "South"], [120, 80])
        apply_chart_chrome(
            ax,
            title="Two regions are carrying the entire quarter while the rest of the portfolio lags",
            subtitle="Sales index by region versus the company-wide average baseline of one hundred",
            source_text="Source: 2024 loyalty transactions",
        )
        fig.set_size_inches(11, 9)  # resize after layout, then export

        buf = io.BytesIO()
        fig.savefig(buf, format=fmt)
        data = buf.getvalue()

        assert signature in data[:1024], f"{fmt} output is missing its format signature; the save did not render"
        assert len(data) > _MIN_RENDERED_VECTOR_BYTES, (
            f"{fmt} output is only {len(data)} bytes; the chrome figure did not render"
        )

    def test_chrome_survives_resize_on_figure_with_colorbar(self):
        """Resizing a heatmap holds the chrome spacing and keeps its colorbar axes.

        Also guards the engine's colorbar-gridspec compatibility: a mismatching engine would
        raise when installed on a figure that already owns a colorbar.
        """
        traffic = pd.DataFrame(
            [[120, 480, 300], [140, 520, 280], [160, 540, 260]],
            index=["Mon", "Tue", "Wed"],
            columns=["Morning", "Midday", "Evening"],
        )
        ax = heatmap.plot(
            traffic,
            cbar_label="Visits",
            eyebrow="Footfall",
            title="Midday peaks across the week",
            subtitle="Store 101",
        )
        fig = ax.figure
        axes_count = len(fig.axes)  # main heatmap axes + colorbar axes

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        eye = _texts_with(fig, "FOOTFALL")[0].get_window_extent(renderer=renderer)
        ttl = _texts_with(fig, "Midday peaks across the week")[0].get_window_extent(renderer=renderer)
        gap_before = eye.y0 - ttl.y1

        fig.set_size_inches(12, 9)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        eye = _texts_with(fig, "FOOTFALL")[0].get_window_extent(renderer=renderer)
        ttl = _texts_with(fig, "Midday peaks across the week")[0].get_window_extent(renderer=renderer)
        gap_after = eye.y0 - ttl.y1

        assert len(fig.axes) == axes_count, "colorbar axes was dropped after resize"
        assert abs(gap_after - gap_before) <= 1.0, (
            f"eyebrow-to-title gap changed from {gap_before:.1f}px to {gap_after:.1f}px after "
            "resizing a heatmap figure with a colorbar"
        )

    def test_tab_width_stays_fixed_across_width_resize(self):
        """The tab mark keeps its absolute design width when the figure is resized horizontally.

        The tab is a fixed-size element; storing only a width fraction lets it scale with the
        figure width on resize.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(["Whole Milk", "Free-Range Eggs"], [120, 88])
        apply_chart_chrome(ax, eyebrow="Category index", title="Premium brands over-index")
        tab = next(p for p in fig.patches if (p.get_gid() or "").startswith("_ors_chrome"))

        fig.canvas.draw()
        width_wide_in = tab.get_width() * fig.get_figwidth()
        fig.set_size_inches(5, 5)
        fig.canvas.draw()
        width_narrow_in = tab.get_width() * fig.get_figwidth()

        assert width_wide_in == pytest.approx(_CHROME_TAB_WIDTH_IN, abs=0.02)
        assert width_narrow_in == pytest.approx(_CHROME_TAB_WIDTH_IN, abs=0.02), (
            f"tab width became {width_narrow_in:.3f}in after narrowing the figure; it must stay "
            f"the fixed {_CHROME_TAB_WIDTH_IN}in design width"
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
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
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

    def test_top_tick_labels_reserve_extra_header_gap(self):
        """Top-rendered tick labels (heatmap/cohort) reserve one tick-label line-height of extra headroom.

        The chrome layout pushes the topmost axes content down by ``tick_size * 1.2 / 72``
        inches when top tick labels are present, so subtitle-to-content whitespace doesn't
        feel cramped relative to the bottom-labels case.
        """
        with_top = _topmost_axes_content_fig_y(labeltop=True)
        without_top = _topmost_axes_content_fig_y(labeltop=False)

        tick_size = get_option("plot.font.tick_size")
        expected_drop_fig = tick_size * _CHROME_TOP_LABEL_HEADROOM_FACTOR / 72.0 / _CHROME_TEST_FIGHEIGHT_IN
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
