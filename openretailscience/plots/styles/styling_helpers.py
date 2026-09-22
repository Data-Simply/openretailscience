"""Module-level helpers that apply styling using the options system."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

from matplotlib.layout_engine import LayoutEngine
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoLocator, FixedFormatter, FixedLocator, Formatter, MaxNLocator

from openretailscience.options import PlotStyleHelper
from openretailscience.plots.styles.font_utils import get_font_properties
from openretailscience.plots.styles.graph_utils import draw_end_of_line_labels

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.container import Container
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from matplotlib.text import Text

GridAxis = Literal["both", "x", "y", "none"]


class _ZeroBlankingFormatter(Formatter):
    """Wrap an axis's major formatter, rendering the value ``0`` as an empty label.

    Blanking is keyed to the value, not the tick index: matplotlib reuses tick
    artists by index, so an index-pinned flag would blank whatever value later
    lands on that index. All other calls delegate to the base formatter, so its
    numeric formatting and offset text (e.g. ``ScalarFormatter``'s ``1e6``) are
    preserved; a caller who sets their own formatter replaces the wrapper, which
    stops the zero-drop.
    """

    def __init__(self, base: Formatter) -> None:
        """Store the wrapped ``base`` formatter to delegate to."""
        super().__init__()
        self._base = base

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Return the base label for ``x``, or ``""`` when ``x`` is exactly ``0``."""
        return "" if x == 0 else self._base(x, pos)

    def set_locs(self, locs: Sequence[float]) -> None:
        """Forward the tick locations so the base formatter can compute its offset and scale."""
        self._base.set_locs(locs)

    def get_offset(self) -> str:
        """Return the base formatter's offset text (e.g. ``ScalarFormatter``'s ``1e6``)."""
        return self._base.get_offset()

    def __getattr__(self, name: str) -> object:
        """Delegate any non-overridden attribute to the base formatter.

        Keeps the wrapper transparent for ``Axes.ticklabel_format``-style setters
        on the wrapped ``ScalarFormatter``. The ``object`` return is deliberate:
        the base can expose any attribute.
        """
        # __getattr__ only fires for names missing on the wrapper; guard ``_base``
        # itself so a half-built instance raises cleanly instead of recursing.
        if name == "_base":
            raise AttributeError(name)
        return getattr(self._base, name)


def _hide_zero_value_ticks(ax: Axes) -> None:
    """Blank the ``0`` tick label on numeric axes (editorial convention).

    Only numeric continuous axes (``MaxNLocator`` / ``AutoLocator``) are touched;
    categorical (``FixedLocator``) and date axes keep their tick-0, which is a
    category index, not a data zero. Idempotent: a repeated styling pass does not
    nest the wrapper. The blank is value-keyed via ``_ZeroBlankingFormatter``.
    """
    for axis in (ax.xaxis, ax.yaxis):
        if not isinstance(axis.get_major_locator(), MaxNLocator | AutoLocator):
            continue
        formatter = axis.get_major_formatter()
        # Idempotent: don't nest the wrapper on a repeated styling pass.
        if isinstance(formatter, _ZeroBlankingFormatter):
            continue
        axis.set_major_formatter(_ZeroBlankingFormatter(formatter))


# Chrome layout constants. Vertical spacings are in absolute inches so the
# visual relationship to the (points-based) chrome text holds at any figure
# height. Horizontal margins remain figure-relative because they're consumed
# by tight_layout's rect.
# Used only when the provisional tight_layout pass fails; normal placement
# anchors chrome to the y-axis spine.
_CHROME_FALLBACK_LEFT_MARGIN = 0.045
_CHROME_TOP_MARGIN_IN = 0.24
_CHROME_BOTTOM_MARGIN_IN = 0.21
_CHROME_LEFT_AXES_MARGIN = 0.015
_CHROME_RIGHT_AXES_MARGIN = 0.015
_CHROME_TAB_WIDTH_IN = 0.47
_CHROME_TAB_HEIGHT_IN = 0.04
_CHROME_TAB_TO_EYEBROW_GAP_IN = 0.13
_MAX_NUMERIC_TICKS = 5
_CHROME_GAP_EYEBROW_TO_TITLE_IN = 0.06
_CHROME_GAP_TITLE_TO_SUBTITLE_IN = 0.072
_CHROME_GAP_HEADER_TO_AXES_IN = 0.108
_CHROME_GAP_SOURCE_TO_AXES_IN = 0.15
_CHROME_TOP_LABEL_HEADROOM_FACTOR = 1.2
# Past half the figure width, a few overflowing legend entries beat a plot squeezed to a sliver.
_MAX_OUTSIDE_LEGEND_WIDTH_FRAC = 0.5

END_OF_LINE_LEGEND_CONFLICT_MSG = (
    "`move_legend_outside` and `legend_title` are ignored when legend_style='end_of_line'."
)


def _resolve_end_of_line_legend_args(
    legend_style: Literal["box", "end_of_line"] | None,
    show_legend: bool,
    legend_title: str | None,
    move_legend_outside: bool,
) -> tuple[bool, str | None, bool]:
    """Neutralise box-legend args when ``legend_style="end_of_line"`` is requested.

    Returns the ``(show_legend, legend_title, move_legend_outside)`` triple so
    the box legend is off before ``apply_legend`` runs. Emits ``UserWarning``
    (stacklevel 4, pointing at user code, message ``END_OF_LINE_LEGEND_CONFLICT_MSG``)
    when ``legend_title`` or ``move_legend_outside`` is combined with ``"end_of_line"``.
    """
    if legend_style != "end_of_line" or not show_legend:
        return show_legend, legend_title, move_legend_outside
    if move_legend_outside or legend_title is not None:
        warnings.warn(END_OF_LINE_LEGEND_CONFLICT_MSG, UserWarning, stacklevel=4)
    return False, None, False


def _rewrap_text_to_width(text: Text, original: str, renderer: RendererBase) -> None:
    """Bake ``original`` into ``text`` as newlines wrapped to the figure's current width.

    Baking is needed because ``get_window_extent`` ignores ``wrap=True``. Uses matplotlib
    internals (``_get_wrapped_text``, ``_renderer``); no public width-wrap API exists, so a
    rename fails loudly here.
    """
    text.set_text(original)
    text.set_wrap(True)
    text._renderer = renderer
    text.set_text(text._get_wrapped_text())
    text.set_wrap(False)


def _active_renderer(fig: Figure) -> RendererBase:
    """Return the renderer for the figure's current draw, across backends.

    ``fig._get_renderer()`` works on every backend, including ``savefig``'s vector canvas;
    ``fig.canvas.get_renderer()`` is Agg-only.
    """
    return fig._get_renderer()


@dataclass
class _ChromeTextSpec:
    """A chrome text element plus the inputs needed to re-place it at any figure width.

    Attributes:
        text: Figure-text artist; holds the baked (newlined) form.
        original: Unwrapped source string, re-wrapped to the current width on each draw.
        gap_after_in: Inch gap from this element's bottom to the next element (the trailing
            slot for the final element).
        wrap: Whether the element re-wraps to the figure width; eyebrows stay single-line.
    """

    text: Text
    original: str
    gap_after_in: float
    wrap: bool = True


def _layout_chrome_header(
    specs: list[_ChromeTextSpec],
    top_offset_in: float,
    fig_h: float,
    dpi: float,
    renderer: RendererBase,
) -> float:
    """Re-wrap and stack the top-anchored header elements; return the block's bottom offset in inches.

    The offset, in inches from the figure top and including the final element's trailing gap,
    anchors the axes top.
    """
    top_in = top_offset_in
    for spec in specs:
        if spec.wrap:
            _rewrap_text_to_width(spec.text, spec.original, renderer)
        spec.text.set_y(1.0 - top_in / fig_h)
        height_in = spec.text.get_window_extent(renderer=renderer).height / dpi
        top_in += height_in + spec.gap_after_in
    return top_in


def _layout_chrome_source(
    source: _ChromeTextSpec,
    fig_h: float,
    dpi: float,
    renderer: RendererBase,
) -> float:
    """Re-wrap the bottom-anchored source line; return its top offset in inches from the figure bottom.

    The source always wraps and is anchored at ``_CHROME_BOTTOM_MARGIN_IN`` from the figure bottom.
    """
    _rewrap_text_to_width(source.text, source.original, renderer)
    source.text.set_y(_CHROME_BOTTOM_MARGIN_IN / fig_h)
    height_in = source.text.get_window_extent(renderer=renderer).height / dpi
    return _CHROME_BOTTOM_MARGIN_IN + height_in


@dataclass
class _ChromeLayout:
    """One axes' chrome placement recipe: resize-invariant inch inputs re-derived on every draw.

    Attributes:
        ax: The axes whose box is reflowed beneath the header.
        header_top_offset_in: Inches from the figure top to the first header element's top.
        header: Top-anchored header elements (eyebrow/title/subtitle) in stacking order.
        header_to_axes_gap_in: Header block bottom to axes top; folds in top-tick headroom.
        source: The bottom-anchored source line, or None.
        source_to_axes_gap_in: Source top to axes bottom edge.
        axes_bottom_offset_in: Figure bottom to axes bottom edge; used when there is no source.
        tab: ``(rectangle, top_offset_in, height_in, width_in)`` for the tab mark, or None.
    """

    ax: Axes
    header_top_offset_in: float
    header: list[_ChromeTextSpec]
    header_to_axes_gap_in: float
    source: _ChromeTextSpec | None
    source_to_axes_gap_in: float
    axes_bottom_offset_in: float
    tab: tuple[Rectangle, float, float, float] | None


def _apply_chrome_layout(
    layout: _ChromeLayout,
    fig_h: float,
    fig_w: float,
    dpi: float,
    renderer: RendererBase,
) -> None:
    """Re-wrap the chrome texts and re-place every artist for the current figure size."""
    header_bottom_in = _layout_chrome_header(layout.header, layout.header_top_offset_in, fig_h, dpi, renderer)
    axes_top_in = header_bottom_in + layout.header_to_axes_gap_in
    if layout.source is not None:
        source_top_in = _layout_chrome_source(layout.source, fig_h, dpi, renderer)
        axes_bottom_in = source_top_in + layout.source_to_axes_gap_in
    else:
        axes_bottom_in = layout.axes_bottom_offset_in
    if layout.tab is not None:
        tab, tab_top_offset_in, tab_height_in, tab_width_in = layout.tab
        tab_height_fig = tab_height_in / fig_h
        tab.set_y(1.0 - tab_top_offset_in / fig_h - tab_height_fig)
        tab.set_height(tab_height_fig)
        tab.set_width(tab_width_in / fig_w)
    pos = layout.ax.get_position()
    axes_top = 1.0 - axes_top_in / fig_h
    axes_bottom = axes_bottom_in / fig_h
    layout.ax.set_position((pos.x0, axes_bottom, pos.width, axes_top - axes_bottom))


def _apply_chrome_layouts(fig: Figure) -> None:
    """Re-apply every stored chrome layout on ``fig`` at its current size."""
    layouts = getattr(fig, "_ors_chrome_layouts", None)
    if layouts is None:
        return
    renderer = _active_renderer(fig)
    fig_h = fig.get_figheight()
    fig_w = fig.get_figwidth()
    dpi = fig.dpi
    for layout in layouts.values():
        _apply_chrome_layout(layout, fig_h, fig_w, dpi, renderer)


def _recompute_chrome_axes_gaps(layout: _ChromeLayout, fig: Figure, renderer: RendererBase) -> None:
    """Re-capture ``layout``'s header/source-to-axes gaps from the axes box's current position.

    Re-wraps and repositions the header and source artists as a side effect, since the gaps
    are measured against their fresh heights.
    """
    fig_h = fig.get_figheight()
    dpi = fig.dpi
    header_bottom_in = _layout_chrome_header(layout.header, layout.header_top_offset_in, fig_h, dpi, renderer)
    pos = layout.ax.get_position()
    layout.header_to_axes_gap_in = (1.0 - pos.y1) * fig_h - header_bottom_in
    axes_bottom_in = pos.y0 * fig_h
    if layout.source is not None:
        source_top_in = _layout_chrome_source(layout.source, fig_h, dpi, renderer)
        layout.source_to_axes_gap_in = axes_bottom_in - source_top_in
    else:
        layout.axes_bottom_offset_in = axes_bottom_in


def _refresh_chrome_axes_offsets(fig: Figure) -> None:
    """Re-derive each layout's axes gaps after a post-install reflow moved the axes box.

    ``_auto_rotate_categorical_x_ticks`` reflows the axes to reserve room for rotated labels;
    without this the engine would re-apply the pre-rotation box and push the labels off-figure.
    """
    layouts = getattr(fig, "_ors_chrome_layouts", None)
    if layouts is None:
        return
    renderer = _active_renderer(fig)
    for layout in layouts.values():
        _recompute_chrome_axes_gaps(layout, fig, renderer)


class _ChromeLayoutEngine(LayoutEngine):
    """Re-applies chrome geometry before each draw so it tracks the figure's current size.

    Matplotlib runs ``execute`` at the start of every draw (including headless ``savefig``),
    before artists render; it never triggers a draw of its own, so it cannot recurse.
    """

    # Match tight_layout's flags: the chrome reflow runs through fig.tight_layout (gridspec
    # based), so matplotlib's engine-compatibility check rejects a mismatching engine once a
    # colorbar (e.g. heatmap) has been created.
    _adjust_compatible = True
    _colorbar_gridspec = True

    def execute(self, fig: Figure) -> None:
        """Apply the stored chrome layouts to ``fig`` at its current size."""
        _apply_chrome_layouts(fig)


def _install_chrome_layout_engine(fig: Figure) -> None:
    """Register the chrome layout engine on ``fig`` unless it is already active."""
    if not isinstance(fig.get_layout_engine(), _ChromeLayoutEngine):
        fig.set_layout_engine(_ChromeLayoutEngine())


def _resolve_chrome_left(fig: Figure, ax: Axes) -> float:
    """Discover the y-axis spine's x in figure coords via a provisional tight_layout.

    Anchoring chrome to the spine keeps the title directly above the data. Falls back to
    ``_CHROME_FALLBACK_LEFT_MARGIN`` when the provisional layout pass fails.
    """
    top = 1.0 - _CHROME_TOP_MARGIN_IN / fig.get_figheight()
    bottom = _CHROME_BOTTOM_MARGIN_IN / fig.get_figheight()
    if not _reflow_axes(fig, top=top, bottom=bottom):
        return _CHROME_FALLBACK_LEFT_MARGIN
    fig.canvas.draw()
    return float(ax.get_position().x0)


def _reflow_axes(fig: Figure, top: float, bottom: float) -> bool:
    """Reflow the axes into the chrome's vertical band. Returns False on failure.

    ``pad=0`` is critical: tight_layout's default pad adds ~3% of figure height
    between the rect top and the axes top, doubling the intended gap. Returns
    False when ``constrained_layout`` is active or the rect is degenerate. The
    chrome layout engine is detached and reinstalls around ``tight_layout``,
    which would otherwise warn and replace it.
    """
    # fig.tight_layout warns and replaces a non-tight engine, so detach ours first (e.g. on
    # _auto_rotate's post-install reflow) and restore it in the finally.
    has_chrome_engine = isinstance(fig.get_layout_engine(), _ChromeLayoutEngine)
    if has_chrome_engine:
        fig.set_layout_engine("none")
    try:
        fig.tight_layout(
            pad=0,
            rect=(_CHROME_LEFT_AXES_MARGIN, bottom, 1.0 - _CHROME_RIGHT_AXES_MARGIN, top),
        )
    except (ValueError, RuntimeError):
        return False
    finally:
        if has_chrome_engine:
            _install_chrome_layout_engine(fig)
            # tight_layout may have moved the axes (e.g. to reserve room for rotated tick
            # labels); resnapshot so the engine preserves the new box instead of reverting it.
            _refresh_chrome_axes_offsets(fig)
    return True


def _clear_prior_chrome(fig: Figure, chrome_gid: str) -> None:
    """Remove chrome artists left on ``fig`` by a previous call for this gid.

    Prevents stacked duplicate chrome on iterative replotting (Jupyter, tests).
    The tab is appended to ``fig.patches`` directly, so it must be popped rather
    than removed.
    """
    for text in [t for t in fig.texts if t.get_gid() == chrome_gid]:
        text.remove()
    for patch in [p for p in fig.patches if p.get_gid() == chrome_gid]:
        fig.patches.remove(patch)


def _track_chrome_axes(fig: Figure, ax: Axes, *, warn_stacklevel: int) -> None:
    """Record ``ax`` on ``fig._ors_chrome_axes`` and warn the first time a second axes appears.

    Chrome is positioned in figure coordinates and is not subplot-aware, so multiple
    chrome-bearing axes on the same figure can shift each other's chrome out of
    alignment. The warning fires once, at the 1→2 tracked-axes transition.
    """
    prior = getattr(fig, "_ors_chrome_axes", None)
    if prior is None:
        fig._ors_chrome_axes = [ax]
        return
    if ax in prior:
        return
    if len(prior) == 1:
        warnings.warn(
            "Chrome (title/eyebrow/subtitle/source/tab) is positioned in figure "
            "coordinates and is not subplot-aware. Calling a plot function with "
            "chrome on multiple axes of the same figure can shift each other's "
            "chrome out of alignment. Render each chrome-bearing plot in its own "
            "figure for predictable layout.",
            stacklevel=warn_stacklevel,
        )
    prior.append(ax)


def apply_chart_chrome(
    ax: Axes,
    *,
    eyebrow: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    source_text: str | None = None,
    warn_stacklevel: int = 4,
) -> None:
    """Place the figure-level chrome (eyebrow, tab, title, subtitle, source) and reflow the axes.

    Each header element is placed after the previous one's measured bbox, so wrapped
    titles push later elements down by their rendered height; absent elements collapse
    their slot. The placement is stored as a resize-invariant recipe and re-derived on
    every draw by ``_ChromeLayoutEngine``, so a post-call ``set_size_inches`` re-flows
    the chrome. The tab mark is controlled by the ``plot.style.show_tab`` option
    (default True).

    Args:
        ax: The plot axes (provides the figure handle).
        eyebrow: Small uppercase label above the title.
        title: Main headline; wraps if it would exceed the figure width.
        subtitle: Supporting copy below the title; wraps.
        source_text: Footer text (rendered italic, muted).
        warn_stacklevel: Stacklevel for the multi-axes chrome warning; 4 points at user
            code via ``standard_graph_styles``, direct public callers (e.g. ``venn.plot``)
            pass 3.
    """
    style = PlotStyleHelper()
    fig = ax.figure
    show_tab = style.show_tab
    has_header = eyebrow is not None or title is not None or subtitle is not None
    any_chrome = has_header or source_text is not None
    chrome_gid = f"_ors_chrome:{id(ax)}"

    _clear_prior_chrome(fig, chrome_gid)
    # Drop any prior layout for this axes too: _clear_prior_chrome just detached its chrome texts
    # (their figure is now None), so the reflow below and the engine must not try to re-wrap them.
    # The fresh layout replaces this entry at the end of the call.
    prior_layouts = getattr(fig, "_ors_chrome_layouts", None)
    if prior_layouts is not None:
        prior_layouts.pop(id(ax), None)

    if any_chrome:
        _track_chrome_axes(fig, ax, warn_stacklevel=warn_stacklevel)

    chrome_x = _resolve_chrome_left(fig, ax) if any_chrome else _CHROME_FALLBACK_LEFT_MARGIN

    fig_h = fig.get_figheight()
    fig_w = fig.get_figwidth()
    dpi = fig.dpi
    renderer = _active_renderer(fig)
    cur_y = 1.0 - _CHROME_TOP_MARGIN_IN / fig_h

    # Always reserve the tab's vertical slot when a header is present, even when
    # the tab is hidden. This keeps the title's distance from the figure edge
    # consistent whether the tab is drawn or hidden, so suppressing the tab
    # doesn't push the title against the top margin.
    tab_spec: tuple[Rectangle, float, float, float] | None = None
    header_top_offset_in = _CHROME_TOP_MARGIN_IN
    if has_header:
        tab_height_fig = _CHROME_TAB_HEIGHT_IN / fig_h
        tab_to_eyebrow_fig = _CHROME_TAB_TO_EYEBROW_GAP_IN / fig_h
        if show_tab:
            tab_width_fig = _CHROME_TAB_WIDTH_IN / fig_w
            tab = Rectangle(
                (chrome_x, cur_y - tab_height_fig),
                tab_width_fig,
                tab_height_fig,
                transform=fig.transFigure,
                facecolor=style.tab_color,
                edgecolor="none",
                clip_on=False,
            )
            tab.set_gid(chrome_gid)
            fig.patches.append(tab)
            tab_spec = (tab, _CHROME_TOP_MARGIN_IN, _CHROME_TAB_HEIGHT_IN, _CHROME_TAB_WIDTH_IN)
        cur_y -= tab_height_fig + tab_to_eyebrow_fig
        header_top_offset_in = (1.0 - cur_y) * fig_h

    # Eyebrow's gap is always applied; the title→subtitle gap only when both
    # are present. Subtitle has no trailing gap — _CHROME_GAP_HEADER_TO_AXES_IN
    # below covers the whitespace down to the plot. Eyebrows stay single-line; the
    # title and subtitle re-wrap to the figure's current width on every draw.
    header_specs = (
        (
            eyebrow.upper() if eyebrow is not None else None,
            style.eyebrow_font,
            style.eyebrow_size,
            style.eyebrow_color,
            False,
            _CHROME_GAP_EYEBROW_TO_TITLE_IN,
        ),
        (
            title,
            style.title_font,
            style.title_size,
            style.title_color,
            True,
            _CHROME_GAP_TITLE_TO_SUBTITLE_IN if subtitle is not None else 0.0,
        ),
        (
            subtitle,
            style.subtitle_font,
            style.subtitle_size,
            style.subtitle_color,
            True,
            0.0,
        ),
    )
    header: list[_ChromeTextSpec] = []
    for text_str, font, size, color, wrap, gap_after_in in header_specs:
        if text_str is None:
            continue
        # y is a placeholder; _layout_chrome_header stacks the elements into their real positions below.
        artist = fig.text(
            chrome_x,
            1.0 - header_top_offset_in / fig_h,
            text_str,
            ha="left",
            va="top",
            fontproperties=get_font_properties(font),
            fontsize=size,
            color=color,
            wrap=wrap,
        )
        artist.set_gid(chrome_gid)
        header.append(_ChromeTextSpec(artist, text_str, gap_after_in, wrap=wrap))

    header_bottom_in = _layout_chrome_header(header, header_top_offset_in, fig_h, dpi, renderer)

    # tight_layout reserves most top tick-label height inside [bottom, header_bottom],
    # but its default label padding leaves a little less visible subtitle-to-content
    # whitespace than the bottom-label case. Add only the residual headroom needed
    # to keep that visible gap consistent.
    has_top_labels = any(t.label2.get_visible() for t in [*ax.xaxis.get_major_ticks(), *ax.yaxis.get_major_ticks()])
    extra_gap_in = style.tick_size * _CHROME_TOP_LABEL_HEADROOM_FACTOR / 72.0 if has_top_labels else 0.0
    rect_top_in = header_bottom_in + _CHROME_GAP_HEADER_TO_AXES_IN + extra_gap_in if has_header else header_bottom_in
    header_bottom = 1.0 - rect_top_in / fig_h

    source_spec: _ChromeTextSpec | None = None
    axes_bottom_in = _CHROME_BOTTOM_MARGIN_IN
    if source_text is not None:
        # y is a placeholder; _layout_chrome_source sets the real position below.
        source_artist = fig.text(
            chrome_x,
            _CHROME_BOTTOM_MARGIN_IN / fig_h,
            source_text,
            ha="left",
            va="bottom",
            fontproperties=get_font_properties(style.source_font),
            fontsize=style.source_size,
            color=style.source_color,
            wrap=True,
        )
        source_artist.set_gid(chrome_gid)
        source_spec = _ChromeTextSpec(source_artist, source_text, 0.0)
        source_top_in = _layout_chrome_source(source_spec, fig_h, dpi, renderer)
        axes_bottom_in = source_top_in + _CHROME_GAP_SOURCE_TO_AXES_IN
    axes_bottom = axes_bottom_in / fig_h

    # Cache the chrome rect so callers (e.g. _auto_rotate_categorical_x_ticks)
    # can re-reflow after they change tick label heights. Without this, a
    # post-chrome rotation/wrap leaves the axes box sized for the original
    # pandas-90° labels and you get a tall empty gap below the data area.
    fig._ors_chrome_rect = (header_bottom, axes_bottom)

    if not _reflow_axes(fig, top=header_bottom, bottom=axes_bottom):
        fig.subplots_adjust(top=header_bottom, bottom=axes_bottom)

    # Store the placement as a resize-invariant recipe and install the engine so the chrome
    # re-wraps and re-flows to the figure's current size on later draws.
    layouts = getattr(fig, "_ors_chrome_layouts", None)
    if layouts is None:
        layouts = {}
        fig._ors_chrome_layouts = layouts
    layout = _ChromeLayout(
        ax=ax,
        header_top_offset_in=header_top_offset_in,
        header=header,
        header_to_axes_gap_in=0.0,
        source=source_spec,
        source_to_axes_gap_in=0.0,
        axes_bottom_offset_in=_CHROME_BOTTOM_MARGIN_IN,
        tab=tab_spec,
    )
    _recompute_chrome_axes_gaps(layout, fig, renderer)
    layouts[id(ax)] = layout
    _install_chrome_layout_engine(fig)


def apply_base_styling(ax: Axes, grid_axis: GridAxis = "both", hide_spines: bool = False) -> None:
    """Apply base plot styling (spines, grid, background) using options.

    Resets any pre-existing (pandas) grid before applying the styled one.

    Args:
        ax: The axes to style.
        grid_axis: ``"x"`` draws only vertical lines, ``"y"`` only horizontal,
            ``"both"`` both, ``"none"`` off.
        hide_spines: Hide all four spines, for plots whose cells define the boundaries
            (heatmap, cohort).
    """
    style = PlotStyleHelper()
    ax.set_facecolor(style.background_color)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False if hide_spines else style.show_top_spine)
    ax.spines["right"].set_visible(False if hide_spines else style.show_right_spine)
    ax.spines["bottom"].set_visible(False if hide_spines else style.show_bottom_spine)
    ax.spines["left"].set_visible(False if hide_spines else style.show_left_spine)

    # Reset any pre-existing grid (pandas can leave one behind) before applying our own.
    ax.grid(False, which="both")
    if grid_axis != "none":
        ax.grid(
            which="major",
            axis=grid_axis,
            color=style.grid_color,
            alpha=style.grid_alpha,
            zorder=1,
        )


def apply_label(ax: Axes, label: str, axis: Literal["x", "y"], pad: int | None = None) -> None:
    """Apply axis label styling using options."""
    style = PlotStyleHelper()
    if pad is None:
        pad = style.x_label_pad if axis == "x" else style.y_label_pad

    font_props = get_font_properties(style.label_font)

    axis_fn = ax.set_xlabel if axis == "x" else ax.set_ylabel
    axis_fn(label, fontproperties=font_props, fontsize=style.label_size, labelpad=pad)


_X_TICK_LABEL_GAP_PX = 4


def _best_two_line_split(text: str) -> str:
    """Split a multi-word label into two balanced lines.

    Picks the whitespace split point that minimises the longer line's character
    count. Single-word labels are returned unchanged: they have no sensible
    split point, so the caller should fall through to rotation instead.
    """
    words = text.split()
    if len(words) <= 1:
        return text
    best_idx = 1
    best_max = float("inf")
    for i in range(1, len(words)):
        line_max = max(len(" ".join(words[:i])), len(" ".join(words[i:])))
        if line_max < best_max:
            best_max = line_max
            best_idx = i
    return " ".join(words[:best_idx]) + "\n" + " ".join(words[best_idx:])


def _set_xtick_rotation(ax: Axes, angle: float) -> None:
    """Rotate every x-tick label on ``ax`` to ``angle`` and align it accordingly.

    Pulls a fresh label list each call: matplotlib redraws can replace tick
    artists, so a cached list goes stale.
    """
    for label in ax.get_xticklabels():
        label.set_rotation(angle)
        if angle == 0:
            label.set_ha("center")
            label.set_rotation_mode("default")
        else:
            label.set_ha("right")
            label.set_rotation_mode("anchor")


def _xtick_labels_overlap(ax: Axes, fig: Figure) -> bool:
    """Return True when adjacent x-tick label bboxes encroach on the configured gap."""
    fig.canvas.draw()
    renderer = _active_renderer(fig)
    bboxes = sorted(
        (label.get_window_extent(renderer=renderer) for label in ax.get_xticklabels()),
        key=lambda b: b.x0,
    )
    return any(a.x1 + _X_TICK_LABEL_GAP_PX > b.x0 for a, b in pairwise(bboxes))


def _rotate_until_no_overlap(ax: Axes, fig: Figure) -> float:
    """Find the smallest rotation in ``(45, 90)`` that clears overlap; fall back to 90°.

    90° is the last resort: when even vertical labels still touch, callers
    accept the overlap rather than refuse to plot.
    """
    for angle in (45, 90):
        _set_xtick_rotation(ax, angle)
        if not _xtick_labels_overlap(ax, fig):
            return angle
    return 90


def _auto_rotate_categorical_x_ticks(ax: Axes) -> None:
    r"""Wrap or rotate categorical x-tick labels only when horizontal placement would overlap.

    Priority: 0° → wrap multi-word labels onto two lines → 45° → 90° (pandas defaults
    bar labels to 90°). Only ``FixedLocator`` (categorical) axes are touched, and only
    when the current rotation is a matplotlib/pandas default, so explicit choices (e.g.
    heatmap's 45°) are respected. Re-reflows the chrome rect when the rotation changes.

    Disable rotation via ``plot.style.auto_rotate_x_ticks``; wrapping independently via
    ``plot.style.auto_wrap_x_ticks``.
    """
    style = PlotStyleHelper()
    if style.auto_rotate_x_ticks is False:
        return
    if not isinstance(ax.xaxis.get_major_locator(), FixedLocator):
        return

    labels = ax.get_xticklabels()
    if len(labels) <= 1:
        return

    # Respect explicit rotation choices (e.g. heatmap's 45°). Only proceed when
    # the current rotation is one of matplotlib/pandas' uncustomised defaults.
    current_rotation = labels[0].get_rotation()
    if current_rotation not in (0, 90):
        return

    fig = ax.figure
    original_formatter = ax.xaxis.get_major_formatter()
    originals = [label.get_text() for label in labels]

    # Priority sequence: 0° → wrap-and-retry-0° (multi-word only) → 45° → 90°.
    # Replacing the formatter for the wrap attempt is required because
    # pandas/matplotlib regenerate tick labels from the formatter on every
    # draw — calling set_text on the artist directly is reverted on the next
    # canvas.draw().
    _set_xtick_rotation(ax, 0)
    if not _xtick_labels_overlap(ax, fig):
        chosen_rotation: float = 0
    elif style.auto_wrap_x_ticks and (wrapped := [_best_two_line_split(t) for t in originals]) != originals:
        ax.xaxis.set_major_formatter(FixedFormatter(wrapped))
        _set_xtick_rotation(ax, 0)
        if not _xtick_labels_overlap(ax, fig):
            chosen_rotation = 0
        else:
            ax.xaxis.set_major_formatter(original_formatter)
            chosen_rotation = _rotate_until_no_overlap(ax, fig)
    else:
        chosen_rotation = _rotate_until_no_overlap(ax, fig)

    # Re-reflow the chrome rect so the axes data area expands to fill the space
    # vacated by shorter (rotated/wrapped) labels. Without this, the chart shows
    # a large empty gap below the data area when un-rotating from pandas' 90°.
    if chosen_rotation != current_rotation:
        chrome_rect = getattr(fig, "_ors_chrome_rect", None)
        if chrome_rect is not None:
            _reflow_axes(fig, top=chrome_rect[0], bottom=chrome_rect[1])


def apply_ticks(ax: Axes) -> None:
    """Apply tick styling using options."""
    style = PlotStyleHelper()
    # length=0 drops the tick marks; pad compensates for the lost label-to-spine gap.
    ax.tick_params(axis="both", which="both", labelsize=style.tick_size, length=0, pad=6)

    # Only AutoLocator-defaulted axes get the cap; FixedLocator (categorical) and
    # AutoDateLocator (time series) keep their callers' choices.
    for axis in (ax.xaxis, ax.yaxis):
        if isinstance(axis.get_major_locator(), AutoLocator):
            axis.set_major_locator(MaxNLocator(nbins=_MAX_NUMERIC_TICKS, prune=None))

    tick_font_props = get_font_properties(style.tick_font)
    for tick in [
        *ax.xaxis.get_major_ticks(),
        *ax.xaxis.get_minor_ticks(),
        *ax.yaxis.get_major_ticks(),
        *ax.yaxis.get_minor_ticks(),
    ]:
        tick.label1.set_fontproperties(tick_font_props)
        tick.label2.set_fontproperties(tick_font_props)

    _hide_zero_value_ticks(ax)


def _build_outside_legend(ax: Axes, handles: list[Artist | Container], labels: list[str], ncols: int) -> Legend:
    """Build the legend anchored outside the axes, spread over ``ncols`` columns."""
    style = PlotStyleHelper()
    return ax.legend(
        handles,
        labels,
        frameon=False,
        bbox_to_anchor=style.legend_bbox_to_anchor,
        loc=style.legend_loc,
        ncols=ncols,
    )


def _style_legend(legend: Legend, title: str | None) -> None:
    """Apply the configured legend font to ``legend``'s entries and optional ``title``."""
    style = PlotStyleHelper()
    legend_font_props = get_font_properties(style.legend_font)
    if title is not None:
        legend.set_title(title)
        legend.get_title().set_fontproperties(legend_font_props)
        legend.get_title().set_fontsize(style.legend_size)

    for text in legend.get_texts():
        text.set_fontproperties(legend_font_props)
        text.set_fontsize(style.legend_size)


def _fit_outside_legend(ax: Axes, title: str | None) -> None:
    """Wrap an over-tall outside legend into columns, capped at half the figure width.

    Must run after chrome, which settles the axes height and sets ``_ors_chrome_rect``;
    the chrome reflow is repeated so ``tight_layout`` reserves the widened legend's slot.
    """
    legend = ax.get_legend()
    fig = ax.figure
    renderer = _active_renderer(fig)
    axes_height = ax.get_window_extent(renderer=renderer).height
    width_cap = _MAX_OUTSIDE_LEGEND_WIDTH_FRAC * fig.bbox.width
    handles = list(legend.legend_handles)
    labels = [text.get_text() for text in legend.get_texts()]

    # Every candidate is measured: a column can cost more width than the one before it (its
    # labels may be wider), so a budget estimated from the first column's width overshoots the cap.
    ncols = 1
    height = legend.get_window_extent(renderer=renderer).height
    while height > axes_height and ncols < len(handles):
        candidate = _build_outside_legend(ax, handles, labels, ncols + 1)
        _style_legend(candidate, title)
        candidate_box = candidate.get_window_extent(renderer=renderer)
        if candidate_box.width > width_cap:
            _style_legend(_build_outside_legend(ax, handles, labels, ncols), title)
            break
        ncols, height = ncols + 1, candidate_box.height
    if ncols == 1:
        return

    chrome_top, chrome_bottom = fig._ors_chrome_rect
    _reflow_axes(fig, top=chrome_top, bottom=chrome_bottom)


def apply_legend(
    ax: Axes,
    title: str | None = None,
    outside: bool = False,
    *,
    reverse: bool = False,
    custom_labels: list[str] | None = None,
) -> None:
    """Apply legend styling using options.

    Handles are read from ``ax`` via ``get_legend_handles_labels()`` and rebuilt in one
    pass, so reversed order (stacked-area / stacked-bar visual stack) and substituted
    labels happen together; ``custom_labels`` is applied after ``reverse``. Must be
    called from inside ``standard_graph_styles`` so chrome's ``tight_layout`` reserves
    the final legend's slot.

    Args:
        ax: The axes whose labelled artists drive the legend.
        title: Legend title; ``None`` leaves it unset.
        outside: Anchor the legend outside the axes.
        reverse: Reverse the handle and label order before rebuilding.
        custom_labels: Override the labels read off ``ax``; the list is the final on-screen order.

    Raises:
        ValueError: If ``custom_labels`` length does not match the number of legend handles on ``ax``.
    """
    handles, labels = ax.get_legend_handles_labels()
    if reverse:
        handles = list(reversed(handles))
        labels = list(reversed(labels))
    if custom_labels is not None:
        if len(custom_labels) != len(handles):
            msg = f"legend_labels length {len(custom_labels)} != number of legend handles {len(handles)}"
            raise ValueError(msg)
        labels = list(custom_labels)

    legend = (
        _build_outside_legend(ax, handles, labels, ncols=1) if outside else ax.legend(handles, labels, frameon=False)
    )
    _style_legend(legend, title)


def standard_graph_styles(  # noqa: PLR0913
    ax: Axes,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_label_pad: int | None = None,
    y_label_pad: int | None = None,
    legend_title: str | None = None,
    move_legend_outside: bool = False,
    show_legend: bool = True,
    legend_style: Literal["box", "end_of_line"] | None = None,
    legend_reverse: bool = False,
    legend_labels: list[str] | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    source_text: str | None = None,
    grid_axis: GridAxis = "both",
    x_margin: float | None = None,
    hide_spines: bool = False,
) -> Axes:
    """Apply standard styles (spines, grid, ticks, legend, chrome) to a Matplotlib graph.

    With ``legend_style="end_of_line"`` the box legend is suppressed and inline labels are
    drawn at each line's right endpoint after the chrome reflow; ``legend_title`` and
    ``move_legend_outside`` are ignored under it (``UserWarning`` if set).

    Args:
        ax: The graph to apply the styles to.
        title: Chart title.
        x_label: X-axis label; when omitted, pandas' auto label is cleared.
        y_label: Y-axis label; when omitted, pandas' auto label is cleared.
        x_label_pad: Padding below the x-axis label.
        y_label_pad: Padding to the left of the y-axis label.
        legend_title: Legend title.
        move_legend_outside: Anchor the legend outside the plot.
        show_legend: Whether to display the legend.
        legend_style: ``"end_of_line"`` suppresses the box legend for inline end labels.
        legend_reverse: Reverse handle and label order before building the legend; used by
            stacked area/bar plots where column order does not match the visual stack.
        legend_labels: Override the labels read from labelled artists (applied after
            ``legend_reverse``).
        eyebrow: Small uppercase label rendered above the title.
        subtitle: Supporting copy rendered below the title.
        source_text: Footer text rendered italic and muted at the bottom-left of the figure.
        grid_axis: Which axis to draw gridlines on.
        x_margin: Override matplotlib's default x-margin; line/area/time/histogram/
            period_on_period pass ``0`` so first/last points touch the spines.
        hide_spines: Hide all four spines, for plots whose cells define the boundaries
            (heatmap, cohort).

    Returns:
        Axes: The graph with the styles applied.

    Raises:
        ValueError: If ``legend_labels`` length does not match the number of legend handles.
    """
    # Suppress box-legend args before apply_legend runs; keep the pre-resolution
    # show_legend for the end-of-line draw call so single-series charts skip it.
    legend_show, legend_title, move_legend_outside = _resolve_end_of_line_legend_args(
        legend_style=legend_style,
        show_legend=show_legend,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
    )

    apply_base_styling(ax, grid_axis=grid_axis, hide_spines=hide_spines)

    if x_margin is not None:
        ax.margins(x=x_margin)

    # Explicitly clear the pandas-auto labels when the caller didn't pass one in.
    if x_label is not None:
        apply_label(ax, x_label, "x", x_label_pad)
    else:
        ax.set_xlabel("")

    if y_label is not None:
        apply_label(ax, y_label, "y", y_label_pad)
    else:
        ax.set_ylabel("")

    apply_ticks(ax)

    # pandas auto-creates legends on multi-series plots, so show_legend=False
    # has to actively remove the existing one.
    legend_present_or_requested = (
        ax.get_legend() is not None
        or legend_title is not None
        or move_legend_outside
        or legend_reverse
        or legend_labels is not None
    )
    if legend_show and legend_present_or_requested:
        apply_legend(
            ax,
            legend_title,
            move_legend_outside,
            reverse=legend_reverse,
            custom_labels=legend_labels,
        )
    elif not legend_show and ax.get_legend() is not None:
        ax.get_legend().remove()

    apply_chart_chrome(
        ax,
        eyebrow=eyebrow,
        title=title,
        subtitle=subtitle,
        source_text=source_text,
    )

    # Before auto-rotation: widening the legend changes the axes width the rotation test reads.
    if legend_show and move_legend_outside:
        _fit_outside_legend(ax, legend_title)

    # Auto-rotate AFTER chrome so the axes width reflects the final layout
    # (legend placement and chrome's tight_layout both shrink the axes).
    # Measuring at the default size missed overlaps that only appeared once
    # the legend was moved outside and tight_layout reserved its slot.
    _auto_rotate_categorical_x_ticks(ax)

    # End-of-line labels run last so the bump algorithm sees the final axes
    # geometry (chrome's tight_layout + any rotation-induced reflow). Uses the
    # pre-resolution show_legend (not legend_show) so single-series charts skip.
    if legend_style == "end_of_line" and show_legend:
        draw_end_of_line_labels(ax)

    return ax
