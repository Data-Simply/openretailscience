"""Module-level helpers that apply styling using the options system."""

import warnings
from itertools import pairwise
from typing import TYPE_CHECKING, Literal, Protocol, cast

from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoLocator, FixedFormatter, FixedLocator, MaxNLocator

from openretailscience.options import PlotStyleHelper
from openretailscience.plots.styles.font_utils import get_font_properties
from openretailscience.plots.styles.graph_utils import draw_end_of_line_labels

if TYPE_CHECKING:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

GridAxis = Literal["both", "x", "y", "none"]


class _WrappableText(Protocol):
    """The matplotlib-private ``Text._get_wrapped_text`` API, absent from the Text stub."""

    def _get_wrapped_text(self) -> str: ...


def _root_figure(ax: Axes) -> Figure:
    """Return the root ``Figure`` that ``ax`` belongs to.

    ``ax.figure`` is typed ``Figure | SubFigure``; chrome layout uses figure-level
    geometry (``get_figheight``/``transFigure``) only available on the root ``Figure``.
    """
    fig = ax.get_figure(root=True)
    if fig is None:
        raise ValueError("Axes is not attached to a figure.")
    return fig


def _get_renderer(fig: Figure) -> RendererBase:
    """Return the figure's Agg renderer.

    ``get_renderer`` lives on the concrete ``FigureCanvasAgg`` (the non-interactive
    backend these plots render with) but not on the ``FigureCanvasBase`` type stub,
    so ``fig.canvas`` is cast to the Agg canvas before the access.
    """
    return cast("FigureCanvasAgg", fig.canvas).get_renderer()


def _hide_zero_value_ticks(ax: Axes) -> None:
    """Hide tick labels at value 0 on numeric axes (Economist editorial convention).

    The ``0`` tick sits in the bottom-left corner where the x and y spines meet,
    crowding the orthogonal axis's first tick label. The spine itself implies
    the baseline, so the editorial convention drops the redundant label.

    Only acts on numeric continuous axes (``MaxNLocator`` / ``AutoLocator``);
    categorical axes (``FixedLocator``) and date axes (``AutoDateLocator``) are
    untouched, since their position-0 tick is a category index, not a data zero.
    """
    for axis in (ax.xaxis, ax.yaxis):
        if not isinstance(axis.get_major_locator(), MaxNLocator | AutoLocator):
            continue
        # get_major_ticks() returns one Tick per locator position, so the
        # lengths match deterministically. get_majorticklabels() filters by
        # visibility and changes length once this function has already hidden
        # a label or under tick_params(labeltop=True)/labelright=True.
        for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs(), strict=True):
            if loc == 0:
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)


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

END_OF_LINE_LEGEND_CONFLICT_MSG = (
    "`move_legend_outside` and `legend_title` are ignored when legend_style='end_of_line'."
)


def _resolve_end_of_line_legend_args(
    legend_style: Literal["box", "end_of_line"] | None,
    show_legend: bool,
    legend_title: str | None,
    move_legend_outside: bool,
) -> tuple[bool, str | None, bool]:
    """Suppress box-legend args when ``legend_style="end_of_line"`` is requested.

    Returns ``(show_legend, legend_title, move_legend_outside)`` so the box
    legend is neutralised before ``apply_legend`` runs. Pure: never draws on
    ``ax`` — label drawing is deferred until after chrome's ``tight_layout``
    at the end of ``standard_graph_styles``. ``stacklevel=4`` puts the
    conflict warning at user code: user → public ``plot()`` →
    ``standard_graph_styles`` → this helper.
    """
    if legend_style != "end_of_line" or not show_legend:
        return show_legend, legend_title, move_legend_outside
    if move_legend_outside or legend_title is not None:
        warnings.warn(END_OF_LINE_LEGEND_CONFLICT_MSG, UserWarning, stacklevel=4)
    return False, None, False


def _place_header_text(
    ax: Axes,
    x_fig: float,
    prev_bottom_fig_y: float,
    text_str: str,
    font_name: str,
    font_size: float,
    color: str,
    wrap: bool,
    gid: str,
) -> float:
    """Place a header text element with its top at ``prev_bottom_fig_y`` and return its measured bottom.

    Each element is placed and drawn in turn so that wrapped lines from
    the previous element are reflected in the y position of the next.
    """
    fig = _root_figure(ax)
    t = fig.text(
        x_fig,
        prev_bottom_fig_y,
        text_str,
        ha="left",
        va="top",
        fontproperties=get_font_properties(font_name),
        fontsize=font_size,
        color=color,
        wrap=wrap,
    )
    t.set_gid(gid)
    fig.canvas.draw()
    # Freeze matplotlib's wrap: bake the wrapped lines into the text content and
    # turn off `wrap` so the wrap isn't recomputed on subsequent draws. Jupyter
    # saves notebook outputs with bbox_inches='tight', which triggers an extra
    # draw cycle with a slightly different effective figure bbox; left dynamic,
    # the wrap can pick a different line count than the chrome layout planned
    # for, opening an unintended gap between title and subtitle.
    if wrap:
        # _get_wrapped_text is matplotlib-private and absent from the Text stub; it
        # returns the text with matplotlib's computed line breaks baked in.
        wrapped = cast("_WrappableText", t)._get_wrapped_text()
        t.set_text(wrapped)
        t.set_wrap(False)
        fig.canvas.draw()
    renderer = _get_renderer(fig)
    return t.get_window_extent(renderer=renderer).y0 / fig.bbox.height


def _resolve_chrome_left(fig: Figure, ax: Axes) -> float:
    """Discover the y-axis spine's x in figure coords via a provisional tight_layout.

    Anchoring chrome to the spine keeps the title directly above the data
    across charts whose y-tick labels vary in width (numeric "10" vs the
    category labels of a horizontal bar). Falls back to the configured
    margin if the provisional layout pass fails — e.g., for axes types
    that ``tight_layout`` can't reflow.
    """
    top = 1.0 - _CHROME_TOP_MARGIN_IN / fig.get_figheight()
    bottom = _CHROME_BOTTOM_MARGIN_IN / fig.get_figheight()
    if not _reflow_axes(fig, top=top, bottom=bottom):
        return _CHROME_FALLBACK_LEFT_MARGIN
    fig.canvas.draw()
    return float(ax.get_position().x0)


def _reflow_axes(fig: Figure, top: float, bottom: float) -> bool:
    """Reflow the axes into the chrome's vertical band. Returns False on failure.

    ``pad=0`` is critical: tight_layout's default pad (1.08 * font size) adds
    ~3% of figure height between the rect top and the axes top, doubling the
    intended gap. With pad=0, ``_CHROME_GAP_HEADER_TO_AXES_IN`` is the *only*
    whitespace between subtitle and plot. Failure happens when
    ``constrained_layout`` is active or the rect is degenerate — callers
    decide whether to fall back to ``subplots_adjust``.
    """
    try:
        fig.tight_layout(
            pad=0,
            rect=(_CHROME_LEFT_AXES_MARGIN, bottom, 1.0 - _CHROME_RIGHT_AXES_MARGIN, top),
        )
    except (ValueError, RuntimeError):
        return False
    return True


def _clear_prior_chrome(fig: Figure, chrome_gid: str) -> None:
    """Remove chrome artists left on ``fig`` by a previous call for this gid.

    Iterative replotting (Jupyter, parametrized tests) otherwise stacks
    duplicate text and tabs at the same figure coordinates. Texts come from
    ``fig.text()`` and support ``.remove()``; the tab is appended to
    ``fig.patches`` directly so it has no ``_remove_method`` and must be
    popped off the list instead.
    """
    for text in [t for t in fig.texts if t.get_gid() == chrome_gid]:
        text.remove()
    for patch in [p for p in fig.patches if p.get_gid() == chrome_gid]:
        fig.patches.remove(patch)


def _track_chrome_axes(fig: Figure, ax: Axes, *, warn_stacklevel: int) -> None:
    """Record ``ax`` on ``fig._ors_chrome_axes`` and warn the first time a second axes appears.

    Chrome is positioned in figure coordinates and is not subplot-aware, so
    multiple chrome-bearing axes on the same figure can shift each other's
    chrome out of alignment. The warning fires once, at the transition from
    one tracked axes to two.
    """
    prior = getattr(fig, "_ors_chrome_axes", None)
    if prior is None:
        # _ors_chrome_axes is a dynamic attribute we stash on the figure; setattr
        # avoids a stub error since Figure doesn't declare it.
        setattr(fig, "_ors_chrome_axes", [ax])  # noqa: B010
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

    Sequential layout: each header element is placed and drawn in turn so
    the next element starts directly below the previous one's measured
    bbox. Wrapped titles therefore push subsequent elements down by their
    actual rendered height, never their estimated height. After the header
    and source are placed, ``tight_layout`` reflows the axes (with their
    tick and axis labels) into the remaining vertical band.

    Each absent element collapses its slot, so a chart with only a title
    takes only the vertical space the title needs.

    Args:
        ax: The plot axes (used to get the figure handle).
        eyebrow: Small uppercase label above the title.
        title: Main headline. Wraps if it would exceed the figure width.
        subtitle: Supporting copy below the title. Wraps.
        source_text: Footer text (rendered italic, muted).
        warn_stacklevel: Stacklevel for the multi-axes chrome warning. Defaults to 4
            so the warning points at user code when reached via
            ``standard_graph_styles`` from a public ``*.plot`` function. Callers
            that invoke ``apply_chart_chrome`` directly from a public entry point
            (e.g. ``venn.plot``) should pass ``3``.

    The small green tab mark above the title block is controlled by the
    ``plot.style.show_tab`` option (default True). Set the option to False to
    suppress it project-wide; use ``option_context`` to scope the change.
    """
    style = PlotStyleHelper()
    fig = _root_figure(ax)
    show_tab = style.show_tab
    has_header = eyebrow is not None or title is not None or subtitle is not None
    any_chrome = has_header or source_text is not None
    chrome_gid = f"_ors_chrome:{id(ax)}"

    _clear_prior_chrome(fig, chrome_gid)

    if any_chrome:
        _track_chrome_axes(fig, ax, warn_stacklevel=warn_stacklevel)

    chrome_x = _resolve_chrome_left(fig, ax) if any_chrome else _CHROME_FALLBACK_LEFT_MARGIN

    fig_h = fig.get_figheight()
    cur_y = 1.0 - _CHROME_TOP_MARGIN_IN / fig_h

    # Always reserve the tab's vertical slot when a header is present, even when
    # the tab is hidden. This keeps the title's distance from the figure edge
    # consistent whether the tab is drawn or hidden, so suppressing the tab
    # doesn't push the title against the top margin.
    if has_header:
        tab_height_fig = _CHROME_TAB_HEIGHT_IN / fig_h
        tab_to_eyebrow_fig = _CHROME_TAB_TO_EYEBROW_GAP_IN / fig_h
        if show_tab:
            tab_width_fig = _CHROME_TAB_WIDTH_IN / fig.get_figwidth()
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
        cur_y -= tab_height_fig + tab_to_eyebrow_fig

    # Eyebrow's gap is always applied; the title→subtitle gap only when both
    # are present. Subtitle has no trailing gap — _CHROME_GAP_HEADER_TO_AXES_IN
    # below covers the whitespace down to the plot.
    header_specs = (
        (
            eyebrow.upper() if eyebrow is not None else None,
            style.eyebrow_font,
            style.eyebrow_size,
            style.eyebrow_color,
            False,
            _CHROME_GAP_EYEBROW_TO_TITLE_IN / fig_h,
        ),
        (
            title,
            style.title_font,
            style.title_size,
            style.title_color,
            True,
            _CHROME_GAP_TITLE_TO_SUBTITLE_IN / fig_h if subtitle is not None else 0.0,
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
    for text, font, size, color, wrap, gap_after in header_specs:
        if text is None:
            continue
        cur_y = _place_header_text(ax, chrome_x, cur_y, text, font, size, color, wrap=wrap, gid=chrome_gid)
        cur_y -= gap_after

    # tight_layout reserves most top tick-label height inside [bottom, header_bottom],
    # but its default label padding leaves a little less visible subtitle-to-content
    # whitespace than the bottom-label case. Add only the residual headroom needed
    # to keep that visible gap consistent.
    has_top_labels = any(t.label2.get_visible() for t in [*ax.xaxis.get_major_ticks(), *ax.yaxis.get_major_ticks()])
    extra_gap_in = style.tick_size * _CHROME_TOP_LABEL_HEADROOM_FACTOR / 72.0 if has_top_labels else 0.0
    header_bottom = cur_y - (_CHROME_GAP_HEADER_TO_AXES_IN + extra_gap_in) / fig_h if has_header else cur_y

    bottom_margin_fig = _CHROME_BOTTOM_MARGIN_IN / fig_h
    axes_bottom = bottom_margin_fig
    if source_text is not None:
        source_t = fig.text(
            chrome_x,
            bottom_margin_fig,
            source_text,
            ha="left",
            va="bottom",
            fontproperties=get_font_properties(style.source_font),
            fontsize=style.source_size,
            color=style.source_color,
            wrap=True,
        )
        source_t.set_gid(chrome_gid)
        fig.canvas.draw()
        source_top_px = source_t.get_window_extent(renderer=_get_renderer(fig)).y1
        axes_bottom = source_top_px / fig.bbox.height + _CHROME_GAP_SOURCE_TO_AXES_IN / fig_h

    # Cache the chrome rect so callers (e.g. _auto_rotate_categorical_x_ticks)
    # can re-reflow after they change tick label heights. Without this, a
    # post-chrome rotation/wrap leaves the axes box sized for the original
    # pandas-90° labels and you get a tall empty gap below the data area.
    # _ors_chrome_rect is a dynamic attribute we stash on the figure; setattr
    # avoids a stub error since Figure doesn't declare it.
    setattr(fig, "_ors_chrome_rect", (header_bottom, axes_bottom))  # noqa: B010

    if not _reflow_axes(fig, top=header_bottom, bottom=axes_bottom):
        fig.subplots_adjust(top=header_bottom, bottom=axes_bottom)


def apply_base_styling(ax: Axes, grid_axis: GridAxis = "both", hide_spines: bool = False) -> None:
    """Apply base plot styling (spines, grid, background) using options.

    Args:
        ax: The axes to style.
        grid_axis: Which axis grid lines to draw. ``"both"`` draws horizontal and vertical
            lines, ``"x"`` draws only vertical (helpful when reading off the x-axis, e.g.
            horizontal bars), ``"y"`` draws only horizontal (the common case for line, area,
            and vertical-bar charts), and ``"none"`` suppresses both.
        hide_spines: If True, hide all four axis spines regardless of the per-spine style
            options. Use for plots where cell colours or shapes already define the
            boundaries (heatmap, cohort) and spines would only repeat that information.
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
            label.set_horizontalalignment("center")
            label.set_rotation_mode("default")
        else:
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")


def _xtick_labels_overlap(ax: Axes, fig: Figure) -> bool:
    """Return True when adjacent x-tick label bboxes encroach on the configured gap."""
    fig.canvas.draw()
    renderer = _get_renderer(fig)
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

    Pandas' ``df.plot(kind="bar")`` defaults to rot=90, which forces vertical
    labels even for short category names like "North"/"South". We override that:
    try 0° first; if neighbours overlap, wrap multi-word labels onto two lines
    ("Camden High St" → "Camden\nHigh St") and re-test; then fall back to 45°
    (right-anchored), and finally 90° only when none of the above fits.

    Disable rotation via ``plot.style.auto_rotate_x_ticks``. Disable wrapping
    independently via ``plot.style.auto_wrap_x_ticks``.
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

    fig = _root_figure(ax)
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


def apply_legend(
    ax: Axes,
    title: str | None = None,
    outside: bool = False,
    *,
    reverse: bool = False,
    custom_labels: list[str] | None = None,
) -> None:
    """Apply legend styling using options.

    Handles are read from ``ax`` via ``get_legend_handles_labels()`` so the legend
    can be rebuilt with reversed order (stacked-area / stacked-bar visual stack)
    or substituted labels (e.g. column ids → human-readable names) in a single
    build. Calling this once from ``standard_graph_styles`` rather than after it
    ensures chrome's ``tight_layout`` reserves the slot matching the final legend.

    Args:
        ax (Axes): The axes whose labelled artists drive the legend.
        title (str | None): Legend title; ``None`` leaves it unset.
        outside (bool): Anchor the legend outside the axes when ``True``.
        reverse (bool): Reverse the handle and label order before rebuilding.
        custom_labels (list[str] | None): Override the labels read off ``ax``.
            Applied after ``reverse``, so the list is the final on-screen order.

    Raises:
        ValueError: If ``custom_labels`` is provided and its length does not
            match the number of legend handles on ``ax``.
    """
    style = PlotStyleHelper()
    handles, labels = ax.get_legend_handles_labels()
    if reverse:
        handles = list(reversed(handles))
        labels = list(reversed(labels))
    if custom_labels is not None:
        if len(custom_labels) != len(handles):
            msg = f"legend_labels length {len(custom_labels)} != number of legend handles {len(handles)}"
            raise ValueError(msg)
        labels = list(custom_labels)

    if outside:
        legend = ax.legend(
            handles,
            labels,
            frameon=False,
            bbox_to_anchor=style.legend_bbox_to_anchor,
            loc=style.legend_loc,
        )
    else:
        legend = ax.legend(handles, labels, frameon=False)

    legend_font_props = get_font_properties(style.legend_font)
    if title is not None:
        legend.set_title(title)
        legend.get_title().set_fontproperties(legend_font_props)
        legend.get_title().set_fontsize(style.legend_size)

    for text in legend.get_texts():
        text.set_fontproperties(legend_font_props)
        text.set_fontsize(style.legend_size)


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
    """Apply standard styles to a Matplotlib graph using styling helpers.

    Args:
        ax (Axes): The graph to apply the styles to.
        title (str, optional): The title of the graph. Defaults to None.
        x_label (str, optional): The x-axis label. Defaults to None.
        y_label (str, optional): The y-axis label. Defaults to None.
        x_label_pad (int, optional): The padding below the x-axis label. Defaults to styling context default.
        y_label_pad (int, optional): The padding to the left of the y-axis label. Defaults to styling context default.
        legend_title (str, optional): The title of the legend. If None, no legend title is applied. Defaults to None.
        move_legend_outside (bool, optional): Whether to move the legend outside the plot. Defaults to False.
        show_legend (bool): Whether to display the legend or not.
        legend_style (Literal["box", "end_of_line"], optional): When ``"end_of_line"``, suppress the box
            legend and draw inline labels at each line's right endpoint after chrome reflow. ``"box"`` and
            ``None`` leave the legend behaviour unchanged. ``legend_title`` and ``move_legend_outside`` are
            ignored under ``"end_of_line"`` and emit a UserWarning if set. Defaults to None.
        legend_reverse (bool, optional): Reverse handle and label order before building the legend.
            Used by stacked area/bar plots where the column-order legend doesn't match the visual
            stack (bottom-up). Defaults to False.
        legend_labels (list[str] | None, optional): Override the labels read from labelled artists
            (e.g. swap column ids for human-readable names). Applied after ``legend_reverse``.
            Length must match the number of legend handles or ``ValueError`` is raised. Defaults to None.
        eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
        subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
        source_text (str, optional): Footer text rendered italic and muted at the bottom-left of the figure.
            The chrome layout engine reserves room for it.
        grid_axis (Literal["both", "x", "y", "none"], optional): Which axis to draw gridlines on.
            Defaults to ``"both"``.
        x_margin (float, optional): If set, override matplotlib's default x-margin. Editorial line/area/time
            charts pass ``0`` so the first data point sits on the spine and the last reaches the right edge.
        hide_spines (bool, optional): If True, hide all four axis spines. Use for plots whose cell
            colours or shapes already define their boundaries (heatmap, cohort). Defaults to False.

    Returns:
        Axes: The graph with the styles applied.
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
    elif not legend_show and (existing_legend := ax.get_legend()) is not None:
        existing_legend.remove()

    apply_chart_chrome(
        ax,
        eyebrow=eyebrow,
        title=title,
        subtitle=subtitle,
        source_text=source_text,
    )

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
