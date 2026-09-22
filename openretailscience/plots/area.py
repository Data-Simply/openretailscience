"""Area and stacked-area plots from a pre-aggregated DataFrame.

No resampling or period aggregation is performed; feed pre-aggregated data. For
time-based series that need resampling or period aggregation use plots.time.
"""

from typing import Any, Literal

import pandas as pd
from matplotlib.axes import Axes, SubplotBase

from openretailscience.plots.styles.colors import get_plot_colors
from openretailscience.plots.styles.styling_helpers import standard_graph_styles


def plot(
    df: pd.DataFrame,
    value_col: str | list[str],
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    x_col: str | None = None,
    group_col: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    legend_title: str | None = None,
    move_legend_outside: bool = False,
    legend_style: Literal["box", "end_of_line"] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> SubplotBase:
    """Plot an area chart for `value_col` over `x_col` (or the index), optionally grouped.

    A `value_col` list and a `group_col` are mutually exclusive; the legend is shown only for
    multi-area charts, and alpha defaults to 0.7.

    Args:
        df (pd.DataFrame or pd.Series): Frame to plot; a Series is converted to a single-column frame.
        value_col (str or list[str]): Column(s) to plot.
        x_label (str, optional): X-axis label.
        y_label (str, optional): Y-axis label.
        title (str, optional): Plot title.
        eyebrow (str, optional): Uppercase label rendered above the title.
        subtitle (str, optional): Supporting copy rendered below the title.
        x_col (str, optional): Column used as the x-axis; if None, the index is used.
        group_col (str, optional): Column used to split the data into separate areas.
        ax (Axes, optional): Axes to plot on.
        source_text (str, optional): Source attribution rendered at the bottom.
        legend_title (str, optional): Legend title.
        move_legend_outside (bool, optional): Move the legend outside the plot.
        legend_style ("box", "end_of_line", optional): ``"box"`` renders the standard
            legend; ``"end_of_line"`` suppresses it and labels each series at the right end
            of its line.
        **kwargs: Forwarded to pandas' plot.

    Raises:
        ValueError: If `value_col` is a list and `group_col` are both provided.
        ValueError: If `legend_style` is not None, "box", or "end_of_line".

    """
    if legend_style not in (None, "box", "end_of_line"):
        msg = f"legend_style must be one of (None, 'box', 'end_of_line'); got {legend_style!r}"
        raise ValueError(msg)

    if isinstance(df, pd.Series):
        df = df.to_frame()

    if isinstance(value_col, list) and group_col:
        raise ValueError("Cannot use both a list for `value_col` and a `group_col`. Choose one.")

    if group_col is None:
        pivot_df = df.set_index(x_col if x_col is not None else df.index)[
            [value_col] if isinstance(value_col, str) else value_col
        ]
    else:
        pivot_df = df.pivot(index=x_col if x_col is not None else None, columns=group_col, values=value_col)

    is_multi_area = (group_col is not None) or (isinstance(value_col, list) and len(value_col) > 1)

    num_colors = len(pivot_df.columns) if is_multi_area else 1
    default_colors = get_plot_colors(num_colors)
    alpha = kwargs.pop("alpha", 0.7)
    color = kwargs.pop("color", default_colors)
    ax = pivot_df.plot(
        ax=ax,
        kind="area",
        alpha=alpha,
        color=color,
        legend=is_multi_area,
        **kwargs,
    )

    # Drop the stroke so the swatch reads as a single translucent block matching the fill.
    for collection in ax.collections:
        collection.set_linewidth(0)

    # pandas labels each band's top-edge Line2D with "_childN" so it stays out
    # of the auto legend in favour of the PolyCollection. Mirror the collection
    # label onto the line so draw_end_of_line_labels can pick the series up.
    if legend_style == "end_of_line":
        collection_labels = [c.get_label() for c in ax.collections]
        lines = ax.get_lines()
        if len(lines) == len(collection_labels):
            for line, label in zip(lines, collection_labels, strict=True):
                line.set_label(label)

    # pandas stacks bottom-up but lists series in column order; legend_reverse
    # routes the rebuild through apply_legend so chrome's tight_layout sees the
    # final ordering. No-op under end-of-line, which suppresses the box legend.
    return standard_graph_styles(
        ax=ax,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
        show_legend=is_multi_area,
        legend_style=legend_style,
        legend_reverse=is_multi_area,
        source_text=source_text,
        grid_axis="y",
        x_margin=0,
    )
