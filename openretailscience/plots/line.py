"""Line plots of pre-aggregated data, with one line per value column or group value.

For x-axes that are relative or sequential values ("days since an event", "months since a
competitor opened"), use ``line.plot``. For an actual datetime x-axis use ``plots.time``, which
resamples and aggregates; this module does not aggregate and only pivots its input.
"""

from typing import Any, Literal

import pandas as pd
from matplotlib.axes import Axes, SubplotBase

from openretailscience.plots.styles.colors import get_named_color, get_plot_colors
from openretailscience.plots.styles.styling_helpers import standard_graph_styles


def _validate_and_prepare_input(
    df: pd.DataFrame | pd.Series,
    value_col: str | list[str] | None,
    x_col: str | None,
    group_col: str | None,
) -> tuple[pd.DataFrame, str | list[str]]:
    """Validate input parameters and convert Series to DataFrame if needed."""
    # Handle Series input
    if isinstance(df, pd.Series):
        if value_col is not None:
            raise ValueError(
                "When df is a pd.Series, value_col must be None. The Series itself represents the values to plot.",
            )
        if x_col is not None:
            raise ValueError(
                "When df is a pd.Series, x_col must be None. The Series index is used as the x-axis.",
            )
        if group_col is not None:
            raise ValueError(
                "When df is a pd.Series, group_col must be None. Cannot group a single series.",
            )
        # Convert Series to DataFrame for uniform processing
        series_name = df.name if df.name is not None else "value"
        df = df.to_frame(name=series_name)
        value_col = series_name

    # Validate value_col for DataFrame input
    if value_col is None:
        raise ValueError("value_col is required when df is a DataFrame")

    if isinstance(value_col, list) and group_col:
        raise ValueError("Cannot use both a list for `value_col` and a `group_col`. Choose one.")

    return df, value_col


def _validate_highlight_parameter(
    highlight: str | list[str] | None,
    value_col: str | list[str],
    group_col: str | None,
    pivot_df: pd.DataFrame,
) -> list[str] | None:
    """Validate and normalize the ``highlight`` parameter.

    Membership is checked against the pivot columns rather than ``ensure_columns`` because, after
    pivoting on a non-string ``group_col`` (e.g. integer store ids), ``pivot_df.columns`` can hold
    non-string labels that ``ensure_columns`` would reject purely on type.
    """
    if highlight is None:
        return None

    is_single_line = group_col is None and (
        isinstance(value_col, str) or (isinstance(value_col, list) and len(value_col) == 1)
    )
    if is_single_line:
        raise ValueError("highlight parameter cannot be used with single-line plots")

    normalized = [highlight] if isinstance(highlight, str) else list(highlight)
    if len(normalized) == 0:
        raise ValueError("highlight must not be an empty list; pass None to disable highlighting.")

    invalid = [h for h in normalized if h not in pivot_df.columns]
    if len(invalid) > 0:
        msg = f"highlight references values not present in the pivoted columns: {invalid}."
        raise ValueError(msg)

    return normalized


def _create_pivot_dataframe(
    df: pd.DataFrame,
    value_col: str | list[str],
    x_col: str | None,
    group_col: str | None,
    fill_na_value: float | None,
) -> pd.DataFrame:
    """Create pivot DataFrame for plotting."""
    if group_col is None:
        pivot_df = df.set_index(x_col if x_col is not None else df.index)[
            [value_col] if isinstance(value_col, str) else value_col
        ]
    else:
        pivot_df = (
            df.pivot(columns=group_col, values=value_col)
            if x_col is None
            else df.pivot(index=x_col, columns=group_col, values=value_col)
        )
        if fill_na_value is not None:
            pivot_df = pivot_df.fillna(fill_na_value)

    return pivot_df


def _categorize_columns(
    pivot_df: pd.DataFrame,
    highlight: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Categorize columns into highlighted and context groups."""
    if highlight is not None:
        highlighted_cols = [col for col in pivot_df.columns if col in highlight]
        context_cols = [col for col in pivot_df.columns if col not in highlight]
    else:
        highlighted_cols = list(pivot_df.columns)
        context_cols = []

    return highlighted_cols, context_cols


def _generate_colors(highlighted_cols: list[str]) -> list[str]:
    """Generate colors for highlighted lines."""
    num_highlighted = len(highlighted_cols) if highlighted_cols else 1
    return get_plot_colors(num_highlighted)


def _render_plot(
    pivot_df: pd.DataFrame,
    highlighted_cols: list[str],
    context_cols: list[str],
    highlighted_colors: list[str],
    is_multi_line: bool,
    ax: Axes | None,
    **kwargs: Any,  # noqa: ANN401
) -> Axes:
    """Render the actual plot with context and highlighted lines."""
    # Context lines first (lower z-order). Underscore-prefix the column names so matplotlib
    # auto-excludes them from any legend rendered later
    if len(context_cols) > 0:
        context_df = pivot_df[context_cols].rename(columns={c: f"_{c}" for c in context_cols})
        context_kwargs = {k: v for k, v in kwargs.items() if k not in ["color", "alpha", "zorder", "linewidth"]}
        ax = context_df.plot(
            ax=ax,
            linewidth=1.0,
            color=get_named_color("context"),
            legend=False,
            zorder=1,
            **context_kwargs,
        )

    highlighted_df = pivot_df[highlighted_cols]
    return highlighted_df.plot(
        ax=ax,
        linewidth=kwargs.pop("linewidth", 3),
        color=kwargs.pop("color", highlighted_colors),
        legend=is_multi_line,
        zorder=2,
        **kwargs,
    )


def plot(  # noqa: PLR0913
    df: pd.DataFrame | pd.Series,
    value_col: str | list[str] | None = None,
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
    fill_na_value: float | None = None,
    highlight: str | list[str] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> SubplotBase:
    """Plot ``value_col`` over ``x_col`` or the index, with a separate line per unique ``group_col`` value.

    When ``df`` is a Series, its values are plotted against its index and ``value_col``,
    ``x_col``, and ``group_col`` must all be None. For a DataFrame, ``value_col`` is required,
    and a list of columns cannot be combined with ``group_col``.

    Args:
        df (pd.DataFrame | pd.Series): The dataframe or series to plot.
        value_col (str | list[str], optional): The column(s) to plot; a list renders one line per
            column. Required when ``df`` is a DataFrame.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        title (str, optional): The title of the plot.
        eyebrow (str, optional): Small uppercase label rendered above the title.
        subtitle (str, optional): Supporting copy rendered below the title.
        x_col (str, optional): The column to use as the x-axis. If None, the index is used.
        group_col (str, optional): The column used to define different lines.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): The source text to add to the plot.
        legend_title (str, optional): The title of the legend.
        move_legend_outside (bool, optional): Move the legend outside the plot.
        legend_style (Literal["box", "end_of_line"], optional): How series are labelled. ``"box"``
            (default when None) renders the standard matplotlib legend. ``"end_of_line"`` suppresses
            the legend and places a colored series label at the right end of each line; it is a no-op
            on single-line plots. When a legend would show, ``move_legend_outside`` and
            ``legend_title`` are ignored and a UserWarning is emitted if either is supplied.
        fill_na_value (float, optional): Value to fill NaNs with after pivoting. Applied only when
            ``group_col`` is set; silently ignored otherwise.
        highlight (str | list[str], optional): Line(s) to emphasize; only for multi-line plots. With
            ``group_col`` these are group values; with a list of ``value_col`` these are column
            names. Highlighted lines use the ``linewidth`` kwarg (default 3) and the ``color`` kwarg
            (default palette colors); non-highlighted context lines are rendered behind them in the
            ``plot.color.context`` color (default #d1d5db, option-configurable) with a 1.0 linewidth.
        **kwargs: Additional keyword arguments for pandas' ``plot`` function. ``linewidth`` and
            ``color`` are consumed by the highlighted lines and do not apply to context lines.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If value_col is a list and group_col is provided.
        ValueError: If df is a Series and value_col is not None.
        ValueError: If df is a Series and x_col is specified (the Series index is the x-axis).
        ValueError: If df is a Series and group_col is specified (cannot group a single series).
        ValueError: If df is a DataFrame and value_col is None.
        ValueError: If highlight is provided for a single-line plot.
        ValueError: If highlight is an empty list.
        ValueError: If highlight values do not match available groups/columns.
        ValueError: If legend_style is not one of ``None``, ``"box"``, or ``"end_of_line"``.
    """
    if legend_style not in (None, "box", "end_of_line"):
        msg = f"legend_style must be one of (None, 'box', 'end_of_line'); got {legend_style!r}"
        raise ValueError(msg)

    df, value_col = _validate_and_prepare_input(df, value_col, x_col, group_col)
    pivot_df = _create_pivot_dataframe(df, value_col, x_col, group_col, fill_na_value)
    highlight = _validate_highlight_parameter(highlight, value_col, group_col, pivot_df)

    highlighted_cols, context_cols = _categorize_columns(pivot_df, highlight)
    highlighted_colors = _generate_colors(highlighted_cols)

    is_multi_line = (group_col is not None) or (isinstance(value_col, list) and len(value_col) > 1)

    ax = _render_plot(pivot_df, highlighted_cols, context_cols, highlighted_colors, is_multi_line, ax, **kwargs)

    return standard_graph_styles(
        ax=ax,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
        show_legend=is_multi_line,
        legend_style=legend_style,
        source_text=source_text,
        grid_axis="y",
        x_margin=0,
    )
