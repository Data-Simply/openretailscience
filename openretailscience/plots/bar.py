"""This module provides flexible functionality for creating bar plots from pandas DataFrames or Series.

It allows you to create bar plots with optional grouping, sorting, orientation, and data labels. The module supports
both single and grouped bar plots, where grouped bars are created by providing a `x_col`, which defines the x-axis
labels or categories.

### Features

- **Single or Grouped Bar Plots**: Plot one or more value columns (`value_col`) as bars. The `x_col` is used to define
categories or groups on the x-axis (e.g., products, categories, or regions). Grouped bars can be created by
specifying both `value_col` (list of columns) and `x_col`.
- **Sorting and Orientation**: Customize the sorting of bars (ascending or descending) and choose between
vertical (`"v"`, `"vertical"`) or horizontal (`"h"`, `"horizontal"`) bar orientations.
- **Data Labels**: Add data labels to bars, with options to show absolute values or percentages.
- **Hatching Patterns**: Apply hatch patterns to the bars for enhanced visual differentiation.
- **Legend Customization**: Move the legend outside the plot for better visibility, especially when dealing
with grouped bars or multiple value columns.


### Use Cases

- **Sales and Revenue Analysis**: Visualize sales or revenue across different products or categories by creating grouped
  bar plots (e.g., revenue across quarters or regions). The `x_col` will define the products or categories displayed on
  the x-axis.
- **Comparative Analysis**: Compare multiple metrics simultaneously by plotting grouped bars. For instance, you can
  compare product sales for different periods side by side, with `x_col` defining the x-axis categories.
- **Distribution Analysis**: Visualize the distribution of categorical data (e.g., product sales) across different
  categories, where `x_col` defines the x-axis labels.

### Limitations and Handling of Data
- **Series Support**: The module can also handle pandas Series, though **`x_col`** cannot be provided when plotting a
Series.
  In this case, the index of the Series will define the x-axis labels.
"""

import warnings
from collections.abc import Iterable
from typing import Any, Literal

import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from matplotlib.container import BarContainer
from matplotlib.patches import Rectangle

import openretailscience.plots.styles.graph_utils as gu
from openretailscience.core.validation import ensure_columns, ensure_value_choice
from openretailscience.options import PlotStyleHelper
from openretailscience.plots.styles.colors import get_plot_colors
from openretailscience.plots.styles.font_utils import get_font_properties
from openretailscience.plots.styles.styling_helpers import standard_graph_styles

VALID_ORIENTATIONS = ("horizontal", "h", "vertical", "v")
VALID_SORT_ORDERS = ("asc", "ascending", "desc", "descending")
VALID_DATA_LABEL_FORMATS = ("absolute", "percentage_by_bar_group", "percentage_by_series")
DEFAULT_BAR_WIDTH = 0.8


def _validate_bar_inputs(
    df: pd.DataFrame | pd.Series,
    x_col: str | None,
    orientation: str,
    sort_order: str | None,
    data_label_format: str | None,
) -> tuple[str, str | None, str | None]:
    """Validate ``plot`` arguments up-front and return the case-normalized enum values."""
    if df.empty:
        raise ValueError("Cannot plot with empty DataFrame")
    if x_col is not None:
        ensure_columns(df, x_col, "x_col")
    orientation = ensure_value_choice(orientation, VALID_ORIENTATIONS, "orientation")
    if sort_order is not None:
        sort_order = ensure_value_choice(sort_order, VALID_SORT_ORDERS, "sort_order")
    if data_label_format is not None:
        data_label_format = ensure_value_choice(data_label_format, VALID_DATA_LABEL_FORMATS, "data_label_format")
    return orientation, sort_order, data_label_format


def plot(  # noqa: PLR0913
    df: pd.DataFrame | pd.Series,
    value_col: str | list[str] | None = None,
    x_col: str | None = None,
    title: str | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    legend_title: str | None = None,
    legend_labels: list[str] | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    move_legend_outside: bool = False,
    orientation: Literal["horizontal", "h", "vertical", "v"] = "vertical",
    sort_order: Literal["asc", "ascending", "desc", "descending"] | None = None,
    data_label_format: Literal["absolute", "percentage_by_bar_group", "percentage_by_series"] | None = None,
    use_hatch: bool = False,
    num_digits: int = 3,
    **kwargs: Any,  # noqa: ANN401
) -> SubplotBase:
    """Creates a customizable bar plot from a DataFrame or Series.

    Supports optional sorting, orientation, and data labels. Grouped bars can be created with a grouping column.

    Args:
        df (pd.DataFrame | pd.Series): The input DataFrame or Series containing the data to be plotted.
        value_col (str | list[str], optional): The column(s) containing values to plot as bars. Multiple value columns
                                               create grouped bars. Defaults to None.
        x_col (str, optional): The column to group data by, used for grouping bars. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None.
        eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
        subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
        x_label (str, optional): The label for the x-axis. Defaults to None.
        y_label (str, optional): The label for the y-axis. Defaults to None.
        legend_title (str, optional): The title for the legend. Defaults to None.
        legend_labels (list[str], optional): Override the legend labels read from the plotted series
            (e.g. swap column ids for human-readable names). Length must match the number of
            value columns or ``ValueError`` is raised. Defaults to None.
        ax (Axes, optional): The Matplotlib Axes object to plot on. Defaults to None.
        source_text (str, optional): Text to be displayed as a source at the bottom of the plot. Defaults to None.
        move_legend_outside (bool, optional): Whether to move the legend outside the plot area. Defaults to False.
        orientation (Literal["horizontal", "h", "vertical", "v"], optional): Orientation of the bars. Can be
                                                                             "horizontal", "h", "vertical", or "v".
                                                                             Defaults to "vertical".
        sort_order (Literal["ascending", "descending"] | None, optional): Sorting order for the bars. Can be
                                                                          "ascending" or "descending". Defaults to None.
        data_label_format (Literal["absolute", "percentage"] | None, optional): Format for displaying data labels.
                                                                                "absolute" shows raw values,
                                                                                "percentage" shows percentage.
                                                                                Defaults to None.
        use_hatch (bool, optional): Whether to apply hatch patterns to the bars. Defaults to False.
        num_digits (int, optional): The number of digits to display in the data labels. Defaults to 3.
        **kwargs (Any): Additional keyword arguments for the Pandas `plot` function.

    Returns:
        SubplotBase: The Matplotlib Axes object with the generated plot.
    """
    orientation, sort_order, data_label_format = _validate_bar_inputs(
        df,
        x_col,
        orientation,
        sort_order,
        data_label_format,
    )

    width = kwargs.pop("width", DEFAULT_BAR_WIDTH)

    if value_col is None:
        value_col = ["Value"]
    elif isinstance(value_col, str):
        value_col = [value_col]

    df = df.to_frame(name=value_col[0]) if isinstance(df, pd.Series) else df

    if data_label_format in ["percentage_by_bar_group", "percentage_by_series"] and (df[value_col] < 0).any().any():
        warnings.warn(
            f"Negative values detected in {value_col}. This may lead to unexpected behavior in terms of the data "
            f"label format '{data_label_format}'.",
            UserWarning,
            stacklevel=2,
        )

    df = df.sort_values(by=value_col[0], ascending=sort_order in ("asc", "ascending")) if sort_order is not None else df

    default_colors = get_plot_colors(len(value_col))

    plot_kind = "bar" if orientation in ["vertical", "v"] else "barh"
    color = kwargs.pop("color", default_colors)
    legend = kwargs.pop("legend", (len(value_col) > 1))

    ax = df.plot(
        kind=plot_kind,
        y=value_col,
        x=x_col,
        ax=ax,
        width=width,
        color=color,
        legend=legend,
        **kwargs,
    )

    if use_hatch:
        ax = gu.apply_hatches(ax=ax, num_segments=len(value_col))

    # Add data labels
    if data_label_format:
        _generate_bar_labels(
            ax=ax,
            plot_kind=plot_kind,
            value_col=value_col,
            df=df,
            data_label_format=data_label_format,
            x_col=x_col if x_col is not None else df.index,
            is_stacked=kwargs.get("stacked", False),
            num_digits=num_digits,
        )

    return standard_graph_styles(
        ax=ax,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        legend_labels=legend_labels,
        move_legend_outside=move_legend_outside,
        source_text=source_text,
        # Gridlines on the value axis only — vertical bars read off y, horizontal off x.
        grid_axis="x" if plot_kind == "barh" else "y",
    )


def _generate_bar_labels(
    ax: Axes,
    plot_kind: str,
    value_col: list[str],
    df: pd.DataFrame,
    data_label_format: Literal[
        "absolute",
        "percentage",
        "percentage_by_bar_group",
        "percentage_by_series",
    ],
    x_col: str | pd.Index,
    is_stacked: bool,
    num_digits: int = 3,
) -> None:
    """Adds bar labels to the bar plot containers."""
    division_by_zero_list = []  # A list to track occurrences of division by zero
    total_sum_per_column = df[value_col].sum()  # Series with a total for each column

    for container, column in zip(ax.containers, value_col, strict=False):
        if not isinstance(container, BarContainer):
            msg = f"Expected BarContainer, got {type(container).__name__}"
            raise TypeError(msg)
        if data_label_format == "absolute":
            container_labels = _generate_absolute_labels(container, plot_kind, num_digits)
        elif data_label_format == "percentage_by_bar_group":
            group_totals = df.groupby(x_col)[value_col].sum().sum(axis=1)
            container_labels = _generate_percentage_labels(
                container,
                group_totals,
                plot_kind,
                num_digits,
                division_by_zero_list,
            )
        elif data_label_format == "percentage_by_series":
            column_total = total_sum_per_column[column]
            container_labels = _generate_percentage_labels(
                container,
                [column_total] * len(container),
                plot_kind,
                num_digits,
                division_by_zero_list,
            )
        else:
            msg = f"Unhandled data_label_format: {data_label_format}"
            raise ValueError(msg)

        _apply_labels_to_container(ax, container, container_labels, data_label_format, is_stacked)

    # Check if any division by zero occurred and how many times
    if division_by_zero_list:
        division_count = len(division_by_zero_list)
        warnings.warn(
            f"Division by zero detected {division_count} time(s), will skip displaying percentages for certain bars.",
            UserWarning,
            stacklevel=2,
        )


def _get_bar_value(v: Rectangle, plot_type: str) -> float:
    """Retrieve the raw value from the bar/rectangle.

    Args:
        v (Rectangle): The bar/rectangle object.
        plot_type (str): The type of plot ('bar' for vertical, 'barh' for horizontal).

    Returns:
        float: The value represented by the bar (height or width).
    """
    return v.get_height() if plot_type == "bar" else v.get_width()


def _generate_absolute_labels(
    container: BarContainer,
    plot_kind: str,
    num_digits: int,
) -> list[str]:
    """Generate absolute value labels for the bars.

    Args:
        container (BarContainer): The container holding the bar objects.
        plot_kind (str): The type of plot ('bar' or 'barh').
        num_digits (int): The number of digits to display in the labels.

    Returns:
        list[str]: A list of formatted labels.
    """
    return [
        gu.truncate_to_x_digits(
            num_str=gu.format_shorthand(num=_get_bar_value(v, plot_kind), decimals=num_digits),
            digits=num_digits,
        )
        for v in container
    ]


def _generate_percentage_labels(
    container: BarContainer,
    denominators: Iterable[float],
    plot_kind: str,
    num_digits: int,
    division_by_zero_list: list,
) -> list[str]:
    """Generate percentage labels for each bar against a per-bar denominator.

    Args:
        container (BarContainer): The container holding the bar objects.
        denominators (Iterable[float]): Per-bar denominators aligned with the container. Pass a Series for
            per-bar-group totals or a broadcast scalar (e.g. ``[total] * len(container)``) for a shared series total.
        plot_kind (str): The type of plot ('bar' or 'barh').
        num_digits (int): The number of digits to display in the labels.
        division_by_zero_list (list): A list to track occurrences of division by zero.

    Returns:
        list[str]: A list of formatted percentage labels.
    """
    labels = []
    for v, denom in zip(container, denominators, strict=True):
        if denom == 0:
            division_by_zero_list.append(True)
            labels.append("")
        else:
            bar_value = _get_bar_value(v, plot_kind)
            percentage_value = (bar_value / denom) * 100
            labels.append(
                gu.truncate_to_x_digits(
                    num_str=gu.format_shorthand(num=percentage_value, decimals=num_digits),
                    digits=num_digits,
                ),
            )
    return labels


def _apply_labels_to_container(
    ax: Axes,
    container: BarContainer,
    container_labels: list[str],
    data_label_format: str,
    is_stacked: bool,
) -> None:
    """Apply the formatted labels to the bar container.

    Args:
        ax (Axes): The matplotlib axes object containing the plot.
        container (BarContainer): The container holding the bar objects.
        container_labels (list[str]): A list of formatted labels to apply.
        data_label_format (str): The format of the labels (e.g., 'absolute', 'percentage').
        is_stacked (bool): Whether the bars are stacked or not.

    Returns:
        None
    """
    formatted_labels = [f"{v}%" if v != "" and data_label_format != "absolute" else v for v in container_labels]
    style = PlotStyleHelper()
    label_padding = 0 if is_stacked else 4
    ax.bar_label(
        container,
        labels=formatted_labels,
        label_type="center" if is_stacked else "edge",
        padding=label_padding,
        fontproperties=get_font_properties(style.data_label_font),
        fontsize=style.data_label_size,
    )
