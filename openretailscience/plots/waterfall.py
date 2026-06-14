"""This module provides functionality to generate a waterfall chart.

A visualization commonly used to illustrate how
different positive and negative values contribute to a cumulative total.Waterfall charts are effective in showing
the incremental impact of individual components, making them particularly useful for financial analysis,
performance tracking, and visualizing changes over time.

### Features

- **Waterfall Chart Creation**: Displays how different positive and negative values affect a starting total.
- **Data Label Formatting**: Supports custom formatting for data labels, including absolute
  values, percentages, or both.
- **Net Line and Bar Display**: Optionally includes a net line and net bar to show the overall
  cumulative result.
- **Customizable Plot Style**: Options to customize chart titles, axis labels, and remove zero
  amounts for better clarity.
- **Handling of Zero Amounts**: Allows removal of zero amounts from the plot to avoid cluttering the chart.
- **Interactive Elements**: Supports custom annotations for the chart with source text.

### Use Cases

- **Financial Analysis**: Show the breakdown of profits and losses over multiple periods, or how
  different cost categories affect overall margin.
- **Revenue Tracking**: Track how revenue or other key metrics change over time, and visualize
  the impact of individual contributing factors.
- **Performance Visualization**: Highlight how various business or product categories affect
  overall performance, such as sales, expenses, or growth metrics.
- **Budget Breakdown**: Visualize how different spending categories contribute to a total budget over a period.

### Functionality Details

- **plot()**: Generates a waterfall chart from a list of amounts and labels. It supports
  additional customization for display settings, labels, and source text.
- **format_data_labels()**: A helper function used to format the data labels according to the
  specified format (absolute, percentage, both).
"""

import warnings
from typing import Any, Literal

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.container import BarContainer

import openretailscience.plots.styles.graph_utils as gu
from openretailscience.core.validation import ensure_value_choice
from openretailscience.options import PlotStyleHelper
from openretailscience.plots.styles.colors import get_named_color
from openretailscience.plots.styles.font_utils import get_font_properties
from openretailscience.plots.styles.styling_helpers import standard_graph_styles

VALID_DATA_LABEL_FORMATS = ("absolute", "percentage", "both")


def plot(
    amounts: list[float],
    labels: list[str],
    title: str | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    y_label: str | None = None,
    x_label: str = "",
    source_text: str | None = None,
    data_label_format: Literal["absolute", "percentage", "both"] | None = None,
    display_net_bar: bool = False,
    display_net_line: bool = False,
    remove_zero_amounts: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Axes:
    """Generates a waterfall chart.

    Waterfall plots are particularly good for showing how different things add or subtract from a starting number. For
    instance:
    - Changes in sales figures from one period to another
    - Breakdown of profit margins
    - Impact of different product categories on overall revenue

    They are often used to identify key drivers of financial performance, highlight areas for
    improvement, and communicate complex data stories to stakeholders in an intuitive manner.

    Args:
        amounts (list[float]): The amounts to plot.
        labels (list[str]): The labels for the amounts.
        title (str, optional): The title of the chart. Defaults to None.
        eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
        subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
        y_label (str, optional): The y-axis label. Defaults to None.
        x_label (str, optional): The x-axis label. Defaults to None.
        source_text (str, optional): The source text to add to the plot. Defaults to None.
        data_label_format (Literal["absolute", "percentage", "both", "none"], optional): The format of the data labels.
            Defaults to "absolute".
        display_net_bar (bool, optional): Whether to display a net bar. Defaults to False.
        display_net_line (bool, optional): Whether to display a net line. Defaults to False.
        remove_zero_amounts (bool, optional): Whether to remove zero amounts from the plot. Defaults to True
        ax (Axes, optional): The matplotlib axes object to plot on. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the Pandas plot function.

    Returns:
        Axes: The matplotlib axes object.
    """
    if len(amounts) != len(labels):
        raise ValueError("The lengths of amounts and labels must be the same")

    if data_label_format is not None:
        ensure_value_choice(data_label_format, VALID_DATA_LABEL_FORMATS, "data_label_format")

    df = pd.DataFrame({"labels": labels, "amounts": amounts})

    if remove_zero_amounts:
        df = df.loc[df["amounts"] != 0]

    amounts_series: pd.Series = df.loc[:, "amounts"]
    amount_total = amounts_series.sum()

    positive_color = get_named_color("positive")
    negative_color = get_named_color("negative")

    default_colors = (amounts_series > 0).map({True: positive_color, False: negative_color}).to_list()
    bottom = amounts_series.cumsum().shift(1).fillna(0).to_list()

    if display_net_bar:
        # Append a row for the net amount
        df.loc[len(df)] = ["Net", amount_total]
        default_colors.append(get_named_color("difference"))
        bottom.append(0)

    # Create the plot
    width = kwargs.pop("width", 0.8)
    color = kwargs.pop("color", default_colors)
    plot_result = df.plot.bar(
        x="labels",
        y="amounts",
        legend=None,
        bottom=bottom,
        color=color,
        width=width,
        ax=ax,
        **kwargs,
    )
    if not isinstance(plot_result, Axes):
        raise TypeError("Expected df.plot.bar to return a single Axes.")
    ax = plot_result

    # Add a black line at the y=0 position
    ax.axhline(y=0, color="black", linewidth=1, zorder=-1)
    if data_label_format is not None:
        decimals = gu.get_decimals(ax.get_ylim(), ax.get_yticks().tolist())
        labels = format_data_labels(
            amounts_series,
            amount_total,
            data_label_format,
            decimals,
        )

        bar_container = ax.containers[0]
        if not isinstance(bar_container, BarContainer):
            raise TypeError("Expected the first container to be a BarContainer.")
        style = PlotStyleHelper()
        ax.bar_label(
            bar_container,
            label_type="edge",
            labels=labels,
            padding=5,
            fontsize=style.data_label_size,
            fontproperties=get_font_properties(style.data_label_font),
        )

    if display_net_line:
        ax.axhline(y=amount_total, color="black", linewidth=1, linestyle="--")

    return standard_graph_styles(
        ax,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        y_label=y_label,
        x_label=x_label,
        source_text=source_text,
        grid_axis="y",
    )


def format_data_labels(
    amounts: pd.Series,
    total_change: float,
    label_format: str,
    decimals: int,
) -> list[str]:
    """Format the data labels based on the specified format.

    Args:
        amounts (pd.Series): The amounts to format.
        total_change (float): The total change (sum of amounts) used for percentage calculations.
        label_format (str): The format of the data labels ("absolute", "percentage", or "both").
        decimals (int): The number of decimal places for formatting.

    Returns:
        list[str]: A list of formatted data label strings.
    """
    if label_format == "absolute":
        return amounts.apply(lambda x: gu.format_shorthand(x, decimals=decimals + 1)).tolist()

    if total_change == 0:
        warnings.warn(
            "Total change is zero, cannot calculate percentages. Percentage labels will be omitted.",
            UserWarning,
            stacklevel=2,
        )
        if label_format == "percentage":
            return [""] * len(amounts)
        return [gu.format_shorthand(x, decimals=decimals + 1) for x in amounts]

    if label_format == "percentage":
        return (amounts / total_change).map("{:.0%}".format).tolist()

    return [f"{gu.format_shorthand(x, decimals=decimals + 1)} ({x / total_change:.0%})" for x in amounts]
