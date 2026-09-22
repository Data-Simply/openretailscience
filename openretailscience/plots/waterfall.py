"""Waterfall chart from `amounts` and `labels` lists (not DataFrame-based).

Each amount is rendered as a bar stacked on the running total, colored by sign.
"""

import warnings
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from matplotlib.axes import Axes

import openretailscience.plots.styles.graph_utils as gu
from openretailscience.core.validation import ensure_value_choice
from openretailscience.options import PlotStyleHelper
from openretailscience.plots.styles.colors import get_named_color
from openretailscience.plots.styles.font_utils import get_font_properties
from openretailscience.plots.styles.styling_helpers import standard_graph_styles

if TYPE_CHECKING:
    from matplotlib.text import Annotation

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
    """Generate a waterfall chart from parallel `amounts` and `labels` lists.

    With remove_zero_amounts (default True) zero bars are dropped before the net total is
    computed. data_label_format=None draws no data labels.

    Args:
        amounts (list[float]): The amounts to plot.
        labels (list[str]): The labels, one per amount.
        title (str, optional): Chart title.
        eyebrow (str, optional): Uppercase label rendered above the title.
        subtitle (str, optional): Supporting copy rendered below the title.
        y_label (str, optional): Y-axis label.
        x_label (str, optional): X-axis label.
        source_text (str, optional): Source attribution rendered at the bottom.
        data_label_format ("absolute", "percentage", "both", optional): Format for the bar
            data labels; None draws no labels.
        display_net_bar (bool, optional): Append a "Net" bar for the total.
        display_net_line (bool, optional): Draw a dashed line at the net total.
        remove_zero_amounts (bool, optional): Drop zero amounts before the net is computed.
        ax (Axes, optional): Axes to plot on.
        **kwargs: Forwarded to df.plot.bar.

    Returns:
        Axes: The matplotlib axes object.

    Raises:
        ValueError: If len(amounts) != len(labels), or data_label_format is not one of
            "absolute", "percentage", "both".

    """
    if len(amounts) != len(labels):
        raise ValueError("The lengths of amounts and labels must be the same")

    if data_label_format is not None:
        data_label_format = ensure_value_choice(data_label_format, VALID_DATA_LABEL_FORMATS, "data_label_format")

    df = pd.DataFrame({"labels": labels, "amounts": amounts})

    if remove_zero_amounts:
        df = df[df["amounts"] != 0].reset_index(drop=True)

    amount_total = df["amounts"].sum()

    positive_color = get_named_color("positive")
    negative_color = get_named_color("negative")

    default_colors = (df["amounts"] > 0).map({True: positive_color, False: negative_color}).to_list()
    bottom = df["amounts"].cumsum().shift(1).fillna(0).to_list()

    if display_net_bar:
        # Append a row for the net amount
        df.loc[len(df)] = ["Net", amount_total]
        default_colors.append(get_named_color("difference"))
        bottom.append(0)

    # Create the plot
    width = kwargs.pop("width", 0.8)
    color = kwargs.pop("color", default_colors)
    ax = df.plot.bar(
        x="labels",
        y="amounts",
        legend=None,
        bottom=bottom,
        color=color,
        width=width,
        ax=ax,
        **kwargs,
    )

    # Add a black line at the y=0 position
    ax.axhline(y=0, color="black", linewidth=1, zorder=-1)
    bar_labels: list[Annotation] = []
    if data_label_format is not None:
        decimals = gu.get_decimals(ax.get_ylim(), ax.get_yticks())
        labels = format_data_labels(
            df["amounts"],
            amount_total,
            data_label_format,
            decimals,
        )

        style = PlotStyleHelper()
        bar_labels = ax.bar_label(
            ax.containers[0],
            label_type="edge",
            labels=labels,
            padding=5,
            fontsize=style.data_label_size,
            fontproperties=get_font_properties(style.data_label_font),
        )

    if display_net_line:
        ax.axhline(y=amount_total, color="black", linewidth=1, linestyle="--")

    ax = standard_graph_styles(
        ax,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        y_label=y_label,
        x_label=x_label,
        source_text=source_text,
        grid_axis="y",
    )

    # Bar sticky edges pin the y-view to the bar extents, so edge labels overflow the axes; grow
    # ylim to bring them inside. Must run after chrome has reflowed the axes.
    gu.expand_ylim_for_bar_labels(ax, bar_labels)

    return ax


def format_data_labels(
    amounts: pd.Series,
    total_change: float,
    label_format: str,
    decimals: int,
) -> list[str]:
    """Format the bar data labels for a waterfall chart.

    label_format is "absolute", "percentage", or "both"; percentages are a share of
    total_change (not of the displayed sum). When total_change is 0 a UserWarning is raised
    and the percentage is omitted ("both" falls back to absolute).
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
