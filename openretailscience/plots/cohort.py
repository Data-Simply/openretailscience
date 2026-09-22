"""Cohort heatmap: wraps heatmap.plot with x tick labels on top and percentage formatting.

Input is a pre-aggregated matrix (rows = cohorts, columns = periods since start); no
aggregation is performed.
"""

from typing import Literal

import pandas as pd
from matplotlib.axes import Axes, SubplotBase

from openretailscience.plots import heatmap


def plot(
    df: pd.DataFrame,
    cbar_label: str,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    percentage: bool = True,
    figsize: tuple[int, int] | None = None,
    colormap_style: Literal["discrete", "continuous"] = "discrete",
) -> SubplotBase:
    """Plot a cohort heatmap by wrapping heatmap.plot.

    Passes x_labels_position="top" (so the chronology reads top-to-bottom) and
    cbar_format="{x:.0%}" when percentage is True, else "{x:g}".

    Args:
        df (pd.DataFrame): Pre-aggregated cohort matrix (rows = cohorts, columns = periods).
        cbar_label (str): Colorbar label.
        x_label (str, optional): X-axis label.
        y_label (str, optional): Y-axis label.
        title (str, optional): Plot title.
        eyebrow (str, optional): Uppercase label rendered above the title.
        subtitle (str, optional): Supporting copy rendered below the title.
        ax (Axes, optional): Axes to plot on.
        source_text (str, optional): Source attribution rendered at the bottom.
        percentage (bool, optional): Display values as percentages (True) or raw numbers.
        figsize (tuple[int, int], optional): Figure size, used only when ax is None.
        colormap_style ("discrete", "continuous", optional): Use "continuous" when retention
            differences between cohorts are small enough that the discrete bins lump them
            together.

    Returns:
        SubplotBase: The matplotlib axes object.

    """
    return heatmap.plot(
        df=df,
        cbar_label=cbar_label,
        x_label=x_label,
        y_label=y_label,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        ax=ax,
        source_text=source_text,
        figsize=figsize,
        x_labels_position="top",
        cbar_format="{x:.0%}" if percentage else "{x:g}",
        colormap_style=colormap_style,
    )
