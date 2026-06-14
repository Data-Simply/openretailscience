"""Bubble chart visualizations for price distribution analysis across categories.

The bubble chart shows price distribution as vertical layers (price bands) with bubble sizes
representing the percentage of products in each price range for different categories like
retailers, countries, etc.

### Core Features

- **Price Band Analysis**: Automatically bins price data into ranges using pandas.cut()
- **Categorical Grouping**: Groups data by categorical columns (retailers, countries, etc.)
- **Bubble Sizing**: Bubble sizes represent percentage of products in each price band per group
- **Flexible Binning**: Supports both integer (equal-width bins) and array (custom boundaries) inputs
- **Grid Layout**: X-axis shows categories, Y-axis shows price bands

### Use Cases

- **Retailer Price Comparison**: Compare price distributions across different retailers
- **Regional Price Analysis**: Analyze price positioning by country/region
- **Competitive Pricing**: Identify pricing gaps and opportunities
- **Price Architecture Visualization**: Visualize competitive pricing landscapes

### Limitations

- **Pandas DataFrame Only**: No Ibis table support
- **Pre-aggregated Data**: Data should be at product level (one row per product)
- **Numeric Price Column**: Requires numeric price/value column for binning
"""

from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from openretailscience.plots.styles.colors import get_plot_colors
from openretailscience.plots.styles.styling_helpers import standard_graph_styles


def _validate_inputs(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    bins: int | list[float],
) -> tuple[pd.DataFrame, int | list[float]]:
    """Validates and processes inputs for price distribution plotting.

    Args:
        df: Input DataFrame containing product-level data.
        value_col: Column containing the price/value data.
        group_col: Column containing the categorical grouping.
        bins: Either number of equal-width bins (int) or custom bin boundaries (list).

    Returns:
        Tuple of (cleaned_dataframe, validated_bins).

    Raises:
        ValueError: If DataFrame is empty, columns don't exist, value column is not numeric,
            or bins parameter is invalid.
        KeyError: If specified columns are not found in DataFrame.
        TypeError: If bins parameter has invalid type.
    """
    # Validate DataFrame is not empty
    if df.empty:
        raise ValueError("Cannot plot with empty DataFrame")

    # Validate columns exist
    if value_col not in df.columns:
        msg = f"value_col '{value_col}' not found in DataFrame"
        raise KeyError(msg)

    if group_col not in df.columns:
        msg = f"group_col '{group_col}' not found in DataFrame"
        raise KeyError(msg)

    # Validate value column is numeric
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        msg = f"value_col '{value_col}' must be numeric for binning"
        raise ValueError(msg)

    # Validate bins parameter
    validated_bins = _validate_bins_parameter(bins)

    # Remove rows with missing values in key columns. pandas-stubs widens DataFrame.dropna() to
    # DataFrame | Series; selecting a column list always yields a DataFrame here.
    df_clean = cast("pd.DataFrame", df[[value_col, group_col]].dropna())

    if df_clean.empty:
        msg = f"No valid data after removing missing values from {value_col} and {group_col}"
        raise ValueError(msg)

    return df_clean, validated_bins


def _validate_bins_parameter(bins: int | list[float]) -> int | list[float]:
    """Validates and processes the bins parameter for price distribution plotting.

    Args:
        bins: Either number of equal-width bins (int) or custom bin boundaries (list).

    Returns:
        Validated and processed bins parameter.

    Raises:
        ValueError: If bins parameter is invalid.
        TypeError: If bins parameter has invalid type.
    """
    if isinstance(bins, int):
        if bins <= 0:
            raise ValueError("bins must be a positive integer")
        return bins
    if isinstance(bins, list):
        min_bins = 2
        if len(bins) < min_bins:
            raise ValueError("bins list must contain at least 2 values")
        if not all(isinstance(x, int | float) for x in bins):
            raise ValueError("All values in bins list must be numeric")
        return sorted(bins)
    msg = "bins must be either an integer or a list of numeric values"
    raise TypeError(msg)


def _fmt_bin_edge(value: float) -> str:
    """Format a bin edge to one decimal, normalising "-0.0" to "0.0".

    pd.cut(..., include_lowest=True) extends the lowest bin's left edge by
    ~0.1% of the data range below the minimum so that minimum is included;
    when that epsilon lands within rounding distance of zero, naive formatting
    produces "-0.0", which is meaningless to readers.
    """
    text = f"{value:.1f}"
    return "0.0" if text == "-0.0" else text


def plot(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    bins: int | list[float],
    title: str | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    legend_title: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    move_legend_outside: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> Axes:
    """Creates a bubble chart visualization showing price distribution analysis across categories.

    The chart displays price bands as vertical layers with bubble sizes representing the percentage
    of products in each price range for different groups (retailers, countries, etc.).

    Args:
        df (pd.DataFrame): Input DataFrame containing product-level data.
        value_col (str): Column containing the price/value data (e.g., "unit_price").
        group_col (str): Column containing the categorical grouping (e.g., "retailer").
        bins (int | list[float]): Either number of equal-width bins (int) or custom bin boundaries (list).
        title (str, optional): The title of the plot. Defaults to None.
        eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
        subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
        x_label (str, optional): The label for the x-axis. Defaults to None.
        y_label (str, optional): The label for the y-axis. Defaults to None.
        legend_title (str, optional): The title for the legend. Defaults to None.
        ax (Axes, optional): The Matplotlib Axes object to plot on. Defaults to None.
        source_text (str, optional): Text to be displayed as a source at the bottom of the plot. Defaults to None.
        move_legend_outside (bool, optional): Whether to move the legend outside the plot area. Defaults to False.
        **kwargs (Any): Additional keyword arguments for the scatter plot function.

    Returns:
        Axes: The Matplotlib Axes object with the generated bubble chart.

    Raises:
        ValueError: If DataFrame is empty, columns don't exist, or bins parameter is invalid.
        KeyError: If specified columns are not found in DataFrame.
        TypeError: If bins parameter has invalid type.
    """
    # Validate inputs and get clean data
    df_clean, bins = _validate_inputs(df, value_col, group_col, bins)

    # Create price bins
    df_clean["price_bin"] = pd.cut(df_clean[value_col], bins=bins, include_lowest=True)

    # Calculate percentage distribution for each group
    group_totals = df_clean.groupby(group_col, observed=True).size()
    bin_counts = df_clean.groupby([group_col, "price_bin"], observed=True).size().unstack(fill_value=0)

    # Convert to proportions (0-1 range)
    proportions = bin_counts.div(group_totals, axis=0)

    ax = ax or plt.gca()

    # Get unique groups and bins
    groups = proportions.index.tolist()
    price_bins = proportions.columns.tolist()

    # Set up color mapping
    colors = get_plot_colors(len(groups))

    alpha = kwargs.pop("alpha", 0.7)
    s_scale = kwargs.pop("s", 2000)
    edge_color = kwargs.pop("edgecolor", "black")  # black stroke around bubbles
    line_width = kwargs.pop("linewidth", 1.5)  # Stroke width

    # Validate that we have some data
    if proportions.max().max() == 0 or pd.isna(proportions.max().max()):
        raise ValueError("All proportions are zero - no data falls within the specified bins")

    # Melt to get all (group, price_bin) combinations with their proportions
    melted = proportions.reset_index().melt(id_vars=group_col, var_name="price_bin", value_name="proportion")
    # Filter out zero proportions to avoid invisible bubbles
    melted = melted[melted["proportion"] > 0]

    if len(melted) > 0:  # Only plot if there are non-zero proportions
        x_positions = [groups.index(group) for group in melted[group_col]]
        y_positions = [price_bins.index(price_bin) for price_bin in melted["price_bin"]]
        # Calculate bubble sizes using absolute proportion values for cross-group comparison
        bubble_sizes = (melted["proportion"] * s_scale).to_numpy()
        bubble_colors = [colors[groups.index(group)] for group in melted[group_col]]

        ax.scatter(
            x_positions,
            y_positions,
            s=bubble_sizes,
            c=bubble_colors,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=line_width,
            **kwargs,
        )

    ax.set_xticks(range(len(groups)))
    # Index.tolist() is typed list[Any]; group labels render via their string form on the axis.
    ax.set_xticklabels([str(group) for group in groups])
    ax.set_yticks(range(len(price_bins)))

    # Reserve half a unit on each side to give bubbles breathing room even at high s_scale.
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(-0.5, len(price_bins) - 0.5)

    # pd.cut(..., include_lowest=True) extends the lowest bin's left edge by ~0.1% of the data range below the minimum
    # so the minimum value is included; when that epsilon lands within rounding distance of zero, naive formatting
    # produces "-0.0", which is meaningless to readers.
    formatted_labels = [f"{_fmt_bin_edge(bin_.left)} - {_fmt_bin_edge(bin_.right)}" for bin_ in price_bins]

    ax.set_yticklabels(formatted_labels)

    # The single ax.scatter() call above draws all bubbles in one collection, so
    # matplotlib's legend auto-discovery has no per-group handles to find. Seed
    # one invisible labeled marker per group so standard_graph_styles can build
    # the legend, and so tight_layout can reserve room for it when
    # move_legend_outside=True. The bubble's edge stroke is omitted on the
    # proxies — at legend marker size the stroke would dominate and wash out
    # the fill color.
    if len(groups) > 1:
        for i, group in enumerate(groups):
            ax.scatter([], [], c=[colors[i]], alpha=alpha, linewidths=0, label=group)

    return standard_graph_styles(
        ax=ax,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
        show_legend=len(groups) > 1,
        source_text=source_text,
        grid_axis="y",
    )
