"""Color Palettes and helper functions.

This module provides functions to create and retrieve color palettes.
"""

from typing import cast

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap

from openretailscience.constants import COLORS
from openretailscience.options import get_option


def get_color_list(name: str, starting_color_code: int = 50, ending_color_code: int = 950) -> list[str]:
    """Returns a filtered list of colors from the Tailwind color palette based on the given range.

    Args:
        name (str): The name of the color palette (e.g., "blue", "red").
        starting_color_code (int): The lowest color shade to use (default: 50).
        ending_color_code (int): The highest color shade to use (default: 950).

    Returns:
        list[str]: A filtered list of colors from the Tailwind color palette.
    """
    if name not in COLORS:
        msg = f"Color pallete {name} not found. Available color palettes are: {', '.join(COLORS.keys())}."
        raise ValueError(msg)
    return [COLORS[name][key] for key in sorted(COLORS[name].keys()) if starting_color_code <= key <= ending_color_code]


def get_listed_cmap(name: str) -> ListedColormap:
    """Returns a ListedColormap from the Tailwind color pallete of the given name.

    Args:
        name (str): The name of the color pallete.

    Returns:
        ListedColormap: The color pallete as a ListedColormap.
    """
    return ListedColormap(get_color_list(name))


def get_linear_cmap(name: str, starting_color_code: int = 50, ending_color_code: int = 950) -> LinearSegmentedColormap:
    """Returns a linear segmented colormap using Tailwind colors.

    This function allows restricting the color range used in the colormap.

    Args:
        name (str): The name of the Tailwind color (e.g., "blue", "red").
        starting_color_code (int): The lowest color shade to use (default: 50).
        ending_color_code (int): The highest color shade to use (default: 950).

    Returns:
        LinearSegmentedColormap: A colormap object for matplotlib.
    """
    return LinearSegmentedColormap.from_list(
        f"{name}_linear_colormap",
        get_color_list(name, starting_color_code, ending_color_code),
    )


def get_base_cmap() -> ListedColormap:
    """Returns a ListedColormap with all the Tailwind colors.

    Returns:
        ListedColormap: A ListedColormap with all the Tailwind colors.
    """
    color_order = [
        "red",
        "orange",
        "yellow",
        "green",
        "teal",
        "sky",
        "indigo",
        "purple",
        "pink",
        "slate",
        "amber",
        "lime",
        "emerald",
        "cyan",
        "blue",
        "violet",
        "fuchsia",
        "rose",
    ]
    color_numbers = [500, 300, 700]
    colors = [COLORS[color][color_number] for color_number in color_numbers for color in color_order]

    return ListedColormap(colors)


def get_plot_colors(num_series: int) -> list[str]:
    """Get appropriate colors for the given number of series.

    Selection logic:
        * 1 series → ``plot.color.primary`` (brand color).
        * 2+ series → cycled multi-color palette.

    Categorical series (e.g. product categories, retailers, segments) read best when each
    has its own hue, which is what the multi-color palette gives.

    Args:
        num_series: Number of series/groups being plotted

    Returns:
        List of hex color strings
    """
    if num_series == 1:
        return [get_option("plot.color.primary")]

    multi_palette = get_option("plot.color.multi_color_palette")
    return [multi_palette[i % len(multi_palette)] for i in range(num_series)]


def get_named_color(color_type: str) -> str:
    """Get a named color from options.

    Args:
        color_type: One of 'positive', 'negative', 'neutral', 'difference', 'context', 'primary'

    Returns:
        Hex color string
    """
    # The dynamic f-string key resolves to a ``plot.color.*`` option, all of which store hex
    # color strings, but the dynamic key only matches the ``str -> OptionTypes`` overload.
    return cast("str", get_option(f"plot.color.{color_type}"))


def get_sequential_cmap() -> Colormap:
    """Resolve the magnitude-ordered colormap (heatmap, cohort, period-on-period).

    Reads ``plot.color.sequential``: a Tailwind hue name (``"green"``, ``"blue"``)
    yields a discrete ``ListedColormap`` over the hue's shades; a matplotlib
    colormap name (``"viridis"``, ``"Greens"``) is handed to ``plt.get_cmap``.

    Returns:
        Colormap: The resolved colormap (a ``ListedColormap`` for Tailwind hues, or
            whatever ``plt.get_cmap`` returns for a matplotlib colormap name).
    """
    cmap_name = get_option("plot.color.sequential")
    return get_listed_cmap(cmap_name) if cmap_name in COLORS else plt.get_cmap(cmap_name)


slate_cmap = get_listed_cmap("slate")
gray_cmap = get_listed_cmap("gray")
zinc_cmap = get_listed_cmap("zinc")
neutral_cmap = get_listed_cmap("neutral")
stone_cmap = get_listed_cmap("stone")
red_cmap = get_listed_cmap("red")
orange_cmap = get_listed_cmap("orange")
amber_cmap = get_listed_cmap("amber")
yellow_cmap = get_listed_cmap("yellow")
lime_cmap = get_listed_cmap("lime")
green_cmap = get_listed_cmap("green")
emerald_cmap = get_listed_cmap("emerald")
teal_cmap = get_listed_cmap("teal")
cyan_cmap = get_listed_cmap("cyan")
sky_cmap = get_listed_cmap("sky")
blue_cmap = get_listed_cmap("blue")
indigo_cmap = get_listed_cmap("indigo")
violet_cmap = get_listed_cmap("violet")
purple_cmap = get_listed_cmap("purple")
fuchsia_cmap = get_listed_cmap("fuchsia")
pink_cmap = get_listed_cmap("pink")


slate_linear_cmap = get_linear_cmap("slate")
gray_linear_cmap = get_linear_cmap("gray")
zinc_linear_cmap = get_linear_cmap("zinc")
neutral_linear_cmap = get_linear_cmap("neutral")
stone_linear_cmap = get_linear_cmap("stone")
red_linear_cmap = get_linear_cmap("red")
orange_linear_cmap = get_linear_cmap("orange")
amber_linear_cmap = get_linear_cmap("amber")
yellow_linear_cmap = get_linear_cmap("yellow")
lime_linear_cmap = get_linear_cmap("lime")
green_linear_cmap = get_linear_cmap("green")
emerald_linear_cmap = get_linear_cmap("emerald")
teal_linear_cmap = get_linear_cmap("teal")
cyan_linear_cmap = get_linear_cmap("cyan")
sky_linear_cmap = get_linear_cmap("sky")
blue_linear_cmap = get_linear_cmap("blue")
indigo_linear_cmap = get_linear_cmap("indigo")
violet_linear_cmap = get_linear_cmap("violet")
purple_linear_cmap = get_linear_cmap("purple")
fuchsia_linear_cmap = get_linear_cmap("fuchsia")
pink_linear_cmap = get_linear_cmap("pink")


base_cmap = get_base_cmap()
