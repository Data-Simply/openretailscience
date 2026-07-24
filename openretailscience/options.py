"""This module provides a simplified implementation of a pandas-like options system.

It allows users to get, set, and reset various options that control the behavior
of data display and processing. The module also includes a context manager for
temporarily changing options.

Example:
    >>> set_option('column.customer_id', 'cust_id')
    >>> print(get_option('column.customer_id'))
    cust_id
    >>> with option_context('column.customer_id', 'shopper_id'):
    ...     print(get_option('column.customer_id'))
    shopper_id
    >>> print(get_option('column.customer_id'))
    cust_id

"""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import toml

from openretailscience.constants import COLORS

OptionTypes = str | int | float | bool | list | dict | None


class Options:
    """A class to manage configurable options."""

    def __init__(self) -> None:
        """Initializes the options with default values."""
        self._options: dict[str, OptionTypes] = {
            # Database columns
            "column.customer_id": "customer_id",
            "column.transaction_id": "transaction_id",
            "column.transaction_date": "transaction_date",
            "column.transaction_time": "transaction_time",
            "column.product_id": "product_id",
            "column.unit_quantity": "unit_quantity",
            "column.unit_price": "unit_price",
            "column.unit_spend": "unit_spend",
            "column.unit_cost": "unit_cost",
            "column.promo_unit_spend": "promo_unit_spend",
            "column.promo_unit_quantity": "promo_unit_quantity",
            "column.store_id": "store_id",
            # Aggregation columns
            "column.agg.customer_id": "customers",
            "column.agg.transaction_id": "transactions",
            "column.agg.product_id": "products",
            "column.agg.unit_quantity": "units",
            "column.agg.unit_price": "prices",
            "column.agg.unit_spend": "spend",
            "column.agg.unit_cost": "costs",
            "column.agg.promo_unit_spend": "promo_spend",
            "column.agg.promo_unit_quantity": "promo_units",
            "column.agg.store_id": "stores",
            # Calculated columns
            "column.calc.price_per_unit": "price_per_unit",
            "column.calc.units_per_transaction": "units_per_transaction",
            "column.calc.spend_per_customer": "spend_per_customer",
            "column.calc.spend_per_transaction": "spend_per_transaction",
            "column.calc.transactions_per_customer": "transactions_per_customer",
            "column.calc.price_elasticity": "price_elasticity",
            "column.calc.frequency_elasticity": "frequency_elasticity",
            # Abbreviation suffix
            "column.suffix.count": "cnt",
            "column.suffix.percent": "pct",
            "column.suffix.difference": "diff",
            "column.suffix.percent_difference": "pct_diff",
            "column.suffix.contribution": "contrib",
            "column.suffix.period_1": "p1",
            "column.suffix.period_2": "p2",
            "column.suffix.unknown_customer": "unknown",
            "column.suffix.total": "total",
            # Performance / query optimization
            "optimization.use_native_sql": True,
            "optimization.use_caching": True,
            # Color palettes
            # 27-color rotation: 9 curated hues at 3 shades. First 5 hues are max-separated on the
            # hue wheel (~70 degrees apart) for 3-5 series. Order: primaries (500), lights (200),
            # darks (700).
            "plot.color.multi_color_palette": [
                COLORS[hue][shade]
                for shade in (500, 200, 700)
                for hue in ("green", "blue", "orange", "violet", "red", "cyan", "pink", "yellow", "teal")
            ],
            "plot.color.positive": COLORS["green"][500],
            "plot.color.negative": COLORS["red"][500],
            "plot.color.neutral": COLORS["gray"][500],
            # Semantic colors (`primary`, `difference`) live one shade darker than their hue's
            # categorical counterpart in `multi_color_palette` (which uses the 500-shade tier as
            # its primary set). The extra weight signals "this means something specific" vs
            # "one of N peers" — keeps the semantic role visually distinct while staying inside
            # the same Tailwind hue family.
            "plot.color.difference": COLORS["blue"][600],
            "plot.color.context": COLORS["gray"][300],
            "plot.color.primary": COLORS["green"][600],
            "plot.color.sequential": "green",
            # Chrome (eyebrow / title / subtitle / tab / source) colors
            "plot.color.eyebrow": COLORS["neutral"][500],
            "plot.color.subtitle": COLORS["neutral"][600],
            "plot.color.title": COLORS["neutral"][950],
            "plot.color.tab": COLORS["green"][600],
            "plot.color.source": COLORS["neutral"][500],
            # Plot font options
            "plot.font.title_font": "poppins_semi_bold",
            "plot.font.eyebrow_font": "poppins_semi_bold",
            "plot.font.subtitle_font": "poppins_regular",
            "plot.font.label_font": "poppins_regular",
            "plot.font.tick_font": "poppins_regular",
            "plot.font.source_font": "poppins_light_italic",
            "plot.font.data_label_font": "poppins_medium",
            "plot.font.legend_font": "poppins_regular",
            "plot.font.title_size": 22.0,
            "plot.font.eyebrow_size": 10.0,
            "plot.font.subtitle_size": 12.0,
            "plot.font.label_size": 11.0,
            "plot.font.tick_size": 10.0,
            "plot.font.source_size": 9.0,
            "plot.font.data_label_size": 10.0,
            "plot.font.legend_size": 10.0,
            # Plot spacing options
            "plot.spacing.x_label_pad": 10,
            "plot.spacing.y_label_pad": 10,
            # Plot style options
            "plot.style.background_color": "white",
            "plot.style.grid_color": COLORS["neutral"][200],
            "plot.style.grid_alpha": 1.0,
            "plot.style.show_top_spine": False,
            "plot.style.show_right_spine": False,
            "plot.style.show_bottom_spine": True,
            "plot.style.show_left_spine": True,
            "plot.style.legend_bbox_to_anchor": [1.02, 1.0],
            "plot.style.legend_loc": "upper left",
            "plot.style.cell_corner_radius": 0.05,
            "plot.style.cell_gap": 3.0,
            "plot.style.auto_rotate_x_ticks": True,
            "plot.style.auto_wrap_x_ticks": True,
            "plot.style.show_tab": True,
        }
        self._descriptions: dict[str, str] = {
            # Database columns
            "column.customer_id": "The name of the column containing customer IDs.",
            "column.transaction_id": "The name of the column containing transaction IDs.",
            "column.transaction_date": "The name of the column containing transaction dates.",
            "column.transaction_time": "The name of the column containing transaction times.",
            "column.product_id": "The name of the column containing product IDs.",
            "column.unit_quantity": "The name of the column containing the number of units sold.",
            "column.unit_price": "The name of the column containing the unit price of the product.",
            "column.unit_spend": (
                "The name of the column containing the total spend of the products in the transaction. "
                "ie, unit_price * units"
            ),
            "column.unit_cost": (
                "The name of the column containing the total cost of the products in the transaction. "
                "ie, single unit cost * units"
            ),
            "column.promo_unit_spend": (
                "The name of the column containing the total spend on promotion of the products in the transaction. "
                "ie, promotional unit price * units"
            ),
            "column.promo_unit_quantity": "The name of the column containing the number of units sold on promotion.",
            "column.store_id": "The name of the column containing store IDs of the transaction.",
            # Aggregation columns
            "column.agg.customer_id": "The name of the column containing the number of unique customers.",
            "column.agg.transaction_id": "The name of the column containing the number of transactions.",
            "column.agg.product_id": "The name of the column containing the number of unique products.",
            "column.agg.unit_quantity": "The name of the column containing the total number of units sold.",
            "column.agg.unit_price": "The name of the column containing the total unit price of products.",
            "column.agg.unit_spend": (
                "The name of the column containing the total spend of the units in the transaction."
            ),
            "column.agg.unit_cost": "The name of the column containing the total unit cost of products.",
            "column.agg.promo_unit_spend": (
                "The name of the column containing the total promotional spend of the units in the transaction."
            ),
            "column.agg.promo_unit_quantity": (
                "The name of the column containing the total number of units sold on promotion."
            ),
            "column.agg.store_id": "The name of the column containing the number of unique stores.",
            # Calculated columns
            "column.calc.price_per_unit": "The name of the column containing the price per unit.",
            "column.calc.units_per_transaction": "The name of the column containing the units per transaction.",
            "column.calc.spend_per_customer": "The name of the column containing the spend per customer.",
            "column.calc.spend_per_transaction": "The name of the column containing the spend per transaction.",
            "column.calc.transactions_per_customer": "The name of the column containing the transactions per customer.",
            "column.calc.price_elasticity": "The name of the column containing the price elasticity calculation.",
            "column.calc.frequency_elasticity": (
                "The name of the column containing the frequency elasticity calculation."
            ),
            # Abbreviation suffixes
            "column.suffix.count": "The suffix to use for count columns.",
            "column.suffix.percent": "The suffix to use for percentage columns.",
            "column.suffix.difference": "The suffix to use for difference columns.",
            "column.suffix.percent_difference": "The suffix to use for percentage difference columns.",
            "column.suffix.contribution": "The suffix to use for revenue contribution columns.",
            "column.suffix.period_1": (
                "The suffix to use for period 1 columns. Often this could represent last year for instance."
            ),
            "column.suffix.period_2": (
                "The suffix to use for period 2 columns. Often this could represent this year for instance."
            ),
            "column.suffix.unknown_customer": "The suffix to use for unknown customer columns.",
            "column.suffix.total": "The suffix to use for total columns.",
            # Performance / query optimization
            "optimization.use_native_sql": (
                "Whether to use backend-native SQL features for faster queries where supported. "
                "When False, or on backends without support, the portable fallback path is used. "
                "Results are identical either way."
            ),
            "optimization.use_caching": (
                "Whether the experimental openretailscience.experimental.cache.cache helper materializes "
                "results. When False, cache() returns the expression unchanged so callers incur no caching "
                "overhead. Results are identical either way."
            ),
            # Color options
            "plot.color.multi_color_palette": (
                "Multi-color palette for plots with many series (default: 27 Tailwind colors, "
                "9 curated hues at shades 500 / 200 / 700)."
            ),
            "plot.color.positive": "Color for positive values (e.g., gains, increases)",
            "plot.color.negative": "Color for negative values (e.g., losses, decreases)",
            "plot.color.neutral": "Color for neutral values (e.g., net amounts)",
            "plot.color.difference": "Color for difference values (e.g., waterfall transitions, tree differences)",
            "plot.color.context": "Color for de-emphasized context lines",
            "plot.color.primary": "Default color for single-series plots",
            "plot.color.sequential": (
                "Tailwind color name (e.g., 'green', 'blue') or matplotlib colormap name"
                " (e.g., 'Greens', 'viridis') for sequential ramps (heatmaps, cohort tables, period-on-period)"
            ),
            "plot.color.eyebrow": "Color for the small uppercase eyebrow text above the title.",
            "plot.color.subtitle": "Color for the subtitle text below the title.",
            "plot.color.title": "Color for the main plot title.",
            "plot.color.tab": "Color for the small rectangular tab mark above the title block.",
            "plot.color.source": "Color for the source text rendered below the plot.",
            # Plot font descriptions
            "plot.font.title_font": "The font family to use for plot titles.",
            "plot.font.eyebrow_font": "The font family to use for the eyebrow text above the title.",
            "plot.font.subtitle_font": "The font family to use for the subtitle text below the title.",
            "plot.font.label_font": "The font family to use for axis labels.",
            "plot.font.tick_font": "The font family to use for axis tick labels.",
            "plot.font.source_font": "The font family to use for source text.",
            "plot.font.data_label_font": "The font family to use for data point labels.",
            "plot.font.legend_font": "The font family to use for legend title and entry text.",
            "plot.font.title_size": "The font size for plot titles (in points).",
            "plot.font.eyebrow_size": "The font size for the eyebrow text above the title (in points).",
            "plot.font.subtitle_size": "The font size for the subtitle text (in points).",
            "plot.font.label_size": "The font size for axis labels (in points).",
            "plot.font.tick_size": "The font size for axis tick labels (in points).",
            "plot.font.source_size": "The font size for source text (in points).",
            "plot.font.data_label_size": "The font size for data point labels (in points).",
            "plot.font.legend_size": "The font size for legend entry text (in points).",
            # Plot spacing descriptions
            "plot.spacing.x_label_pad": "The padding below the x-axis label (in points).",
            "plot.spacing.y_label_pad": "The padding to the left of the y-axis label (in points).",
            # Plot style descriptions
            "plot.style.background_color": "The background color of the plot area.",
            "plot.style.grid_color": "The color of the grid lines.",
            "plot.style.grid_alpha": "The transparency of the grid lines (0.0 to 1.0).",
            "plot.style.show_top_spine": "Whether to show the top border of the plot.",
            "plot.style.show_right_spine": "Whether to show the right border of the plot.",
            "plot.style.show_bottom_spine": "Whether to show the bottom border of the plot.",
            "plot.style.show_left_spine": "Whether to show the left border of the plot.",
            "plot.style.legend_bbox_to_anchor": "The bounding box anchor for legend positioning when moved outside.",
            "plot.style.legend_loc": "The location of the legend when moved outside the plot.",
            "plot.style.cell_corner_radius": (
                "Corner radius (in cell-size data units) for cell-based plots like heatmap and cohort. "
                "Values around 0.03 give a subtle softening; larger values produce a pill-chip look."
            ),
            "plot.style.cell_gap": (
                "Width (in points) of the visual gap between adjacent cells in cell-based plots. "
                "Drawn as a background-coloured edge on each cell so the gap stays a fixed pixel size "
                "regardless of axes aspect or grid dimensions."
            ),
            "plot.style.auto_rotate_x_ticks": (
                "When True, categorical x-axis tick labels render horizontal and only rotate "
                "to 45° or 90° when neighbouring labels would overlap. When False, the rotation "
                "set by matplotlib (or pandas, for bar plots) is left unchanged."
            ),
            "plot.style.auto_wrap_x_ticks": (
                "When True, multi-word categorical x-tick labels are split onto two balanced "
                "lines before falling back to rotation when neighbours would overlap (e.g. "
                "'Camden High St' becomes 'Camden\\nHigh St'). Single-word labels are left "
                "alone and proceed straight to rotation."
            ),
            "plot.style.show_tab": (
                "When True (default), the small green tab mark renders above the title block "
                "on charts with header chrome. Set to False to suppress the tab project-wide; "
                "use ``option_context`` to scope the change to a block."
            ),
        }
        self._default_options: dict[str, OptionTypes] = self._options.copy()

    def set_option(self, pat: str, val: OptionTypes) -> None:
        """Set the value of the specified option.

        Args:
            pat: The option name.
            val: The value to set the option to.

        Raises:
            ValueError: If the option name is unknown.
        """
        if pat not in self._options:
            msg = f"Unknown option: {pat}"
            raise ValueError(msg)

        self._options[pat] = val

    def get_option(self, pat: str) -> OptionTypes:
        """Get the value of the specified option.

        Args:
            pat: The option name.

        Returns:
            The value of the option.

        Raises:
            ValueError: If the option name is unknown.
        """
        if pat in self._options:
            return self._options[pat]

        msg = f"Unknown option: {pat}"
        raise ValueError(msg)

    def reset_option(self, pat: str) -> None:
        """Reset the specified option to its default value.

        Args:
            pat: The option name.

        Raises:
            ValueError: If the option name is unknown.
        """
        if pat not in self._options:
            msg = f"Unknown option: {pat}"
            raise ValueError(msg)

        self._options[pat] = self._default_options[pat]

    def list_options(self) -> list[str]:
        """List all available options.

        Returns:
            A list of all option names.
        """
        return list(self._options.keys())

    def describe_option(self, pat: str) -> str:
        """Describe the specified option.

        Args:
            pat: The option name.

        Returns:
            A string describing the option and its current value.

        Raises:
            ValueError: If the option name is unknown.
        """
        if pat in self._descriptions:
            return f"{pat}: {self._descriptions[pat]} (current value: {self._options[pat]})"

        msg = f"Unknown option: {pat}"
        raise ValueError(msg)

    @staticmethod
    def flatten_options(k: str, v: OptionTypes, parent_key: str = "") -> dict[str, OptionTypes]:
        """Flatten nested options into a single dictionary."""
        if parent_key != "":
            parent_key += "."

        if isinstance(v, dict):
            ret_dict = {}
            for sub_key, sub_value in v.items():
                ret_dict.update(Options.flatten_options(sub_key, sub_value, parent_key=f"{parent_key}{k}"))
            return ret_dict

        return {f"{parent_key}{k}": v}

    @classmethod
    def load_from_project(cls) -> "Options":
        """Try to load options from an openretailscience.toml file in the project root directory.

        If the project root directory cannot be found, return a default Options instance.

        Returns:
            An Options instance with options loaded from the openretailscience.toml file or default
        """
        options_instance = cls()

        project_root = find_project_root()
        if project_root is None:
            return options_instance

        toml_file = Path(project_root) / "openretailscience.toml"
        if toml_file.is_file():
            return Options.load_from_toml(toml_file)

        return options_instance

    @classmethod
    def load_from_toml(cls, file_path: str) -> "Options":
        """Load options from a TOML file.

        Args:
            file_path: The path to the TOML file.

        Raises:
            ValueError: If the TOML file contains unknown options.
        """
        options_instance = cls()

        with open(file_path) as f:
            toml_data = toml.load(f)

        for section, options in toml_data.items():
            for option_name, option_value in Options.flatten_options(section, options).items():
                if option_name in options_instance._options:
                    options_instance.set_option(option_name, option_value)
                else:
                    msg = f"Unknown option in TOML file: {option_name}"
                    raise ValueError(msg)

        return options_instance


def find_project_root() -> str | None:
    """Returns the directory containing .git, .hg, or pyproject.toml, starting from the current working directory."""
    current_dir = Path.cwd()

    while True:
        if (Path(current_dir / ".git")).is_dir() or (Path(current_dir / "openretailscience.toml")).is_file():
            return current_dir

        parent_dir = Path(current_dir).parent
        reached_root = parent_dir == current_dir
        if reached_root:
            return None

        current_dir = parent_dir


# Global instance of Options
_global_options = Options().load_from_project()


def set_option(pat: str, val: OptionTypes) -> None:
    """Set the value of the specified option.

    This is a global function that delegates to the _global_options instance.

    Args:
        pat: The option name.
        val: The value to set the option to.

    Raises:
        ValueError: If the option name is unknown.
    """
    _global_options.set_option(pat, val)


def get_option(pat: str) -> OptionTypes:
    """Get the value of the specified option.

    This is a global function that delegates to the _global_options instance.

    Args:
        pat: The option name.

    Returns:
        The value of the option.

    Raises:
        ValueError: If the option name is unknown.
    """
    return _global_options.get_option(pat)


def reset_option(pat: str) -> None:
    """Reset the specified option to its default value.

    This is a global function that delegates to the _global_options instance.

    Args:
        pat: The option name.

    Raises:
        ValueError: If the option name is unknown.
    """
    _global_options.reset_option(pat)


def list_options() -> list[str]:
    """List all available options.

    This is a global function that delegates to the _global_options instance.

    Returns:
        A list of all option names.
    """
    return _global_options.list_options()


def describe_option(pat: str) -> str:
    """Describe the specified option.

    This is a global function that delegates to the _global_options instance.

    Args:
        pat: The option name.

    Returns:
        A string describing the option and its current value.

    Raises:
        ValueError: If the option name is unknown.
    """
    return _global_options.describe_option(pat)


@contextmanager
def option_context(*args: OptionTypes) -> Generator[None, None, None]:
    """Context manager to temporarily set options.

    Temporarily set options and restore them to their previous values after the
    context exits. Options may be supplied either as alternating option names and
    values, or as a single mapping of option names to values.

    Args:
        *args: Either an even number of arguments alternating between option names
               (str) and their corresponding values, or a single dict mapping
               option names to values.

    Yields:
        None

    Raises:
        ValueError: If an odd number of arguments is supplied (positional form), or if an
            unknown option name is supplied (either form).

    Example:
        >>> with option_context('column.customer_id', 'cust_id', 'column.store_id', 'outlet_id'):
        ...     # Do something with modified options
        ...     pass
        >>> with option_context({'column.customer_id': 'cust_id', 'column.store_id': 'outlet_id'}):
        ...     # Equivalent dict form
        ...     pass
        >>> # Options are restored to their previous values here
    """
    if len(args) == 1 and isinstance(args[0], dict):
        items = args[0].items()
    elif len(args) % 2 != 0:
        raise ValueError("The context manager requires an even number of arguments")
    else:
        items = zip(args[::2], args[1::2], strict=True)

    old_options: dict[str, OptionTypes] = {}
    try:
        for pat, val in items:
            old_options[pat] = get_option(pat)
            set_option(pat, val)
        yield
    finally:
        for pat, val in old_options.items():
            set_option(pat, val)


class AggColumns:
    """Aggregation columns accessed via cols.agg.*."""

    def __init__(self) -> None:
        """Initialize aggregation columns."""
        # Base aggregation columns
        self.customer_id: str = get_option("column.agg.customer_id")
        self.transaction_id: str = get_option("column.agg.transaction_id")
        self.unit_spend: str = get_option("column.agg.unit_spend")
        self.unit_qty: str = get_option("column.agg.unit_quantity")
        self.unit_cost: str = get_option("column.agg.unit_cost")
        self.promo_unit_spend: str = get_option("column.agg.promo_unit_spend")
        self.promo_unit_qty: str = get_option("column.agg.promo_unit_quantity")

        # Period variants
        self.customer_id_p1: str = ColumnHelper.join_options("column.agg.customer_id", "column.suffix.period_1")
        self.customer_id_p2: str = ColumnHelper.join_options("column.agg.customer_id", "column.suffix.period_2")
        self.transaction_id_p1: str = ColumnHelper.join_options("column.agg.transaction_id", "column.suffix.period_1")
        self.transaction_id_p2: str = ColumnHelper.join_options("column.agg.transaction_id", "column.suffix.period_2")
        self.unit_spend_p1: str = ColumnHelper.join_options("column.agg.unit_spend", "column.suffix.period_1")
        self.unit_spend_p2: str = ColumnHelper.join_options("column.agg.unit_spend", "column.suffix.period_2")
        self.unit_qty_p1: str = ColumnHelper.join_options("column.agg.unit_quantity", "column.suffix.period_1")
        self.unit_qty_p2: str = ColumnHelper.join_options("column.agg.unit_quantity", "column.suffix.period_2")
        self.unit_cost_p1: str = ColumnHelper.join_options("column.agg.unit_cost", "column.suffix.period_1")
        self.unit_cost_p2: str = ColumnHelper.join_options("column.agg.unit_cost", "column.suffix.period_2")
        self.promo_unit_spend_p1: str = ColumnHelper.join_options(
            "column.agg.promo_unit_spend",
            "column.suffix.period_1",
        )
        self.promo_unit_spend_p2: str = ColumnHelper.join_options(
            "column.agg.promo_unit_spend",
            "column.suffix.period_2",
        )
        self.promo_unit_qty_p1: str = ColumnHelper.join_options(
            "column.agg.promo_unit_quantity",
            "column.suffix.period_1",
        )
        self.promo_unit_qty_p2: str = ColumnHelper.join_options(
            "column.agg.promo_unit_quantity",
            "column.suffix.period_2",
        )

        # Diff variants
        self.customer_id_diff: str = ColumnHelper.join_options("column.agg.customer_id", "column.suffix.difference")
        self.transaction_id_diff: str = ColumnHelper.join_options(
            "column.agg.transaction_id",
            "column.suffix.difference",
        )
        self.unit_spend_diff: str = ColumnHelper.join_options("column.agg.unit_spend", "column.suffix.difference")
        self.unit_qty_diff: str = ColumnHelper.join_options("column.agg.unit_quantity", "column.suffix.difference")
        self.unit_cost_diff: str = ColumnHelper.join_options("column.agg.unit_cost", "column.suffix.difference")
        self.promo_unit_spend_diff: str = ColumnHelper.join_options(
            "column.agg.promo_unit_spend",
            "column.suffix.difference",
        )
        self.promo_unit_qty_diff: str = ColumnHelper.join_options(
            "column.agg.promo_unit_quantity",
            "column.suffix.difference",
        )

        # Percent diff variants
        self.customer_id_pct_diff: str = ColumnHelper.join_options(
            "column.agg.customer_id",
            "column.suffix.percent_difference",
        )
        self.transaction_id_pct_diff: str = ColumnHelper.join_options(
            "column.agg.transaction_id",
            "column.suffix.percent_difference",
        )
        self.unit_spend_pct_diff: str = ColumnHelper.join_options(
            "column.agg.unit_spend",
            "column.suffix.percent_difference",
        )
        self.unit_qty_pct_diff: str = ColumnHelper.join_options(
            "column.agg.unit_quantity",
            "column.suffix.percent_difference",
        )
        self.unit_cost_pct_diff: str = ColumnHelper.join_options(
            "column.agg.unit_cost",
            "column.suffix.percent_difference",
        )
        self.promo_unit_spend_pct_diff: str = ColumnHelper.join_options(
            "column.agg.promo_unit_spend",
            "column.suffix.percent_difference",
        )
        self.promo_unit_qty_pct_diff: str = ColumnHelper.join_options(
            "column.agg.promo_unit_quantity",
            "column.suffix.percent_difference",
        )

        # Contribution variants
        self.customer_id_contrib: str = ColumnHelper.join_options(
            "column.agg.customer_id",
            "column.suffix.contribution",
        )

        # Percent variants
        self.customers_pct: str = ColumnHelper.join_options("column.agg.customer_id", "column.suffix.percent")

        # Unknown/Total variants
        self.unit_spend_unknown: str = ColumnHelper.join_options(
            "column.agg.unit_spend",
            "column.suffix.unknown_customer",
        )
        self.unit_spend_total: str = ColumnHelper.join_options("column.agg.unit_spend", "column.suffix.total")
        self.transaction_id_unknown: str = ColumnHelper.join_options(
            "column.agg.transaction_id",
            "column.suffix.unknown_customer",
        )
        self.transaction_id_total: str = ColumnHelper.join_options("column.agg.transaction_id", "column.suffix.total")
        self.unit_qty_unknown: str = ColumnHelper.join_options(
            "column.agg.unit_quantity",
            "column.suffix.unknown_customer",
        )
        self.unit_qty_total: str = ColumnHelper.join_options("column.agg.unit_quantity", "column.suffix.total")


class CalcColumns:
    """Calculated columns accessed via cols.calc.*."""

    def __init__(self) -> None:
        """Initialize calculated columns."""
        # Base calculated columns
        self.spend_per_cust: str = get_option("column.calc.spend_per_customer")
        self.trans_per_cust: str = get_option("column.calc.transactions_per_customer")
        self.spend_per_trans: str = get_option("column.calc.spend_per_transaction")
        self.units_per_trans: str = get_option("column.calc.units_per_transaction")
        self.price_per_unit: str = get_option("column.calc.price_per_unit")
        self.price_elasticity: str = get_option("column.calc.price_elasticity")
        self.frequency_elasticity: str = get_option("column.calc.frequency_elasticity")

        # Period variants
        self.spend_per_cust_p1: str = ColumnHelper.join_options(
            "column.calc.spend_per_customer",
            "column.suffix.period_1",
        )
        self.spend_per_cust_p2: str = ColumnHelper.join_options(
            "column.calc.spend_per_customer",
            "column.suffix.period_2",
        )
        self.trans_per_cust_p1: str = ColumnHelper.join_options(
            "column.calc.transactions_per_customer",
            "column.suffix.period_1",
        )
        self.trans_per_cust_p2: str = ColumnHelper.join_options(
            "column.calc.transactions_per_customer",
            "column.suffix.period_2",
        )
        self.spend_per_trans_p1: str = ColumnHelper.join_options(
            "column.calc.spend_per_transaction",
            "column.suffix.period_1",
        )
        self.spend_per_trans_p2: str = ColumnHelper.join_options(
            "column.calc.spend_per_transaction",
            "column.suffix.period_2",
        )
        self.units_per_trans_p1: str = ColumnHelper.join_options(
            "column.calc.units_per_transaction",
            "column.suffix.period_1",
        )
        self.units_per_trans_p2: str = ColumnHelper.join_options(
            "column.calc.units_per_transaction",
            "column.suffix.period_2",
        )
        self.price_per_unit_p1: str = ColumnHelper.join_options("column.calc.price_per_unit", "column.suffix.period_1")
        self.price_per_unit_p2: str = ColumnHelper.join_options("column.calc.price_per_unit", "column.suffix.period_2")

        # Diff variants
        self.spend_per_cust_diff: str = ColumnHelper.join_options(
            "column.calc.spend_per_customer",
            "column.suffix.difference",
        )
        self.trans_per_cust_diff: str = ColumnHelper.join_options(
            "column.calc.transactions_per_customer",
            "column.suffix.difference",
        )
        self.spend_per_trans_diff: str = ColumnHelper.join_options(
            "column.calc.spend_per_transaction",
            "column.suffix.difference",
        )
        self.units_per_trans_diff: str = ColumnHelper.join_options(
            "column.calc.units_per_transaction",
            "column.suffix.difference",
        )
        self.price_per_unit_diff: str = ColumnHelper.join_options(
            "column.calc.price_per_unit",
            "column.suffix.difference",
        )

        # Percent diff variants
        self.spend_per_cust_pct_diff: str = ColumnHelper.join_options(
            "column.calc.spend_per_customer",
            "column.suffix.percent_difference",
        )
        self.trans_per_cust_pct_diff: str = ColumnHelper.join_options(
            "column.calc.transactions_per_customer",
            "column.suffix.percent_difference",
        )
        self.spend_per_trans_pct_diff: str = ColumnHelper.join_options(
            "column.calc.spend_per_transaction",
            "column.suffix.percent_difference",
        )
        self.units_per_trans_pct_diff: str = ColumnHelper.join_options(
            "column.calc.units_per_transaction",
            "column.suffix.percent_difference",
        )
        self.price_per_unit_pct_diff: str = ColumnHelper.join_options(
            "column.calc.price_per_unit",
            "column.suffix.percent_difference",
        )

        # Contribution variants
        self.spend_per_cust_contrib: str = ColumnHelper.join_options(
            "column.calc.spend_per_customer",
            "column.suffix.contribution",
        )
        self.trans_per_cust_contrib: str = ColumnHelper.join_options(
            "column.calc.transactions_per_customer",
            "column.suffix.contribution",
        )
        self.spend_per_trans_contrib: str = ColumnHelper.join_options(
            "column.calc.spend_per_transaction",
            "column.suffix.contribution",
        )
        self.units_per_trans_contrib: str = ColumnHelper.join_options(
            "column.calc.units_per_transaction",
            "column.suffix.contribution",
        )
        self.price_per_unit_contrib: str = ColumnHelper.join_options(
            "column.calc.price_per_unit",
            "column.suffix.contribution",
        )

        # Unknown/Total variants
        self.spend_per_trans_unknown: str = ColumnHelper.join_options(
            "column.calc.spend_per_transaction",
            "column.suffix.unknown_customer",
        )
        self.spend_per_trans_total: str = ColumnHelper.join_options(
            "column.calc.spend_per_transaction",
            "column.suffix.total",
        )
        self.price_per_unit_unknown: str = ColumnHelper.join_options(
            "column.calc.price_per_unit",
            "column.suffix.unknown_customer",
        )
        self.price_per_unit_total: str = ColumnHelper.join_options("column.calc.price_per_unit", "column.suffix.total")
        self.units_per_trans_unknown: str = ColumnHelper.join_options(
            "column.calc.units_per_transaction",
            "column.suffix.unknown_customer",
        )
        self.units_per_trans_total: str = ColumnHelper.join_options(
            "column.calc.units_per_transaction",
            "column.suffix.total",
        )


class ColumnHelper:
    """A class to help with column naming conventions.

    Access patterns:
    - Base columns: cols.transaction_date, cols.customer_id, etc.
    - Aggregation columns: cols.agg.unit_spend, cols.agg.customer_id_p1, etc.
    - Calculated columns: cols.calc.spend_per_cust, cols.calc.price_per_unit_diff, etc.
    """

    def __init__(self) -> None:
        """Initialize column helper with base columns and nested column groups."""
        # Base columns (stay flat on ColumnHelper)
        self.transaction_date = get_option("column.transaction_date")
        self.transaction_time = get_option("column.transaction_time")
        self.customer_id = get_option("column.customer_id")
        self.transaction_id = get_option("column.transaction_id")
        self.product_id = get_option("column.product_id")
        self.store_id = get_option("column.store_id")
        self.unit_spend = get_option("column.unit_spend")
        self.unit_qty = get_option("column.unit_quantity")
        self.unit_cost = get_option("column.unit_cost")
        self.promo_unit_spend = get_option("column.promo_unit_spend")
        self.promo_unit_qty = get_option("column.promo_unit_quantity")

        # Nested column groups
        self.agg = AggColumns()
        self.calc = CalcColumns()

    @staticmethod
    def join_options(*args: str, sep: str = "_") -> str:
        """Join multiple option values together with a separator.

        This method resolves option keys to their configured values and joins them.
        Commonly used to create column names with suffixes like period indicators.

        Args:
            *args: Option keys to resolve and join (e.g., "column.agg.unit_spend", "column.suffix.period_1")
            sep: Separator to use when joining values (default: "_")

        Returns:
            A string with all resolved option values joined together.

        Example:
            >>> join_options("column.agg.unit_spend", "column.suffix.period_1")
            "spend_p1"  # Assuming default options
        """
        return sep.join(map(get_option, args))


class PlotStyleHelper:
    """A class to help with plot styling options.

    Access patterns:
    - Font options: style.title_font, style.title_size, etc.
    - Spacing options: style.x_label_pad, style.y_label_pad, etc.
    - Style options: style.background_color, style.grid_alpha, etc.
    """

    def __init__(self) -> None:
        """Initialize plot style helper with font, spacing, and style options."""
        # Font options
        self.title_font: str = get_option("plot.font.title_font")
        self.eyebrow_font: str = get_option("plot.font.eyebrow_font")
        self.subtitle_font: str = get_option("plot.font.subtitle_font")
        self.label_font: str = get_option("plot.font.label_font")
        self.tick_font: str = get_option("plot.font.tick_font")
        self.source_font: str = get_option("plot.font.source_font")
        self.data_label_font: str = get_option("plot.font.data_label_font")
        self.legend_font: str = get_option("plot.font.legend_font")
        self.title_size: float = get_option("plot.font.title_size")
        self.eyebrow_size: float = get_option("plot.font.eyebrow_size")
        self.subtitle_size: float = get_option("plot.font.subtitle_size")
        self.label_size: float = get_option("plot.font.label_size")
        self.tick_size: float = get_option("plot.font.tick_size")
        self.source_size: float = get_option("plot.font.source_size")
        self.data_label_size: float = get_option("plot.font.data_label_size")
        self.legend_size: float = get_option("plot.font.legend_size")

        # Chrome colors
        self.eyebrow_color: str = get_option("plot.color.eyebrow")
        self.subtitle_color: str = get_option("plot.color.subtitle")
        self.title_color: str = get_option("plot.color.title")
        self.tab_color: str = get_option("plot.color.tab")
        self.source_color: str = get_option("plot.color.source")

        # Spacing options
        self.x_label_pad: int = get_option("plot.spacing.x_label_pad")
        self.y_label_pad: int = get_option("plot.spacing.y_label_pad")

        # Style options
        self.background_color: str = get_option("plot.style.background_color")
        self.grid_color: str = get_option("plot.style.grid_color")
        self.grid_alpha: float = get_option("plot.style.grid_alpha")
        self.show_top_spine: bool = get_option("plot.style.show_top_spine")
        self.show_right_spine: bool = get_option("plot.style.show_right_spine")
        self.show_bottom_spine: bool = get_option("plot.style.show_bottom_spine")
        self.show_left_spine: bool = get_option("plot.style.show_left_spine")
        self.legend_bbox_to_anchor: list[float] = get_option("plot.style.legend_bbox_to_anchor")
        self.legend_loc: str = get_option("plot.style.legend_loc")
        self.cell_corner_radius: float = get_option("plot.style.cell_corner_radius")
        self.cell_gap: float = get_option("plot.style.cell_gap")
        self.show_tab: bool = get_option("plot.style.show_tab")
        self.auto_rotate_x_ticks: bool = get_option("plot.style.auto_rotate_x_ticks")
        self.auto_wrap_x_ticks: bool = get_option("plot.style.auto_wrap_x_ticks")
