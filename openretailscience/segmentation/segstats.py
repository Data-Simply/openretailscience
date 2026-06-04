"""Segment Performance Analysis for Retail Business Intelligence.

## Business Context

Retailers need to understand performance differences across various business dimensions -
whether comparing customer segments, store locations, product categories, brands, channels,
or any other grouping. This module transforms transactional data into actionable insights
by calculating key performance metrics for any segment or combination of segments.

## The Business Problem

Business stakeholders receive segment data but struggle to answer performance questions:
- Which stores/categories/customer segments generate the most revenue?
- How do transaction patterns differ between segments?
- What's the customer density and spending behavior by segment?
- Are certain combinations of segments more valuable than others?

Without segment performance analysis, decisions are made on incomplete information
rather than data-driven insights about segment value and behavior.

## Real-World Applications

### Customer Segment Analysis
- Compare RFM segments: Which customer types drive the most revenue?
- Analyze geographic segments: Regional performance differences
- Age/demographic segments: Spending patterns by customer characteristics

### Store/Location Analysis
- Store performance comparison: Revenue per customer, transaction frequency
- Regional analysis: Market penetration and customer behavior by area
- Channel analysis: Online vs in-store performance metrics

### Product/Category Analysis
- Category performance: Which product lines drive customer frequency?
- Brand analysis: Private label vs national brand customer behavior
- SKU analysis: Performance metrics for product rationalization decisions

### Multi-Dimensional Analysis
- Store + Customer segment: High-value customers by location
- Category + Channel: Product performance across sales channels
- Brand + Geography: Regional brand performance variations

This module calculates comprehensive statistics including spend, customer counts,
transaction frequency, average basket size, and custom business metrics for any
segment combination.
"""

import functools
import warnings
from itertools import chain, combinations
from typing import Any, Literal

import ibis
import pandas as pd

from openretailscience.core.validation import ensure_columns, ensure_data_has_columns, ensure_ibis_table
from openretailscience.options import ColumnHelper, get_option

__all__ = ["SegTransactionStats", "cube", "rollup"]

# Maximum number of dimensions for CUBE mode before warning about exponential growth.
# CUBE generates 2^n grouping sets, so 6 dimensions = 64 sets (reasonable), but 7+ dimensions
# = 128+ sets (potentially expensive). This threshold balances flexibility with performance awareness.
MAX_CUBE_DIMENSIONS_WITHOUT_WARNING = 6


def cube(*columns: str) -> list[tuple[str, ...]]:
    """Generate CUBE grouping sets (all possible combinations).

    CUBE generates all 2^n combinations of the specified columns, from full detail down to
    grand total. Returns a list of tuples that can be passed directly to grouping_sets,
    or used with fixed columns in a nested list specification.

    This matches SQL's GROUP BY CUBE(A, B), C syntax.

    Args:
        *columns (str): Column names to include in the CUBE operation

    Returns:
        list[tuple[str, ...]]: List of tuples representing all CUBE combinations

    Raises:
        ValueError: If no columns are provided
        TypeError: If any column is not a string
        UserWarning: If more than MAX_CUBE_DIMENSIONS_WITHOUT_WARNING columns

    Example:
        >>> from openretailscience.segmentation import cube
        >>>
        >>> # Simple CUBE - returns list of tuples
        >>> cube("store", "region")
        [("store", "region"), ("store",), ("region",), ()]
        >>>
        >>> # Use directly (equivalent to explicit list of tuples)
        >>> stats = SegTransactionStats(
        ...     data=df,
        ...     segment_col=["store", "region", "date"],
        ...     grouping_sets=cube("store", "region", "date")
        ... )
        >>>
        >>> # CUBE with fixed columns - wrap in tuple
        >>> stats = SegTransactionStats(
        ...     data=df,
        ...     segment_col=["store", "region", "date"],
        ...     grouping_sets=[(cube("store", "region"), "date")]
        ... )
        >>> # Produces 4 grouping sets (2^2 from CUBE):
        >>> # [("store", "region", "date"), ("store", "date"), ("region", "date"), ("date",)]
    """
    if len(columns) == 0:
        raise ValueError("cube() requires at least one column")

    # Validate all columns are strings
    for col in columns:
        if not isinstance(col, str):
            msg = f"All column names must be strings. Got {type(col).__name__}: {col}"
            raise TypeError(msg)

    # Validation: warn if too many dimensions
    num_grouping_sets = 2 ** len(columns)
    if len(columns) > MAX_CUBE_DIMENSIONS_WITHOUT_WARNING:
        warnings.warn(
            f"CUBE with {len(columns)} dimensions will generate {num_grouping_sets} grouping sets, "
            f"which may be computationally expensive. Consider using ROLLUP mode or limiting to "
            f"{MAX_CUBE_DIMENSIONS_WITHOUT_WARNING} dimensions.",
            UserWarning,
            stacklevel=2,
        )

    # Expansion: generate all 2^n combinations and return as list
    return list(
        chain.from_iterable(combinations(columns, size) for size in range(len(columns), -1, -1)),
    )


def rollup(*columns: str) -> list[tuple[str, ...]]:
    """Generate ROLLUP grouping sets (hierarchical aggregation levels).

    ROLLUP generates n+1 hierarchical levels from right to left. Returns a list of tuples
    that can be passed directly to grouping_sets, or used with fixed columns in a nested
    list specification.

    This matches SQL's GROUP BY ROLLUP(A, B), C syntax.

    Args:
        *columns (str): Column names in hierarchical order (left = highest level)

    Returns:
        list[tuple[str, ...]]: List of tuples representing ROLLUP hierarchy levels

    Raises:
        ValueError: If no columns are provided
        TypeError: If any column is not a string

    Example:
        >>> from openretailscience.segmentation import rollup
        >>>
        >>> # Simple ROLLUP - returns list of tuples
        >>> rollup("year", "quarter", "month")
        [("year", "quarter", "month"), ("year", "quarter"), ("year",), ()]
        >>>
        >>> # Use directly (equivalent to explicit list of tuples)
        >>> stats = SegTransactionStats(
        ...     data=df,
        ...     segment_col=["year", "quarter", "month"],
        ...     grouping_sets=rollup("year", "quarter", "month")
        ... )
        >>>
        >>> # ROLLUP with fixed column - wrap in tuple
        >>> stats = SegTransactionStats(
        ...     data=df,
        ...     segment_col=["year", "quarter", "month", "store"],
        ...     grouping_sets=[(rollup("year", "quarter", "month"), "store")]
        ... )
        >>> # Produces 4 grouping sets (3+1 from ROLLUP):
        >>> # [("year", "quarter", "month", "store"), ("year", "quarter", "store"),
        >>> #  ("year", "store"), ("store",)]
    """
    if len(columns) == 0:
        raise ValueError("rollup() requires at least one column")

    # Validate all columns are strings
    for col in columns:
        if not isinstance(col, str):
            msg = f"All column names must be strings. Got {type(col).__name__}: {col}"
            raise TypeError(msg)

    # Expansion: generate n+1 hierarchical levels and return as list
    return [tuple(columns[:i]) for i in range(len(columns), -1, -1)]


class SegTransactionStats:
    """Calculates transaction performance statistics for any business segment or dimension.

    Analyzes transaction data across segments like customer types, store locations,
    product categories, brands, channels, or any combination to reveal performance
    differences and guide business decisions.

    The class automatically calculates key retail metrics including total spend,
    unique customers, transaction frequency, spend per customer, and custom
    aggregations for comparison across segments.
    """

    def __init__(
        self,
        data: pd.DataFrame | ibis.Table,
        segment_col: str | list[str] = "segment_name",
        calc_total: bool | None = None,
        extra_aggs: dict[str, tuple[str, str]] | None = None,
        calc_rollup: bool | None = None,
        rollup_value: Any | list[Any] = "Total",  # noqa: ANN401 - Any is required for ibis.literal typing
        unknown_customer_value: int | str | ibis.Scalar | ibis.expr.types.BooleanColumn | None = None,
        grouping_sets: Literal["rollup", "cube", "total"] | list[tuple[str, ...]] | None = None,
        sort_by: bool = False,
    ) -> None:
        """Calculates transaction statistics by segment.

        Args:
            data (pd.DataFrame | ibis.Table): The transaction data. The dataframe must contain the columns
                customer_id, unit_spend and transaction_id. If the dataframe contains the column unit_quantity, then
                the columns unit_spend and unit_quantity are used to calculate the price_per_unit and
                units_per_transaction.
            segment_col (str | list[str], optional): The column or list of columns to use for the segmentation.
                Defaults to "segment_name".
            calc_total (bool | None, optional): Whether to include the total row. Defaults to True if grouping_sets is
                None. Cannot be used with grouping_sets parameter.
                Note: This parameter is planned for deprecation. Use grouping_sets parameter for new code.
            extra_aggs (dict[str, tuple[str, str]], optional): Additional aggregations to perform.
                The keys in the dictionary will be the column names for the aggregation results.
                The values are tuples with (column_name, aggregation_function), where:
                - column_name is the name of the column to aggregate
                - aggregation_function is a string name of an Ibis aggregation function (e.g., "nunique", "sum")
                Example: {"stores": ("store_id", "nunique")} would count unique store_ids.
            calc_rollup (bool | None, optional): Whether to calculate rollup totals. Defaults to False if grouping_sets
                is None. When True and multiple segment columns are provided, the method generates subtotal rows for
                both:
                - Prefix rollups: progressively aggregating left-to-right (e.g., [A, B, Total], [A, Total, Total]).
                - Suffix rollups: progressively aggregating right-to-left (e.g., [Total, B, C], [Total, Total, C]).
                A grand total row is also included when calc_total is True.
                Note: This differs from grouping_sets='rollup' which generates only prefix rollups (SQL standard).
                Performance: adds O(n) extra aggregation passes where n is the number of segment
                columns. For large hierarchies, consider disabling rollups or reducing columns.
                Cannot be used with grouping_sets parameter.
                Note: This parameter is planned for deprecation. Use grouping_sets parameter for new code.
            rollup_value (Any | list[Any], optional): The value to use for rollup totals. Can be a single value
                applied to all columns or a list of values matching the length of segment_col, with each value
                cast to match the corresponding column type. Defaults to "Total".
            unknown_customer_value
                (int | str | ibis.Scalar | ibis.expr.types.BooleanColumn | None, optional):
                Value or expression identifying unknown customers for separate tracking.
                When provided, metrics are split into identified, unknown, and total variants.
                Accepts simple values (e.g., -1), ibis literals, or boolean expressions
                (e.g., data["customer_id"] < 0). Requires customer_id column. Defaults to None.
            grouping_sets (Literal["rollup", "cube"] | list[list[str] | tuple[str, ...]] | None, optional):
                Grouping sets mode. Mutually exclusive with calc_total/calc_rollup when explicitly set.
                - "rollup": SQL ROLLUP (hierarchical aggregation from right to left). Generates [A,B,C], [A,B], [A], [].
                - "cube": SQL CUBE (all possible combinations). Generates 2^n grouping sets for n dimensions.
                - list: Custom grouping sets (list of lists/tuples). Specify arbitrary dimension combinations.
                  Each element must be a list or tuple of column names from segment_col. Empty list/tuple ()
                  represents grand total. Automatically deduplicates and validates column names.
                - None: Use calc_total/calc_rollup behavior (default).
                Defaults to None.
            sort_by (bool, optional): When True, sort the output rows ascending by ``segment_col`` using
                ibis ``.order_by()`` so the sort runs at the backend. Defaults to False (no ordering applied).

        Raises:
            ValueError: If grouping_sets is used with explicit calc_total or calc_rollup.
            ValueError: If grouping_sets is not a valid value.

        Example:
            >>> # Hierarchical rollup using grouping_sets
            >>> stats = SegTransactionStats(
            ...     data=df,
            ...     segment_col=["region", "store", "product"],
            ...     grouping_sets="rollup",
            ... )
            >>>
            >>> # All combinations using CUBE
            >>> stats = SegTransactionStats(
            ...     data=df,
            ...     segment_col=["region", "store", "product"],
            ...     grouping_sets="cube",
            ... )
            >>>
            >>> # Custom grouping sets for specific dimension combinations
            >>> stats = SegTransactionStats(
            ...     data=df,
            ...     segment_col=["region", "store", "product"],
            ...     grouping_sets=[
            ...         ("region", "product"),  # Regional product performance (skip store)
            ...         ("product",),           # Product-only totals
            ...         ()                      # Grand total
            ...     ],
            ... )
            >>>
            >>> # Legacy behavior (backward compatible)
            >>> stats = SegTransactionStats(
            ...     data=df,
            ...     segment_col=["region", "store"],
            ...     calc_total=True,
            ...     calc_rollup=False,
            ... )
        """
        data = ensure_ibis_table(data, "data")

        cols = ColumnHelper()

        segment_col = ensure_columns(data, segment_col, "segment_col")

        # segment_col is already validated above; only the function's hard-coded requirements remain.
        ensure_data_has_columns(
            data,
            [
                cols.unit_spend,
                cols.transaction_id,
                *filter(lambda x: x in data.columns, [cols.unit_qty, cols.customer_id]),
            ],
        )

        # Validate extra_aggs if provided
        self._validate_extra_aggs(data, extra_aggs)

        self.segment_col = segment_col
        self.extra_aggs = {} if extra_aggs is None else extra_aggs
        self.rollup_value = rollup_value
        self.unknown_customer_value = unknown_customer_value

        # Validate grouping_sets parameter
        self._validate_grouping_sets_params(grouping_sets, calc_total, calc_rollup)

        # Normalize parameters as local variables (only in legacy mode)
        if grouping_sets is None:
            calc_total = True if calc_total is None else calc_total
            calc_rollup = False if calc_rollup is None else calc_rollup

        self.table = self._calc_seg_stats(
            data,
            segment_col,
            calc_total,
            self.extra_aggs,
            calc_rollup,
            rollup_value,
            unknown_customer_value,
            grouping_sets,
        )

        if sort_by:
            self.table = self.table.order_by(segment_col)

    @staticmethod
    def _get_col_order(include_quantity: bool, include_customer: bool, include_unknown: bool = False) -> list[str]:
        """Returns the default column order.

        Args:
            include_quantity (bool): Whether to include the columns related to quantity.
            include_customer (bool): Whether to include customer-based columns.
            include_unknown (bool, optional): Whether to include unknown customer columns. Defaults to False.

        Returns:
            list[str]: The default column order.
        """
        cols = ColumnHelper()

        column_configs = [
            (cols.agg.unit_spend, True),
            (cols.agg.transaction_id, True),
            (cols.agg.customer_id, include_customer),
            (cols.agg.unit_qty, include_quantity),
            (cols.calc.spend_per_cust, include_customer),
            (cols.calc.spend_per_trans, True),
            (cols.calc.trans_per_cust, include_customer),
            (cols.calc.price_per_unit, include_quantity),
            (cols.calc.units_per_trans, include_quantity),
        ]

        # Add unknown customer columns if tracking unknown customers
        if include_unknown:
            unknown_configs = [
                (cols.agg.unit_spend_unknown, True),
                (cols.agg.transaction_id_unknown, True),
                (cols.agg.unit_qty_unknown, include_quantity),
                (cols.calc.spend_per_trans_unknown, True),
                (cols.calc.price_per_unit_unknown, include_quantity),
                (cols.calc.units_per_trans_unknown, include_quantity),
            ]
            column_configs.extend(unknown_configs)

            # Add total columns
            total_configs = [
                (cols.agg.unit_spend_total, True),
                (cols.agg.transaction_id_total, True),
                (cols.agg.unit_qty_total, include_quantity),
                (cols.calc.spend_per_trans_total, True),
                (cols.calc.price_per_unit_total, include_quantity),
                (cols.calc.units_per_trans_total, include_quantity),
            ]
            column_configs.extend(total_configs)

        return [col for col, condition in column_configs if condition]

    @staticmethod
    def _create_typed_literals(
        data: ibis.Table,
        columns: list[str],
        values: list[Any],
    ) -> dict[str, ibis.Scalar]:
        """Create a dictionary of ibis literals with proper column types.

        Args:
            data (ibis.Table): The data table containing column type information
            columns (list[str]): List of column names
            values (list[Any]): List of values to convert to typed literals

        Returns:
            dict[str, ibis.Scalar]: Dictionary mapping column names to typed literals
        """
        mutations = {}
        for i, col in enumerate(columns):
            col_type = data[col].type()
            mutations[col] = ibis.literal(values[i], type=col_type)
        return mutations

    @staticmethod
    def _validate_extra_aggs(data: ibis.Table, extra_aggs: dict[str, tuple[str, str]] | None) -> None:
        """Validate extra_aggs parameter.

        Args:
            data (ibis.Table): The data table to validate against
            extra_aggs (dict[str, tuple[str, str]] | None): Extra aggregations to validate

        Raises:
            ValueError: If column doesn't exist or aggregation function is not available
        """
        if extra_aggs is None:
            return

        for col_tuple in extra_aggs.values():
            col, func = col_tuple
            if col not in data.columns:
                msg = f"Column '{col}' specified in extra_aggs does not exist in the data"
                raise ValueError(msg)
            if not hasattr(data[col], func):
                msg = f"Aggregation function '{func}' not available for column '{col}'"
                raise ValueError(msg)

    @staticmethod
    def _validate_grouping_sets_params(
        grouping_sets: Literal["rollup", "cube", "total"] | list[tuple[str, ...]] | None,
        calc_total: bool | None,
        calc_rollup: bool | None,
    ) -> None:
        """Validate grouping_sets parameter (type checking only).

        Column validation happens in _generate_grouping_sets() since it requires segment_col.

        Args:
            grouping_sets: The grouping_sets parameter value
            calc_total (bool | None): Whether to include grand total
            calc_rollup (bool | None): Whether to generate rollup subtotals

        Raises:
            ValueError: If grouping_sets is used with explicit calc_total or calc_rollup
            ValueError: If grouping_sets is not a valid value
            TypeError: If grouping_sets has invalid type
        """
        if grouping_sets is None:
            # Warn if relying on implicit calc_total=True default (calc_total will be removed)
            if calc_total is None and calc_rollup is None:
                warnings.warn(
                    "The calc_total parameter is deprecated and will be removed in a future version. "
                    "To maintain the current behavior of including a grand total, use grouping_sets='total' "
                    "or grouping_sets=[()] instead. "
                    "See documentation for more flexible aggregation control with the grouping_sets parameter.",
                    FutureWarning,
                    stacklevel=3,
                )
            return

        # Mutual exclusivity check
        if calc_total is not None or calc_rollup is not None:
            raise ValueError("Cannot use grouping_sets with calc_total or calc_rollup")

        # String validation
        if isinstance(grouping_sets, str):
            if grouping_sets not in ["rollup", "cube", "total"]:
                msg = (
                    f"grouping_sets must be 'rollup', 'cube', 'total', a list of tuples,"
                    f" or None. Got: '{grouping_sets}'"
                )
                raise ValueError(msg)

        # List validation - only accept tuples for consistency (Ticket 5 design)
        elif isinstance(grouping_sets, list):
            if len(grouping_sets) == 0:
                raise ValueError("grouping_sets list cannot be empty")

            # Validate each element is a tuple (consistency: always list of tuples)
            for item in grouping_sets:
                if not isinstance(item, tuple):
                    msg = f"Each element must be a tuple. Got: {type(item).__name__}"
                    raise TypeError(msg)

    @staticmethod
    def _flatten_item(item: tuple) -> list[tuple[str, ...]]:
        """Flatten a single item into grouping sets.

        Uses structural detection to distinguish explicit sets from specifications:
        - Tuple of strings only → explicit grouping set (return as-is)
        - Tuple containing a list → specification to expand (cube()/rollup() result + optional fixed columns)

        The cube()/rollup() functions return lists, so we detect them by checking if the tuple
        contains a list element.

        Args:
            item (tuple): A tuple that is either an explicit grouping set or a specification

        Returns:
            list[tuple[str, ...]]: List of one or more grouping sets

        Raises:
            ValueError: If specification tuple contains multiple cube()/rollup() calls or is empty
            TypeError: If specification tuple contains invalid types

        Example:
            >>> # Explicit set (tuple of strings only)
            >>> _flatten_item(("region", "store"))
            [("region", "store")]
            >>>
            >>> # Specification (tuple containing cube() result + fixed column)
            >>> cube_result = [("region", "store"), ("region",), ("store",), ()]
            >>> _flatten_item((cube_result, "date"))
            [
                ("region", "store", "date"),
                ("region", "date"),
                ("store", "date"),
                ("date",)
            ]
            >>>
            >>> # Invalid: Multiple cube()/rollup() calls
            >>> _flatten_item((cube("region"), rollup("store")))  # ValueError
            >>>
            >>> # Invalid: Mixed types (integers not allowed)
            >>> _flatten_item((cube("region"), 123))  # TypeError
            >>>
            >>> # Invalid: Empty cube()/rollup() result
            >>> _flatten_item(([],))  # ValueError
        """
        # Check if tuple contains a list (cube()/rollup() result)
        has_list = any(isinstance(elem, list) for elem in item)

        if not has_list:
            # Explicit grouping set - all elements are strings
            return [item]

        # Specification to flatten - must contain exactly one cube()/rollup() result + optional fixed columns
        grouping_sets_list = None
        fixed_cols = []

        for element in item:
            if isinstance(element, list):
                # This is a cube()/rollup() result (list of tuples)
                if grouping_sets_list is not None:
                    raise ValueError("Only one cube()/rollup() call allowed per specification")
                grouping_sets_list = element
            elif isinstance(element, str):
                # Fixed column
                fixed_cols.append(element)
            else:
                msg = f"Invalid type in specification tuple: {type(element).__name__}"
                raise TypeError(msg)

        # Validate cube()/rollup() result is not empty
        if len(grouping_sets_list) == 0:
            raise ValueError("Specification tuple must contain non-empty cube() or rollup() result")

        # Flatten: append fixed columns to each set
        fixed_suffix = tuple(fixed_cols)
        return [gs + fixed_suffix for gs in grouping_sets_list]

    @staticmethod
    def _generate_grouping_sets(
        segment_col: list[str],
        calc_total: bool | None = None,
        calc_rollup: bool | None = None,
        grouping_sets: (
            Literal["rollup", "cube", "total"] | list[tuple[str, ...] | tuple[list | str, ...]] | None
        ) = None,
    ) -> list[tuple[str, ...]]:
        """Generate grouping sets based on grouping_sets parameter or calc_total/calc_rollup settings.

        Args:
            segment_col (list[str]): The segment columns to generate grouping sets for
            calc_total (bool | None): Whether to include grand total (ignored if grouping_sets is not None)
            calc_rollup (bool | None): Whether to generate rollup subtotals (ignored if grouping_sets is not None)
            grouping_sets: Grouping sets mode ('rollup', 'cube', list of tuples, or None)

        Returns:
            list[tuple[str, ...]]: List of grouping set tuples. Each tuple contains the
                column names to group by for that grouping set. Empty tuple () represents
                grand total.

        Raises:
            ValueError: If custom grouping set contains column not in segment_col

        Example:
            >>> # ROLLUP mode
            >>> _generate_grouping_sets(["region", "store", "product"], grouping_sets="rollup")
            [
                ("region", "store", "product"),  # full detail
                ("region", "store"),             # rollup level 1
                ("region",),                     # rollup level 2
                (),                              # grand total
            ]

            >>> # Custom grouping sets
            >>> _generate_grouping_sets(
            ...     ["region", "store", "product"],
            ...     grouping_sets=[("region", "product"), ("product",), ()]
            ... )
            [
                ("region", "product"),  # Regional product performance
                ("product",),           # Product-only totals
                (),                     # Grand total
            ]

            >>> # Legacy mode (calc_total/calc_rollup)
            >>> _generate_grouping_sets(["region", "store", "product"], True, True, None)
            [
                ("region", "store", "product"),  # base grouping
                ("region", "store"),             # prefix rollup
                ("region",),                     # prefix rollup
                ("store", "product"),            # suffix rollup
                ("product",),                    # suffix rollup
                (),                              # grand total
            ]
        """
        # Handle string shortcuts - delegate to helper functions
        if grouping_sets == "rollup":
            return rollup(*segment_col)

        if grouping_sets == "cube":
            return cube(*segment_col)

        if grouping_sets == "total":
            return [tuple(segment_col), ()]

        # Handle list of tuples - flatten each item
        if isinstance(grouping_sets, list):
            expanded = []
            for item in grouping_sets:
                expanded.extend(SegTransactionStats._flatten_item(item))

            # Deduplicate (order not preserved, but not needed)
            expanded = list(set(expanded))

            # Validate columns (applies to list modes)
            all_mentioned_cols = {col for gs in expanded for col in gs}
            invalid_cols = all_mentioned_cols - set(segment_col)
            if invalid_cols:
                msg = (
                    f"Columns {sorted(invalid_cols)} in grouping_sets not found in segment_col {segment_col}. "
                    f"All grouping set columns must be in segment_col."
                )
                raise ValueError(msg)

            unmentioned_cols = set(segment_col) - all_mentioned_cols
            if unmentioned_cols:
                msg = (
                    f"Columns {sorted(unmentioned_cols)} in segment_col are not mentioned in any grouping set. "
                    f"All segment_col columns must appear in at least one grouping set. "
                    f"Either remove these columns from segment_col or include them in at least one grouping set."
                )
                raise ValueError(msg)

            return expanded

        # Legacy mode - delegate to helper
        return SegTransactionStats._generate_legacy_grouping_sets(segment_col, calc_total, calc_rollup)

    @staticmethod
    def _generate_legacy_grouping_sets(
        segment_col: list[str],
        calc_total: bool | None,
        calc_rollup: bool | None,
    ) -> list[tuple[str, ...]]:
        """Generate grouping sets using legacy calc_total/calc_rollup parameters.

        Args:
            segment_col (list[str]): The segment columns
            calc_total (bool | None): Whether to include grand total
            calc_rollup (bool | None): Whether to generate rollup subtotals

        Returns:
            list[tuple[str, ...]]: List of grouping set tuples
        """
        grouping_sets_list = [tuple(segment_col)]  # Base grouping always included

        if calc_rollup:
            # Prefix rollups: progressively remove from the right
            grouping_sets_list.extend(tuple(segment_col[:i]) for i in range(1, len(segment_col)))

            # Suffix rollups: progressively remove from the left (only if calc_total=True)
            if calc_total:
                grouping_sets_list.extend(tuple(segment_col[i:]) for i in range(1, len(segment_col)))

        if calc_total:
            grouping_sets_list.append(())  # Empty tuple = grand total

        return grouping_sets_list

    @staticmethod
    def _execute_grouping_sets(
        data: ibis.Table,
        grouping_sets: list[tuple[str, ...]],
        segment_col: list[str],
        rollup_value: list[Any],
        aggs: dict[str, Any],
    ) -> ibis.Table:
        """Execute all grouping sets and union results.

        This method handles ALL grouping set execution uniformly, including:
        - Base grouping (full segment_col)
        - Rollup groupings (subsets of segment_col)
        - Grand total (empty tuple)

        Each grouping set is executed independently and results are unioned together.

        Args:
            data (ibis.Table): The data table to aggregate
            grouping_sets (list[tuple[str, ...]]): List of grouping set tuples to execute.
                Each tuple contains column names to group by. Empty tuple () means grand total.
            segment_col (list[str]): All segment columns (used for mutation)
            rollup_value (list[Any]): Rollup values for each segment column
            aggs (dict[str, Any]): Aggregation specifications

        Returns:
            ibis.Table: Union of all grouping set results

        Example:
            >>> grouping_sets = [
            ...     ("region", "store", "product"),  # base
            ...     ("region", "store"),             # rollup
            ...     ("region",),                     # rollup
            ...     ()                               # grand total
            ... ]
            >>> _execute_grouping_sets(data, grouping_sets, segment_col, rollup_value, aggs)
        """
        results = []

        for gs in grouping_sets:
            if len(gs) == 0:
                # Grand total: aggregate all data, no GROUP BY
                result = data.aggregate(**aggs)
                # Mutate ALL segment columns to rollup_value
                mutations = SegTransactionStats._create_typed_literals(data, segment_col, rollup_value)
                result = result.mutate(**mutations)
            else:
                # Regular grouping: group by specified columns
                group_cols = list(gs)
                result = data.group_by(group_cols).aggregate(**aggs)

                # Mutate columns NOT in this grouping set to rollup_value
                mutation_cols = [col for col in segment_col if col not in gs]
                if len(mutation_cols) > 0:
                    mutation_values = [rollup_value[segment_col.index(col)] for col in mutation_cols]
                    mutations = SegTransactionStats._create_typed_literals(data, mutation_cols, mutation_values)
                    result = result.mutate(**mutations)

            results.append(result)

        # Union all results - first result is the base, union the rest
        return results[0].union(*results[1:]) if len(results) > 1 else results[0]

    @staticmethod
    def _create_unknown_flag(
        data: ibis.Table,
        unknown_customer_value: int | str | ibis.Scalar | ibis.expr.types.BooleanColumn,
    ) -> ibis.expr.types.BooleanColumn:
        """Create a boolean flag identifying unknown customers.

        Args:
            data (ibis.Table): The data table
            unknown_customer_value (int | str | ibis.Scalar | ibis.expr.types.BooleanColumn):
                The value or expression identifying unknown customers

        Returns:
            ibis.expr.types.BooleanColumn: Boolean expression identifying unknown customers
        """
        cols = ColumnHelper()

        if isinstance(unknown_customer_value, ibis.expr.types.BooleanColumn):
            return unknown_customer_value
        if isinstance(unknown_customer_value, ibis.Scalar):
            return data[cols.customer_id] == unknown_customer_value
        # Simple value (int/str)
        return data[cols.customer_id] == ibis.literal(unknown_customer_value)

    @staticmethod
    def _build_standard_aggs(
        data: ibis.Table,
        extra_aggs: dict[str, tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Build standard aggregations without unknown customer tracking.

        Args:
            data (ibis.Table): The data table
            extra_aggs (dict[str, tuple[str, str]] | None): Additional aggregations

        Returns:
            dict[str, Any]: Aggregation specifications
        """
        cols = ColumnHelper()
        agg_specs = [
            (cols.agg.unit_spend, cols.unit_spend, "sum"),
            (cols.agg.transaction_id, cols.transaction_id, "nunique"),
            (cols.agg.unit_qty, cols.unit_qty, "sum"),
            (cols.agg.customer_id, cols.customer_id, "nunique"),
        ]

        aggs = {agg_name: getattr(data[col], func)() for agg_name, col, func in agg_specs if col in data.columns}

        # Add extra aggregations if provided
        if extra_aggs:
            aggs.update({agg_name: getattr(data[col], func)() for agg_name, (col, func) in extra_aggs.items()})

        return aggs

    @staticmethod
    def _build_unknown_aggs(
        data: ibis.Table,
        unknown_flag: ibis.expr.types.BooleanColumn,
        extra_aggs: dict[str, tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Build aggregations with unknown customer tracking.

        Args:
            data (ibis.Table): The data table
            unknown_flag (ibis.expr.types.BooleanColumn): Boolean flag identifying unknown customers
            extra_aggs (dict[str, tuple[str, str]] | None): Additional aggregations

        Returns:
            dict[str, Any]: Aggregation specifications for identified, unknown, and total variants
        """
        cols = ColumnHelper()
        aggs = {}

        # Identified customers only (where NOT unknown)
        # Use coalesce to ensure proper types: int for counts, float for sums
        aggs[cols.agg.unit_spend] = data[cols.unit_spend].sum(where=~unknown_flag).coalesce(0.0)
        aggs[cols.agg.transaction_id] = data[cols.transaction_id].nunique(where=~unknown_flag).coalesce(0)
        aggs[cols.agg.customer_id] = data[cols.customer_id].nunique(where=~unknown_flag).coalesce(0)
        if cols.unit_qty in data.columns:
            aggs[cols.agg.unit_qty] = data[cols.unit_qty].sum(where=~unknown_flag).coalesce(0)

        # Unknown customers (where unknown)
        # Use coalesce to ensure proper types: int for counts, float for sums
        aggs[cols.agg.unit_spend_unknown] = data[cols.unit_spend].sum(where=unknown_flag).coalesce(0.0)
        aggs[cols.agg.transaction_id_unknown] = data[cols.transaction_id].nunique(where=unknown_flag).coalesce(0)
        if cols.unit_qty in data.columns:
            aggs[cols.agg.unit_qty_unknown] = data[cols.unit_qty].sum(where=unknown_flag).coalesce(0)

        # Total (all customers)
        aggs[cols.agg.unit_spend_total] = data[cols.unit_spend].sum()
        aggs[cols.agg.transaction_id_total] = data[cols.transaction_id].nunique()
        if cols.unit_qty in data.columns:
            aggs[cols.agg.unit_qty_total] = data[cols.unit_qty].sum()

        # Add extra aggregations with three variants
        if extra_aggs:
            suffix_unknown = get_option("column.suffix.unknown_customer")
            suffix_total = get_option("column.suffix.total")
            for agg_name, (col, func) in extra_aggs.items():
                # Use coalesce with 0 for count functions, 0.0 for others
                coalesce_value = 0 if func in ("nunique", "count") else 0.0
                aggs[agg_name] = getattr(data[col], func)(where=~unknown_flag).coalesce(coalesce_value)
                aggs[f"{agg_name}_{suffix_unknown}"] = getattr(data[col], func)(where=unknown_flag).coalesce(
                    coalesce_value,
                )
                aggs[f"{agg_name}_{suffix_total}"] = getattr(data[col], func)()

        return aggs

    @staticmethod
    def _calc_seg_stats(
        data: ibis.Table,
        segment_col: list[str],
        calc_total: bool | None,
        extra_aggs: dict[str, tuple[str, str]] | None = None,
        calc_rollup: bool | None = None,
        rollup_value: Any | list[Any] = "Total",  # noqa: ANN401 - Any is required for ibis.literal typing
        unknown_customer_value: int | str | ibis.Scalar | ibis.expr.types.BooleanColumn | None = None,
        grouping_sets: Literal["rollup", "cube", "total"] | list[tuple[str, ...]] | None = None,
    ) -> ibis.Table:
        """Calculates the transaction statistics by segment.

        Args:
            data (ibis.Table): The transaction data.
            segment_col (list[str]): The columns to use for the segmentation.
            calc_total (bool | None): Whether to include the total row (ignored if grouping_sets is not None).
            extra_aggs (dict[str, tuple[str, str]], optional): Additional aggregations to perform.
                The keys in the dictionary will be the column names for the aggregation results.
                The values are tuples with (column_name, aggregation_function).
            calc_rollup (bool | None, optional): Whether to calculate rollup totals (ignored if grouping_sets is not
                None). When True with multiple segment columns, subtotal rows are added for all non-empty prefixes and
                suffixes of the hierarchy. For example, with [A, B, C], prefixes include [A, B, Total], [A, Total,
                Total]; suffixes include [Total, B, C], [Total, Total, C]. Performance: O(n) additional aggregation
                passes for suffixes, where n is the number of segment columns.
            rollup_value (Any | list[Any], optional): The value to use for rollup totals. Can be a single value
                applied to all columns or a list of values matching the length of segment_col, with each value
                cast to match the corresponding column type. Defaults to "Total".
            unknown_customer_value
                (int | str | ibis.Scalar | ibis.expr.types.BooleanColumn | None, optional):
                Value or expression identifying unknown customers for separate tracking.
                When provided, metrics are split into identified, unknown, and total variants.
                Accepts simple values (e.g., -1), ibis literals, or boolean expressions.
                Defaults to None.
            grouping_sets (Literal["rollup", "cube", "total"] | list[tuple[str, ...]] | None, optional):
                Grouping sets mode ('rollup', 'cube', list of tuples, or None). Defaults to None.

        Returns:
            pd.DataFrame: The transaction statistics by segment.

        """
        cols = ColumnHelper()

        # segment_col arrives pre-validated from __init__; no re-validation needed here.
        # Normalize rollup_value to always be a list matching segment_col length
        rollup_value = [rollup_value] * len(segment_col) if not isinstance(rollup_value, list) else rollup_value

        # Validate rollup_value list length
        if len(rollup_value) != len(segment_col):
            msg = (
                f"If rollup_value is a list, its length must match the number of segment"
                f" columns. Expected {len(segment_col)}, got {len(rollup_value)}"
            )
            raise ValueError(msg)

        # Validate and create unknown flag if unknown_customer_value is provided
        unknown_flag = None
        if unknown_customer_value is not None:
            if cols.customer_id not in data.columns:
                msg = f"Column '{cols.customer_id}' is required when unknown_customer_value parameter is specified"
                raise ValueError(msg)
            unknown_flag = SegTransactionStats._create_unknown_flag(data, unknown_customer_value)

        # Build aggregations based on unknown customer tracking
        aggs = (
            SegTransactionStats._build_unknown_aggs(data, unknown_flag, extra_aggs)
            if unknown_flag is not None
            else SegTransactionStats._build_standard_aggs(data, extra_aggs)
        )

        # Generate ALL grouping sets based on current parameters
        grouping_sets_list = SegTransactionStats._generate_grouping_sets(
            segment_col,
            calc_total,
            calc_rollup,
            grouping_sets,
        )

        # Execute all grouping sets uniformly - no special cases
        final_metrics = SegTransactionStats._execute_grouping_sets(
            data,
            grouping_sets_list,
            segment_col,
            rollup_value,
            aggs,
        )

        # Calculate derived metrics
        # Note: .nullif(0) converts zero denominators to NULL, preventing division by zero errors.
        # Some database backends (e.g., SQL Server) raise errors on X/0 rather than returning NULL.
        final_metrics = final_metrics.mutate(
            **{
                cols.calc.spend_per_trans: ibis._[cols.agg.unit_spend] / ibis._[cols.agg.transaction_id].nullif(0),
            },
        )

        if cols.unit_qty in data.columns:
            final_metrics = final_metrics.mutate(
                **{
                    cols.calc.price_per_unit: ibis._[cols.agg.unit_spend] / ibis._[cols.agg.unit_qty].nullif(0),
                    cols.calc.units_per_trans: ibis._[cols.agg.unit_qty]
                    / ibis._[cols.agg.transaction_id].nullif(0).cast("float"),
                },
            )

        if cols.customer_id in data.columns:
            final_metrics = final_metrics.mutate(
                **{
                    cols.calc.spend_per_cust: ibis._[cols.agg.unit_spend] / ibis._[cols.agg.customer_id].nullif(0),
                    cols.calc.trans_per_cust: ibis._[cols.agg.transaction_id]
                    / ibis._[cols.agg.customer_id].nullif(0).cast("float"),
                },
            )

        # Add derived metrics for unknown and total when tracking unknown customers
        if unknown_flag is not None:
            # Unknown customer derived metrics
            final_metrics = final_metrics.mutate(
                **{
                    cols.calc.spend_per_trans_unknown: ibis._[cols.agg.unit_spend_unknown]
                    / ibis._[cols.agg.transaction_id_unknown].nullif(0),
                },
            )

            # Total derived metrics
            final_metrics = final_metrics.mutate(
                **{
                    cols.calc.spend_per_trans_total: ibis._[cols.agg.unit_spend_total]
                    / ibis._[cols.agg.transaction_id_total].nullif(0),
                },
            )

            # Quantity-based derived metrics for unknown and total
            if cols.unit_qty in data.columns:
                final_metrics = final_metrics.mutate(
                    **{
                        cols.calc.price_per_unit_unknown: ibis._[cols.agg.unit_spend_unknown]
                        / ibis._[cols.agg.unit_qty_unknown].nullif(0),
                        cols.calc.units_per_trans_unknown: ibis._[cols.agg.unit_qty_unknown]
                        / ibis._[cols.agg.transaction_id_unknown].nullif(0).cast("float"),
                        cols.calc.price_per_unit_total: ibis._[cols.agg.unit_spend_total]
                        / ibis._[cols.agg.unit_qty_total].nullif(0),
                        cols.calc.units_per_trans_total: ibis._[cols.agg.unit_qty_total]
                        / ibis._[cols.agg.transaction_id_total].nullif(0).cast("float"),
                    },
                )

        return final_metrics

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """Returns the dataframe with the transaction statistics by segment."""
        cols = ColumnHelper()
        include_quantity = cols.agg.unit_qty in self.table.columns
        include_customer = cols.agg.customer_id in self.table.columns
        include_unknown = self.unknown_customer_value is not None
        col_order = [
            *self.segment_col,
            *SegTransactionStats._get_col_order(
                include_quantity=include_quantity,
                include_customer=include_customer,
                include_unknown=include_unknown,
            ),
        ]

        # Add any extra aggregation columns to the column order
        if hasattr(self, "extra_aggs") and self.extra_aggs:
            if include_unknown:
                # Add identified, unknown, and total variants for each extra agg
                suffix_unknown = get_option("column.suffix.unknown_customer")
                suffix_total = get_option("column.suffix.total")
                for agg_name in self.extra_aggs:
                    col_order.append(agg_name)
                    col_order.append(f"{agg_name}_{suffix_unknown}")
                    col_order.append(f"{agg_name}_{suffix_total}")
            else:
                col_order.extend(self.extra_aggs.keys())

        return self.table.execute()[col_order]
