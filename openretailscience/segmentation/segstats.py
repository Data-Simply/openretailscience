"""Transaction KPIs (spend, transactions, customers, units, ratios) for any segment or dimension.

Supports grouping-set rollup/cube via `grouping_sets` and the `cube()`/`rollup()` helpers.
The sibling segmentation classes (`RFMSegmentation`, `HMLSegmentation`, `NLRSegmentation`,
`ThresholdSegmentation`) produce the segment labels this class aggregates.
"""

import functools
import warnings
from itertools import chain, combinations
from typing import Any, Literal

import ibis
import ibis.expr.operations as ops
import pandas as pd
from ibis.backends.sql import compilers as sql_compilers
from sqlglot import exp

from openretailscience.core.validation import (
    ensure_columns,
    ensure_data_has_columns,
    ensure_ibis_table,
    ensure_valid_extra_aggs,
)
from openretailscience.options import ColumnHelper, get_option

__all__ = ["SegTransactionStats", "cube", "rollup"]

# Maximum number of dimensions for CUBE mode before warning about exponential growth.
# CUBE generates 2^n grouping sets, so 6 dimensions = 64 sets (reasonable), but 7+ dimensions
# = 128+ sets (potentially expensive). This threshold balances flexibility with performance awareness.
MAX_CUBE_DIMENSIONS_WITHOUT_WARNING = 6

# Backends whose dialect supports GROUP BY GROUPING SETS + GROUPING(); these use the single-scan native path.
_NATIVE_GROUPING_SETS_BACKENDS = frozenset(
    {"duckdb", "mssql", "oracle", "snowflake", "bigquery", "pyspark", "databricks"},
)

# Prefix for the transient GROUPING() flag columns; letter-led so every dialect (incl. Oracle) accepts it.
_GROUPING_FLAG_PREFIX = "grouping_flag_"


def _resolve_group_key(key: exp.Expression, selects: list[exp.Expression]) -> exp.Expression:
    """Resolve a GROUP BY key to a concrete column expression.

    Some ibis backends emit positional group keys (``GROUP BY 1, 2`` on DuckDB/Snowflake/BigQuery/
    Databricks); others emit the named columns (SQL Server/Oracle, which reject positional group-by).
    ``GROUPING()`` and ``GROUPING SETS`` need real column expressions, so a positional ordinal is
    mapped back to its SELECT-list column and a named key is returned unchanged.

    Args:
        key (exp.Expression): A single GROUP BY key expression (a positional ordinal or a column).
        selects (list[exp.Expression]): The query's SELECT-list expressions.

    Returns:
        exp.Expression: The resolved column expression.
    """
    if isinstance(key, exp.Literal) and key.is_int:
        col = selects[int(key.name) - 1]
        return (col.this if isinstance(col, exp.Alias) else col).copy()
    return key.copy()


def _rewrite_to_grouping_sets(
    tree: exp.Select,
    grouping_sets: list[tuple[str, ...]],
    segment_col: list[str],
    flag_names: list[str],
) -> None:
    """Rewrite an ibis-compiled aggregation AST in place into ``GROUP BY GROUPING SETS`` with ``GROUPING()`` flags.

    Operates on the sqlglot AST ibis produced (so aggregate expressions, quoting, and dialect quirks
    are already correct); swaps the grouping clause and appends one ``GROUPING(col)`` flag column per
    segment column, used downstream to apply rollup labels.

    Args:
        tree (exp.Select): The base aggregation SELECT, grouped by all of ``segment_col``. Mutated in place.
        grouping_sets (list[tuple[str, ...]]): The grouping sets to emit (tuples of column names).
        segment_col (list[str]): All segment columns, in the order they appear in the GROUP BY.
        flag_names (list[str]): Output names for the per-segment GROUPING() flag columns.
    """
    resolved = [_resolve_group_key(k, tree.selects) for k in tree.args["group"].expressions]
    key_by_name = dict(zip(segment_col, resolved, strict=True))

    for col, flag in zip(segment_col, flag_names, strict=True):
        tree.select(exp.func("GROUPING", key_by_name[col].copy()).as_(flag, quoted=True), copy=False)

    tuples = [exp.Tuple(expressions=[key_by_name[col].copy() for col in gs]) for gs in grouping_sets]
    tree.set("group", exp.Group(expressions=[exp.GroupingSets(expressions=tuples)]))


def cube(*columns: str) -> list[tuple[str, ...]]:
    """Generate CUBE grouping sets (all 2^n combinations, from full detail down to grand total).

    Matches SQL's GROUP BY CUBE(A, B), C syntax. The result can be passed directly to
    `grouping_sets`, or wrapped with fixed columns in a nested spec
    (`[(cube("store", "region"), "date")]`).

    Args:
        *columns (str): Column names to include in the CUBE operation.

    Returns:
        list[tuple[str, ...]]: All 2^n combinations, largest first, ending with the () grand total.

    Raises:
        ValueError: If no columns are provided.
        TypeError: If any column is not a string.
        UserWarning: If more than MAX_CUBE_DIMENSIONS_WITHOUT_WARNING columns.

    Example:
        >>> cube("store", "region")
        [("store", "region"), ("store",), ("region",), ()]
        >>>
        >>> # CUBE with fixed columns - wrap in tuple
        >>> SegTransactionStats(
        ...     data=df,
        ...     segment_col=["store", "region", "date"],
        ...     grouping_sets=[(cube("store", "region"), "date")]
        ... )
        # Produces 4 grouping sets (2^2 from CUBE):
        # [("store", "region", "date"), ("store", "date"), ("region", "date"), ("date",)]
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
    """Generate ROLLUP grouping sets (n+1 hierarchical aggregation levels).

    Levels are built right to left by dropping the rightmost column at each step, matching
    SQL's GROUP BY ROLLUP(A, B), C syntax. The result can be passed directly to `grouping_sets`,
    or wrapped with fixed columns in a nested spec (`[(rollup("a", "b"), "c")]`).

    Args:
        *columns (str): Column names in hierarchical order (left = highest level).

    Returns:
        list[tuple[str, ...]]: n+1 tuples from the full column list down to the () grand total.

    Raises:
        ValueError: If no columns are provided.
        TypeError: If any column is not a string.

    Example:
        >>> rollup("year", "quarter", "month")
        [("year", "quarter", "month"), ("year", "quarter"), ("year",), ()]
        >>>
        >>> # ROLLUP with fixed column - wrap in tuple
        >>> SegTransactionStats(
        ...     data=df,
        ...     segment_col=["year", "quarter", "month", "store"],
        ...     grouping_sets=[(rollup("year", "quarter", "month"), "store")]
        ... )
        # Produces 4 grouping sets (3+1 from ROLLUP):
        # [("year", "quarter", "month", "store"), ("year", "quarter", "store"),
        #  ("year", "store"), ("store",)]
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

    Standard metrics: spend, transactions, customers (only when customer_id is present),
    units (only when unit_quantity is present), and derived ratios. Results are a lazy
    ibis Table (`.table`), materialized via `.df`.
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
            data (pd.DataFrame | ibis.Table): The transaction data. Must contain unit_spend and
                transaction_id; customer_id and unit_quantity are optional and add the
                customer-based and quantity-based columns when present.
            segment_col (str | list[str], optional): The column or list of columns to use for
                the segmentation. Defaults to "segment_name".
            calc_total (bool | None, optional): Whether to include the grand total row.
                Defaults to True when grouping_sets is None. Cannot be used with grouping_sets.
                Deprecated: a FutureWarning is emitted when grouping_sets is None and both
                calc_total and calc_rollup are omitted. Use grouping_sets='total' instead.
            extra_aggs (dict[str, tuple[str, str]], optional): Additional aggregations. Keys are
                output column names; values are (column_name, aggregation_function) tuples, e.g.
                {"stores": ("store_id", "nunique")} counts unique store_ids.
            calc_rollup (bool | None, optional): Whether to add subtotal rows. Defaults to False
                when grouping_sets is None. Unlike grouping_sets='rollup' (prefix rollups only),
                True generates prefix and suffix rollups; suffix rollups only when calc_total=True.
                Adds O(n) extra aggregation passes for n segment columns.
                Cannot be used with grouping_sets.
                Deprecated: a FutureWarning is emitted when grouping_sets is None and both
                calc_total and calc_rollup are omitted. Use grouping_sets='rollup' instead.
            rollup_value (Any | list[Any], optional): The value used for rollup rows. A single
                value applies to all columns, or a list with one value per segment_col column,
                each cast to the corresponding column's type. Defaults to "Total".
            unknown_customer_value (int | str | ibis.Scalar | ibis.expr.types.BooleanColumn | None, optional):
                Value, ibis literal, or boolean expression identifying unknown customers
                (requires the customer_id column). When set, metrics are split into identified,
                `_unknown`, and `_total` variants, with suffixes from the options
                column.suffix.unknown_customer / column.suffix.total (defaults "unknown"/"total").
            grouping_sets (Literal["rollup", "cube", "total"] | list[tuple[str, ...]] | None, optional):
                Grouping sets mode, mutually exclusive with calc_total/calc_rollup:
                - "rollup": SQL ROLLUP — [A,B,C], [A,B], [A], ().
                - "cube": SQL CUBE — all 2^n combinations.
                - "total": the full grouping plus the () grand total.
                - list of tuples: custom grouping sets; each tuple is either explicit columns or
                  a cube()/rollup() result plus fixed columns (see cube()). Sets are deduplicated
                  (order not preserved); every segment_col column must appear in at least one set
                  and every set column must be in segment_col.
                - None: use calc_total/calc_rollup behavior.
            sort_by (bool, optional): When True, sort the output rows ascending by segment_col
                using ibis .order_by() so the sort runs at the backend. Defaults to False.

        Raises:
            ValueError: If grouping_sets is used with explicit calc_total or calc_rollup; if
                grouping_sets is not a valid value or is an empty list; if the grouping sets and
                segment_col do not cover each other; if rollup_value is a list whose length does
                not match segment_col; or if unknown_customer_value is set without a customer_id
                column.
            TypeError: If a grouping_sets list element is not a tuple.
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

        ensure_valid_extra_aggs(data, extra_aggs)

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
        """Create a dictionary of ibis literals, each cast to the corresponding column's type.

        Args:
            data (ibis.Table): The data table containing column type information.
            columns (list[str]): List of column names.
            values (list[Any]): List of values to convert to typed literals.

        Returns:
            dict[str, ibis.Scalar]: Column names mapped to typed literals, used as rollup labels.
        """
        mutations = {}
        for i, col in enumerate(columns):
            col_type = data[col].type()
            mutations[col] = ibis.literal(values[i], type=col_type)
        return mutations

    @staticmethod
    def _validate_grouping_sets_params(
        grouping_sets: Literal["rollup", "cube", "total"] | list[tuple[str, ...]] | None,
        calc_total: bool | None,
        calc_rollup: bool | None,
    ) -> None:
        """Validate grouping_sets parameter (type checking only).

        Column validation happens in _generate_grouping_sets() since it requires segment_col.
        Emits a FutureWarning when grouping_sets is None and both calc_total and calc_rollup
        are omitted (the implicit calc_total=True default is deprecated).

        Args:
            grouping_sets (Literal["rollup", "cube", "total"] | list[tuple[str, ...]] | None): The
                grouping_sets parameter value.
            calc_total (bool | None): Whether to include grand total.
            calc_rollup (bool | None): Whether to generate rollup subtotals.

        Raises:
            ValueError: If grouping_sets is used with explicit calc_total or calc_rollup.
            ValueError: If grouping_sets is not a valid value or is an empty list.
            TypeError: If grouping_sets is a list containing a non-tuple element.
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
        """Flatten a single grouping_sets item into explicit grouping sets.

        Structural detection: a tuple of strings is an explicit grouping set (returned as-is);
        a tuple containing a list is a specification — exactly one cube()/rollup() result plus
        str fixed columns, which are appended as a suffix to every set.

        Args:
            item (tuple): A tuple that is either an explicit grouping set or a specification.

        Returns:
            list[tuple[str, ...]]: List of one or more grouping sets.

        Raises:
            ValueError: If the specification contains multiple cube()/rollup() results or an
                empty cube()/rollup() result.
            TypeError: If the specification contains a non-str, non-list element.

        Example:
            >>> _flatten_item(("region", "store"))
            [("region", "store")]
            >>>
            >>> # Specification (cube() result + fixed column)
            >>> cube_result = [("region", "store"), ("region",), ("store",), ()]
            >>> _flatten_item((cube_result, "date"))
            [("region", "store", "date"), ("region", "date"), ("store", "date"), ("date",)]
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

        "rollup"/"cube" delegate to rollup()/cube(); "total" yields the full grouping plus the
        () grand total. List mode flattens each item, deduplicates via set() (order not
        preserved), and validates that the sets and segment_col cover each other. Any other
        value delegates to _generate_legacy_grouping_sets().

        Args:
            segment_col (list[str]): The segment columns to generate grouping sets for.
            calc_total (bool | None): Whether to include grand total (ignored if grouping_sets is not None).
            calc_rollup (bool | None): Whether to generate rollup subtotals (ignored if grouping_sets is not None).
            grouping_sets (Literal["rollup", "cube", "total"] | list[tuple[str, ...]] | None): Grouping sets mode.

        Returns:
            list[tuple[str, ...]]: List of grouping set tuples; the empty tuple () represents
                the grand total.

        Raises:
            ValueError: If a grouping set references a column not in segment_col, or a
                segment_col column is missing from every grouping set.

        Example:
            >>> # Legacy mode (calc_total/calc_rollup): base, prefix and suffix rollups, grand total
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

        The base grouping (all columns) is always included; calc_rollup adds prefix rollups and,
        only when calc_total is also true, suffix rollups; calc_total appends the () grand total.

        Args:
            segment_col (list[str]): The segment columns.
            calc_total (bool | None): Whether to include grand total.
            calc_rollup (bool | None): Whether to generate rollup subtotals.

        Returns:
            list[tuple[str, ...]]: List of grouping set tuples.
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
        """Execute all grouping sets and union the results.

        Handles all grouping sets uniformly: the base grouping, rollup subsets, and the ()
        grand total (no GROUP BY). Each set is aggregated independently and the results are
        unioned; segment columns not in a set are mutated to the corresponding rollup_value
        (typed literals).

        Args:
            data (ibis.Table): The data table to aggregate.
            grouping_sets (list[tuple[str, ...]]): List of grouping set tuples to execute;
                the empty tuple () means grand total.
            segment_col (list[str]): All segment columns (used for mutation).
            rollup_value (list[Any]): Rollup values for each segment column.
            aggs (dict[str, Any]): Aggregation specifications.

        Returns:
            ibis.Table: Union of all grouping set results.
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
    def _execute_grouping_sets_native(
        data: ibis.Table,
        grouping_sets: list[tuple[str, ...]],
        segment_col: list[str],
        rollup_value: list[Any],
        aggs: dict[str, Any],
    ) -> ibis.Table:
        """Execute all grouping sets in a single backend-native ``GROUP BY GROUPING SETS`` pass.

        Equivalent in output to :meth:`_execute_grouping_sets`, but scans the source once instead of
        once per grouping set. Only valid for backends in :data:`_NATIVE_GROUPING_SETS_BACKENDS` whose
        data lives in a real table (not an in-memory table); the caller is responsible for that gate.

        Args:
            data (ibis.Table): The data table to aggregate.
            grouping_sets (list[tuple[str, ...]]): Grouping set tuples; ``()`` means grand total.
            segment_col (list[str]): All segment columns.
            rollup_value (list[Any]): Rollup label per segment column.
            aggs (dict[str, Any]): Aggregation specifications.

        Returns:
            ibis.Table: Aggregated table with rollup labels applied. Same rows and columns as the
                union path; column order is normalized later by the ``.df`` property.
        """
        backend = data.get_backend()
        compiler = getattr(sql_compilers, backend.name).compiler

        base = data.group_by(segment_col).aggregate(**aggs)

        # Lengthen the flag prefix until it can't collide with a real output column (else schema merge breaks).
        prefix = _GROUPING_FLAG_PREFIX
        while any(f"{prefix}{i}" in base.columns for i in range(len(segment_col))):
            prefix = f"_{prefix}"
        flag_names = [f"{prefix}{i}" for i in range(len(segment_col))]

        # Rewrite ibis's own compiled AST directly (no SQL-string round trip; reuses ibis's per-backend dialect).
        tree = compiler.to_sqlglot(base.unbind())
        # to_sqlglot returns a list of statements on some backends (e.g. BigQuery scalar UDFs); the SELECT is last.
        tree = tree[-1] if isinstance(tree, list) else tree
        _rewrite_to_grouping_sets(tree, grouping_sets, segment_col, flag_names)
        native_sql = tree.sql(dialect=compiler.dialect)

        # Pass the known schema so con.sql skips inference (it mis-types float aggregates on e.g. Oracle).
        out_schema = ibis.schema({**base.schema(), **dict.fromkeys(flag_names, "int32")})
        gs_table = backend.sql(native_sql, schema=out_schema)

        # Rolled-up columns come back NULL; GROUPING(col) == 1 marks them, so replace with rollup_value.
        typed_values = SegTransactionStats._create_typed_literals(data, segment_col, rollup_value)
        labelled = gs_table.mutate(
            **{
                col: ibis.ifelse(gs_table[flag] == 1, typed_values[col], gs_table[col])
                for col, flag in zip(segment_col, flag_names, strict=True)
            },
        )
        return labelled.drop(*flag_names)

    @staticmethod
    def _create_unknown_flag(
        data: ibis.Table,
        unknown_customer_value: int | str | ibis.Scalar | ibis.expr.types.BooleanColumn,
    ) -> ibis.expr.types.BooleanColumn:
        """Create a boolean flag identifying unknown customers.

        A BooleanColumn is used as-is; an ibis.Scalar or plain value compares customer_id
        against it.

        Args:
            data (ibis.Table): The data table.
            unknown_customer_value (int | str | ibis.Scalar | ibis.expr.types.BooleanColumn):
                The value or expression identifying unknown customers.

        Returns:
            ibis.expr.types.BooleanColumn: Boolean expression identifying unknown customers.
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

        Only aggregates columns present in the data (hence the customer and quantity columns
        are conditional); extra_aggs are merged in.

        Args:
            data (ibis.Table): The data table.
            extra_aggs (dict[str, tuple[str, str]] | None): Additional aggregations.

        Returns:
            dict[str, Any]: Aggregation specifications.
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

        Every metric (including suffixed extra_aggs) gets identified, `_unknown`, and `_total`
        variants. coalesce(0/0.0) forces int vs float types when a segment has no rows.

        Args:
            data (ibis.Table): The data table.
            unknown_flag (ibis.expr.types.BooleanColumn): Boolean flag identifying unknown customers.
            extra_aggs (dict[str, tuple[str, str]] | None): Additional aggregations.

        Returns:
            dict[str, Any]: Aggregation specifications for identified, unknown, and total variants.
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
        """Calculate the transaction statistics by segment as a lazy ibis expression.

        Aggregates each grouping set (see _generate_grouping_sets) and unions the results, then
        derives the ratio metrics with nullif(0) so zero denominators yield NULL instead of a
        division error. Uses the native single-scan GROUPING SETS path when the
        optimization.use_native_sql option is set, there is more than one grouping set, the
        backend supports GROUPING SETS, and the data is not an in-memory table.

        Args:
            data (ibis.Table): The transaction data.
            segment_col (list[str]): The columns to use for the segmentation.
            calc_total (bool | None): Whether to include the total row (ignored if grouping_sets is not None).
            extra_aggs (dict[str, tuple[str, str]] | None): Additional aggregations (see __init__).
            calc_rollup (bool | None): Whether to calculate rollup totals (ignored if grouping_sets is not None).
            rollup_value (Any | list[Any]): Rollup label(s); a list must match the segment_col length.
            unknown_customer_value (int | str | ibis.Scalar | ibis.expr.types.BooleanColumn | None):
                Value or expression identifying unknown customers (see __init__).
            grouping_sets (Literal["rollup", "cube", "total"] | list[tuple[str, ...]] | None): Grouping sets mode.

        Returns:
            ibis.Table: The transaction statistics by segment.

        Raises:
            ValueError: If rollup_value is a list whose length does not match segment_col, or if
                unknown_customer_value is set but the customer_id column is missing.
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

        # Native GROUPING SETS only pays off by fusing 2+ sets into one scan; a single set has no benefit.
        use_native = (
            get_option("optimization.use_native_sql")
            and len(grouping_sets_list) > 1
            and data.get_backend().name in _NATIVE_GROUPING_SETS_BACKENDS
            and len(data.op().find(ops.InMemoryTable)) == 0
        )
        execute = (
            SegTransactionStats._execute_grouping_sets_native
            if use_native
            else SegTransactionStats._execute_grouping_sets
        )
        final_metrics = execute(
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
        """Returns the materialized dataframe of transaction statistics by segment.

        Executes the lazy `.table`. Column order: segment_col, then the standard metrics
        (option names), then the extra_aggs columns.
        """
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
        if self.extra_aggs:
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
