"""Composite ranking across multiple metrics.

Ranks each metric column (ascending or descending per column), then aggregates the
per-column ranks — not the raw values — into a `composite_rank` column (lower is
better overall).
"""

import functools

import ibis
import ibis.expr.types as ir
import pandas as pd

from openretailscience.core.validation import VALID_SORT_ORDERS, ensure_columns, ensure_ibis_table, ensure_value_choice

VALID_AGG_FUNCS = ("mean", "sum", "min", "max")


class CompositeRank:
    """Creates composite rankings by aggregating per-metric ranks.

    Ranks are 1-based with rank 1 = best in the given sort direction, so a lower
    `composite_rank` is better overall. `composite_rank` aggregates the per-column
    ranks, not the raw values: mean = average rank, sum = total rank, min = `ibis.least`
    (best single-metric rank dominates), max = `ibis.greatest` (worst single-metric rank
    dominates). `.table` is a lazy ibis Table; `.df` materializes on first access and is
    cached.
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        rank_cols: list[tuple[str, str] | str],
        agg_func: str,
        ignore_ties: bool = False,
        group_col: str | list[str] | None = None,
    ) -> None:
        """Initialize a CompositeRank over `df`.

        Args:
            df (pd.DataFrame | ibis.Table): Data to rank.
            rank_cols (list[tuple[str, str] | str]): Metrics to rank with their sort
                direction. Each entry is a column name (defaults to "asc") or a
                (column, sort_order) tuple; sort_order is "asc", "ascending", "desc", or
                "descending" (case-insensitive).
            agg_func (str): How to combine the per-column ranks (not raw values):
                "mean" = average rank, "sum" = total rank, "min" = `ibis.least`,
                "max" = `ibis.greatest`.
            ignore_ties (bool, optional): False (default) uses `ibis.rank` — tied values
                share a rank. True uses `ibis.row_number` — every row gets a unique rank.
            group_col (str | list[str], optional): Column(s) to partition the ranking by.
                None (default) ranks globally; otherwise ranks are computed independently
                within each group.

        Raises:
            TypeError: If `df` is not a pandas DataFrame or an Ibis Table.
            ValueError: If `rank_cols` is empty.
            ValueError: If a specified metric is not in the data or the sort order is invalid.
            ValueError: If the aggregation function is not supported.
            ValueError: If `group_col` is specified but doesn't exist in the data.

        Examples:
            >>> # Group-based ranking: each product competes with its own category
            >>> ranker = CompositeRank(
            ...     df=product_data,
            ...     rank_cols=[("weekly_sales", "desc"), ("margin_percentage", "desc"), ("stock_cover_days", "asc")],
            ...     agg_func="mean",
            ...     group_col="product_category",
            ... )
        """
        df = ensure_ibis_table(df)

        if group_col is not None:
            group_col = ensure_columns(df, group_col, "group_col")
        # Validate agg_func up-front so an invalid value fails before any per-column work runs.
        agg_func = ensure_value_choice(agg_func, VALID_AGG_FUNCS, "agg_func")
        rank_mutates = self._process_rank_columns(rank_cols, df, group_col, ignore_ties)
        df = df.mutate(**rank_mutates)
        self.table = self._create_composite_ranking(df, rank_mutates, agg_func)

    def _process_rank_columns(
        self,
        rank_cols: list[tuple[str, str] | str],
        df: ibis.Table,
        group_col: list[str] | None,
        ignore_ties: bool,
    ) -> dict[str, ir.IntegerColumn]:
        """Build an ibis ranking expression for each rank column spec.

        Args:
            rank_cols (list[tuple[str, str] | str]): Column specifications to rank. Each element is
                either a string (column name, defaults to ascending) or a tuple of (column_name, sort_order).
            df (ibis.Table): The table containing the columns to rank.
            group_col (list[str] | None): Columns to partition the ranking window by, or None
                for global ranking. Pre-normalized to a list by the caller.
            ignore_ties (bool): If True, uses row_number for unique ranks. If False, uses rank
                which assigns the same rank to tied values.

        Returns:
            dict[str, ir.IntegerColumn]: Mapping of output column names (`{col_name}_rank`)
                to ibis ranking expressions.

        Raises:
            ValueError: If `rank_cols` is empty.
            ValueError: If a specified column is not found in the DataFrame.
            ValueError: If a sort order is not one of "asc", "ascending", "desc", or "descending".
        """
        if len(rank_cols) == 0:
            msg = "rank_cols must contain at least one column specification"
            raise ValueError(msg)

        rank_mutates = {}

        for col_spec in rank_cols:
            col_name, sort_order = self._parse_column_spec(col_spec)

            ensure_columns(df, col_name, "rank_cols")
            sort_order = ensure_value_choice(sort_order, VALID_SORT_ORDERS, "sort_order")

            order_by = ibis.asc(df[col_name]) if sort_order in ["asc", "ascending"] else ibis.desc(df[col_name])
            window = (
                ibis.window(order_by=order_by)
                if group_col is None
                else ibis.window(group_by=[df[col] for col in group_col], order_by=order_by)
            )

            # Calculate rank based on ignore_ties parameter (using 1-based ranks)
            rank_col = ibis.row_number().over(window) + 1 if ignore_ties else ibis.rank().over(window) + 1
            rank_mutates[f"{col_name}_rank"] = rank_col

        return rank_mutates

    def _parse_column_spec(self, col_spec: tuple[str, str] | str) -> tuple[str, str]:
        """Parse a column specification into a column name and sort order.

        Args:
            col_spec (tuple[str, str] | str): Either a column name string (defaults to "asc")
                or a tuple of (column_name, sort_order).

        Returns:
            tuple[str, str]: A tuple of (column_name, sort_order).

        Raises:
            ValueError: If a tuple is provided but does not contain exactly two elements.
        """
        if isinstance(col_spec, str):
            return col_spec, "asc"
        if len(col_spec) != 2:  # noqa: PLR2004 - Error message below explains the value
            msg = f"Column specification must be a string or a tuple of (column_name, sort_order). Got {col_spec}"
            raise ValueError(msg)
        return col_spec

    def _create_composite_ranking(
        self,
        df: ibis.Table,
        rank_mutates: dict[str, ir.IntegerColumn],
        agg_func: str,
    ) -> ibis.Table:
        """Aggregate per-metric rank columns into a `composite_rank` column.

        Args:
            df (ibis.Table): The table with individual rank columns already added.
            rank_mutates (dict[str, ir.IntegerColumn]): Mapping of rank column names to their expressions,
                used to identify which columns to aggregate.
            agg_func (str): Aggregation function to combine ranks. Must be one of
                "mean", "sum", "min", or "max".

        Returns:
            ibis.Table: The input table with an additional composite_rank column.
        """
        column_refs = [df[col] for col in rank_mutates]
        agg_expr = {
            "mean": sum(column_refs) / len(column_refs),
            "sum": sum(column_refs),
            "min": ibis.least(*column_refs),
            "max": ibis.greatest(*column_refs),
        }
        return df.mutate(composite_rank=agg_expr[agg_func])

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """Materialized ranked data (computed lazily on first access, then cached).

        Returns:
            pd.DataFrame: The input metrics plus one `{col}_rank` column per ranked
                metric and the final `composite_rank` column.
        """
        return self.table.execute()
