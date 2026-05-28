"""Composite Rank Analysis Module for Multi-Factor Retail Decision Making.

## Business Context

In retail, critical decisions like product ranging, supplier selection, and store
performance evaluation require balancing multiple competing factors. A product might
have high sales but low margin, or a supplier might offer great prices but poor
delivery reliability. Composite ranking enables data-driven decisions by combining
multiple performance metrics into a single, actionable score.

## Real-World Applications

1. **Product Range Optimization**: Rank products for listing/delisting decisions based on:
   - Sales velocity (units per week)
   - Gross margin percentage
   - Stock turn rate
   - Customer satisfaction scores
   - Return rates

2. **Supplier Performance Management**: Evaluate suppliers using:
   - On-time delivery percentage
   - Price competitiveness
   - Quality scores (defect rates)
   - Payment terms flexibility
   - Order fill rates

3. **Store Performance Assessment**: Rank stores for investment decisions based on:
   - Sales per square foot
   - Conversion rates
   - Labor productivity
   - Customer satisfaction (NPS)
   - Shrinkage rates

4. **Category Management**: Prioritize categories for space allocation using:
   - Category growth rates
   - Market share
   - Profitability
   - Cross-category purchase influence
   - Seasonal consistency

## How It Works

The module creates individual rankings for each metric, then combines these rankings
using aggregation functions (mean, sum, min, max) to produce a final composite score.
This approach normalizes metrics with different scales and ensures each factor contributes
appropriately to the final decision.

## Business Value

- **Objective Decision Making**: Removes bias by systematically weighing all factors
- **Scalability**: Can evaluate thousands of products/stores/suppliers simultaneously
- **Transparency**: Clear methodology that stakeholders can understand and trust
- **Flexibility**: Different aggregation methods suit different business strategies
- **Actionable Output**: Direct ranking enables clear cut-off decisions

Key Features:
- Creates individual ranks for multiple columns with business metrics
- Supports both ascending and descending sort orders for each metric
- Combines individual ranks using business-appropriate aggregation functions
- Handles tie values for fair comparison
- Utilizes Ibis for efficient query execution on large retail datasets
"""

from typing import Any

import ibis
import ibis.expr.types as ir
import pandas as pd

from openretailscience.core.validation import ensure_columns, ensure_value_choice


class CompositeRank:
    """Creates multi-factor composite rankings for retail decision-making.

    The CompositeRank class enables retailers to make data-driven decisions by combining
    multiple performance metrics into a single, actionable ranking. This is essential for
    scenarios where no single metric tells the complete story.

    ## Business Problem Solved

    Retailers face complex trade-offs daily: Should we keep the high-volume product with
    low margins or the high-margin product with slow sales? Which supplier offers the best
    overall value when considering price, quality, and reliability? This class provides a
    systematic approach to these multi-dimensional decisions.

    ## Example Use Case: Product Range Review

    When conducting quarterly range reviews, a retailer might rank products by:
    - Sales performance (higher is better → descending order)
    - Days of inventory (lower is better → ascending order)
    - Customer rating (higher is better → descending order)
    - Return rate (lower is better → ascending order)

    The composite rank identifies products that perform well across ALL metrics, not just
    excel in one area. Products with the best composite scores are clear "keep" decisions,
    while those with the worst scores are candidates for delisting.

    ## Aggregation Strategies

    Different business contexts require different aggregation approaches:
    - **Mean**: Balanced scorecard approach, all factors equally important
    - **Min**: Conservative approach, focus on worst-performing metric
    - **Max**: Optimistic approach, highlight strength in any area
    - **Sum**: Cumulative performance across all dimensions

    ## Actionable Outcomes

    The composite rank directly supports decisions like:
    - Top 20% composite rank → Increase inventory investment
    - Bottom 20% composite rank → Consider delisting or markdown
    - Middle 60% → Maintain current strategy, monitor for changes
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        rank_cols: list[tuple[str, str] | str],
        agg_func: str,
        ignore_ties: bool = False,
        group_col: str | list[str] | None = None,
    ) -> None:
        """Initialize the CompositeRank class for multi-criteria retail analysis.

        Args:
            df (pd.DataFrame | ibis.Table): Product, store, or supplier performance data.
            rank_cols (List[Union[Tuple[str, str], str]]): Metrics to rank with their optimization direction.
                Examples for product ranging:
                - ("sales_units", "desc") - Higher sales are better
                - ("days_inventory", "asc") - Lower inventory days are better
                - ("margin_pct", "desc") - Higher margins are better
                - ("return_rate", "asc") - Lower returns are better
                If just a string is provided, ascending order is assumed.
            agg_func (str): How to combine individual rankings:
                - "mean": Balanced scorecard (most common for range reviews)
                - "sum": Total performance score (for bonus calculations)
                - "min": Worst-case performance (for risk assessment)
                - "max": Best-case performance (for opportunity identification)
            ignore_ties (bool, optional): How to handle identical values:
                - False (default): Products with same sales get same rank (fair comparison)
                - True: Force unique ranks even for ties (strict ordering needed)
            group_col (str | list[str], optional): Column(s) to partition rankings by group.
                - None (default): Rank across entire dataset (current behavior)
                - If specified: Calculate ranks independently within each group
                Examples for group-based ranking:
                - "product_category": Rank products within each category
                - "store_region": Rank stores within their regions
                - "supplier_type": Rank suppliers within their specialization

        Raises:
            ValueError: If specified metrics are not in the data or sort order is invalid.
            ValueError: If aggregation function is not supported.
            ValueError: If group_col is specified but doesn't exist in the data.

        Examples:
            >>> # Global ranking: Rank all products together (current behavior)
            >>> ranker = CompositeRank(
            ...     df=product_data,
            ...     rank_cols=[
            ...         ("weekly_sales", "desc"),
            ...         ("margin_percentage", "desc"),
            ...         ("stock_cover_days", "asc"),
            ...         ("customer_rating", "desc")
            ...     ],
            ...     agg_func="mean"
            ... )
            >>> # Products with lowest composite_rank should be reviewed for delisting

            >>> # Group-based ranking: Rank products within each category
            >>> ranker = CompositeRank(
            ...     df=product_data,
            ...     rank_cols=[
            ...         ("weekly_sales", "desc"),
            ...         ("margin_percentage", "desc"),
            ...         ("stock_cover_days", "asc")
            ...     ],
            ...     agg_func="mean",
            ...     group_col="product_category"
            ... )
            >>> # Electronics products ranked against other electronics
            >>> # Apparel products ranked against other apparel
        """
        self._df: pd.DataFrame | None = None
        if isinstance(df, pd.DataFrame):
            df = ibis.memtable(df)

        self._validate_group_col(group_col, df)
        rank_mutates = self._process_rank_columns(rank_cols, df, group_col, ignore_ties)
        df = df.mutate(**rank_mutates)
        self.table = self._create_composite_ranking(df, rank_mutates, agg_func)

    def _validate_group_col(self, group_col: str | list[str] | None, df: ibis.Table) -> None:
        """Validate group_col parameter against the DataFrame columns.

        Args:
            group_col (str | list[str] | None): Column name or list of column names to partition rankings by.
                None skips validation entirely.
            df (ibis.Table): The table whose columns are validated against.

        Raises:
            TypeError: If group_col is not a string, list, or None.
            ValueError: If group_col is an empty list.
            ValueError: If any specified group columns are not found in the DataFrame.
        """
        if group_col is not None:
            ensure_columns(df, group_col)

    def _process_rank_columns(
        self,
        rank_cols: list[tuple[str, str] | str],
        df: ibis.Table,
        group_col: str | list[str] | None,
        ignore_ties: bool,
    ) -> dict[str, ir.IntegerColumn]:
        """Process rank columns and create ranking expressions.

        Validates each column specification, then builds an ibis ranking expression
        for each metric using the appropriate window function and tie-handling strategy.

        Args:
            rank_cols (list[tuple[str, str] | str]): Column specifications to rank. Each element is
                either a string (column name, defaults to ascending) or a tuple of (column_name, sort_order).
            df (ibis.Table): The table containing the columns to rank.
            group_col (str | list[str] | None): Column(s) to partition the ranking window by,
                or None for global ranking.
            ignore_ties (bool): If True, uses row_number for unique ranks. If False, uses rank
                which assigns the same rank to tied values.

        Returns:
            dict[str, ir.IntegerColumn]: Mapping of rank column names (e.g., "sales_rank") to ibis ranking expressions.

        Raises:
            ValueError: If a specified column is not found in the DataFrame.
            ValueError: If a sort order is not one of "asc", "ascending", "desc", or "descending".
        """
        if len(rank_cols) == 0:
            msg = "rank_cols must contain at least one column specification"
            raise ValueError(msg)

        valid_sort_orders = ["asc", "ascending", "desc", "descending"]
        rank_mutates = {}

        for col_spec in rank_cols:
            col_name, sort_order = self._parse_column_spec(col_spec)

            ensure_columns(df, col_name)
            sort_order = ensure_value_choice(sort_order, valid_sort_orders, "sort_order")

            order_by = ibis.asc(df[col_name]) if sort_order in ["asc", "ascending"] else ibis.desc(df[col_name])
            window = self._create_window(group_col, df, order_by)

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

    def _create_window(
        self,
        group_col: str | list[str] | None,
        df: ibis.Table,
        order_by: ir.Column,
    ) -> Any:  # noqa: ANN401 - WindowedExpr is an internal ibis type; Any is safest here
        """Create an ibis window specification for ranking.

        Builds the appropriate window function based on whether rankings should be
        global or partitioned by group column(s).

        Args:
            group_col (str | list[str] | None): Column(s) to partition the window by.
                None creates a global window, a string or list creates a grouped window.
            df (ibis.Table): The table containing the group columns.
            order_by (ir.Column): An ibis ordering expression (e.g., ibis.asc or ibis.desc) for the window.

        Returns:
            Any: An ibis window specification for use with ranking functions.
        """
        if group_col is None:
            return ibis.window(order_by=order_by)
        if isinstance(group_col, str):
            return ibis.window(group_by=df[group_col], order_by=order_by)
        return ibis.window(group_by=[df[col] for col in group_col], order_by=order_by)

    def _create_composite_ranking(
        self,
        df: ibis.Table,
        rank_mutates: dict[str, ir.IntegerColumn],
        agg_func: str,
    ) -> ibis.Table:
        """Create the final composite ranking by aggregating individual rank columns.

        Combines the individual ranking columns into a single composite_rank column
        using the specified aggregation function.

        Args:
            df (ibis.Table): The table with individual rank columns already added.
            rank_mutates (dict[str, ir.IntegerColumn]): Mapping of rank column names to their expressions,
                used to identify which columns to aggregate.
            agg_func (str): Aggregation function to combine ranks. Must be one of
                "mean", "sum", "min", or "max".

        Returns:
            ibis.Table: The input table with an additional composite_rank column.

        Raises:
            ValueError: If agg_func is not one of the supported aggregation functions.
        """
        column_refs = [df[col] for col in rank_mutates]
        agg_expr = {
            "mean": sum(column_refs) / len(column_refs),
            "sum": sum(column_refs),
            "min": ibis.least(*column_refs),
            "max": ibis.greatest(*column_refs),
        }

        agg_func = ensure_value_choice(agg_func, list(agg_expr.keys()), "agg_func")
        return df.mutate(composite_rank=agg_expr[agg_func])

    @property
    def df(self) -> pd.DataFrame:
        """Returns ranked data ready for business decision-making.

        Returns:
            pd.DataFrame: Performance data with ranking columns added:
                - Original metrics (sales, margin, etc.)
                - Individual rank columns (e.g., sales_rank, margin_rank)
                - composite_rank: Final combined ranking for decisions
        """
        if self._df is None:
            self._df = self.table.execute()
        return self._df
