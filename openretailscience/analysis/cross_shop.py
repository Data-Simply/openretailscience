"""Cross-Shopping Analysis Module for Category and Brand Interaction Insights.

## Business Context

Cross-shopping analysis reveals how customers navigate between different product
categories, brands, or store locations. This intelligence drives critical retail
decisions from store layout and category adjacencies to promotional bundling and
targeted marketing campaigns.

## Real-World Problem

Retailers often make incorrect assumptions about customer behavior. They might place
baby products far from beer, not realizing these categories have high cross-shopping
rates (the famous "diapers and beer" phenomenon). Cross-shop analysis replaces
assumptions with data-driven insights about actual customer purchase patterns.

## Business Applications

1. **Store Layout Optimization**
   - Place frequently cross-shopped categories near each other
   - Create logical customer journey paths through the store
   - Reduce friction in the shopping experience

2. **Promotional Strategy**
   - Bundle products from highly cross-shopped categories
   - Time promotions to capture cross-category purchases
   - Design "buy from A, get discount on B" offers

3. **Category Management**
   - Understand category interdependencies
   - Identify opportunity categories for existing shoppers
   - Spot categories at risk when others decline

4. **Multi-Channel Strategy**
   - Analyze cross-shopping between online and physical stores
   - Optimize channel-specific assortments
   - Design omnichannel customer journeys

5. **Competitive Analysis**
   - Understand customer loyalty across competing brands
   - Identify vulnerable competitor segments
   - Design conquest strategies for shared customers

## Business Value

- **Increased Basket Size**: Strategic placement increases impulse purchases
- **Customer Retention**: Better store experience reduces defection
- **Marketing Efficiency**: Target promotions to actual behavior patterns
- **Strategic Insights**: Understand true category relationships
- **Competitive Advantage**: Leverage unique customer behavior insights

## Visualization Output

The module generates Venn diagrams showing:
- Exclusive shoppers for each category/brand
- Overlap segments with cross-shopping behavior
- Relative size of each segment by customer count or spend
- Percentage breakdowns for strategic planning
"""

from collections.abc import Callable
from typing import Any

import ibis
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

from openretailscience.core.validation import ensure_columns
from openretailscience.options import get_option
from openretailscience.plots import venn


class CrossShop:
    """Analyzes customer cross-shopping behavior between categories, brands, or locations.

    The CrossShop class reveals hidden relationships in customer purchasing patterns,
    enabling retailers to optimize everything from store layouts to marketing campaigns.
    By understanding which products customers buy together across shopping trips, retailers
    can make smarter decisions about product placement, promotions, and assortment.

    ## Business Problem Solved

    Many retail decisions assume customer behavior that may not reflect reality. For example:
    - Are organic shoppers also buying conventional products?
    - Do online grocery shoppers still visit physical stores?
    - Which private label categories attract national brand buyers?

    This analysis provides definitive answers with actionable percentages.
    """

    @staticmethod
    def _generate_default_labels(count: int) -> list[str]:
        """Generate default alphabetical labels for groups.

        Args:
            count (int): Number of labels to generate.

        Returns:
            list[str]: List of alphabetical labels (A, B, C, etc.).
        """
        return [chr(65 + i) for i in range(count)]

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        group_1_col: str,
        group_1_val: str,
        group_2_val: str,
        group_2_col: str | None = None,
        group_3_col: str | None = None,
        group_3_val: str | None = None,
        labels: list[str] | None = None,
        group_col: str | None = None,
        value_col: str | None = None,
        agg_func: str = "sum",
    ) -> None:
        """Initialize cross-shopping analysis between retail categories, brands, or locations.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data with customer purchases.
            group_1_col (str): Column identifying first segment (e.g., "category", "brand", "channel").
            group_1_val (str): Value to analyze for first segment (e.g., "organic", "Brand_A", "online").
            group_2_val (str): Value to analyze for second segment.
            group_2_col (str, optional): Column identifying second segment.
                Defaults to group_1_col.
            group_3_col (str, optional): Column for three-way analysis.
                Defaults to group_1_col when group_3_val provided.
            group_3_val (str, optional): Value for third segment. Defaults to None.
            labels (list[str], optional): Custom labels for diagram (e.g., ["Organic", "Local"]).
                Defaults to alphabetical labels [A, B, C].
            group_col (str, optional): Grouping column (e.g., customer_id, store_id,
                segment_name). Defaults to customer_id from options.
            value_col (str, optional): Metric to analyze (sales, units, visits).
                Defaults to spend column from options.
            agg_func (str, optional): How to combine customer values ("sum", "mean", "count").
                Defaults to "sum" for total opportunity sizing.

        Returns:
            None

        Raises:
            ValueError: If required columns missing or label count doesn't match groups.

        Business Examples:
            >>> # Analyze organic vs conventional shoppers
            >>> cross_shop = CrossShop(
            ...     df=transactions,
            ...     group_1_col="product_type",
            ...     group_1_val="organic",
            ...     group_2_val="conventional",
            ...     labels=["Organic", "Conventional"]
            ... )
            ...
            >>> # Three-way analysis
            >>> cross_shop = CrossShop(
            ...     df=transactions,
            ...     group_1_col="channel",
            ...     group_1_val="online",
            ...     group_2_val="store",
            ...     group_3_val="mobile",
            ...     labels=["Online", "Store", "Mobile"]
            ... )
            ...
            >>> # Custom customer column
            >>> cross_shop = CrossShop(
            ...     df=transactions,
            ...     group_1_col="brand",
            ...     group_1_val="Nike",
            ...     group_2_val="Adidas",
            ...     group_col="user_id"
            ... )
        """
        # Apply smart defaults for simplified interface
        group_col = group_col or get_option("column.customer_id")
        value_col = value_col or get_option("column.unit_spend")

        # Default group_2_col and group_3_col to group_1_col when columns are not provided
        if group_2_col is None:
            group_2_col = group_1_col
        if group_3_val is not None and group_3_col is None:
            group_3_col = group_1_col

        required_cols = [group_col, value_col]
        ensure_columns(df, required_cols, "required_cols")

        self.group_count = 2 if group_3_col is None else 3

        if (labels is not None) and (len(labels) != self.group_count):
            raise ValueError("The number of labels must be equal to the number of group indexes given")

        self.labels = labels if labels is not None else self._generate_default_labels(self.group_count)

        self.cross_shop_df = self._calc_cross_shop(
            df=df,
            group_1_col=group_1_col,
            group_1_val=group_1_val,
            group_2_col=group_2_col,
            group_2_val=group_2_val,
            group_3_col=group_3_col,
            group_3_val=group_3_val,
            group_col=group_col,
            value_col=value_col,
            agg_func=agg_func,
            labels=labels,
        )
        self.cross_shop_table_df = self._calc_cross_shop_table(
            df=self.cross_shop_df,
            value_col=value_col,
        )

    @staticmethod
    def _calc_cross_shop(
        df: pd.DataFrame | ibis.Table,
        group_1_col: str,
        group_1_val: str,
        group_2_col: str,
        group_2_val: str,
        group_3_col: str | None = None,
        group_3_val: str | None = None,
        group_col: str | None = None,
        value_col: str | None = None,
        agg_func: str = "sum",
        labels: list[str] | None = None,
    ) -> pd.DataFrame:
        """Calculate the cross-shop dataframe that will be used to plot the diagram.

        Args:
            df (pd.DataFrame | ibis.Table):  The input DataFrame or ibis Table containing transactional data.
            group_1_col (str): Column name for the first group.
            group_1_val (str): Value to filter for the first group.
            group_2_col (str): Column name for the second group.
            group_2_val (str): Value to filter for the second group.
            group_3_col (str, optional): Column name for the third group. Defaults to None.
            group_3_val (str, optional): Value to filter for the third group. Defaults to None.
            group_col (str, optional): Grouping column (e.g., customer_id, store_id,
                segment_name). Defaults to customer_id from options.
            value_col (str, optional): The column to aggregate. Defaults to option column.unit_spend.
            agg_func (str, optional): The aggregation function. Defaults to "sum".
            labels (list[str], optional): The labels for the groups. Defaults to None.

        Returns:
            pd.DataFrame: The cross-shop dataframe.

        Raises:
            ValueError: If group_3_col or group_3_val is populated, then the other must be as well.
        """
        if isinstance(df, pd.DataFrame):
            df: ibis.Table = ibis.memtable(df)
        if (group_3_col is None) != (group_3_val is None):
            raise ValueError("If group_3_col or group_3_val is populated, then the other must be as well")

        # Apply defaults for group_col and value_col
        group_col = group_col or get_option("column.customer_id")
        value_col = value_col or get_option("column.unit_spend")

        # Using a temporary value column to avoid duplicate column errors during selection.
        # This happens when `value_col` has the same name as `group_col`, causing conflicts
        # in `.select()`.
        temp_value_col = "temp_value_col"
        df = df.mutate(**{temp_value_col: df[value_col]})

        group_1 = (df[group_1_col] == group_1_val).cast("int32").name("group_1")
        group_2 = (df[group_2_col] == group_2_val).cast("int32").name("group_2")
        group_3 = (df[group_3_col] == group_3_val).cast("int32").name("group_3") if group_3_col else None

        group_cols = ["group_1", "group_2"]
        select_cols = [df[group_col], group_1, group_2]
        if group_3 is not None:
            group_cols.append("group_3")
            select_cols.append(group_3)

        cs_df = df.select([*select_cols, df[temp_value_col]]).order_by(group_col)
        cs_df = (
            cs_df.group_by(group_col)
            .aggregate(
                **{col: cs_df[col].max().name(col) for col in group_cols},
                **{temp_value_col: getattr(cs_df[temp_value_col], agg_func)().name(temp_value_col)},
            )
            .order_by(group_col)
        ).execute()

        cs_df["groups"] = cs_df[group_cols].apply(tuple, axis=1)

        # Use default alphabetical labels if none provided
        if labels is None:
            labels = CrossShop._generate_default_labels(len(group_cols))

        group_label_series = cs_df[group_cols].apply(
            lambda x: [labels[i] for i, grp_val in enumerate(x) if grp_val == 1],
            axis=1,
        )
        cs_df["group_labels"] = group_label_series.map(lambda x: "No Groups" if len(x) == 0 else ", ".join(x))

        column_order = [group_col, *group_cols, "groups", "group_labels", temp_value_col]
        cs_df = cs_df[column_order]
        cs_df = cs_df.set_index(group_col)
        return cs_df.rename(columns={temp_value_col: value_col})

    @staticmethod
    def _calc_cross_shop_table(
        df: pd.DataFrame,
        value_col: str = get_option("column.unit_spend"),
    ) -> pd.DataFrame:
        """Calculate the aggregated cross-shop table that will be used to plot the diagram.

        Args:
            df (pd.DataFrame): The cross-shop dataframe.
            value_col (str, optional): The column to aggregate. Defaults to option column.unit_spend.

        Returns:
            pd.DataFrame: The cross-shop table.
        """
        df = df.groupby(["groups", "group_labels"], dropna=False)[value_col].sum().reset_index().copy()
        df["percent"] = df[value_col] / df[value_col].sum()
        return df

    def plot(
        self,
        title: str | None = None,
        eyebrow: str | None = None,
        subtitle: str | None = None,
        source_text: str | None = None,
        vary_size: bool = False,
        figsize: tuple[int, int] | None = None,
        ax: Axes | None = None,
        subset_label_formatter: Callable | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> SubplotBase:
        """Generate Venn diagram showing customer segment overlaps.

        Args:
            title (str, optional): Chart title (e.g., "Cross-Shopping: Organic vs Conventional").
            eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
            subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
            source_text (str, optional): Data source attribution. Defaults to None.
            vary_size (bool, optional): Scale circles by segment value for visual impact.
                True = larger segments appear bigger. Defaults to False.
            figsize (tuple[int, int], optional): Plot dimensions. Defaults to None.
            ax (Axes, optional): Existing axes for subplot integration. Defaults to None.
            subset_label_formatter (callable, optional): Custom formatting for percentages.
                Default shows one decimal place (e.g., "34.5%").
            **kwargs (Any): Additional diagram customization options.

        Returns:
            SubplotBase: Matplotlib axes containing the cross-shop visualization.
        """
        return venn.plot(
            df=self.cross_shop_table_df,
            labels=self.labels,
            title=title,
            eyebrow=eyebrow,
            subtitle=subtitle,
            source_text=source_text,
            vary_size=vary_size,
            figsize=figsize,
            ax=ax,
            subset_label_formatter=subset_label_formatter or (lambda x: f"{x:.1%}"),
            **kwargs,
        )
