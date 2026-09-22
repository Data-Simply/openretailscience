"""Cross-shop analysis: per-customer overlap (Venn) of 2-3 segments defined by column=value filters."""

from collections.abc import Callable
from typing import Any

import ibis
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

from openretailscience.core.validation import ensure_data_has_columns, ensure_ibis_table
from openretailscience.options import get_option
from openretailscience.plots import venn


class CrossShop:
    """Per-customer overlap of 2-3 segments defined by column=value filters.

    Computed eagerly at construction (``.execute()``); ``.cross_shop_df`` and
    ``.cross_shop_table_df`` are pandas DataFrames.
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
        """Initialize cross-shop analysis for 2 or 3 segments defined by column=value filters.

        Each entity appears once: a segment flag is 1 if the entity has ANY transaction
        matching that segment (max of the 0/1 flags), and `value_col` is the entity's
        `agg_func` over ALL of its transactions, not just matching ones.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data with customer purchases.
            group_1_col (str): Column defining the first segment (e.g., "category", "brand", "channel").
            group_1_val (str): Value of `group_1_col` defining the first segment (e.g., "organic").
            group_2_val (str): Value of `group_2_col` defining the second segment.
            group_2_col (str, optional): Column defining the second segment.
                Defaults to `group_1_col`.
            group_3_col (str, optional): Column for three-way analysis.
                Defaults to `group_1_col` when `group_3_val` is provided.
            group_3_val (str, optional): Value of `group_3_col` defining the third segment. Defaults to None.
            labels (list[str], optional): Segment labels. Defaults to alphabetical labels [A, B, C].
            group_col (str, optional): Entity column to overlap on. Defaults to option
                `column.customer_id`.
            value_col (str, optional): Metric to aggregate per entity. Defaults to option
                `column.unit_spend`.
            agg_func (str, optional): How to combine the entity's values ("sum", "mean", "count").
                Defaults to "sum".

        Raises:
            TypeError: If `df` is not a pandas DataFrame or an Ibis Table.
            ValueError: If required columns are missing or the label count doesn't match the groups.
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
        ensure_data_has_columns(df, required_cols)

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
        """Calculate the per-entity cross-shop dataframe used to plot the diagram.

        Args:
            df (pd.DataFrame | ibis.Table): The input DataFrame or ibis Table containing transactional data.
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
            pd.DataFrame: One row per `group_col` value, indexed by it. Columns: int32 0/1
                flags `group_1`/`group_2` (and `group_3` for three-way), `groups` (tuple of
                the flags), `group_labels` (comma-joined labels, or "No Groups"), and the
                aggregated `value_col`.

        Raises:
            ValueError: If group_3_col or group_3_val is populated, then the other must be as well.
        """
        df = ensure_ibis_table(df)
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

        # ifelse (not casting the boolean directly) keeps this portable to engines with no
        # boolean type (SQL Server, Oracle < 23c); the cast preserves the int32 dtype.
        group_1 = (df[group_1_col] == group_1_val).ifelse(1, 0).cast("int32").name("group_1")
        group_2 = (df[group_2_col] == group_2_val).ifelse(1, 0).cast("int32").name("group_2")
        group_3 = (df[group_3_col] == group_3_val).ifelse(1, 0).cast("int32").name("group_3") if group_3_col else None

        group_cols = ["group_1", "group_2"]
        select_cols = [df[group_col], group_1, group_2]
        if group_3 is not None:
            group_cols.append("group_3")
            select_cols.append(group_3)

        cs_df = df.select([*select_cols, df[temp_value_col]])
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
        """Calculate the aggregated cross-shop table used to plot the diagram.

        The `value_col` default is evaluated at import time; the only caller passes it
        explicitly, so a future caller running after `set_option` would get a stale default.

        Args:
            df (pd.DataFrame): The cross-shop dataframe.
            value_col (str, optional): The column to aggregate. Defaults to option column.unit_spend.

        Returns:
            pd.DataFrame: One row per (`groups`, `group_labels`) with the summed `value_col`
                and `percent`, the segment's share of the total value.
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
        """Generate a Venn diagram of customer segment overlaps.

        Delegates to `openretailscience.plots.venn.plot` with `self.cross_shop_table_df`.

        Args:
            title (str, optional): Chart title (e.g., "Cross-Shopping: Organic vs Conventional").
            eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
            subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
            source_text (str, optional): Data source attribution. Defaults to None.
            vary_size (bool, optional): Scale circles by segment value.
                True = larger segments appear bigger. Defaults to False.
            figsize (tuple[int, int], optional): Plot dimensions. Defaults to None.
            ax (Axes, optional): Existing axes for subplot integration. Defaults to None.
            subset_label_formatter (Callable, optional): Custom formatting for subset percentages.
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
