"""Gain loss (switching) analysis of customer movement between a focus and a comparison group across two periods.

Pandas-only: all period/group selectors are boolean masks or lists aligned to ``df.index``.
"""

from typing import Any

import pandas as pd
from matplotlib.axes import Axes, SubplotBase

from openretailscience.core.validation import ensure_columns, ensure_data_has_columns
from openretailscience.options import ColumnHelper, get_option
from openretailscience.plots import bar
from openretailscience.plots.styles.colors import COLORS


class GainLoss:
    """Gain loss (switching) analysis between a focus group and a comparison group.

    All selectors are boolean masks or lists aligned to ``df.index`` (pandas-only).
    Results are computed at initialization: ``gain_loss_df`` holds the per-customer
    decomposition and ``gain_loss_table_df`` the aggregated table; there are no
    ``df``/``table`` attributes.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        p1_index: list[bool] | pd.Series,
        p2_index: list[bool] | pd.Series,
        focus_group_index: list[bool] | pd.Series,
        focus_group_name: str,
        comparison_group_index: list[bool] | pd.Series,
        comparison_group_name: str,
        group_col: str | None = None,
        value_col: str = get_option("column.unit_spend"),
        agg_func: str = "sum",
    ) -> None:
        """Calculate the gain loss decomposition at the customer level.

        The four index arguments are boolean masks or lists aligned to ``df.index``.
        The decomposition columns are ``new``, ``lost``, ``increased_focus``,
        ``decreased_focus``, ``switch_from_comparison``, ``switch_to_comparison``.

        Args:
            df (pd.DataFrame): The data to analyze.
            p1_index (list[bool] | pd.Series): Boolean mask selecting the first period rows.
            p2_index (list[bool] | pd.Series): Boolean mask selecting the second period rows.
            focus_group_index (list[bool] | pd.Series): Boolean mask selecting the focus group rows.
            focus_group_name (str): Display name of the focus group.
            comparison_group_index (list[bool] | pd.Series): Boolean mask selecting the comparison group rows.
            comparison_group_name (str): Display name of the comparison group.
            group_col (str | None): Optional column to break the analysis down by.
            value_col (str): Value column to aggregate; defaults to option ``column.unit_spend``.
            agg_func (str): Aggregation function; defaults to "sum".

        Raises:
            ValueError: If ``p1_index`` and ``p2_index`` overlap.
            ValueError: If ``focus_group_index`` and ``comparison_group_index`` overlap.
            ValueError: If the four index masks do not all have the same length.
            ValueError: If the option-derived customer_id or ``value_col`` columns are missing,
                or if a given ``group_col`` is not in ``df``.
        """
        # # Ensure no overlap between p1 and p2
        if not df[p1_index].index.intersection(df[p2_index].index).empty:
            raise ValueError("p1_index and p2_index should not overlap")

        if not df[focus_group_index].index.intersection(df[comparison_group_index].index).empty:
            raise ValueError("focus_group_index and comparison_group_index should not overlap")

        if not len(p1_index) == len(p2_index) == len(focus_group_index) == len(comparison_group_index):
            raise ValueError(
                "p1_index, p2_index, focus_group_index, and comparison_group_index should have the same length",
            )

        if group_col is not None:
            # group_col is typed str | None here, so unpack the single-element list back to a string.
            group_col = ensure_columns(df, group_col, "group_col")[0]
        ensure_data_has_columns(df, [get_option("column.customer_id"), value_col])

        self.focus_group_name = focus_group_name
        self.comparison_group_name = comparison_group_name
        self.group_col = group_col
        self.value_col = value_col

        self.gain_loss_df = self._calc_gain_loss(
            df=df,
            p1_index=p1_index,
            p2_index=p2_index,
            focus_group_index=focus_group_index,
            comparison_group_index=comparison_group_index,
            group_col=group_col,
            value_col=value_col,
            agg_func=agg_func,
        )
        self.gain_loss_table_df = self._calc_gains_loss_table(
            gain_loss_df=self.gain_loss_df,
            group_col=group_col,
        )

    @staticmethod
    def process_customer_group(
        focus_p1: float,
        comparison_p1: float,
        focus_p2: float,
        comparison_p2: float,
        focus_diff: float,
        comparison_diff: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Decompose a group's focus-period change into the six gain loss components.

        Returns ``(new, lost, increased_focus, decreased_focus, switch_from_comparison,
        switch_to_comparison)`` in that order. ``new``, ``increased_focus``, and
        ``switch_from_comparison`` are >= 0; ``lost``, ``decreased_focus``, and
        ``switch_to_comparison`` are <= 0. When both groups are zero in p1 the entire
        focus p2 value counts as ``new``; when both are zero in p2 the entire focus
        p1 value counts as ``lost``.

        Args:
            focus_p1 (float): Focus group value in period 1.
            comparison_p1 (float): Comparison group value in period 1.
            focus_p2 (float): Focus group value in period 2.
            comparison_p2 (float): Comparison group value in period 2.
            focus_diff (float): Focus group change, p2 - p1.
            comparison_diff (float): Comparison group change, p2 - p1.

        Returns:
            tuple[float, float, float, float, float, float]: The six components, in the unpack order above.
        """
        if focus_p1 == 0 and comparison_p1 == 0:
            return focus_p2, 0, 0, 0, 0, 0
        if focus_p2 == 0 and comparison_p2 == 0:
            return 0, -1 * focus_p1, 0, 0, 0, 0

        if focus_diff > 0:
            focus_inc_dec = focus_diff if comparison_diff > 0 else max(0, comparison_diff + focus_diff)
        elif comparison_diff < 0:
            focus_inc_dec = focus_diff
        else:
            focus_inc_dec = min(0, comparison_diff + focus_diff)

        increased_focus = max(0, focus_inc_dec)
        decreased_focus = min(0, focus_inc_dec)

        transfer = focus_diff - focus_inc_dec
        switch_from_comparison = max(0, transfer)
        switch_to_comparison = min(0, transfer)

        return 0, 0, increased_focus, decreased_focus, switch_from_comparison, switch_to_comparison

    @staticmethod
    def _calc_gain_loss(
        df: pd.DataFrame,
        p1_index: list[bool],
        p2_index: list[bool],
        focus_group_index: list[bool],
        comparison_group_index: list,
        group_col: str | None = None,
        value_col: str = get_option("column.unit_spend"),
        agg_func: str = "sum",
    ) -> pd.DataFrame:
        """Calculate the per-customer gain loss decomposition (argument semantics as in ``GainLoss.__init__``).

        Returns:
            pd.DataFrame: Indexed by customer, with ``group_col`` as the leading level when given.
                Columns: ``focus_p1``/``focus_p2``, ``comparison_p1``/``comparison_p2``,
                ``total_p1``/``total_p2``, ``focus_diff``, ``comparison_diff``, ``total_diff``
                plus the six decomposition columns. Rows that are all zero after grouping
                are dropped; missing periods are filled with 0.
        """
        cols = ColumnHelper()
        df = df[p1_index | p2_index].copy()
        df[cols.customer_id] = df[cols.customer_id].astype("category")

        grp_cols = [cols.customer_id] if group_col is None else [group_col, cols.customer_id]

        p1_df = pd.concat(
            [
                df[focus_group_index & p1_index].groupby(grp_cols, observed=False)[value_col].agg(agg_func),
                df[comparison_group_index & p1_index].groupby(grp_cols, observed=False)[value_col].agg(agg_func),
                df[(focus_group_index | comparison_group_index) & p1_index]
                .groupby(grp_cols, observed=False)[value_col]
                .agg(agg_func),
            ],
            axis=1,
        )
        p1_df.columns = ["focus", "comparison", "total"]

        p2_df = pd.concat(
            [
                df[focus_group_index & p2_index].groupby(grp_cols, observed=False)[value_col].agg(agg_func),
                df[comparison_group_index & p2_index].groupby(grp_cols, observed=False)[value_col].agg(agg_func),
                df[(focus_group_index | comparison_group_index) & p2_index]
                .groupby(grp_cols, observed=False)[value_col]
                .agg(agg_func),
            ],
            axis=1,
        )
        p2_df.columns = ["focus", "comparison", "total"]

        gl_df = p1_df.merge(p2_df, on=grp_cols, how="outer", suffixes=("_p1", "_p2")).fillna(0)

        # Remove rows that are all 0 due to grouping by customer_id as a categorical with observed=False
        gl_df = gl_df[~(gl_df == 0).all(axis=1)]

        gl_df["focus_diff"] = gl_df["focus_p2"] - gl_df["focus_p1"]
        gl_df["comparison_diff"] = gl_df["comparison_p2"] - gl_df["comparison_p1"]
        gl_df["total_diff"] = gl_df["total_p2"] - gl_df["total_p1"]

        (
            gl_df["new"],
            gl_df["lost"],
            gl_df["increased_focus"],
            gl_df["decreased_focus"],
            gl_df["switch_from_comparison"],
            gl_df["switch_to_comparison"],
        ) = zip(
            *gl_df.apply(
                lambda x: GainLoss.process_customer_group(
                    focus_p1=x["focus_p1"],
                    comparison_p1=x["comparison_p1"],
                    focus_p2=x["focus_p2"],
                    comparison_p2=x["comparison_p2"],
                    focus_diff=x["focus_diff"],
                    comparison_diff=x["comparison_diff"],
                ),
                axis=1,
            ),
            strict=False,
        )

        return gl_df

    @staticmethod
    def _calc_gains_loss_table(
        gain_loss_df: pd.DataFrame,
        group_col: str | None = None,
    ) -> pd.DataFrame:
        """Aggregate the per-customer gain loss table to total gains and losses.

        Args:
            gain_loss_df (pd.DataFrame): The per-customer gain loss table.
            group_col (str | None): Grouping column, if any.

        Returns:
            pd.DataFrame: A single-row frame when ``group_col`` is None, otherwise one row per group.
        """
        if group_col is None:
            return gain_loss_df.sum().to_frame("").T

        return gain_loss_df.groupby(level=0).sum()

    def plot(
        self,
        title: str | None = None,
        eyebrow: str | None = None,
        subtitle: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        ax: Axes | None = None,
        source_text: str | None = None,
        move_legend_outside: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> SubplotBase:
        """Plot the gain loss table as a stacked horizontal bar of the six decomposition columns.

        Green segments are ``new`` / ``increased_focus`` / ``switch_from_comparison``; red
        segments are ``lost`` / ``decreased_focus`` / ``switch_to_comparison``. Legend labels
        use the focus and comparison group names. ``stacked`` and ``color`` kwargs are
        silently popped and cannot be overridden.

        Args:
            title (str | None): Plot title; defaults to "Gain Loss from {focus} to {comparison}".
            eyebrow (str | None): Small uppercase label rendered above the title.
            subtitle (str | None): Supporting copy rendered below the title.
            x_label (str | None): X-axis label; defaults to the value column name.
            y_label (str | None): Y-axis label; defaults to the focus group name, or ``group_col`` when grouped.
            ax (Axes | None): Axes to plot on.
            source_text (str | None): Source annotation.
            move_legend_outside (bool): Whether to move the legend outside the plot.
            **kwargs (Any): Additional keyword arguments forwarded to the bar plot.

        Returns:
            SubplotBase: The plot axes.
        """
        increase_cols = ["new", "increased_focus", "switch_from_comparison"]
        decrease_cols = ["lost", "decreased_focus", "switch_to_comparison"]
        all_cols = increase_cols + decrease_cols
        segment_colors = [
            COLORS["green"][700],
            COLORS["green"][500],
            COLORS["green"][300],
            COLORS["red"][700],
            COLORS["red"][500],
            COLORS["red"][300],
        ]
        legend_labels = [
            "New",
            f"Increased {self.focus_group_name}",
            f"Switch From {self.comparison_group_name}",
            "Lost",
            f"Decreased {self.focus_group_name}",
            f"Switch To {self.comparison_group_name}",
        ]

        default_y_label = self.focus_group_name if self.group_col is None else self.group_col
        plot_data = self.gain_loss_table_df.copy()

        kwargs.pop("stacked", None)
        kwargs.pop("color", None)

        resolved_title = (
            title if title is not None else f"Gain Loss from {self.focus_group_name} to {self.comparison_group_name}"
        )

        ax = bar.plot(
            df=plot_data,
            value_col=all_cols,
            title=resolved_title,
            eyebrow=eyebrow,
            subtitle=subtitle,
            y_label=y_label if y_label is not None else default_y_label,
            x_label=x_label if x_label is not None else self.value_col,
            orientation="horizontal",
            ax=ax,
            source_text=source_text,
            move_legend_outside=move_legend_outside,
            legend_labels=legend_labels,
            stacked=True,
            color=segment_colors,
            **kwargs,
        )

        ax.axvline(0, color="black", linewidth=0.5)

        return ax
