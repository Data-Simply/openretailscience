"""Index analysis: how over- or under-represented each group is in a subset of rows, against all rows."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from openretailscience.core.validation import (
    ensure_agg_func,
    ensure_columns,
    ensure_ibis_table,
    ensure_value_choice,
)

if TYPE_CHECKING:
    import ibis
    import pandas as pd

BASELINE_INDEX = 100

VALID_AGG_FUNCS = ("sum", "mean", "max", "min", "nunique")


class IndexAnalysis:
    r"""Each group's share of a subset of rows, indexed against its share of all rows (100 = parity).

    The subset is the rows where ``index_col == value_to_index``; the baseline is every row. For each
    group, ``index = 100 * subset_share / baseline_share``.

    ``table`` (Ibis) and ``df`` (pandas, sorted by the group columns) have one row per group found in the
    baseline, or per ``subgroup_col`` and group pair found in it; a group with no rows in a subgroup gets no
    row there, not one at 0. The columns are the group columns, then ``subset_value`` and
    ``baseline_value`` (``value_col`` aggregated with ``agg_func``), ``subset_share`` and ``baseline_share``
    (fractions of the subset's and baseline's totals, within each subgroup when one is set) and ``index``.
    A group the subset has no rows in has a ``subset_value``, ``subset_share`` and ``index`` of 0; one whose
    subset rows all have a null ``value_col`` has them null. ``index`` is also null when the group's baseline
    share or the subset's total is 0. Reading ``df`` raises a ValueError when no row with a non-null group is
    in the subset.

    Rows with a null ``group_col`` or ``subgroup_col`` are left out of both totals, as if filtered out first; to
    count them, fill the nulls with a label such as "Unknown" before passing the data.

    Examples:
        >>> from openretailscience.analysis import IndexAnalysis
        >>> ia = IndexAnalysis(
        ...     transactions, value_col="customer_id", group_col="category", index_col="brand",
        ...     value_to_index="Meadow Farm", agg_func="nunique",
        ... )
        >>> ia.df
              category  subset_value  baseline_value  subset_share  baseline_share  \
        0       Bakery             0              67      0.000000        0.206790
        ...
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        value_col: str,
        group_col: str,
        index_col: str,
        value_to_index: str | int,
        agg_func: str = "sum",
        subgroup_col: str | None = None,
    ) -> None:
        """Build the index query; it runs when ``df`` is read.

        Args:
            df (pd.DataFrame | ibis.Table): The data, e.g. one row per transaction line or per customer.
            value_col (str): The column to aggregate, e.g. ``unit_spend``, or ``customer_id`` with
                ``agg_func="nunique"`` to count customers.
            group_col (str): The column whose values are indexed, e.g. ``age_band``.
            index_col (str): The column that defines the subset, e.g. ``brand``.
            value_to_index (str | int): The ``index_col`` value whose rows form the subset, e.g. "Meadow Farm".
            agg_func (str, optional): How to aggregate ``value_col``: "sum", "mean", "max", "min" or
                "nunique". Defaults to "sum".
            subgroup_col (str, optional): Compute shares and the index within each value of this column,
                e.g. ``region``. Defaults to None.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame or an Ibis Table.
            ValueError: If a column is missing, ``subgroup_col`` is ``group_col``, or ``agg_func`` is not one
                of the valid options or has no numeric result for ``value_col`` (e.g. "sum" of a text column).
            ibis.common.exceptions.IbisTypeError: If ``value_to_index`` can't be compared with ``index_col``
                (e.g. an int on a text column).
        """
        table = ensure_ibis_table(df)
        ensure_columns(table, value_col, "value_col")
        ensure_columns(table, group_col, "group_col")
        ensure_columns(table, index_col, "index_col")
        if subgroup_col is not None:
            ensure_columns(table, subgroup_col, "subgroup_col")
            if subgroup_col == group_col:
                raise ValueError("subgroup_col must differ from group_col")
        agg_func = ensure_value_choice(agg_func, VALID_AGG_FUNCS, "agg_func")
        ensure_agg_func(table, value_col, agg_func, numeric=True)

        self._group_cols = [group_col] if subgroup_col is None else [subgroup_col, group_col]
        self._index_col = index_col
        self._value_to_index = value_to_index

        table = table.filter(*[table[col].notnull() for col in self._group_cols])  # noqa: PD004 (ibis API)
        aggregate = getattr(table[value_col], agg_func)
        in_subset = table[index_col] == value_to_index
        values = table.group_by(self._group_cols).aggregate(
            subset_value=aggregate(where=in_subset),
            baseline_value=aggregate(),
            # Tells a group with no subset rows (0) from one whose subset values are all null (null), and
            # lets df detect an empty subset, without another query
            subset_rows=table[index_col].count(where=in_subset),
        )
        values = values.mutate(subset_value=(values.subset_rows == 0).ifelse(0, values.subset_value))
        totals_by = [] if subgroup_col is None else [subgroup_col]
        shares = values.mutate(
            subset_share=values.subset_value / values.subset_value.sum().over(group_by=totals_by).nullif(0),
            baseline_share=values.baseline_value / values.baseline_value.sum().over(group_by=totals_by).nullif(0),
        )
        self._with_subset_rows = shares.mutate(
            index=shares.subset_share / shares.baseline_share.nullif(0) * BASELINE_INDEX,
        )
        self.table = self._with_subset_rows.drop("subset_rows")

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """The index table as a pandas DataFrame, sorted by the group columns.

        Raises:
            ValueError: If no row with a non-null group has ``index_col == value_to_index``.
        """
        result = self._with_subset_rows.order_by(self._group_cols).execute()
        if result["subset_rows"].sum() == 0:
            group_cols = " or ".join(repr(col) for col in self._group_cols)
            pandas_fills = " and ".join(f"df[{col!r}].fillna('Unknown')" for col in self._group_cols)
            ibis_fills = " and ".join(f"table[{col!r}].fill_null('Unknown')" for col in self._group_cols)
            msg = (
                f"value_to_index={self._value_to_index!r} matches no rows of index_col {self._index_col!r} with a "
                f"non-null {group_cols}, so there is no subset to index. Pass a value that exists in "
                f"{self._index_col!r}, matching its case and spacing exactly. If its rows have a null "
                f"{group_cols}, fill those first, e.g. {pandas_fills} in pandas or {ibis_fills} in Ibis."
            )
            raise ValueError(msg)
        return result.drop(columns="subset_rows")
