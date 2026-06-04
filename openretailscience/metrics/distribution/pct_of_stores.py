"""% of Stores (Numeric Distribution) metric.

% of Stores measures the share of total stores in the dataset that sell a given product.
Every store counts equally regardless of its sales volume.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import ibis
from ibis import _

if TYPE_CHECKING:
    import pandas as pd

from openretailscience.core.validation import ensure_columns, ensure_data_has_columns, ensure_ibis_table
from openretailscience.metrics.base import ratio_metric
from openretailscience.options import ColumnHelper, get_option

_TEMP_TOTAL_STORES = "__prs_temp_total_stores__"


class PctOfStores:
    """Calculates the percentage of stores selling each product.

    This is the simplest, unweighted distribution metric (numeric distribution).
    It answers the question: "What fraction of stores carry this product?"

    Results are accessible via the ``table`` attribute (ibis Table) or the ``df`` property
    (materialized pandas DataFrame).

    Args:
        df (pd.DataFrame | ibis.Table): Transaction-level data containing at least
            store_id and product_id columns.
        product_col (str | list[str] | None, optional): Column(s) defining product granularity.
            Defaults to ``get_option("column.product_id")``.
        group_col (str | list[str] | None, optional): Additional grouping dimensions
            (e.g., ``"category_0_name"``). Defaults to None.
        within_group (bool, optional): Controls the denominator when ``group_col`` is specified.
            When ``False`` (default), the percentage is relative to all stores in the dataset.
            When ``True``, the percentage is relative to stores within each group independently.
            Has no effect when ``group_col`` is None. Defaults to False.

    Raises:
        TypeError: If df is not a pandas DataFrame or an Ibis Table.
        ValueError: If required columns are missing from the data, or if product_col
            appears in group_col.
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        *,
        product_col: str | list[str] | None = None,
        group_col: str | list[str] | None = None,
        within_group: bool = False,
    ) -> None:
        """Initializes the % of Stores calculation."""
        self.table: ibis.Table

        df = ensure_ibis_table(df)

        store_id_col = get_option("column.store_id")

        if product_col is None:
            product_col = [get_option("column.product_id")]
        else:
            product_col = ensure_columns(df, product_col, "product_col")

        if group_col is not None:
            group_col = ensure_columns(df, group_col, "group_col")

        group_cols = list(product_col)
        if group_col is not None:
            overlap = set(product_col) & set(group_col)
            if len(overlap) > 0:
                msg = f"product_col {overlap} must not also appear in group_col"
                raise ValueError(msg)
            group_cols.extend(group_col)
        # store_id_col + any unvalidated product_col defaults still need to exist in df;
        # already-validated user inputs (group_col, an explicitly-passed product_col) are
        # excluded to avoid redundant set-difference work.
        ensure_data_has_columns(df, [store_id_col, *product_col])

        store_product = df.select([store_id_col, *group_cols]).distinct()

        agg_stores_col = get_option("column.agg.store_id")
        per_group = store_product.group_by(group_cols).aggregate(
            **{agg_stores_col: _[store_id_col].count()},
        )

        use_within_group = within_group and group_col is not None
        if use_within_group:
            total_stores = store_product.group_by(group_col).aggregate(
                **{_TEMP_TOTAL_STORES: _[store_id_col].nunique()},
            )
            per_group = per_group.inner_join(total_stores, group_col)
            denominator = _[_TEMP_TOTAL_STORES]
        else:
            denominator = store_product[store_id_col].nunique()

        pct_stores_col = ColumnHelper.join_options("column.agg.store_id", "column.suffix.percent")
        final_cols = [*group_cols, agg_stores_col, pct_stores_col]
        self.table = per_group.mutate(
            **{pct_stores_col: ratio_metric(_[agg_stores_col], denominator)},
        ).select(final_cols)

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """Returns the materialized pandas DataFrame of % of Stores results.

        Returns:
            pd.DataFrame: DataFrame with % of stores values. Cached after first access.
        """
        return self.table.execute()
