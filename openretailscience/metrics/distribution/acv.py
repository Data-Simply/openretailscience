"""ACV (All Commodity Volume) metric.

ACV measures total dollar sales across all products in a set of stores,
expressed in millions ($MM).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import ibis
from ibis import _

if TYPE_CHECKING:
    import pandas as pd

from openretailscience.core.validation import ensure_columns, ensure_data_has_columns, ensure_ibis_table
from openretailscience.options import get_option


class Acv:
    """Calculates ACV (All Commodity Volume) for a set of stores.

    ACV represents total dollar sales across all products, expressed in millions ($MM).
    NaN values in the spend column are excluded from the sum.

    Results are accessible via the `table` attribute (ibis Table) or the `df` property
    (materialized pandas DataFrame).

    Args:
        df (pd.DataFrame | ibis.Table): Transaction data containing at least a unit_spend column.
        group_col (str | list[str] | None, optional): Optional column(s) to group the ACV calculation by
            (e.g., store_id). Defaults to None for total ACV.
        acv_scale_factor (float, optional): Factor to scale the ACV result (default is 1,000,000 for $MM).

    Raises:
        TypeError: If df is not a pandas DataFrame or an Ibis Table.
        ValueError: If required columns are missing from the data or if acv_scale_factor is not positive.
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        *,
        group_col: str | list[str] | None = None,
        acv_scale_factor: float = 1_000_000,
    ) -> None:
        """Initializes the ACV calculation."""
        self.table: ibis.Table

        df = ensure_ibis_table(df)

        if acv_scale_factor <= 0:
            raise ValueError("acv_scale_factor must be positive.")

        unit_spend_col = get_option("column.unit_spend")

        if group_col is not None:
            group_col = ensure_columns(df, group_col, "group_col")

        # group_col is already validated above; only the function's hard-coded requirement remains.
        ensure_data_has_columns(df, [unit_spend_col])

        if group_col is not None:
            df = df.group_by(group_col)

        self.table = df.aggregate(acv=_[unit_spend_col].sum() / acv_scale_factor)

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """Returns the materialized pandas DataFrame of ACV results.

        Returns:
            pd.DataFrame: DataFrame with ACV values.
        """
        return self.table.execute()
