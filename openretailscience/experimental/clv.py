"""Customer-lifetime-value (BTYD) model input preparation.

This module adapts OpenRetailScience transaction data into the per-customer summary
frame consumed by the "buy-till-you-die" models in
`pymc-marketing <https://www.pymc-marketing.io/>`_ — ``ParetoNBDModel`` (churn and
purchase frequency) and ``GammaGammaModel`` (per-transaction spend).

The summary follows the standard BTYD convention, where each customer is "born" at their
first purchase and observed until ``observation_period_end``:

- ``frequency`` — number of *repeat* purchase occasions (total occasions minus one). A
  purchase occasion is a distinct calendar day, so multiple baskets on the same day count
  once.
- ``recency`` — time from the first purchase to the last purchase.
- ``T`` — the customer's "age": time from the first purchase to ``observation_period_end``.
- ``monetary_value`` — mean spend across the *repeat* purchases (the first purchase is
  excluded, matching the Gamma-Gamma assumption). ``NaN`` for one-time buyers, who cannot
  be fit by the standard Gamma-Gamma model and fall back to the population mean downstream.

``recency`` and ``T`` are expressed in ``period`` units (``"day"`` or ``"week"``) and are
fractional (e.g. 2.5 weeks). The elapsed time is measured in whole days and then divided
by the days-per-period, because the Oracle backend supports only day-granularity date
deltas (see ``openretailscience.analysis.cohort`` for the same constraint).

The first purchase is always the earliest transaction *in the supplied data*, exactly as
`pymc_marketing.clv.rfm_summary` treats it. A customer who was already active before the
data window is therefore treated as born at the window's start — a known limitation of a
truncated observation period, not something this adapter corrects.
"""

from __future__ import annotations

import datetime
import functools
from typing import TYPE_CHECKING, ClassVar

import ibis

from openretailscience.core.validation import (
    ensure_data_has_columns,
    ensure_ibis_table,
    ensure_tznaive_datetime,
    ensure_value_choice,
)
from openretailscience.options import ColumnHelper

if TYPE_CHECKING:
    import pandas as pd


def _coerce_to_date(value: str | datetime.date) -> datetime.date:
    """Normalize a date-like value to a ``datetime.date``.

    Args:
        value (str | datetime.date): An ISO-8601 date string, a ``datetime.date``, or a
            ``datetime.datetime`` (its date part is used).

    Returns:
        datetime.date: The normalized date.

    Raises:
        TypeError: If ``value`` is not a string or ``datetime.date``.
    """
    if isinstance(value, str):
        return datetime.date.fromisoformat(value)
    # ``datetime.datetime`` is a ``datetime.date`` subclass; take its date part explicitly.
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    msg = "observation_period_end must be an ISO-8601 date string, a datetime.date, or None."
    raise TypeError(msg)


class CLVStats:
    """Prepares the per-customer BTYD summary for pymc-marketing CLV models.

    Aggregates transaction data into the ``frequency`` / ``recency`` / ``T`` /
    ``monetary_value`` summary that ``ParetoNBDModel`` and ``GammaGammaModel`` consume.
    Results are accessible via the ``table`` attribute (ibis Table) or the ``df`` property
    (materialized pandas DataFrame). See the module docstring for column definitions.

    Constructing ``CLVStats`` runs one aggregate query against the backend to resolve and
    validate the observation window; the full per-customer summary stays lazy and is
    materialized only on first ``df`` access.

    Args:
        data (pd.DataFrame | ibis.Table): Transaction data containing the ``customer_id``,
            ``transaction_date`` (a date/datetime type), and ``unit_spend`` columns.
        period (str, optional): The time unit for ``recency`` and ``T``. One of ``"day"``
            or ``"week"``. Defaults to ``"week"``.
        observation_period_end (str | datetime.date | None, optional): The end of the
            observation window, from which each customer's age ``T`` is measured. An
            ISO-8601 string or a ``datetime.date``. Defaults to the latest transaction date
            in ``data``.

    Raises:
        TypeError: If ``data`` is not a pandas DataFrame or an Ibis Table, if
            ``transaction_date`` is not a date/datetime type, or if
            ``observation_period_end`` is not a valid date-like value.
        ValueError: If required columns are missing, if ``period`` is not ``"day"`` or
            ``"week"``, or if ``observation_period_end`` is before the latest transaction.
    """

    #: Supported period values. Restricted to the units for which a fractional
    #: (day-delta / days-per-period) age is well defined; month/quarter/year have
    #: irregular day counts and are not meaningful for continuous BTYD time.
    VALID_PERIODS: ClassVar[tuple[str, ...]] = ("day", "week")
    _DAYS_PER_PERIOD: ClassVar[dict[str, int]] = {"day": 1, "week": 7}

    def __init__(
        self,
        data: pd.DataFrame | ibis.Table,
        *,
        period: str = "week",
        observation_period_end: str | datetime.date | None = None,
    ) -> None:
        """Initializes and computes the BTYD summary."""
        self.table: ibis.Table

        cols = ColumnHelper()
        data = ensure_ibis_table(data, "data")
        ensure_data_has_columns(data, [cols.customer_id, cols.transaction_date, cols.unit_spend])
        ensure_tznaive_datetime(data, cols.transaction_date)
        period = ensure_value_choice(period, self.VALID_PERIODS, "period")

        # Cast the single max scalar to a date (not every row) — same idiom as segmentation/rfm.py.
        latest_transaction = _coerce_to_date(data[cols.transaction_date].max().cast("date").execute())
        if observation_period_end is None:
            observation_period_end = latest_transaction
        else:
            observation_period_end = _coerce_to_date(observation_period_end)
            if observation_period_end < latest_transaction:
                msg = (
                    f"observation_period_end ({observation_period_end}) must be on or after the latest "
                    f"transaction date ({latest_transaction}); an earlier end would produce a negative age."
                )
                raise ValueError(msg)

        self.table = self._compute_summary(data, cols, period, observation_period_end)

    @classmethod
    def _compute_summary(
        cls,
        data: ibis.Table,
        cols: ColumnHelper,
        period: str,
        observation_period_end: datetime.date,
    ) -> ibis.Table:
        """Computes the BTYD summary table.

        Args:
            data (ibis.Table): The validated transaction data.
            cols (ColumnHelper): Resolved column names.
            period (str): The validated period unit (``"day"`` or ``"week"``).
            observation_period_end (datetime.date): The resolved observation window end.

        Returns:
            ibis.Table: One row per customer with the BTYD summary columns.
        """
        # Collapse to one purchase occasion per customer per calendar day, summing spend
        # within the day (same-day baskets are a single occasion under the BTYD convention).
        day = data[cols.transaction_date].cast("date").name("_day")
        daily = data.group_by([cols.customer_id, day]).aggregate(_day_spend=data[cols.unit_spend].sum())

        # A repeat occasion is any purchase day after the customer's first purchase day.
        first_day = daily["_day"].min().over(ibis.window(group_by=daily[cols.customer_id]))
        daily = daily.mutate(_is_repeat=daily["_day"] > first_day)

        summary = daily.group_by(cols.customer_id).aggregate(
            _first_day=daily["_day"].min(),
            _last_day=daily["_day"].max(),
            _occasions=daily.count(),
            _repeat_spend=daily._day_spend.sum(where=daily._is_repeat),
        )

        days_per_period = cls._DAYS_PER_PERIOD[period]
        observation_end = ibis.literal(observation_period_end)
        frequency = summary._occasions - 1

        # frequency / recency / T / monetary_value are pymc-marketing's required literal
        # column names (ParetoNBDModel / GammaGammaModel), deliberately not options.py names.
        # The customer id keeps its configured column name; if column.customer_id has been
        # overridden, the downstream model call must be given the matching id column.
        return summary.mutate(
            frequency=frequency.cast("int64"),
            # Day-granularity delta (Oracle-safe) divided by days-per-period for a fractional age.
            recency=summary._last_day.delta(summary._first_day, unit="day") / days_per_period,
            T=observation_end.delta(summary._first_day, unit="day") / days_per_period,
            monetary_value=summary._repeat_spend / frequency.nullif(0),
        ).select(
            cols.customer_id,
            "frequency",
            "recency",
            "T",
            "monetary_value",
        )

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """Returns the materialized BTYD summary as a pandas DataFrame.

        The ``customer_id`` is a column (not the index) so the frame can be passed straight
        to ``ParetoNBDModel`` / ``GammaGammaModel``, which expect it as a column.

        Returns:
            pd.DataFrame: One row per customer with columns ``customer_id``, ``frequency``,
                ``recency``, ``T``, and ``monetary_value``.
        """
        return self.table.execute()
