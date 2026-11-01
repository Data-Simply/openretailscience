"""Utility functions for time-related operations in retail analysis."""

from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta

import ibis
import pandas as pd

from openretailscience.core.validation import ensure_data_has_columns, ensure_ibis_table, ensure_tznaive_datetime
from openretailscience.options import get_option


def _period_end(end_date: datetime | str) -> datetime:
    """Return a period's inclusive end: a date-only string covers that whole day, not its first instant.

    Args:
        end_date (datetime | str): A datetime, or a ``YYYY-MM-DD`` string.

    Returns:
        datetime: ``end_date`` itself, or the string's day at 23:59:59.999999.
    """
    if isinstance(end_date, str):
        return datetime.combine(datetime.strptime(end_date, "%Y-%m-%d"), time.max)  # noqa: DTZ007
    return end_date


def _normalize_datetime(date_val: datetime | str) -> datetime:
    """Convert a string or datetime to a timezone-aware datetime object.

    Strings are parsed as ``%Y-%m-%d`` and localized to UTC.  Naive datetimes
    are made aware by attaching UTC.  Already-aware datetimes are returned
    unchanged.

    Args:
        date_val (datetime | str): A date string (``YYYY-MM-DD``) or datetime to normalize.

    Returns:
        datetime: A timezone-aware datetime in UTC.

    Raises:
        TypeError: If *date_val* is neither a ``str`` nor a ``datetime``.
    """
    if isinstance(date_val, str):
        # Convert string to timezone-aware datetime
        return datetime.strptime(date_val, "%Y-%m-%d").replace(tzinfo=UTC)
    if isinstance(date_val, datetime):
        # If datetime is timezone-naive, make it timezone-aware (UTC)
        if date_val.tzinfo is None:
            return date_val.replace(tzinfo=UTC)
        return date_val
    error_msg = f"Expected str or datetime, got {type(date_val)}"
    raise TypeError(error_msg)


def _is_naive(d: datetime | str) -> bool:
    """Check whether a datetime-like input is timezone-naive.

    Args:
        d (datetime | str): A datetime object or date string to check.

    Returns:
        bool: True if the input is a string or a naive datetime, False if tz-aware.

    Raises:
        TypeError: If d is not a str or datetime instance.
    """
    if isinstance(d, str):
        return True
    if isinstance(d, datetime):
        return d.tzinfo is None
    msg = f"Expected str or datetime, got {type(d)}"
    raise TypeError(msg)


def _validate_and_normalize_periods(
    period_ranges: Mapping[str, tuple[datetime | str, datetime | str]],
) -> dict[str, tuple[datetime, datetime]]:
    """Validates and normalizes period ranges, returning timezone-aware datetime tuples."""
    if len(period_ranges) == 0:
        msg = "period_ranges must contain at least one period"
        raise ValueError(msg)

    normalized = {}

    for period_name, date_range in period_ranges.items():
        date_range_length = 2
        if not (isinstance(date_range, tuple) and len(date_range) == date_range_length):
            msg = f"Period '{period_name}' must have a (start_date, end_date) tuple"
            raise ValueError(msg)

        start_date, end_date = date_range
        for side, bound in (("start", start_date), ("end", end_date)):
            if isinstance(bound, str):
                try:
                    datetime.strptime(bound, "%Y-%m-%d")  # noqa: DTZ007
                except ValueError as err:
                    msg = f"Period '{period_name}' {side} '{bound}' is not a valid 'YYYY-MM-DD' date."
                    raise ValueError(msg) from err

        start_dt = _normalize_datetime(start_date)
        end_dt = _normalize_datetime(_period_end(end_date))

        if start_dt > end_dt:
            msg = f"Period '{period_name}': start date ({start_date}) must be <= end date ({end_date})"
            raise ValueError(msg)

        normalized[period_name] = (start_dt, end_dt)

    _ensure_no_overlap(normalized)
    return normalized


def _ensure_no_overlap(normalized: dict[str, tuple[datetime, datetime]]) -> None:
    """Raise if any two normalized periods overlap.

    Args:
        normalized (dict[str, tuple[datetime, datetime]]): Period names mapped to inclusive ``(start, end)``.

    Raises:
        ValueError: If two periods overlap.
    """
    period_list = list(normalized.items())
    for i, (name1, (start1, end1)) in enumerate(period_list):
        for name2, (start2, end2) in period_list[i + 1 :]:
            if start1 <= end2 and start2 <= end1:
                overlap_msg = (
                    f"Periods '{name1}' ({start1.date()}-{end1.date()}) and"
                    f" '{name2}' ({start2.date()}-{end2.date()}) overlap"
                )
                raise ValueError(overlap_msg)


def _normalize_bound(value: datetime | date | str, on_date_column: bool) -> datetime | str:
    """Return a bound as the overlap check and the filter must both read it.

    The column is tz-naive wall-clock time, so an aware datetime drops its offset (as ``ensure_date`` keeps a
    datetime's local date). A ``date`` always covers whole days, and so does a ``datetime`` on a date column,
    which compares it by its day; those become ``"YYYY-MM-DD"`` strings.

    Args:
        value (datetime | date | str): A period bound.
        on_date_column (bool): Whether the transaction date column is a date (not a timestamp) column.

    Returns:
        datetime | str: A naive datetime, or a ``"YYYY-MM-DD"`` string.
    """
    if isinstance(value, datetime):
        return value.date().isoformat() if on_date_column else value.replace(tzinfo=None)
    if isinstance(value, date):
        return value.isoformat()
    return value


def filter_and_label_by_periods(
    transactions: pd.DataFrame | ibis.Table,
    period_ranges: Mapping[str, tuple[datetime | date | str, datetime | date | str]],
    period_col: str = "period_name",
) -> ibis.Table:
    """Filter transactions to named time periods and label each row with its period.

    Rows outside every period are dropped. The date column is read from the
    ``column.transaction_date`` option.

    Args:
        transactions (pd.DataFrame | ibis.Table): Transactions containing the transaction date column.
        period_ranges (Mapping[str, tuple[datetime | date | str, datetime | date | str]]): Period
            names mapped to inclusive ``(start_date, end_date)`` tuples. A ``date`` or
            ``"YYYY-MM-DD"`` string end includes that whole day. A datetime end is an exact instant
            on a timestamp column; on a date column it is cast to its day, so that whole day is
            included. A timezone-aware datetime is read as wall-clock time: its offset is ignored.
        period_col (str): Name of the label column to add. Defaults to ``"period_name"``.

    Returns:
        ibis.Table: Filtered table with the ``period_col`` label column.

    Raises:
        TypeError: If ``transactions`` is neither a pandas DataFrame nor an ibis Table, if its
            transaction date column is not a date or datetime type, if ``period_ranges`` is not a
            mapping, or if a period bound is not a string, date or datetime.
        ValueError: If ``period_ranges`` is empty, if a period value is not a ``(start, end)``
            tuple, if a bound string is not a valid ``YYYY-MM-DD`` date, if a start date is after
            its end date, if two periods overlap, if ``period_col`` is already a column, or if the
            transaction date column is missing or timezone-aware.

    Examples:
        >>> from openretailscience.utils import filter_and_label_by_periods
        >>> periods = {"H1 2024": ("2024-01-01", "2024-06-30"), "H1 2025": ("2025-01-01", "2025-06-30")}
        >>> h1 = filter_and_label_by_periods(transactions, period_ranges=periods)
        >>> h1.group_by("period_name").aggregate(spend=h1.unit_spend.sum()).order_by("period_name").to_pandas()
          period_name    spend
        0     H1 2024  1194.05
        1     H1 2025  1350.15
    """
    if not isinstance(period_ranges, Mapping):
        msg = (
            "period_ranges must be a mapping of period names to (start_date, end_date) tuples. "
            f"Got {type(period_ranges).__name__}."
        )
        raise TypeError(msg)
    transactions = ensure_ibis_table(transactions, "transactions")
    date_col = get_option("column.transaction_date")
    ensure_data_has_columns(transactions, [date_col])
    if period_col in transactions.columns:
        msg = f"period_col '{period_col}' is already a column in transactions. Pass a different period_col."
        raise ValueError(msg)
    ensure_tznaive_datetime(transactions, date_col)
    date_column = transactions[date_col]
    date_col_dtype = date_column.type()

    period_ranges = {
        name: tuple(_normalize_bound(bound, date_col_dtype.is_date()) for bound in bounds)
        if isinstance(bounds, tuple)
        else bounds
        for name, bounds in period_ranges.items()
    }
    _validate_and_normalize_periods(period_ranges)

    branches = []
    for period_name, date_range in period_ranges.items():
        start_date, end_date = date_range

        start = ibis.literal(start_date, type=date_col_dtype)
        if isinstance(end_date, str):
            # "< next day" rather than <= 23:59:59.999999, which a millisecond backend (e.g. Postgres
            # TIMESTAMP(3)) rounds up to the next midnight.
            next_day = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)  # noqa: DTZ007
            end = ibis.literal(next_day, type=date_col_dtype)
            period_condition = (date_column >= start) & (date_column < end)
        else:
            period_condition = date_column.between(start, ibis.literal(end_date, type=date_col_dtype))
        branches.append((period_condition, ibis.literal(period_name)))

    conditions = ibis.or_(*[condition[0] for condition in branches])
    return transactions.filter(conditions).mutate(**{period_col: ibis.cases(*branches)})


def find_overlapping_periods(
    start_date: datetime | str,
    end_date: datetime | str,
    return_str: bool = True,
) -> list[tuple[str | datetime, str | datetime]]:
    """Split a date range into year-aligned overlapping periods.

    The first period starts at ``start_date``; each subsequent period starts on the same
    month and day one year later. Each period ends on ``end_date``'s month and day in the
    following year, so the last period ends exactly at ``end_date``. Both dates in the
    same year yield an empty list.

    Note:
        No leap-year adjustment: a Feb 29 start or end date raises ``ValueError`` in a
        non-leap year.

    Args:
        start_date (datetime | str): Range start, a datetime or ``YYYY-MM-DD`` string.
        end_date (datetime | str): Range end, a datetime or ``YYYY-MM-DD`` string.
        return_str (bool): If True, dates are returned as ISO-formatted strings; if
            False, as datetime objects. Defaults to True.

    Returns:
        list[tuple[str | datetime, str | datetime]]: ``(start, end)`` pairs; datetime
            outputs preserve the input timezone-awareness (string inputs give naive outputs).

    Raises:
        TypeError: If ``start_date`` and ``end_date`` have mismatched timezone awareness.
        ValueError: If the start date is after the end date.

    Examples:
        >>> from openretailscience.utils import find_overlapping_periods
        >>> find_overlapping_periods("2023-10-01", "2025-12-31")
        [('2023-10-01', '2024-12-31'), ('2024-10-01', '2025-12-31')]
    """
    # Track whether outputs should be tz-naive to preserve backward compatibility.
    # String inputs and naive datetime inputs both produced naive outputs before.
    start_is_naive = _is_naive(start_date)
    end_is_naive = _is_naive(end_date)

    if start_is_naive != end_is_naive:
        msg = "start_date and end_date must have matching timezone awareness. Got naive and aware (or vice versa)."
        raise TypeError(msg)

    start_date = _normalize_datetime(start_date)
    end_date = _normalize_datetime(end_date)

    if start_date > end_date:
        raise ValueError("Start date must be before end date")

    if start_date.year == end_date.year:
        return []

    if start_is_naive:
        output_tz = None
        start_date = start_date.replace(tzinfo=None)
    else:
        output_tz = start_date.tzinfo

    period_starts = [
        start_date if year == start_date.year else datetime(year, start_date.month, start_date.day, tzinfo=output_tz)
        for year in range(start_date.year, end_date.year)
    ]
    period_ends = [
        datetime(year + 1, end_date.month, end_date.day, tzinfo=output_tz)
        for year in range(start_date.year, end_date.year)
    ]

    if return_str:
        return [
            (s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")) for s, e in zip(period_starts, period_ends, strict=True)
        ]

    return list(zip(period_starts, period_ends, strict=True))
