"""Utility functions for time-related operations in retail analysis."""

from collections.abc import Mapping
from datetime import datetime, timezone

import ibis

from openretailscience.options import get_option


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
        return datetime.strptime(date_val, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if isinstance(date_val, datetime):
        # If datetime is timezone-naive, make it timezone-aware (UTC)
        if date_val.tzinfo is None:
            return date_val.replace(tzinfo=timezone.utc)
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
    normalized = {}

    for period_name, date_range in period_ranges.items():
        date_range_length = 2
        if not (isinstance(date_range, tuple) and len(date_range) == date_range_length):
            msg = f"Period '{period_name}' must have a (start_date, end_date) tuple"
            raise ValueError(msg)

        start_date, end_date = date_range

        start_dt = _normalize_datetime(start_date)
        end_dt = _normalize_datetime(end_date)

        if start_dt > end_dt:
            msg = f"Period '{period_name}': start date ({start_date}) must be <= end date ({end_date})"
            raise ValueError(msg)

        normalized[period_name] = (start_dt, end_dt)

    # Check for overlapping periods
    period_list = list(normalized.items())
    for i, (name1, (start1, end1)) in enumerate(period_list):
        for name2, (start2, end2) in period_list[i + 1 :]:
            if start1 <= end2 and start2 <= end1:
                overlap_msg = (
                    f"Periods '{name1}' ({start1.date()}-{end1.date()}) and"
                    f" '{name2}' ({start2.date()}-{end2.date()}) overlap"
                )
                raise ValueError(overlap_msg)

    return normalized


def filter_and_label_by_periods(
    transactions: ibis.Table,
    period_ranges: dict[str, tuple[datetime, datetime] | tuple[str, str]],
    period_col: str = "period_name",
) -> ibis.Table:
    """Filter transactions to named time periods and label each row with its period.

    Rows outside every period are dropped. The date column is read from the
    ``column.transaction_date`` option; string dates in ``period_ranges`` are
    parsed as ``%Y-%m-%d`` and localized to UTC.

    Args:
        transactions (ibis.Table): Table containing the transaction date column.
        period_ranges (dict[str, tuple[datetime, datetime] | tuple[str, str]]): Period names
            mapped to ``(start_date, end_date)`` tuples.
        period_col (str): Name of the label column to add. Defaults to ``"period_name"``.

    Returns:
        ibis.Table: Filtered table with the ``period_col`` label column.

    Raises:
        ValueError: If a period value is not a ``(start, end)`` tuple, if a start date is
            after its end date, or if two periods overlap.
    """
    # Validate periods first
    _validate_and_normalize_periods(period_ranges)

    branches = []
    date_column = transactions[get_option("column.transaction_date")]

    date_col_dtype = date_column.type()

    for period_name, date_range in period_ranges.items():
        start_date, end_date = date_range

        period_condition = date_column.between(
            ibis.literal(start_date, type=date_col_dtype),
            ibis.literal(end_date, type=date_col_dtype),
        )
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
