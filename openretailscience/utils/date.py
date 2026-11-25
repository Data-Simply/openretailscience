"""Utility functions for time-related operations in retail analysis."""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from enum import Enum
from typing import TYPE_CHECKING

import ibis

from openretailscience.core.validation import (
    ensure_columns,
    ensure_data_has_columns,
    ensure_date,
    ensure_ibis_table,
    ensure_tznaive_datetime,
)
from openretailscience.options import get_option
from openretailscience.utils.filter_and_label import filter_and_label_by_condition

if TYPE_CHECKING:
    import pandas as pd

_DAY = timedelta(days=1)


class _BeforeEvent(Enum):
    """Type of ``BEFORE_EVENT``, the window that ends the day before the event."""

    BEFORE_EVENT = "BEFORE_EVENT"


BEFORE_EVENT = _BeforeEvent.BEFORE_EVENT
_MAX_BAD_EVENT_ROWS = 5
_RANGE_BOUNDS = 2


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
        if not (isinstance(date_range, tuple) and len(date_range) == _RANGE_BOUNDS):
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

    _ensure_no_overlap(normalized, "Periods")
    return normalized


def _ensure_no_overlap(
    ranges: Mapping[str, tuple[datetime, datetime]] | Mapping[str | int, tuple[int, int]],
    kind: str,
) -> None:
    """Raise if any two inclusive ``(start, end)`` ranges overlap.

    Args:
        ranges (Mapping[str, tuple[datetime, datetime]] | Mapping[str | int, tuple[int, int]]): Names mapped to
            inclusive ``(start, end)`` datetimes, or to day offsets from an event date.
        kind (str): What the ranges are called in the error, such as ``"Periods"``.

    Raises:
        ValueError: If two ranges overlap.
    """
    range_list = list(ranges.items())
    for i, (name1, (start1, end1)) in enumerate(range_list):
        for name2, (start2, end2) in range_list[i + 1 :]:
            if start1 <= end2 and start2 <= end1:
                # Compared as formatted, so a string period end (23:59:59.999999) matches a start on the same day.
                s1, e1, s2, e2 = (
                    b.date().isoformat() if isinstance(b, datetime) else f"{b}d" for b in (start1, end1, start2, end2)
                )
                msg = f"{kind} {name1!r} [{s1}, {e1}] and {name2!r} [{s2}, {e2}] overlap."
                if (start1 < start2 and s2 == e1) or (start2 < start1 and s1 == e2):
                    earlier, later = (name1, name2) if start1 < start2 else (name2, name1)
                    msg += f" Both ends are included, so end {earlier!r} before {later!r} starts."
                raise ValueError(msg)


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

    conditions = {}
    for period_name, date_range in period_ranges.items():
        start_date, end_date = date_range

        start = ibis.literal(start_date, type=date_col_dtype)
        if isinstance(end_date, str):
            # "< next day" rather than <= 23:59:59.999999, which a millisecond backend (e.g. Postgres
            # TIMESTAMP(3)) rounds up to the next midnight.
            next_day = datetime.strptime(end_date, "%Y-%m-%d") + _DAY  # noqa: DTZ007
            end = ibis.literal(next_day, type=date_col_dtype)
            period_condition = (date_column >= start) & (date_column < end)
        else:
            period_condition = date_column.between(start, ibis.literal(end_date, type=date_col_dtype))
        conditions[period_name] = period_condition

    return filter_and_label_by_condition(transactions, conditions, label_col=period_col)


def _ensure_window_labels(windows: Mapping, values: str) -> None:
    """Raise unless ``windows`` is a non-empty mapping whose labels are all strings or all integers."""
    if not isinstance(windows, Mapping):
        msg = f"windows must be a mapping of labels to {values}. Got {type(windows).__name__}."
        raise TypeError(msg)
    if len(windows) == 0:
        raise ValueError("windows must contain at least one window.")
    if not (
        all(isinstance(label, str) for label in windows)
        or all(isinstance(label, numbers.Integral) for label in windows)
    ):
        msg = "windows labels must be all strings or all integers. They become one label column, which has one type."
        raise ValueError(msg)


def _window_days(
    windows: Mapping[str, tuple[timedelta, timedelta]] | Mapping[int, tuple[timedelta, timedelta]],
) -> dict[str | int, tuple[int, int]]:
    """Validate ``windows`` and return each label's inclusive ``(start, end)`` offsets in days."""
    _ensure_window_labels(windows, "(start, end) timedelta offsets from the event date")
    days = {}
    for label, bounds in windows.items():
        if not (
            isinstance(bounds, tuple) and len(bounds) == _RANGE_BOUNDS and all(isinstance(b, timedelta) for b in bounds)
        ):
            msg = f"Window {label!r} must be a (start, end) tuple of timedeltas. Got {bounds!r}."
            raise TypeError(msg)
        start, end = bounds
        if start % _DAY or end % _DAY:
            msg = f"Window {label!r} offsets must be whole days. Got {start} and {end}."
            raise ValueError(msg)
        if start > end:
            msg = f"Window {label!r} starts after it ends: start {start.days}d must be <= end {end.days}d."
            raise ValueError(msg)
        days[label] = (start.days, end.days)
    _ensure_no_overlap(days, "Windows")
    return days


def _shift_days(
    windows: Mapping[str, timedelta | _BeforeEvent] | Mapping[int, timedelta | _BeforeEvent],
) -> dict[str | int, int | _BeforeEvent]:
    """Validate event-to-date ``windows`` and return each label's shift in days, or ``BEFORE_EVENT``."""
    _ensure_window_labels(windows, "timedelta moves of the test period, or BEFORE_EVENT")
    shifts = {}
    for label, shift in windows.items():
        if shift is BEFORE_EVENT:
            shifts[label] = shift
            continue
        if not isinstance(shift, timedelta):
            msg = f"Window {label!r} must be a timedelta or BEFORE_EVENT. Got {shift!r}."
            raise TypeError(msg)
        if shift % _DAY:
            msg = f"Window {label!r} must move by whole days. Got {shift}."
            raise ValueError(msg)
        shifts[label] = shift.days
    return shifts


def _longest_test_period(
    shifts: dict[str | int, int | _BeforeEvent],
) -> tuple[int, str | int, str | int] | None:
    """Return the longest test period, in days, for which no two windows overlap, and the pair that sets it.

    None when no pair overlaps at any length, so only events after ``end_date`` are invalid.

    Every window is as long as the test period. Two windows moved by ``a`` and ``b`` days overlap once that
    length exceeds ``|a - b|``. ``BEFORE_EVENT`` ends the day before the event, so it overlaps a window moved
    back ``a`` days once the length exceeds ``a // 2``.

    Raises:
        ValueError: If two windows overlap at any length.
    """
    limits = []
    labels = list(shifts)
    for i, label1 in enumerate(labels):
        for label2 in labels[i + 1 :]:
            shift1, shift2 = shifts[label1], shifts[label2]
            if shift1 is BEFORE_EVENT and shift2 is BEFORE_EVENT:
                msg = f"Windows {label1!r} and {label2!r} are both BEFORE_EVENT, so they always overlap."
                raise ValueError(msg)
            if shift1 is BEFORE_EVENT or shift2 is BEFORE_EVENT:
                moved = shift2 if shift1 is BEFORE_EVENT else shift1
                if moved == -1:
                    msg = (
                        f"Windows {label1!r} and {label2!r} both hold the day before the event, so they always overlap."
                    )
                    raise ValueError(msg)
                if moved < 0:
                    limits.append((-moved // 2, label1, label2))
                continue
            if shift1 == shift2:
                msg = f"Windows {label1!r} and {label2!r} move by the same amount, so they always overlap."
                raise ValueError(msg)
            limits.append((abs(shift1 - shift2), label1, label2))
    return min(limits, key=lambda limit: limit[0]) if len(limits) > 0 else None


def _ensure_one_event_per_key(
    events: ibis.Table,
    keys: list[str],
    event_date_col: str,
    date_rule: tuple[ibis.expr.types.BooleanColumn, str] | None = None,
) -> None:
    """Raise if a key has several events, a null value or a null event date; a duplicate would fan out the join.

    ``date_rule`` is ``(outside, rule)``: events where ``outside`` is true raise too, in the same query, and
    ``rule`` says which event dates are allowed.
    """
    aggregates = {
        "rows": events.count(),
        "null_event_dates": events.count() - events[event_date_col].count(),
    }
    if date_rule is not None:
        aggregates["dates_outside"] = ibis.cases((date_rule[0], 1), else_=0).sum()
    per_key = events.group_by(keys).aggregate(**aggregates)
    has_null_key = ibis.or_(*[per_key[key].isnull() for key in keys])  # noqa: PD003 (ibis API, not pandas)
    is_bad = (per_key.rows > 1) | (per_key.null_event_dates > 0) | has_null_key
    if date_rule is not None:
        is_bad |= per_key.dates_outside > 0
    bad = per_key.filter(is_bad).limit(_MAX_BAD_EVENT_ROWS).to_pandas()
    if len(bad) > 0:
        rule = f"events must have one row per {', '.join(keys)}, with no null keys or event dates"
        if date_rule is not None:
            rule += f", and {date_rule[1]}"
        rows = "; ".join(str(row) for row in bad.to_dict("records"))
        msg = f"{rule}. Rows to fix (at most {_MAX_BAD_EVENT_ROWS} shown): {rows}"
        raise ValueError(msg)


def _prepare_event_inputs(
    transactions: pd.DataFrame | ibis.Table,
    events: pd.DataFrame | ibis.Table,
    on: str | list[str],
    event_date_col: str,
    period_col: str,
) -> tuple[ibis.Table, ibis.Table, list[str]]:
    """Validate the arguments both event functions share; return the two tables and the key columns."""
    transactions = ensure_ibis_table(transactions, "transactions")
    events = ensure_ibis_table(events, "events")
    keys = ensure_columns(transactions, on, "on")
    ensure_columns(events, keys, "on (in events)")
    ensure_columns(events, event_date_col, "event_date_col")
    date_col = get_option("column.transaction_date")
    ensure_data_has_columns(transactions, [date_col])
    if period_col in transactions.columns:
        msg = f"period_col '{period_col}' is already a column in transactions. Pass a different period_col."
        raise ValueError(msg)
    ensure_tznaive_datetime(transactions, date_col)
    ensure_tznaive_datetime(events, event_date_col)
    return transactions, events, keys


def _label_by_key_bounds(
    transactions: ibis.Table,
    events: ibis.Table,
    keys: list[str],
    bounds: dict[str | int, tuple[ibis.expr.types.DateValue, ibis.expr.types.DateValue]],
    period_col: str,
    extra_cols: dict[str, ibis.Value],
) -> ibis.Table:
    """Label each transaction with the window its key's ``[start, stop)`` dates contain.

    ``bounds`` maps each label to start and stop dates built from ``events``; stop is the day after the
    window's last day. They are computed on the events table so the join leaves only comparisons for the
    transactions. ``extra_cols`` are further per-key columns, built from ``events``, to keep in the output.
    """
    date_col = get_option("column.transaction_date")
    date_type = transactions[date_col].type()
    bound_cols = {}
    for i, (start, stop) in enumerate(bounds.values()):
        bound_cols[f"__window_{i}_start"] = start.cast(date_type)
        bound_cols[f"__window_{i}_stop"] = stop.cast(date_type)
    key_bounds = events.select(*keys, **bound_cols, **extra_cols)
    joined = transactions.join(key_bounds, keys)

    date_column = joined[date_col]
    # Read through key_bounds, not joined: a transactions column with a bound's name makes the join rename it.
    conditions = {
        label: (date_column >= key_bounds[f"__window_{i}_start"]) & (date_column < key_bounds[f"__window_{i}_stop"])
        for i, label in enumerate(bounds)
    }
    labelled = filter_and_label_by_condition(joined, conditions, label_col=period_col)
    if not isinstance(next(iter(bounds)), str):
        # An integer literal takes the narrowest type that fits, so week numbers would come back as int8.
        labelled = labelled.mutate(**{period_col: labelled[period_col].cast("int64")})
    return labelled.select(*transactions.columns, period_col, *extra_cols)


def filter_and_label_by_event_periods(
    transactions: pd.DataFrame | ibis.Table,
    events: pd.DataFrame | ibis.Table,
    on: str | list[str],
    windows: Mapping[str, tuple[timedelta, timedelta]] | Mapping[int, tuple[timedelta, timedelta]],
    event_date_col: str = "event_date",
    period_col: str = "period_name",
) -> ibis.Table:
    """Filter transactions to windows around each key's own event date and label each row with its window.

    Each store, customer, product or other key in ``events`` has its own event date, such as a competitor
    opening or a loyalty sign-up, and the same windows are placed around each one. The date column is read
    from the ``column.transaction_date`` option.

    Transactions whose key has no row in ``events``, or whose key is null, are dropped. So are rows outside
    every window. A window the data covers only in part returns the rows it has: make sure the data spans
    every window for every key, or a part-filled window will be compared with a full one.

    Both offsets of a window are included as whole days, so windows that follow on from each other end one
    day before the next starts. For the same window a year earlier, subtract 52 weeks from both offsets,
    which keeps weekdays aligned. For a week-by-week curve, use integer labels:
    ``{n: (n * wk, (n + 1) * wk - day) for n in range(-12, 12)}`` labels each row with its week number,
    where week 0 starts on the event day.

    ``events`` is checked with one query when this function is called. The returned table is lazy.

    Args:
        transactions (pd.DataFrame | ibis.Table): Transactions with the transaction date column and the
            ``on`` columns.
        events (pd.DataFrame | ibis.Table): One row per ``on`` key, with its event date.
        on (str | list[str]): Column, or columns, in both tables that say what each event belongs to,
            such as ``"store_id"`` or ``["store_id", "brand_id"]``.
        windows (Mapping[str, tuple[timedelta, timedelta]] | Mapping[int, tuple[timedelta, timedelta]]):
            Labels mapped to inclusive ``(start, end)`` offsets from the event date. The labels must be all
            strings or all integers.
        event_date_col (str): Date column in ``events``. A timestamp counts from the start of its day.
            Defaults to ``"event_date"``.
        period_col (str): Name of the label column to add. Defaults to ``"period_name"``.

    Returns:
        ibis.Table: The transactions inside a window, with the ``period_col`` label column.

    Raises:
        TypeError: If ``transactions`` or ``events`` is neither a pandas DataFrame nor an ibis Table, if
            ``on`` is not a string or list of strings, if ``windows`` is not a mapping, if a window is not a
            ``(start, end)`` tuple of timedeltas, or if either date column is not a date or datetime type.
        ValueError: If ``windows`` is empty, mixes string and integer labels, has an offset that is not a
            whole day, a window that starts after it ends or two windows that overlap; if ``on`` is empty;
            if a column is missing; if ``period_col`` is already a column in ``transactions``; if either date
            column is timezone-aware; or if ``events`` has more than one row for a key, a null key or a null
            event date.

    Examples:
        >>> from datetime import timedelta
        >>> from openretailscience.utils import filter_and_label_by_event_periods
        >>> events = pd.DataFrame({"store_id": [101, 102], "event_date": pd.to_datetime(["2024-09-02", "2025-01-06"])})
        >>> wk, day = timedelta(weeks=1), timedelta(days=1)
        >>> windows = {"pre": (-12 * wk, -day), "post": (0 * wk, 12 * wk - day)}
        >>> labelled = filter_and_label_by_event_periods(transactions, events=events, on="store_id", windows=windows)
        >>> labelled.group_by("period_name").agg(spend=labelled.unit_spend.sum()).order_by("period_name").to_pandas()
          period_name   spend
        0        post  363.90
        1         pre  544.65
    """
    window_days = _window_days(windows)
    transactions, events, keys = _prepare_event_inputs(transactions, events, on, event_date_col, period_col)
    _ensure_one_event_per_key(events, keys, event_date_col)

    # Truncated whatever its type: ibis types an Oracle DATE, which holds a time of day, as a date.
    event_day = events[event_date_col].truncate("D").cast("date")
    bounds = {
        # The day after the end is the stop, so a timestamp late on the end day is still inside.
        label: (event_day + ibis.interval(days=start), event_day + ibis.interval(days=end + 1))
        for label, (start, end) in window_days.items()
    }
    return _label_by_key_bounds(transactions, events, keys, bounds, period_col, {})


def filter_and_label_by_event_to_date(
    transactions: pd.DataFrame | ibis.Table,
    events: pd.DataFrame | ibis.Table,
    on: str | list[str],
    end_date: date | str,
    windows: Mapping[str, timedelta | _BeforeEvent] | Mapping[int, timedelta | _BeforeEvent],
    event_date_col: str = "event_date",
    period_col: str = "period_name",
    include_duration: bool = False,
    duration_col: str = "period_days",
) -> ibis.Table:
    """Label transactions with a test period from each key's event date to one end date, and copies of it.

    Each key's test period runs from its own event date, such as a loyalty sign-up, to ``end_date``, such as the
    last day of the data. Every window has the same length as its key's test period. A ``timedelta`` moves both
    ends of the test period: ``0 * wk`` is the test period itself, and ``-52 * wk`` the same weeks a year earlier.
    ``BEFORE_EVENT`` is the window that ends the day before the event.

    Windows of the same length overlap once the test period is long enough. With ``BEFORE_EVENT`` and
    ``-52 * wk``, a test period over 182 days overlaps, so every event must be at most 181 days before
    ``end_date``. ``events`` is checked for this, and for one row per key, with one query when this function
    is called. Filter early events out first to leave them out knowingly. The returned table is lazy.

    Transactions whose key has no row in ``events``, or whose key is null, are dropped. So are rows outside
    every window. A window the data covers only in part returns the rows it has: make sure the data spans
    every window for every key.

    Args:
        transactions (pd.DataFrame | ibis.Table): Transactions with the transaction date column and the
            ``on`` columns.
        events (pd.DataFrame | ibis.Table): One row per ``on`` key, with its event date.
        on (str | list[str]): Column, or columns, in both tables that say what each event belongs to,
            such as ``"customer_id"`` or ``"store_id"``.
        end_date (date | str): The last day of every key's test period, included.
        windows (Mapping[str, timedelta | BEFORE_EVENT] | Mapping[int, timedelta | BEFORE_EVENT]): Labels mapped
            to a whole-day ``timedelta`` added to both ends of the test period, or ``BEFORE_EVENT``. The labels
            must be all strings or all integers.
        event_date_col (str): Date column in ``events``. A timestamp counts from the start of its day.
            Defaults to ``"event_date"``.
        period_col (str): Name of the label column to add. Defaults to ``"period_name"``.
        include_duration (bool): Whether to add ``duration_col``, the key's period length in days, both ends
            included. Defaults to False.
        duration_col (str): Name of the duration column. Defaults to ``"period_days"``.

    Returns:
        ibis.Table: The transactions inside a window, with the ``period_col`` label column, and
            ``duration_col`` if ``include_duration``.

    Raises:
        TypeError: If ``transactions`` or ``events`` is neither a pandas DataFrame nor an ibis Table, if
            ``on`` is not a string or list of strings, if ``windows`` is not a mapping, if a window is neither a
            ``timedelta`` nor ``BEFORE_EVENT``, if ``end_date`` is not a date or string, or if either date column
            is not a date or datetime type.
        ValueError: If ``windows`` is empty, mixes string and integer labels, moves by part of a day or has two
            windows that always overlap; if ``end_date`` is not an ISO date; if ``on`` is empty; if a column is
            missing; if ``period_col`` or ``duration_col`` is already a column in ``transactions``, or
            ``duration_col`` equals ``period_col``; if either date column is timezone-aware; or if ``events`` has
            more than one row for a key, a null key or a null event date, or an event after ``end_date`` or too
            early for its windows not to overlap.

    Examples:
        >>> from datetime import timedelta
        >>> from openretailscience.utils import BEFORE_EVENT, filter_and_label_by_event_to_date
        >>> events = pd.DataFrame({"store_id": [101, 102], "event_date": pd.to_datetime(["2025-03-03", "2025-04-07"])})
        >>> windows = {"post": timedelta(0), "pre": BEFORE_EVENT, "post_py1": timedelta(weeks=-52)}
        >>> labelled = filter_and_label_by_event_to_date(
        ...     transactions, events=events, on="store_id", end_date="2025-06-30", windows=windows
        ... )
        >>> labelled.group_by("period_name").agg(spend=labelled.unit_spend.sum()).order_by("period_name").to_pandas()
          period_name   spend
        0        post  476.95
        1    post_py1  388.80
        2         pre  521.85
    """
    end_date = ensure_date(end_date, "end_date")
    shifts = _shift_days(windows)
    longest = _longest_test_period(shifts)
    transactions, events, keys = _prepare_event_inputs(transactions, events, on, event_date_col, period_col)
    if include_duration and duration_col in transactions.columns:
        msg = f"duration_col '{duration_col}' is already a column in transactions. Pass a different duration_col."
        raise ValueError(msg)
    if include_duration and duration_col == period_col:
        msg = f"duration_col and period_col are both '{period_col}'. Pass a different duration_col."
        raise ValueError(msg)

    # Truncated whatever its type: ibis types an Oracle DATE, which holds a time of day, as a date.
    event_day = events[event_date_col].truncate("D").cast("date")
    if longest is None:
        outside, dates = event_day > end_date, f"event dates on or before {end_date}"
    else:
        max_days, label1, label2 = longest
        earliest = end_date - timedelta(days=max_days - 1)
        outside = (event_day < earliest) | (event_day > end_date)
        dates = (
            f"event dates from {earliest} to {end_date}. An earlier event gives a test period over {max_days} "
            f"days, where windows {label1!r} and {label2!r} overlap"
        )
    _ensure_one_event_per_key(events, keys, event_date_col, (outside, dates))

    end = ibis.literal(end_date, type="date")
    length = end.delta(event_day, unit="day") + 1
    bounds = {}
    for label, shift in shifts.items():
        if shift is BEFORE_EVENT:
            bounds[label] = (event_day - length.as_interval("D"), event_day)
        else:
            # The day after the end is the stop, so a timestamp late on end_date is still inside.
            bounds[label] = (event_day + ibis.interval(days=shift), end + ibis.interval(days=shift + 1))
    extra_cols = {duration_col: length.cast("int64")} if include_duration else {}
    return _label_by_key_bounds(transactions, events, keys, bounds, period_col, extra_cols)


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
