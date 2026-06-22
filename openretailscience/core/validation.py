"""Internal validation helpers consumed by other openretailscience modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable


VALID_SORT_ORDERS = ("asc", "ascending", "desc", "descending")


def ensure_ibis_table(df: pd.DataFrame | ibis.Table, param_name: str = "df") -> ibis.Table:
    """Convert pandas DataFrame to ibis Table, or validate input is an ibis Table.

    Args:
        df (pd.DataFrame | ibis.Table): Input data to convert or validate.
        param_name (str): The caller's parameter name to surface in error messages.
            Defaults to ``"df"``.

    Returns:
        ibis.Table: An ibis Table representation of the input data.

    Raises:
        TypeError: If df is neither a pandas DataFrame nor an ibis Table.
    """
    if isinstance(df, pd.DataFrame):
        return ibis.memtable(df)
    if isinstance(df, ibis.Table):
        return df
    msg = f"{param_name} must be either a pandas DataFrame or an Ibis Table."
    raise TypeError(msg)


def ensure_columns(
    df: pd.DataFrame | ibis.Table,
    columns: str | list[str],
    param_name: str,
) -> list[str]:
    """Normalize a column parameter to a list and validate it against a DataFrame/Table.

    Combines the four steps every column parameter needs: accept either a single
    string or a list of strings, normalize to a list, validate that all elements
    are strings, and validate that every column exists in the input.

    Args:
        df (pd.DataFrame | ibis.Table): The data whose columns must contain the requested names.
        columns (str | list[str]): Column name or list of column names to validate.
        param_name (str): The caller's parameter name to surface in error messages
            (e.g. ``"group_col"`` or ``"segment_col"``).

    Returns:
        list[str]: A fresh list of validated column names.

    Raises:
        TypeError: If ``columns`` is neither a string nor a list.
        TypeError: If ``columns`` is a list whose elements are not all strings.
        ValueError: If ``columns`` is an empty list.
        ValueError: If any requested column is missing from ``df``.
    """
    if isinstance(columns, str):
        normalized = [columns]
    elif isinstance(columns, list):
        normalized = list(columns)
    else:
        msg = f"{param_name} must be a string or list of strings. Got {type(columns).__name__}."
        raise TypeError(msg)

    if len(normalized) == 0:
        msg = f"{param_name} must not be an empty list."
        raise ValueError(msg)

    if not all(isinstance(col, str) for col in normalized):
        bad_types = sorted({type(col).__name__ for col in normalized if not isinstance(col, str)})
        msg = f"{param_name} must contain only strings. Got element types: {bad_types}."
        raise TypeError(msg)

    missing_cols = sorted(set(normalized) - set(df.columns))
    if len(missing_cols) > 0:
        msg = f"{param_name} references columns not present in the data: {missing_cols}."
        raise ValueError(msg)

    return normalized


def ensure_data_has_columns(df: pd.DataFrame | ibis.Table, columns: list[str]) -> None:
    """Verify the input data contains every column the function requires.

    Use when the column list is built internally from defaults or aggregated from
    multiple sources (i.e. there is no single user-facing parameter to blame).
    For validating a single column parameter supplied by the caller, use
    ``ensure_columns`` instead so the parameter name appears in error messages.

    Args:
        df (pd.DataFrame | ibis.Table): The data being inspected.
        columns (list[str]): Column names that must be present in ``df``.

    Raises:
        TypeError: If ``columns`` is not a list (e.g. a bare string would otherwise be
            iterated as characters and silently produce a meaningless "missing columns" error).
        ValueError: If any of the listed columns is missing from ``df``.
    """
    if not isinstance(columns, list):
        msg = f"columns must be a list of column names. Got {type(columns).__name__}."
        raise TypeError(msg)
    missing = sorted(set(columns) - set(df.columns))
    if len(missing) > 0:
        msg = f"Input data is missing required columns: {missing}."
        raise ValueError(msg)


def ensure_value_choice(
    value: str,
    choices: Iterable[str],
    param_name: str,
) -> str:
    """Validate a string parameter against a fixed set of allowed values.

    Matching is case-insensitive; the returned value is the canonical entry
    from ``choices`` (so callers can use it directly in branching or lookup
    logic without further normalization).

    Args:
        value (str): The value supplied by the caller.
        choices (Iterable[str]): The full set of allowed values. Any iterable of
            strings — list, tuple, dict (iteration yields keys), set, etc. —
            is accepted; the iterable is materialized into a list once so a
            one-shot iterator works as well.
        param_name (str): The parameter name to surface in error messages.

    Returns:
        str: The canonical entry from ``choices`` matching ``value``.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If ``value`` is not one of ``choices``.
    """
    if not isinstance(value, str):
        msg = f"{param_name} must be a string. Got {type(value).__name__}."
        raise TypeError(msg)

    choices_list = list(choices)
    lowered = value.lower()
    for choice in choices_list:
        if choice.lower() == lowered:
            return choice

    msg = f"{param_name} must be one of {choices_list}. Got '{value}'."
    raise ValueError(msg)


def ensure_unit_interval(value: float, param_name: str) -> None:
    """Validate that a numeric parameter lies in the closed unit interval [0, 1].

    Args:
        value (float): The value supplied by the caller (e.g. a percentile).
        param_name (str): The parameter name to surface in error messages.

    Raises:
        TypeError: If ``value`` is not an int or float (``bool`` is rejected).
        ValueError: If ``value`` is outside the inclusive range [0, 1].
    """
    # ``bool`` is an ``int`` subclass, so it would otherwise pass as 0.0/1.0 and silently
    # return the min/max instead of erroring on a misrouted flag.
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{param_name} must be a number between 0 and 1. Got {type(value).__name__}."
        raise TypeError(msg)
    if not 0.0 <= value <= 1.0:
        msg = f"{param_name} must be between 0 and 1 (inclusive). Got {value}."
        raise ValueError(msg)


def ensure_positive(value: float, param_name: str) -> None:
    """Validate that a numeric parameter is strictly greater than zero.

    Args:
        value (float): The value supplied by the caller (e.g. a churn period in days).
        param_name (str): The parameter name to surface in error messages.

    Raises:
        TypeError: If ``value`` is not an int or float (``bool`` is rejected).
        ValueError: If ``value`` is not strictly positive.
    """
    # ``bool`` is an ``int`` subclass; reject it so a misrouted flag does not become a 1-day window.
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{param_name} must be a number. Got {type(value).__name__}."
        raise TypeError(msg)
    if value <= 0:
        msg = f"{param_name} must be positive. Got {value}."
        raise ValueError(msg)


def ensure_tznaive_datetime(df: ibis.Table, column: str) -> None:
    """Validate that a column is a timezone-naive date or datetime type.

    Day-level operations (truncation, lag/lead over ordered days) require a temporal
    column. Timezone-aware timestamps are rejected because the backend normalizes them
    to UTC, which silently shifts day boundaries for non-UTC data.

    Args:
        df (ibis.Table): The table whose column is being validated.
        column (str): The name of the column that must be a tz-naive temporal type.

    Raises:
        TypeError: If ``column`` is not a date or datetime type.
        ValueError: If ``column`` is a timezone-aware timestamp.
    """
    col_type = df[column].type()
    if not col_type.is_temporal():
        msg = f"Column '{column}' must be a date or datetime type. Got {col_type}."
        raise TypeError(msg)
    if col_type.is_timestamp() and col_type.timezone is not None:
        msg = (
            f"Column '{column}' is timezone-aware ({col_type.timezone}). The backend normalizes "
            f"timezone-aware timestamps to UTC, which can shift day boundaries. Convert to a "
            f"timezone-naive wall-clock time first, e.g. "
            f"df['{column}'] = df['{column}'].dt.tz_localize(None)."
        )
        raise ValueError(msg)
