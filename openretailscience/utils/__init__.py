"""Ibis helpers for labelling rows and splitting transactions into time periods.

- keep rows matching any of several conditions, each tagged with its condition's name
  → `filter_and_label_by_condition`
- keep transactions in named date ranges, each tagged with its period's name → `filter_and_label_by_periods`
- split a date range into year-aligned periods for year-on-year comparison → `find_overlapping_periods`
- label each group by whether its items meet a condition (contains, not contains, mixed) → `label_by_condition`

Call help() on a function, such as `help(openretailscience.utils.label_by_condition)`, for its parameters, output
and an example.
"""

from openretailscience.utils.date import filter_and_label_by_periods as filter_and_label_by_periods
from openretailscience.utils.date import find_overlapping_periods as find_overlapping_periods
from openretailscience.utils.filter_and_label import filter_and_label_by_condition as filter_and_label_by_condition
from openretailscience.utils.label import label_by_condition as label_by_condition

# No `__all__`: pydoc would render every listed function in full on this package's help() page.
_EXPORTS = (
    "filter_and_label_by_condition",
    "filter_and_label_by_periods",
    "find_overlapping_periods",
    "label_by_condition",
)


def __dir__() -> list[str]:
    return list(_EXPORTS)
