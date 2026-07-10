"""Example script demonstrating find_overlapping_periods usage."""

from datetime import datetime
from openretailscience.utils.date import find_overlapping_periods

# Example 1: Q4 year-over-year periods
q4_periods = find_overlapping_periods("2020-10-01", "2024-12-31", return_str=True)

# Example 2: Datetime objects in, datetime objects out (return_str=False)
summer_periods = find_overlapping_periods(
    start_date=datetime(2020, 6, 1),
    end_date=datetime(2024, 8, 31),
    return_str=False
)

# Example 3: Single year range (edge case) - returns [] when start and end fall in the same year
single_year = find_overlapping_periods("2024-01-01", "2024-12-31", return_str=True)

# Example 4: Named period dictionary for analysis
periods = find_overlapping_periods("2021-01-01", "2024-12-31", return_str=True)
period_dict = {f"Year_{2021+i}": period for i, period in enumerate(periods)}
