"""Example script demonstrating filter_and_label_by_event_periods usage."""

from datetime import timedelta

import numpy as np
import pandas as pd
from openretailscience.utils.date import filter_and_label_by_event_periods

rng = np.random.default_rng(7)
transactions = pd.DataFrame({
    "transaction_id": range(1, 2001),
    "store_id": rng.choice([101, 102, 103], 2000),
    "transaction_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, 2000), unit="D"),
    "unit_spend": rng.uniform(5, 80, 2000).round(2),
})

# The date a competitor opened near each store. Store 103 has none, so its rows are dropped.
openings = pd.DataFrame({
    "store_id": [101, 102],
    "event_date": pd.to_datetime(["2025-03-10", "2025-06-02"]),
})

wk, day = timedelta(weeks=1), timedelta(days=1)

# Example 1: before and after each store's opening, plus the post weeks a year earlier
labelled = filter_and_label_by_event_periods(
    transactions,
    events=openings,
    on="store_id",
    windows={
        "pre": (-12 * wk, -day),
        "post": (0 * wk, 12 * wk - day),
        "post_py1": (-52 * wk, -40 * wk - day),
    },
)
spend_by_window = (
    labelled.group_by(["store_id", "period_name"])
    .aggregate(spend=labelled.unit_spend.sum())
    .order_by(["store_id", "period_name"])
    .execute()
)

# Example 2: a week-by-week curve, labelled with the week number (week 0 starts on the opening day)
weekly = filter_and_label_by_event_periods(
    transactions,
    events=openings,
    on="store_id",
    windows={n: (n * wk, (n + 1) * wk - day) for n in range(-12, 12)},
)
weekly_spend = weekly.group_by("period_name").aggregate(spend=weekly.unit_spend.sum()).order_by("period_name").execute()
