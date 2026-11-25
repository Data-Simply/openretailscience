"""Example script demonstrating filter_and_label_by_event_to_date usage."""

from datetime import timedelta

import numpy as np
import pandas as pd
from openretailscience.utils.date import BEFORE_EVENT, filter_and_label_by_event_to_date

rng = np.random.default_rng(7)
transactions = pd.DataFrame({
    "transaction_id": range(1, 3001),
    "customer_id": rng.choice([1042, 2210, 3378, 4415], 3000),
    "transaction_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 547, 3000), unit="D"),
    "unit_spend": rng.uniform(5, 80, 3000).round(2),
})

# The day each customer joined the loyalty programme. Customer 4415 never joined, so their rows are dropped.
signups = pd.DataFrame({
    "customer_id": [1042, 2210, 3378],
    "event_date": pd.to_datetime(["2025-01-20", "2025-03-15", "2025-05-10"]),
})

wk = timedelta(weeks=1)

# Each customer's test period runs from sign-up to the last day of the data. pre is the same number of days
# just before sign-up, and post_py1 is the test period's dates a year earlier.
labelled = filter_and_label_by_event_to_date(
    transactions,
    events=signups,
    on="customer_id",
    end_date="2025-06-30",
    windows={"post": 0 * wk, "pre": BEFORE_EVENT, "post_py1": -52 * wk},
    include_duration=True,
)
spend_by_window = (
    labelled.group_by(["customer_id", "period_name", "period_days"])
    .aggregate(spend=labelled.unit_spend.sum())
    .order_by(["customer_id", "period_name"])
    .execute()
)
