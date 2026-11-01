"""Example script demonstrating filter_and_label_by_periods usage."""

import numpy as np
import pandas as pd
from datetime import datetime
from openretailscience.utils.date import filter_and_label_by_periods

rng = np.random.default_rng(42)
n_transactions = 2000

df = pd.DataFrame({
    "transaction_id": range(1, n_transactions + 1),
    "transaction_date": pd.date_range("2023-01-01", periods=n_transactions, freq="D")[:n_transactions],
})

# Example 1: Named period ranges (string dates)
quarters = {
    "Q1_2023": ("2023-01-01", "2023-03-31"),
    "Q2_2023": ("2023-04-01", "2023-06-30"),
    "Q3_2023": ("2023-07-01", "2023-09-30"),
    "Q4_2023": ("2023-10-01", "2023-12-31")
}
quarterly_data = filter_and_label_by_periods(df, quarters)
quarterly_result = quarterly_data.execute()

# Example 2: Period ranges as datetime objects instead of date strings
# A datetime end is an exact instant: datetime(2023, 6, 30) drops sales after midnight on Jun 30 when
# transaction_date has a time part. Use "2023-06-30" to include the whole end day.
campaigns = {
    "Summer_Sale": (datetime(2023, 6, 1), datetime(2023, 6, 30)),
    "Back_to_School": (datetime(2023, 8, 1), datetime(2023, 8, 31)),
    "Black_Friday": (datetime(2023, 11, 24), datetime(2023, 11, 27))
}
campaign_data = filter_and_label_by_periods(df, campaigns)
campaign_result = campaign_data.execute()
