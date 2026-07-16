"""Example: build the BTYD summary for pymc-marketing CLV models with CLVStats.

CLVStats turns transaction data into the per-customer frequency / recency / T /
monetary_value frame that pymc-marketing's ParetoNBDModel and GammaGammaModel consume. It
does not fit the models (and does not depend on pymc-marketing); it prepares their input.
"""

import numpy as np
import pandas as pd

from openretailscience.experimental.clv import CLVStats

rng = np.random.default_rng(42)

# ~200 customers shopping across 2023. Each starts on a random day in the first ~200 days
# and makes 1-11 purchases; one-time buyers arise naturally and get frequency 0.
n_customers = 200
epoch = np.datetime64("2023-01-01")
last_day_offset = 364  # 2023-12-31

first_day = rng.integers(0, 200, size=n_customers)
purchases_per_customer = rng.integers(1, 12, size=n_customers)

customer_id = np.repeat(np.arange(1, n_customers + 1), purchases_per_customer)
first_day_per_txn = np.repeat(first_day, purchases_per_customer)
days_after_first = (rng.random(customer_id.size) * 330).astype(int)
day_offset = np.minimum(first_day_per_txn + days_after_first, last_day_offset)

transactions = pd.DataFrame({
    "customer_id": customer_id,
    "transaction_date": pd.to_datetime(epoch + day_offset.astype("timedelta64[D]")),
    "unit_spend": rng.uniform(5, 250, size=customer_id.size).round(2),
})

# Weekly summary: recency and T are in weeks; monetary_value is the mean spend across each
# customer's repeat purchases (NaN for one-time buyers). Columns: customer_id, frequency,
# recency, T, monetary_value.
summary = CLVStats(transactions, period="week").df

# Pin the observation window to an explicit "as of" date rather than the default (the latest
# transaction date). A wrong horizon unit later is a common silent bug, so keep the model's
# time unit (here, weeks) fixed and known.
summary_asof = CLVStats(transactions, period="week", observation_period_end="2023-12-31").df

# The repeat-buyer subset is what the Gamma-Gamma model is fit on.
repeat_buyers = summary[summary["frequency"] > 0]

# Feed these straight to pymc-marketing (install it separately; not imported here):
#
#     from pymc_marketing.clv import ParetoNBDModel, GammaGammaModel
#
#     pareto = ParetoNBDModel(data=summary)      # uses frequency, recency, T
#     pareto.fit()
#
#     gamma_gamma = GammaGammaModel(data=repeat_buyers)   # uses frequency, monetary_value
#     gamma_gamma.fit()
