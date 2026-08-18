"""Example: build the BTYD summary for pymc-marketing CLV models with CLVStats.

CLVStats turns transaction data into the per-customer frequency / recency / T /
monetary_value frame that pymc-marketing's ParetoNBDModel and GammaGammaModel consume. It
does not fit the models (and does not depend on pymc-marketing); it prepares their input.
"""

import numpy as np
import pandas as pd

from openretailscience.experimental.clv import CLVStats

rng = np.random.default_rng(42)

# ~600 customers shopping across 2023. Each starts on a random day in the first ~200 days
# and makes 1-11 purchases; one-time buyers arise naturally and get frequency 0. 600 rather than
# fewer: the frequency/monetary correlation must stay inside repeat_buyers' independence check for
# the sample drawn below too.
n_customers = 600
epoch = np.datetime64("2023-01-01")
last_day_offset = 364  # 2023-12-31

first_day = rng.integers(0, 200, size=n_customers)
purchases_per_customer = rng.integers(1, 12, size=n_customers)

customer_id = np.repeat(np.arange(1, n_customers + 1), purchases_per_customer)
first_day_per_txn = np.repeat(first_day, purchases_per_customer)
days_after_first = (rng.random(customer_id.size) * 330).astype(int)
day_offset = np.minimum(first_day_per_txn + days_after_first, last_day_offset)

# A customer-constant acquisition channel (one value per customer, repeated across their rows) and a
# per-transaction store. The channel is a candidate one-hot covariate; the store is aggregated per
# customer into customer_attributes below.
channels = np.array(["email", "paid_search", "organic", None], dtype=object)
signup_channel = np.repeat(rng.choice(channels, size=n_customers), purchases_per_customer)

# Each customer has a typical basket size; per-transaction spend varies mildly around it and is
# independent of how often they shop (Gamma-Gamma assumes spend and frequency are uncorrelated).
basket_size = rng.uniform(5, 250, size=n_customers)
unit_spend = (np.repeat(basket_size, purchases_per_customer) * rng.uniform(0.8, 1.2, size=customer_id.size)).round(2)

transactions = pd.DataFrame({
    "customer_id": customer_id,
    "transaction_date": pd.to_datetime(epoch + day_offset.astype("timedelta64[D]")),
    "unit_spend": unit_spend,
    "store_id": rng.integers(1, 6, size=customer_id.size),
    "signup_channel": signup_channel,
})

# Weekly BTYD summary: one row per customer (customer_id, frequency, recency, T, monetary_value).
clv = CLVStats(transactions, period="week")

# observation_period_end pins the window instead of defaulting to the latest transaction date.
summary_asof = CLVStats(transactions, period="week", observation_period_end="2023-12-31").df

# sample() draws a random customer subset as another CLVStats: fit on it, score everyone from clv.df.
# Pass n (exact count) or frac (share), never both; the draw is deterministic given random_state.
train = clv.sample(n=150)
train_repeat_buyers = train.repeat_buyers  # frequency > 0: the GammaGammaModel fitting input

# The same property over the full population: the frame fitted spend is scored on.
repeat_buyers = clv.repeat_buyers

# pymc_time_unit ("W" here) is the time_unit for expected_customer_lifetime_value; its default "D"
# silently misreads a weekly/monthly horizon.
time_unit = clv.pymc_time_unit

# customer_attributes (one row per customer) is left-joined onto the summary; one_hot_col encodes one
# of its columns into 0/1 ParetoNBDModel covariates. Build it with any per-customer aggregation.
customer_attributes = (
    transactions.groupby("customer_id")
    .agg(stores_shopped=("store_id", "nunique"), signup_channel=("signup_channel", "first"))
    .reset_index()
)
clv_covariates = CLVStats(
    transactions,
    period="week",
    customer_attributes=customer_attributes,
    one_hot_col="signup_channel",
)
summary_covariates = clv_covariates.df
# covariate_cols = the attached columns (one-hot dummies + stores_shopped) to pass as covariates.
covariate_cols = clv_covariates.covariate_cols

# Feed to pymc-marketing (install separately): fit ParetoNBDModel on train.df and GammaGammaModel on
# train_repeat_buyers, then score the full population with clv.df / repeat_buyers, passing time_unit
# for finite-horizon CLV. For covariates, fit on clv_covariates (or a sample of it) and pass
# covariate_cols as ParetoNBDModel's purchase/dropout covariate columns.
