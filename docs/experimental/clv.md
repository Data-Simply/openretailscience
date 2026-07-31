---
title: Customer Lifetime Value
social:
  cards_layout_options:
    title: OpenRetailScience | Customer Lifetime Value
---

!!! warning "Experimental"
    CLV preparation is experimental. Its API and import path (under `openretailscience.experimental`)
    may change without notice.

## CLVStats (BTYD model input)

`CLVStats` turns transaction data into the per-customer summary that the "buy-till-you-die" (BTYD)
models in [pymc-marketing](https://www.pymc-marketing.io/) consume: `ParetoNBDModel` (churn and
purchase frequency) and `GammaGammaModel` (per-transaction spend). It does not fit the models and does
not depend on pymc-marketing.

Each customer is "born" at their first purchase in the data and observed until the
`observation_period_end`. A purchase occasion is a distinct calendar day with positive net spend, so
multiple baskets on the same day count once and a returns-only day (net spend zero or less) is not an
occasion. Undated rows are also dropped, and a customer left with no occasion drops out entirely.

| Column | Meaning |
| --- | --- |
| `customer_id` | Your configured customer id column. |
| `frequency` | Number of *repeat* purchase occasions (distinct positive-spend days minus one). |
| `recency` | Time from the first purchase to the last purchase, in `period` units. |
| `T` | Customer age: time from the first purchase to `observation_period_end`, in `period` units. |
| `monetary_value` | Mean spend across the *repeat* purchases; `NaN` for one-time buyers. |

`recency` and `T` are fractional (for example, 2.5 weeks). The elapsed time is measured in whole days
and divided by the days-per-period, because some backends (notably Oracle) support only
day-granularity date deltas:

$$
T = \frac{\text{days}(\text{first purchase} \rightarrow \text{observation end})}{\text{days per period}}
\qquad
\text{recency} = \frac{\text{days}(\text{first purchase} \rightarrow \text{last purchase})}{\text{days per period}}
$$

`period` may be `"day"`, `"week"`, or `"month"` (case-insensitive, with short forms like `"d"`/`"m"`
accepted). A month is a fixed 365.25/12-day unit, not a calendar month, so continuous age stays well
defined. Day, week, and month map to pymc-marketing's `"D"`/`"W"`/`"M"` time units.

`monetary_value` excludes the first purchase, matching the Gamma-Gamma assumption. One-time buyers
(`frequency` of 0) have no repeat spend, cannot be fit by the standard Gamma-Gamma model, and are
excluded from it (their expected spend falls back to the population mean). They remain valid rows for
the Pareto/NBD model.

The `clv.repeat_buyers` property returns the Gamma-Gamma-ready subset: the `frequency > 0` rows.
`monetary_value` is always positive here (an occasion is a positive-spend day), so no spend filter is
needed. It warns if `frequency` and `monetary_value` correlate (`|Pearson r| > 0.15`), which breaks
Gamma-Gamma's independence assumption and biases its spend estimates.

`frequency`, `recency`, `T`, and `monetary_value` are pymc-marketing's required literal column names,
so the frame can be passed straight to the models.

Example:

```python
import pandas as pd
from openretailscience.experimental.clv import CLVStats

transactions = pd.DataFrame({
    "customer_id": [101, 101, 101, 102, 103, 103],
    "transaction_date": pd.to_datetime(
        ["2023-01-01", "2023-01-08", "2023-01-15", "2023-01-01", "2023-01-10", "2023-01-24"]
    ),
    "unit_spend": [100.0, 50.0, 70.0, 200.0, 50.0, 80.0],
})

# .df rows are one-per-customer in engine order; sort for a stable view.
clv = CLVStats(transactions, period="week", observation_period_end="2023-01-29")
print(clv.df.sort_values("customer_id").reset_index(drop=True))
#    customer_id  frequency  recency         T  monetary_value
# 0          101          2      2.0  4.000000            60.0
# 1          102          0      0.0  4.000000             NaN
# 2          103          1      2.0  2.714286            80.0
```

The summary feeds pymc-marketing directly (install it separately):

```python
from pymc_marketing.clv import ParetoNBDModel, GammaGammaModel

clv = CLVStats(transactions, period="week", observation_period_end="2023-01-29")
pareto = ParetoNBDModel(data=clv.df) # frequency, recency, T
pareto.fit()

gamma_gamma = GammaGammaModel(data=clv.repeat_buyers) # frequency > 0
gamma_gamma.fit()
```

<!-- markdownlint-disable MD046 -->
!!! warning "Finite-horizon CLV: pass `time_unit`, or it fails silently"
    `GammaGammaModel.expected_customer_lifetime_value(future_t=12, time_unit=...)` reads `future_t` in
    **months**. The default `time_unit="D"` is wrong for a weekly or monthly summary and scales the
    horizon by ~7x (weekly) to ~30x (monthly) with **no error raised**. Pass `time_unit=clv.pymc_time_unit`:

    ```python
    clv_12mo = gamma_gamma.expected_customer_lifetime_value(
        transaction_model=pareto, data=clv.df, future_t=12, time_unit=clv.pymc_time_unit,
    )
    ```

    Want an undiscounted CLV over a fixed horizon set directly in the model's units (e.g. exactly the
    next 52 weeks)?
    `pareto.expected_purchases(data=clv.df, future_t=52) * gamma_gamma.expected_customer_spend(data=clv.repeat_buyers)`
    computes it with no month conversion and no `time_unit`.
<!-- markdownlint-enable MD046 -->

## Extra columns and covariates

`customer_attributes` attaches a caller-supplied per-customer table (one row per `customer_id`) to the
summary via a left join: customer descriptors (signup channel, region) or pre-computed aggregates
(stores shopped). Build it however you like, e.g. `SegTransactionStats` grouped on `customer_id`, or a
plain `group_by`. `one_hot_col` names column(s) of that table to one-hot encode into `0`/`1` dummy
columns suitable for `ParetoNBDModel` covariates, and the original column is removed.

```python
transactions = pd.DataFrame({...})  # customer_id, transaction_date, unit_spend

# One row per customer: a pre-computed count joined as-is, plus a categorical to one-hot encode.
customer_attributes = pd.DataFrame({
    "customer_id": [101, 102, 103, 104],
    "stores_shopped": [3, 1, 2, 1],
    "signup_channel": ["email", "paid_search", "organic", None],  # None -> the dropped reference level
})

clv = CLVStats(
    transactions, period="week", customer_attributes=customer_attributes, one_hot_col="signup_channel",
)
# adds: stores_shopped, signup_channel_email, signup_channel_organic, signup_channel_paid_search

# covariate_cols lists the attached columns (stores_shopped + the one-hot dummies), no prefix matching.
covariate_cols = clv.covariate_cols
pareto = ParetoNBDModel(
    data=clv.df,
    model_config={"purchase_covariate_cols": covariate_cols, "dropout_covariate_cols": covariate_cols},
)
```

`customer_attributes` must have one row per `customer_id`. When it is an Ibis table ensure it is on
the same backend as the transactions data. A pandas frame also works, but do not mix live
backends (e.g. BigQuery transactions with a separate DuckDB attributes file).

Every customer needs a row in `customer_attributes`: a missing one leaves NULL covariates (one-hot
dummies included) that silently break `ParetoNBDModel`'s fit. A NULL attribute *value* is a NULL
covariate too, except in a one-hot source column, where NULL is just the reference level (all dummies 0).
`CLVStats` rejects NULL covariates only on `.df`; feed `.table` to the model and you must check yourself.

For each one-hot column, one level is dropped as the reference level, because a full set of
dummies would be collinear with the model intercept: NULL when the column contains NULLs, otherwise
the first value in sorted order. Values are used verbatim in the column names, so keep categorical
values short and free of characters you would not want in a column name. A column producing more than 32
dummies emits a `UserWarning` (one-hot is for low-cardinality categoricals, not IDs). A single-category
column produces zero dummies and also warns, since it contributes no covariate. `GammaGammaModel` has no covariate
support in pymc-marketing, so spend-by-channel needs a separate Gamma-Gamma fit per channel.

!!! note "Truncated history"
    The first purchase is always the earliest transaction *in the supplied data*, exactly as
    `pymc_marketing.clv.rfm_summary` treats it. A customer who was already active before the data
    window is treated as born at the window's start, which understates their true age. This is a
    limitation of a truncated observation period, not something `CLVStats` corrects. Substituting an
    earlier first-purchase date while the intervening transactions are missing would break the model's
    frequency/recency/T consistency and silently deflate the estimated purchase rate.
