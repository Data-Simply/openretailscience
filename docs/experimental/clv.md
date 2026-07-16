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
models in [pymc-marketing](https://www.pymc-marketing.io/) consume — `ParetoNBDModel` (churn and
purchase frequency) and `GammaGammaModel` (per-transaction spend). It prepares the model **input**;
it does not fit the models and does not depend on pymc-marketing.

Each customer is "born" at their first purchase in the data and observed until the
`observation_period_end`. A purchase occasion is a distinct calendar day, so multiple baskets on the
same day count once.

| Column | Meaning |
| --- | --- |
| `customer_id` | Your configured customer id column. |
| `frequency` | Number of *repeat* purchase occasions (distinct purchase days minus one). |
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

`monetary_value` excludes the first purchase, matching the Gamma-Gamma assumption. One-time buyers
(`frequency` of 0) have no repeat spend, cannot be fit by the standard Gamma-Gamma model, and are
excluded from it (their expected spend falls back to the population mean). They remain valid rows for
the Pareto/NBD model.

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
summary = CLVStats(transactions, period="week", observation_period_end="2023-01-29").df
print(summary.sort_values("customer_id").reset_index(drop=True))
#    customer_id  frequency  recency         T  monetary_value
# 0          101          2      2.0  4.000000            60.0
# 1          102          0      0.0  4.000000             NaN
# 2          103          1      2.0  2.714286            80.0
```

The summary feeds pymc-marketing directly (install it separately):

```python
from pymc_marketing.clv import ParetoNBDModel, GammaGammaModel

pareto = ParetoNBDModel(data=summary)                              # frequency, recency, T
pareto.fit()

repeat_buyers = summary[summary["frequency"] > 0]
gamma_gamma = GammaGammaModel(data=repeat_buyers)                  # frequency, monetary_value
gamma_gamma.fit()
```

!!! note "Truncated history"
    The first purchase is always the earliest transaction *in the supplied data*, exactly as
    `pymc_marketing.clv.rfm_summary` treats it. A customer who was already active before the data
    window is treated as born at the window's start, which understates their true age. This is a
    limitation of a truncated observation period, not something `CLVStats` corrects — substituting an
    earlier first-purchase date while the intervening transactions are missing would break the model's
    frequency/recency/T consistency and silently deflate the estimated purchase rate.
