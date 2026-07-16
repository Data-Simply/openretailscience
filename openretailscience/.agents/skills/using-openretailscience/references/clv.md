# Customer lifetime value (BTYD) reference

`openretailscience.experimental.clv.CLVStats` prepares the per-customer summary that the
"buy-till-you-die" models in [pymc-marketing](https://www.pymc-marketing.io/) consume —
`ParetoNBDModel` (churn + purchase frequency) and `GammaGammaModel` (per-transaction
spend). It **prepares the input; it does not fit the models** and does not depend on
pymc-marketing. Flagged experimental — the API may change.

```python
from openretailscience.experimental.clv import CLVStats

summary = CLVStats(data, period="week", observation_period_end=None).df
```

- `CLVStats(data, period="week", observation_period_end=None)` — reads `customer_id`,
  `transaction_date` (must be a date/datetime type), and `unit_spend` from the options
  system. Read `.table` (Ibis) / `.df` (pandas). `help(CLVStats)` for full Args/Raises.

## Output columns

One row per customer, using pymc-marketing's **required literal** column names (not
options.py names). Each customer is "born" at their first purchase in the data:

| column | meaning |
| --- | --- |
| `customer_id` | your configured id column |
| `frequency` | number of *repeat* purchase occasions (distinct purchase days minus one) |
| `recency` | first-purchase → last-purchase, in `period` units |
| `T` | customer age: first-purchase → `observation_period_end`, in `period` units |
| `monetary_value` | mean spend across the *repeat* purchases; `NaN` for one-time buyers |

`recency` and `T` are fractional (e.g. `2.5` weeks). Same-day baskets count as one
occasion. `monetary_value` excludes the first purchase (the Gamma-Gamma convention), so
one-time buyers (`frequency == 0`) have no value and are excluded from the Gamma-Gamma fit
— filter `summary[summary["frequency"] > 0]` for that model.

## Arguments

- `period` — `"day"` or `"week"` only (the units for which a fractional age is well
  defined). Fixes the model's time unit; keep it consistent when later converting a
  business horizon (e.g. 12 months → 52 weeks).
- `observation_period_end` — ISO-8601 string or `datetime.date`; defaults to the latest
  transaction date in the data. Must be on or after every customer's last purchase.

## Feeding pymc-marketing

```python
from pymc_marketing.clv import ParetoNBDModel, GammaGammaModel

pareto = ParetoNBDModel(data=summary)                       # frequency, recency, T
pareto.fit()

gamma_gamma = GammaGammaModel(data=summary[summary["frequency"] > 0])   # frequency, monetary_value
gamma_gamma.fit()
```

If you have overridden `column.customer_id`, give the downstream model the matching id
column name.

## Caveat: truncated history

The first purchase is always the earliest transaction **in the supplied data**, exactly as
`pymc_marketing.clv.rfm_summary` treats it. A customer who was already active before the
data window is treated as born at the window's start, which understates their true age.
This is a limitation of a truncated observation period, not something `CLVStats` corrects —
substituting an earlier first-purchase date from a customer dimension while the intervening
transactions are missing would break the model's frequency/recency/T consistency and
silently deflate the estimated purchase rate.
