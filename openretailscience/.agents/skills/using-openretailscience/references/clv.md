# Customer lifetime value (BTYD) reference

`openretailscience.experimental.clv.CLVStats` prepares the per-customer summary that the
"buy-till-you-die" models in [pymc-marketing](https://www.pymc-marketing.io/) consume —
`ParetoNBDModel` (churn + purchase frequency) and `GammaGammaModel` (per-transaction
spend). It **prepares the input; it does not fit the models** and does not depend on
pymc-marketing. Flagged experimental — the API may change.

```python
from openretailscience.experimental.clv import CLVStats

clv = CLVStats(data, period="week", observation_period_end=None)
```

Reads `customer_id`, `transaction_date` (must be a date/datetime type), and `unit_spend`
from the options system. Undated rows and returns-only days (net spend zero or less) are dropped
(a customer left with none drops out). Read `.table` (Ibis) / `.df` (pandas); `.repeat_buyers` is the
GammaGamma-ready subset (`frequency > 0`); `.sample()` draws a customer subset to fit on;
`.covariate_cols`
lists the attached customer_attributes / one-hot columns; `.pymc_time_unit` is the matching
 pymc-marketing `time_unit`. `help(CLVStats)` for full Args/Raises.

## Output columns

One row per customer, using pymc-marketing's **required literal** column names (**not** your
configured options.py names). Each customer is "born" at their first purchase in the data:

| column | meaning |
| --- | --- |
| `customer_id` | your configured id column |
| `frequency` | number of *repeat* purchase occasions (distinct positive-spend days minus one) |
| `recency` | first-purchase → last-purchase, in `period` units |
| `T` | customer age: first-purchase → `observation_period_end`, in `period` units |
| `monetary_value` | mean spend across the *repeat* purchases; `NaN` for one-time buyers |

`recency` and `T` are fractional (e.g. `2.5` weeks). Same-day baskets count as one occasion; a day
whose net spend is zero or less (returns) is not an occasion. `monetary_value` excludes the first
purchase (the Gamma-Gamma convention), so one-time buyers (`frequency == 0`) have no value. Use
`clv.repeat_buyers` for the Gamma-Gamma fit: the `frequency > 0` rows (one-time buyers fall back to
the population mean). It warns if `frequency` and `monetary_value` correlate (`|Pearson r| > 0.15`),
which breaks GammaGamma's independence assumption.

## Arguments

- `period` — `"day"`, `"week"`, or `"month"` (case-insensitive; short forms like `"d"`/`"m"`
  accepted). A month is a fixed 365.25/12-day unit, so continuous age stays well defined. Fixes the
  model's time unit; pass `clv.pymc_time_unit` downstream (day/week/month map to `"D"`/`"W"`/`"M"`).
- `observation_period_end` — ISO-8601 string or `datetime.date`; defaults to the latest
  transaction date in the data. Must be on or after every customer's last purchase.
- `customer_attributes` — a pandas/Ibis table, one row per `customer_id`, of extra columns to
  left-join onto the summary (covariates such as signup channel, region, or a pre-computed
  `stores_shopped` count). Build it however you like (e.g. `SegTransactionStats` grouped on
  `customer_id`). Must have a unique `customer_id` and a row per customer: a missing customer leaves NULL
  covariates (one-hot dummies included) that silently break ParetoNBD. A NULL attribute *value* counts
  too, except a one-hot source column (NULL = reference level, all dummies 0). Rejected on `.df` only;
  check yourself if you consume `.table`. Joins in-database when it is an Ibis table on the same backend
  as the transactions (a pandas frame also works, but don't mix live backends).
- `one_hot_col` — `str` or list of columns **of `customer_attributes`** to one-hot encode into
  `{col}_{value}` 0/1 dummy columns for `ParetoNBDModel` covariates. One level is dropped as the
  reference (collinear with the intercept otherwise): NULL when the column has NULLs, else the first
  value in sorted order. The original column is dropped. Warns if a column yields more than 32 dummies
  (a likely high-cardinality mistake) or zero dummies (a single-category column, no covariate).

## Extra columns and covariates

```python
# One row per customer_id: a numeric covariate joined as-is + a categorical to one-hot encode.
customer_attributes = pd.DataFrame(
    {"customer_id": [...], "stores_shopped": [...], "signup_channel": [...]},  # None -> dropped reference
)
clv = CLVStats(
    data,
    period="week",
    customer_attributes=customer_attributes,
    one_hot_col="signup_channel", # -> signup_channel_<value> dummy columns, original dropped
)
covariate_cols = clv.covariate_cols # attached columns (customer_attributes + one-hot dummies)

from pymc_marketing.clv import ParetoNBDModel
ParetoNBDModel(data=clv.df, model_config={
    "purchase_covariate_cols": covariate_cols,
    "dropout_covariate_cols": covariate_cols,
})
```

Values are used verbatim in column names (same as ibis `pivot_wider`, no sanitisation).
`GammaGammaModel` has **no** covariate support in pymc-marketing, so for spend-by-channel, fit a
separate Gamma-Gamma per channel.

## Fitting on a sample, scoring everyone

`.sample()` returns **another `CLVStats`** over a random subset of customers, so the sample keeps
`.df`, `.repeat_buyers`, `.covariate_cols`, and `.pymc_time_unit`. Fit on the sample when the full
population is too large for MCMC; predict on the full `.df`.

```python
clv = CLVStats(data, period="week")
train = clv.sample(n=50_000)  # or frac=0.1; exactly one of the two

pareto = ParetoNBDModel(data=train.df)
pareto.fit()
gamma_gamma = GammaGammaModel(data=train.repeat_buyers)
gamma_gamma.fit()

everyone = pareto.expected_purchases(data=clv.df, future_t=52)  # score the full population
```

- `n` — exact number of customers; yields every customer if it exceeds the population (no error).
  Costs a backend top-N sort.
- `frac` — share in `(0, 1]`, drawn per customer, so the count lands *near* `frac` * population, not
  on it. One pushed-down predicate, no sort.
- `random_state` — defaults to `42`. Customers are selected by a deterministic hash of `customer_id`
  salted with it, so the same arguments always draw the same customers and a different `random_state`
  draws an independent sample.

Sampling the summary is equivalent to sampling customers before aggregating (a customer's row depends
only on their own transactions), so one aggregate pass serves both the fit and the scoring, and a
sampled row is the row that customer has in the full `.df`. Row order is not meaningful.

## Feeding pymc-marketing

```python
from openretailscience.experimental.clv import CLVStats
from pymc_marketing.clv import ParetoNBDModel, GammaGammaModel

clv = CLVStats(data)

pareto = ParetoNBDModel(data=clv.df) # frequency, recency, T
pareto.fit()

gamma_gamma = GammaGammaModel(data=clv.repeat_buyers) # frequency > 0
gamma_gamma.fit()
```

**Finite-horizon CLV: pass `time_unit`, or it fails silently.**
`GammaGammaModel.expected_customer_lifetime_value(future_t=12, time_unit=...)` reads `future_t` in
**months**; its default `"D"` is wrong for a weekly/monthly summary and scales the horizon ~7x (weekly)
to ~30x (monthly) with **no error**. Pass `time_unit=clv.pymc_time_unit`:

```python
clv_12mo = gamma_gamma.expected_customer_lifetime_value(
    transaction_model=pareto, data=clv.df, future_t=12, time_unit=clv.pymc_time_unit,
)
```

Want an undiscounted CLV over a fixed horizon set directly in the model's units (e.g. exactly the
next 52 weeks)?
`pareto.expected_purchases(data=clv.df, future_t=52) * gamma_gamma.expected_customer_spend(data=clv.repeat_buyers)`
computes it with no month conversion and no `time_unit`.

## Caveat: truncated history

The first purchase is always the earliest transaction **in the supplied data**, exactly as
`pymc_marketing.clv.rfm_summary` treats it. A customer who was already active before the
data window is treated as born at the window's start, which understates their true age.
This is a limitation of a truncated observation period, not something `CLVStats` corrects —
substituting an earlier first-purchase date from a customer dimension while the intervening
transactions are missing would break the model's frequency/recency/T consistency and
silently deflate the estimated purchase rate.
