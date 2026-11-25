# Analysis, segmentation & utils reference

Every class accepts a pandas `DataFrame` or an Ibis `Table` and reads column
names from the options system unless an argument overrides them. Results come
from a `.df` (pandas) and/or `.table` (Ibis) property; some classes also render
with `.plot()` / `.draw_tree()`. Result frames carry their keys (`customer_id`, `group_col`, ...) as
columns on a RangeIndex; merge on them (`on="customer_id"`).
Call `help(Cls)` for the authoritative
Google-style Args/Returns/Raises — the summaries below list only the key
arguments.

## Segmentation (`openretailscience.segmentation`)

Transaction KPIs by segment or dimension:

```python
from openretailscience.segmentation.segstats import SegTransactionStats, cube, rollup
```

- `SegTransactionStats(data, *, segment_col="segment_name", grouping_sets=None, extra_aggs=None, ...)` —
  spend/transactions/customers and ratios per segment. `grouping_sets` is
  `"total"`, `"rollup"`, `"cube"`, or a list of column tuples. `extra_aggs` is
  `{out: (col, agg)}`. Read `.table` / `.df` (RangeIndex, segment columns as columns, row order unspecified
  unless `sort_by=True`; columns a grouping set omits hold `rollup_value`; zero-denominator ratios are null).
  Rolled-up rows are labeled `"Total"` by default, which only fits string columns: a rolled-up int or boolean
  segment column raises unless you pass a `rollup_value` of its type. `rollup_value` requires `grouping_sets`.
  With the `data.unknown_customer_value` option set (`references/configuration.md`), plain metrics cover identified
  customers only, and spend/transactions/units (+ their ratios and extra_aggs) gain `_unknown`/`_total` variants;
  customers do not.
  For a trend, pass `expected_segments`: a pandas DataFrame of the combinations that must appear, with exactly the
  `segment_col` columns (e.g. `categories.merge(weeks, how="cross")`, weeks from a calendar, not the data).
  Missing combinations get rows with 0 for counts/sums (incl. extra_aggs `sum`/`count`/`nunique`/`approx_nunique`)
  and null for ratios and other aggregations; don't pad `.df` yourself. Subtotals get the listed values of their own
  columns (a weekly total covers every listed week); list detail combinations only, never rollup labels. A
  combination with a null isn't padded.
- `cube(*columns)` → all 2ⁿ grouping-set tuples; `rollup(*columns)` → n+1
  hierarchical tuples.

`grouping_sets` worked examples (a set is a tuple of `segment_col` columns; `()` is the
grand total):

- `None` (default) — one row per `segment_col` combination, no subtotals or total.
- `"rollup"` — hierarchical subtotals over `segment_col`: `[A,B,C], [A,B], [A], ()`.
- `"cube"` — all 2ⁿ combinations of `segment_col`, including the `()` grand total.
- `"total"` — the full `[A,B,C]` set plus the `()` grand total.
- `[("region", "product"), ("product",), ()]` — custom list of tuples (`()` is the grand total).
- `[(cube("store", "region"), "date")]` — nested spec: one `cube()`/`rollup()` result per
  tuple plus str fixed columns, which are appended as a suffix to every set (produces
  `[("store", "region", "date"), ("store", "date"), ("region", "date"), ("date",)]`).

Customer segments:

```python
from openretailscience.segmentation.rfm import RFMSegmentation
from openretailscience.segmentation.hml import HMLSegmentation
from openretailscience.segmentation.threshold import ThresholdSegmentation
from openretailscience.segmentation.nlr import NLRSegmentation
```

- `RFMSegmentation(df, current_date=None, r_segments=10, f_segments=10, m_segments=10, ...)` —
  Recency/Frequency/Monetary; `transaction_date` must be date or datetime (string columns are rejected;
  `current_date` may be an ISO string). `.df` has `customer_id`, `recency_days`, `frequency`, `monetary`,
  `r_score`/`f_score`/`m_score` (0-based, higher is better: 0..n-1 for `*_segments=n`, 0..k for k cut points),
  `rfm_segment` = R*100+F*10+M and `fm_segment` = F*10+M.
- `HMLSegmentation(df, value_col=None, agg_func="sum", ...)` — Heavy/Medium/Light: bottom 50% / next 30% /
  top 20% of customers by spend (percentile ranks, not spend values); subclass of `ThresholdSegmentation`.
- `ThresholdSegmentation(df, thresholds, segments, value_col=None, ...)` — generic
  percentile thresholds, strictly ascending and ending at `1.0` (else `ValueError`);
  `.df` has `customer_id` (+ `group_col`), the aggregated `value_col` and `segment_name`. Tied values get
  distinct ranks by `customer_id`, so equal spenders can split across segments. `zero_value_customers`:
  `"separate_segment"` (own segment, unranked), `"exclude"` (dropped), `"include_with_light"` (ranked with
  everyone; tied zeros can spill past the first segment).
- `NLRSegmentation(df, period_col, p1_value, p2_value, ...)` — New / Lapsed /
  Repeating across two periods. With `count`/`nunique` the positive-value rule measures
  transaction presence, not spend.

## Analysis (`openretailscience.analysis`)

```python
from openretailscience.analysis.cross_shop import CrossShop
from openretailscience.analysis.gain_loss import GainLoss
from openretailscience.analysis.index_analysis import IndexAnalysis
from openretailscience.analysis.cohort import CohortAnalysis
from openretailscience.analysis.product_association import ProductAssociation
from openretailscience.analysis.revenue_tree import RevenueTree, calc_tree_kpis
from openretailscience.analysis.customer_decision_hierarchy import CustomerDecisionHierarchy
from openretailscience.analysis.customer import PurchasesPerCustomer, DaysBetweenPurchases, TransactionChurn
from openretailscience.analysis.composite_rank import CompositeRank
from openretailscience.analysis.haversine import haversine_distance
```

- `CrossShop(df, group_1_col, group_1_val, group_2_val, ...)` — 2/3-way overlap.
  `.cross_shop_df` (one row per customer, incl. "No Groups"), `.cross_shop_table_df` (`percent` = fraction of
  summed `value_col`, spend by default; use `value_col="customer_id", agg_func="nunique"` for customer shares,
  which names the count column `customer_id_nunique` in both frames);
  `.plot(vary_size=False, ...)` → Venn.
- `GainLoss(df, period_col, p1_value, p2_value, compare_col, focus_value, comparison_value, group_col=None,
  value_col=None, agg_func="sum")` — `agg_func` is "sum", "count" or "nunique". To compare against several
  values (e.g. every other brand in the category) or show display names, label a column first with
  `filter_and_label_by_condition` and pass it as `compare_col`; label date ranges with
  `filter_and_label_by_periods` for `period_col`. `.customer_table` (lazy, unordered) has one row
  per `customer_id` (per `group_col` and `customer_id` when grouped); its six components sum to `focus_diff`.
  `.df` / `.table` are one row ungrouped, else one per `group_col` value (null is its own group); `.df` raises
  if a value matches no rows. Null customer IDs, and the `data.unknown_customer_value` ID when set, are left
  out. Values are netted per customer and period: a return of an item bought outside both periods shows as
  negative `new` or positive `lost`, so drop such returns or move them to the sale's period first. `.plot(...)`.
- `IndexAnalysis(df, value_col, group_col, index_col, value_to_index, agg_func="sum", subgroup_col=None)` —
  how over- or under-represented each `group_col` value is among the rows where `index_col == value_to_index`,
  against all rows: `index = 100 * subset_share / baseline_share`. Use it instead of computing shares and
  indexes by hand. For customers, pass `value_col="customer_id", agg_func="nunique"`. `.df` / `.table` have one
  row per baseline group (per `subgroup_col` and group pair present in the data, with no row for a pair that has
  no rows; shares are then within each subgroup):
  `subset_value`, `baseline_value`, `subset_share`, `baseline_share`, `index`. A group with no subset rows has
  index 0; one whose subset rows all have a null `value_col` has a null index (0 under "nunique"), as does one
  whose baseline share or subset total is 0. If no row with a group matches
  `value_to_index`, reading `.df` raises a ValueError: the value must exist in `index_col` exactly (case,
  spacing), or its rows have null groups to fill; a value of the wrong type (int on a text column) raises
  `IbisTypeError` on construction. Null `group_col`/`subgroup_col` rows leave both totals; fill
  them with a label first to count them. There is no `.plot()`; chart it with `plots.index.plot` and the same
  arguments (`series_col` for `subgroup_col`).
- `CohortAnalysis(df, aggregation_column, agg_func="nunique", period="month", percentage=False)` —
  `period` ∈ year/quarter/month/week/day; `transaction_date` must be date or datetime (string
  columns are rejected). For retention use
  `aggregation_column="customer_id"` (nunique on a value column counts distinct values). `.df` is the cohort
  matrix: `min_period_shopped` column = first-purchase period, then integer-labelled columns = periods since, from 0
  to the first cohort's age at the end of the data. A covered period with no buyers is 0; one past the end of
  the data is NaN, so `.melt(id_vars="min_period_shopped").dropna()` gives one row per covered cohort and period;
  `percentage=True` gives fractions of period 0 rounded to 2 (pair with `plots.cohort.plot`).
- `ProductAssociation(df, value_col, group_col=None, target_item=None, min_support=0.0, ...)` —
  `.table` / `.df` with support, confidence, uplift. `group_col` is the basket: the default `customer_id`
  pools all of a customer's purchases; pass `group_col="transaction_id"` for per-transaction baskets.
  `target_item` must match `value_col`'s type: integer IDs for a numeric column, strings otherwise.
- `RevenueTree(df, period_col, p1_value, p2_value, group_col=None)` — `period_col` holds a
  caller-assigned period label per row (not a raw date); both values must exist and differ. `.df` is the
  KPI table, one row per group, `group_col` columns first (`_pct_diff` columns are fractions);
  `.draw_tree(row_index=0, ...)` → tree diagram. `calc_tree_kpis(p1_df, p2_df, group_col=None)` computes the
  same table from pre-aggregated per-period frames (customers/transactions/spend[/units] columns, one row per
  group), pairing rows on `group_col`; for an ungrouped comparison give each one row.
  With the `data.unknown_customer_value` option set, plain metrics cover identified customers only, and
  `_unknown` and `_total` sets of transactions, spend[, units] and their per-transaction ratios follow, each
  with its own `_contrib` columns; `draw_tree` splits Revenue into Unidentified and Identified Revenue.
  `calc_tree_kpis` does the same when both frames also have the `_unknown` and `_total` transactions, spend[,
  units] columns.
- `CustomerDecisionHierarchy(df, product_col, method="yules_q", ...)` —
  `.pairs_df`, `.distances` ([0, 1], ordered by `pairs_df[product_col].cat.categories`; 0 = every buyer of
  one also buys the other, 1 = never the same buyer, or no customer bought neither; by default near =
  substitutes); `.plot(...)` → dendrogram.
- `PurchasesPerCustomer(df)` — `.df` (`customer_id`, `purchase_count`); `.purchases_percentile(0.5)`,
  `.find_purchase_percentile(n, comparison="less_than_equal_to")`.
- `DaysBetweenPurchases(df)` — `.df` (`customer_id`, `avg_days_between_purchases`);
  `.purchases_percentile(0.5)`.
- `TransactionChurn(df, churn_period)` — `.df` by 1-based purchase-day number (`transaction_number`, `retained`,
  `churned`, `churned_pct` = churned / (churned + retained)); row N counts only customers whose N-th purchase
  day is more than `churn_period` days before the last day in the data. `.n_unique_customers`.
- `CompositeRank(df, rank_cols, agg_func, ignore_ties=False, group_col=None)` —
  `rank_cols` is a list of `(col, "asc"|"desc")`; `.df` ranked. Ranks are 1-based,
  1 = best per sort direction; ties share the lowest rank; nulls rank last (first, i.e. best, in an "asc"
  column on BigQuery); lower `composite_rank` is better. Row order is unspecified, so sort by `composite_rank`.
- `haversine_distance(lat_col, lon_col, target_lat_col, target_lon_col, radius=6371.0)` —
  returns an Ibis Column expression (not materialized). For one fixed target point pass `ibis.literal(lat)` /
  `ibis.literal(lon)`; a bare float raises `AttributeError`.

## Utils (`openretailscience.utils`)

```python
from openretailscience.utils.date import (
    BEFORE_EVENT,
    filter_and_label_by_event_periods,
    filter_and_label_by_event_to_date,
    filter_and_label_by_periods,
    find_overlapping_periods,
)
from openretailscience.utils.filter_and_label import filter_and_label_by_condition
from openretailscience.utils.label import label_by_condition
```

- `filter_and_label_by_periods(transactions, period_ranges, period_col="period_name")` —
  filter a DataFrame or Ibis table to named inclusive `{name: (start, end)}` ranges and tag each row, returning an
  Ibis table; validates non-overlap. `transaction_date` must be a tz-naive date or datetime (string and tz-aware
  columns are rejected). A `"YYYY-MM-DD"` string or `date` end includes that whole day; on a timestamp column a
  datetime end is an exact instant, so `datetime(2023, 3, 31)` drops sales later on Mar 31. A tz-aware bound is read
  as wall-clock time (its offset is ignored).
- `filter_and_label_by_event_periods(transactions, events, on, windows, event_date_col="event_date",
  period_col="period_name")` — tag each row with a window around its own key's event date (each store's competitor
  opening, each customer's sign-up), returning an Ibis table. `events` has one row per `on` key (a column or a list
  of columns); a duplicate, null key or null date raises, checked by one query on `events` at call time. `windows`
  maps labels (all str or all int) to inclusive whole-day `(start, end)` `timedelta` offsets; overlaps raise. Windows
  that follow on end a day early:
  `wk, day = timedelta(weeks=1), timedelta(days=1)`, `{"pre": (-12 * wk, -day), "post": (0 * wk, 12 * wk - day)}`.
  Prior year: subtract `52 * wk` from both offsets. Weekly curve: `{n: (n * wk, (n + 1) * wk - day) for n in
  range(-12, 12)}` gives an int week label. Rows whose key has no event, or a null key, are dropped; a window the
  data only partly covers returns what it has, so check coverage first.
- `filter_and_label_by_event_to_date(transactions, events, on, end_date, windows, event_date_col="event_date",
  period_col="period_name", include_duration=False, duration_col="period_days")` — each key's test period runs from
  its own event date to one `end_date` (inclusive), so its length differs per key. `windows` maps labels to a
  whole-day `timedelta` added to both ends of the test period (`0 * wk` is the test period, `-52 * wk` the same
  weeks a year earlier), or `BEFORE_EVENT` for the same number of days ending the day before the event. Same-length
  windows overlap once the test period is long enough: with `BEFORE_EVENT` and `-52 * wk`, every event must be at
  most 181 days before `end_date`. One query on `events` at call time raises on that, on events after `end_date`, and on
  duplicate or null keys, naming the cutoff date. Filter early events out first to drop them knowingly.
  `include_duration=True` adds each row's period length in days. Use `filter_and_label_by_event_periods` instead when
  every key gets the same fixed-length windows.
- `find_overlapping_periods(start_date, end_date, return_str=True)` — split a range
  into year-aligned `(start, end)` tuples for `plots.period_on_period.plot`.
- `filter_and_label_by_condition(table, conditions, label_col="label")` — keep rows
  matching any `{label: ibis boolean expr}` and tag with its label.
- `label_by_condition(table, condition, label_col=None, labeling_strategy="binary", ...)` —
  label each `label_col` group (default customer) by whether its items meet a condition. Returns only
  `label_col` and `return_col` (one row per group); join back for other columns.

## Experimental (`openretailscience.experimental`)

Flagged experimental — APIs may change. `experimental.metrics.base.ratio_metric(...)`,
and the distribution metrics `experimental.metrics.distribution.acv.Acv` /
`experimental.metrics.distribution.pct_of_stores.PctOfStores` (`stores_pct` is 0-100, not a fraction).
