# Analysis, segmentation & utils reference

Every class accepts a pandas `DataFrame` or an Ibis `Table` and reads column
names from the options system unless an argument overrides them. Results come
from a `.df` (pandas) and/or `.table` (Ibis) property; some classes also render
with `.plot()` / `.draw_tree()`. Call `help(Cls)` for the authoritative
Google-style Args/Returns/Raises — the summaries below list only the key
arguments.

## Segmentation (`openretailscience.segmentation`)

Transaction KPIs by segment or dimension:

```python
from openretailscience.segmentation.segstats import SegTransactionStats, cube, rollup
```

- `SegTransactionStats(data, segment_col="segment_name", grouping_sets=None, extra_aggs=None, ...)` —
  spend/transactions/customers and ratios per segment. `grouping_sets` is
  `"total"`, `"rollup"`, `"cube"`, or a list of column tuples. `extra_aggs` is
  `{out: (col, agg)}`. Read `.table` / `.df` (RangeIndex, segment columns as columns, row order unspecified
  unless `sort_by=True`; columns a grouping set omits hold `rollup_value`; zero-denominator ratios are null).
  With `unknown_customer_value`, plain metrics cover identified customers only, and spend/transactions/units
  (+ their ratios and extra_aggs) gain `_unknown`/`_total` variants; customers do not.
- `cube(*columns)` → all 2ⁿ grouping-set tuples; `rollup(*columns)` → n+1
  hierarchical tuples.

`grouping_sets` worked examples (a set is a tuple of `segment_col` columns; `()` is the
grand total):

- `"rollup"` — hierarchical subtotals over `segment_col`: `[A,B,C], [A,B], [A], ()`.
- `"cube"` — all 2ⁿ combinations of `segment_col`, including the `()` grand total.
- `"total"` — the full `[A,B,C]` set plus the `()` grand total.
- `[("region", "product"), ("product",), ()]` — custom list of tuples (`()` is the grand total).
- `[(cube("store", "region"), "date")]` — nested spec: one `cube()`/`rollup()` result per
  tuple plus str fixed columns, which are appended as a suffix to every set (produces
  `[("store", "region", "date"), ("store", "date"), ("region", "date"), ("date",)]`).

`calc_total`/`calc_rollup` are deprecated: a `FutureWarning` is emitted when
`grouping_sets=None` and both are omitted. Legacy `calc_rollup=True` generates prefix AND
suffix rollups, and suffix rollups are generated only when `calc_total=True`.

Customer segments:

```python
from openretailscience.segmentation.rfm import RFMSegmentation
from openretailscience.segmentation.hml import HMLSegmentation
from openretailscience.segmentation.threshold import ThresholdSegmentation
from openretailscience.segmentation.nlr import NLRSegmentation
```

- `RFMSegmentation(df, current_date=None, r_segments=10, f_segments=10, m_segments=10, ...)` —
  Recency/Frequency/Monetary; `.df` is indexed by `customer_id` with `recency_days`, `frequency`, `monetary`,
  `r_score`/`f_score`/`m_score` (0-based, higher is better: 0..n-1 for `*_segments=n`, 0..k for k cut points),
  `rfm_segment` = R*100+F*10+M and `fm_segment` = F*10+M.
- `HMLSegmentation(df, value_col=None, agg_func="sum", ...)` — Heavy/Medium/Light: bottom 50% / next 30% /
  top 20% of customers by spend (percentile ranks, not spend values); subclass of `ThresholdSegmentation`.
- `ThresholdSegmentation(df, thresholds, segments, value_col=None, ...)` — generic
  percentile thresholds, strictly ascending and ending at `1.0` (else `ValueError`);
  `.df` is indexed by `customer_id` (+ `group_col`) with `segment_name`. Tied values get distinct ranks by
  `customer_id`, so equal spenders can split across segments. `zero_value_customers`: `"separate_segment"`
  (own segment, unranked), `"exclude"` (dropped), `"include_with_light"` (ranked with everyone; tied zeros
  can spill past the first segment).
- `NLRSegmentation(df, period_col, p1_value, p2_value, ...)` — New / Lapsed /
  Repeating across two periods. With `count`/`nunique` the positive-value rule measures
  transaction presence, not spend.

## Analysis (`openretailscience.analysis`)

```python
from openretailscience.analysis.cross_shop import CrossShop
from openretailscience.analysis.gain_loss import GainLoss
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
  summed `value_col`, spend by default; use `value_col="customer_id", agg_func="nunique"` for customer shares);
  `.plot(vary_size=False, ...)` → Venn.
- `GainLoss(df, p1_index, p2_index, focus_group_index, focus_group_name, comparison_group_index, ...)` —
  period/group index args are boolean lists, arrays or Series with one entry per row of `df` (matched by
  position; a Series must have `df`'s index, so build it from `df`). `.gain_loss_df` is indexed by
  `customer_id` (`(group_col, customer_id)` when grouped); its six components sum to `focus_diff`.
  `.gain_loss_table_df` is one row ungrouped; `.plot(...)`.
- `CohortAnalysis(df, aggregation_column, agg_func="nunique", period="month", percentage=False)` —
  `period` ∈ year/quarter/month/week/day; `transaction_date` must be date or datetime (strings aren't
  parsed). For retention use
  `aggregation_column="customer_id"` (nunique on a value column counts distinct values). `.df` is the cohort
  matrix: index = first-purchase period, int columns = periods since (empty cells 0, not necessarily square);
  `percentage=True` gives fractions of period 0 rounded to 2 (pair with `plots.cohort.plot`).
- `ProductAssociation(df, value_col, group_col=None, target_item=None, min_support=0.0, ...)` —
  `.table` / `.df` with support, confidence, uplift. `group_col` is the basket: the default `customer_id`
  pools all of a customer's purchases; pass `group_col="transaction_id"` for per-transaction baskets.
- `RevenueTree(df, period_col, p1_value, p2_value, group_col=None)` — `period_col` holds a
  caller-assigned period label per row (not a raw date); both values must exist and differ. `.df` is the
  KPI table, one row per group (`_pct_diff` columns are fractions); `.draw_tree(row_index=0, ...)` → tree
  diagram. `calc_tree_kpis(p1_df, p2_df)` computes the same table from pre-aggregated per-period frames
  (customers/transactions/spend[/units] columns, one row per group), pairing rows by index label; for an
  ungrouped comparison give both one row labelled 0.
- `CustomerDecisionHierarchy(df, product_col, method="yules_q", ...)` —
  `.pairs_df`, `.distances` ([0, 1], ordered by `pairs_df[product_col].cat.categories`; 0 = every buyer of
  one also buys the other, 1 = never the same buyer, or no customer bought neither; by default near =
  substitutes); `.plot(...)` → dendrogram.
- `PurchasesPerCustomer(df)` — `.df` (`purchase_count`); `.purchases_percentile(0.5)`,
  `.find_purchase_percentile(n, comparison="less_than_equal_to")`.
- `DaysBetweenPurchases(df)` — `.df` (`avg_days_between_purchases`);
  `.purchases_percentile(0.5)`.
- `TransactionChurn(df, churn_period)` — `.df` by 1-based purchase-day number (`retained`,
  `churned`, `churned_pct` = churned / (churned + retained)); row N counts only customers whose N-th purchase
  day is more than `churn_period` days before the last day in the data. `.n_unique_customers`.
- `CompositeRank(df, rank_cols, agg_func, ignore_ties=False, group_col=None)` —
  `rank_cols` is a list of `(col, "asc"|"desc")`; `.df` ranked. Ranks are 1-based,
  1 = best per sort direction; ties share the lowest rank; nulls rank last (first, i.e. best, in an "asc"
  column on BigQuery); lower `composite_rank` is better. Row order is unspecified, so sort by `composite_rank`.
- `haversine_distance(lat_col, lon_col, target_lat_col, target_lon_col, radius=6371.0)` —
  returns an Ibis Column expression (not materialized).

## Utils (`openretailscience.utils`)

```python
from openretailscience.utils.date import filter_and_label_by_periods, find_overlapping_periods
from openretailscience.utils.filter_and_label import filter_and_label_by_condition
from openretailscience.utils.label import label_by_condition
```

- `filter_and_label_by_periods(transactions, period_ranges, period_col="period_name")` —
  filter an Ibis table to named inclusive `{name: (start, end)}` ranges and tag each row;
  validates non-overlap. A `"YYYY-MM-DD"` string or `date` end includes that whole day; on a timestamp column a
  datetime end is an exact instant, so `datetime(2023, 3, 31)` drops sales later on Mar 31.
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
