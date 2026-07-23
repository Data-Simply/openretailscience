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
  `{out: (col, agg)}`. Read `.table` / `.df`.
- `cube(*columns)` → all 2ⁿ grouping-set tuples; `rollup(*columns)` → n+1
  hierarchical tuples.

Customer segments:

```python
from openretailscience.segmentation.rfm import RFMSegmentation
from openretailscience.segmentation.hml import HMLSegmentation
from openretailscience.segmentation.threshold import ThresholdSegmentation
from openretailscience.segmentation.nlr import NLRSegmentation
```

- `RFMSegmentation(df, current_date=None, r_segments=10, f_segments=10, m_segments=10, ...)` —
  Recency/Frequency/Monetary; `.df` segment = R*100+F*10+M.
- `HMLSegmentation(df, value_col=None, agg_func="sum", ...)` — Heavy/Medium/Light
  (fixed 50/80/100 spend cuts); subclass of `ThresholdSegmentation`.
- `ThresholdSegmentation(df, thresholds, segments, value_col=None, ...)` — generic
  percentile thresholds; `.df` has `segment_name`.
- `NLRSegmentation(df, period_col, p1_value, p2_value, ...)` — New / Lapsed /
  Repeating across two periods.

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
  `.cross_shop_df`, `.cross_shop_table_df`; `.plot(vary_size=False, ...)` → Venn.
- `GainLoss(df, p1_index, p2_index, focus_group_index, focus_group_name, comparison_group_index, ...)` —
  period/group index args are boolean Series/masks. `.gain_loss_df`,
  `.gain_loss_table_df`; `.plot(...)`.
- `CohortAnalysis(df, aggregation_column, agg_func="nunique", period="month", percentage=False)` —
  `period` ∈ year/quarter/month/week/day. `.df` is the cohort matrix (pair with
  `plots.cohort.plot`).
- `ProductAssociation(df, value_col, group_col=None, target_item=None, min_support=0.0, ...)` —
  `.table` / `.df` with support, confidence, uplift.
- `RevenueTree(df, period_col, p1_value, p2_value, group_col=None)` — `.df` KPI
  table; `.draw_tree(row_index=0, ...)` → tree diagram. `calc_tree_kpis(df, p1_index, p2_index)`
  is the functional core returning a pandas DataFrame.
- `CustomerDecisionHierarchy(df, product_col, method="yules_q", ...)` —
  `.pairs_df`, `.distances`; `.plot(...)` → dendrogram.
- `PurchasesPerCustomer(df)` — `.df` (`purchase_count`); `.purchases_percentile(0.5)`,
  `.find_purchase_percentile(n, comparison="less_than_equal_to")`.
- `DaysBetweenPurchases(df)` — `.df` (`avg_days_between_purchases`);
  `.purchases_percentile(0.5)`.
- `TransactionChurn(df, churn_period)` — `.df` by transaction number (`retained`,
  `churned`, `churned_pct`), `.n_unique_customers`.
- `CompositeRank(df, rank_cols, agg_func, ignore_ties=False, group_col=None)` —
  `rank_cols` is a list of `(col, "asc"|"desc")`; `.df` ranked.
- `haversine_distance(lat_col, lon_col, target_lat_col, target_lon_col, radius=6371.0)` —
  returns an Ibis Column expression (not materialized).

## Utils (`openretailscience.utils`)

```python
from openretailscience.utils.date import filter_and_label_by_periods, find_overlapping_periods
from openretailscience.utils.filter_and_label import filter_and_label_by_condition
from openretailscience.utils.label import label_by_condition
```

- `filter_and_label_by_periods(transactions, period_ranges, period_col="period_name")` —
  filter an Ibis table to named `{name: (start, end)}` ranges and tag each row;
  validates non-overlap.
- `find_overlapping_periods(start_date, end_date, return_str=True)` — split a range
  into year-aligned `(start, end)` tuples for `plots.period_on_period.plot`.
- `filter_and_label_by_condition(table, conditions, label_col="label")` — keep rows
  matching any `{label: ibis boolean expr}` and tag with its label.
- `label_by_condition(table, condition, label_col=None, labeling_strategy="binary", ...)` —
  label each `label_col` group (default customer) by whether its items meet a condition.

## Experimental (`openretailscience.experimental`)

Flagged experimental — APIs may change. `experimental.metrics.base.ratio_metric(...)`,
and the distribution metrics `experimental.metrics.distribution.acv.Acv` /
`experimental.metrics.distribution.pct_of_stores.PctOfStores`. Also
`experimental.cache.cache(expr)` — a generic caching helper mirroring Ibis's `Table.cache()`
that transparently works around the broken native cache on Spark Connect (Databricks); see
`references/configuration.md`.
