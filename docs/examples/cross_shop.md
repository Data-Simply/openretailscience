---
title: Which Customers Buy Both? Cross-Shopping Analysis
description: >-
  How many customers buy from two or three categories, brands or stores, what customers who buy both are worth, and
  which buyers of one have not bought the other yet. Cross-shopping analysis in Python.
---

# Which Customers Buy Both? Cross-Shopping Analysis

The coffee team at a homewares retailer is planning a pod promotion. A coffee machine is a high-value sale that a
customer makes only every few years, while the pods that go in it are bought every few weeks and generate most of the
long-term revenue from a machine owner. The team suspects that many customers buy their machine from the retailer but
their pods elsewhere, and wants the promotion to win that pod spend back. The analysis needs to answer three
questions:

- How many machine buyers never buy pods from the retailer?
- How much more is a customer who buys both worth than one who buys only a machine?
- Which customers should receive the offer?

Cross-shopping analysis, which counts how many customers buy from two or three categories and in which combinations,
answers all three.

```python exec="on" session="cross_shop" workdir="docs/examples/data"
import base64
from io import BytesIO

import matplotlib.pyplot as plt


def show_table(df):
    dates = df.select_dtypes("datetime").columns
    print(df.assign(**{col: df[col].dt.date for col in dates}).to_markdown(index=False))


def show_figure(alt):
    # The docs build's notebook_assets hook moves the data URI into a linked file.
    buffer = BytesIO()
    plt.gcf().savefig(buffer, format="png", dpi=110)
    plt.close("all")
    print(f'<img alt="{alt}" src="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}">')
```

## Your transaction data

The analysis starts from transaction data, with one row per item sold, the customer who bought it and the item's
category. This example uses a synthetic year of sales from a coffee and kitchen
range, covering 5,000 customers.

```bash
pip install openretailscience
```

```python exec="on" source="above" session="cross_shop" workdir="docs/examples/data"
import pandas as pd

df = pd.read_parquet("coffee_transactions.parquet")
df.head(3)
show_table(df.head(3))  # markdown-exec: hide
```

If your columns have different names, rename them or [map your column names once](../getting_started/options_guide.md).
To run the same analysis on a table in your data warehouse rather than a pandas DataFrame, see
[Connecting to your data](../getting_started/connecting_to_data.md).

## Just over half of machine buyers never buy pods from the retailer

Keep the two categories in question and pass them to `CrossShop`. Without the filter, customers who bought neither
category would appear as a fourth group, and each customer's spend would include everything else they bought.

```python exec="on" source="above" session="cross_shop" workdir="docs/examples/data"
from openretailscience.analysis import CrossShop

coffee = df[df["category_name"].isin(["Coffee Machines", "Coffee Pods"])]
cs = CrossShop(
    df=coffee,
    group_1_col="category_name",
    group_1_val="Coffee Machines",
    group_2_val="Coffee Pods",
    labels=["Coffee machines", "Coffee pods"],
)
```

`cs.cross_shop_df` has one row per customer, showing which of the two categories each customer bought and their
combined machine and pod spend for the year. Grouping it gives the number of customers in each combination and their
average spend.

```python exec="on" source="above" session="cross_shop" workdir="docs/examples/data"
summary = cs.cross_shop_df.groupby("group_labels")["unit_spend"].agg(customers="count", avg_spend="mean")
summary
show_table(summary.round(2).reset_index())  # markdown-exec: hide
```

404 of the 771 machine buyers (52%) never bought pods from the retailer. Customers who bought both spent £253.12 on
average on machines and pods over the year, £78 more than the £175.61 spent by machine-only customers. A further 524
customers bought pods but no machine this year.

## Customers who buy both bring in 45% of machine and pod sales

Every customer in the filtered data bought at least one of the two categories, so the shares add up to 100%.

```python exec="on" source="above" session="cross_shop" workdir="docs/examples/data"
cs.plot(
    eyebrow="Coffee customers, 2025",
    title="Customers who buy both bring in 45% of machine and pod sales",
    subtitle="Share of machine and pod sales in each combination",
    subset_label_formatter=lambda x: f"{x:.0%}",
)
show_figure(  # markdown-exec: hide
    "Venn diagram of machine and pod sales. Customers who buy both bring in 45%, "  # markdown-exec: hide
    "machine-only customers 35% and pod-only customers 20%."  # markdown-exec: hide
)  # markdown-exec: hide
```

Customers who buy both are only 28% of the 1,295 customers who bought either category.

## The pod offer list contains 404 customers

Rename the category columns to readable names, then keep the machine buyers who have never bought pods.

```python exec="on" source="above" session="cross_shop" workdir="docs/examples/data"
customers = cs.cross_shop_df.rename(columns={"group_1": "machines", "group_2": "pods"})
offer_list = customers[(customers["machines"] == 1) & (customers["pods"] == 0)]
len(offer_list)
print(f"```\n{len(offer_list)}\n```")  # markdown-exec: hide
```

Two checks should come before the offer. Customers who bought their machine in the last weeks of the year have had
little time to buy pods, so leave recent machine buyers out or repeat the analysis over a longer period. Pods also fit
only one machine system, so confirm that the range stocks pods for the machines these customers bought.

!!! success "Recommendation"
    Send a pod offer to the 404 machine buyers who have never bought pods from the retailer, after removing recent
    machine buyers and confirming pod compatibility. Customers who buy both are worth £78 a year more on average and
    already generate 45% of machine and pod sales. `offer_list["customer_id"]` is the customer list.

## Adapting the analysis

The same call answers related questions:

- To add a third category, such as descaler, pass `group_3_val`.
- To compare brands or stores, set `group_1_col="brand_name"` or `group_1_col="store_id"` and choose the values to
  compare. Each group can use a different column through `group_2_col` and `group_3_col`.
- To count customers instead of spend in the table and diagram, pass `value_col="customer_id"` and
  `agg_func="nunique"`.
- To look at shopping trips instead of customers, pass `group_col="transaction_id"`, `value_col="transaction_id"`
  and `agg_func="nunique"`. The result shows how often the categories appear in the same basket.
- To compare more than three brands, use a [cross-brand affinity table](brand_affinity.md), which covers every pair at
  once.

See also [Product Association](product_association.ipynb) and the
[`CrossShop` API reference](../api/analysis/cross_shop.md).
