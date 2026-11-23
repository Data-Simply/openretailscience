---
title: Which Brands Share Customers? Building a Cross-Brand Affinity Table
description: >-
  Which brands share customers? Build a brand affinity table with overlap percentages and an affinity index in Python,
  to judge the risk of delisting a brand and where its customers would go.
---

# Which Brands Share Customers? Building a Cross-Brand Affinity Table

A department store's jeans fixture has room for four brands next season but currently carries five. The category
manager plans to drop Lee, the brand with the lowest sales, but only if its customers are likely to stay with the store
by buying a brand that remains. The analysis needs to answer two questions:

- Do Lee's customers already buy a brand the store is keeping?
- Which brand should take Lee's space?

A cross-brand affinity table, which shows how many customers each pair of brands shares, answers both.

```python exec="on" session="brand_affinity" workdir="docs/examples/data"
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
brand. This example uses a synthetic year of jeans sales covering 20,000 customers
and five brands.

```bash
pip install openretailscience
```

```python exec="on" source="above" session="brand_affinity" workdir="docs/examples/data"
import pandas as pd

jeans = pd.read_parquet("jeans_transactions.parquet")
jeans.head(3)
show_table(jeans.head(3))  # markdown-exec: hide
```

If your columns have different names, rename them or [map your column names once](../getting_started/options_guide.md).
To run the same analysis on a table in your data warehouse rather than a pandas DataFrame, see
[Connecting to your data](../getting_started/connecting_to_data.md).

## Counting the customers each pair of brands shares

```python exec="on" source="above" session="brand_affinity" workdir="docs/examples/data"
from openretailscience.analysis import ProductAssociation

pairs = ProductAssociation(df=jeans, value_col="brand_name").df
pairs.head(3)
show_table(pairs.head(3).round(3))  # markdown-exec: hide
```

Each row is a pair of brands. `occurrences_1` and `occurrences_2` count the customers who bought each brand, and
`cooccurrences` counts those who bought both. `confidence` is the share of the first brand's customers who also bought
the second. `uplift` compares that share with the average customer, where 1.0 is typical and 2.0 is twice as likely.

A pair counts when the same customer bought both brands at any point in the year. Run the analysis on one category's
sales only. The comparison is against every customer in the data, so including other categories makes every pair look
stronger than it is.

## Lee's customers already buy brands the store is keeping

Start with each brand's sales.

```python exec="on" source="above" session="brand_affinity" workdir="docs/examples/data"
jeans.groupby("brand_name")["unit_spend"].sum().sort_values()
show_table(jeans.groupby("brand_name")["unit_spend"].sum().sort_values().reset_index())  # markdown-exec: hide
```

Lee has the lowest sales of the five brands, at £172,125. To see whether its customers already buy another brand,
pivot the shares into a table with one row per brand and read across Lee's row.

```python exec="on" source="above" session="brand_affinity" workdir="docs/examples/data"
from openretailscience.plots import heatmap

overlap = pairs.pivot(index="brand_name_1", columns="brand_name_2", values="confidence")
heatmap.plot(
    overlap.mul(100),
    cbar_label="% of buyers",
    cbar_format="{x:.0f}%",
    eyebrow="Jeans customers, 2025",
    figsize=(9, 6),
    title="Lee's customers already buy Levi's and Wrangler",
    subtitle="% of the row brand's buyers who also bought the column brand",
)
show_figure(  # markdown-exec: hide
    "Overlap heatmap. 26% of Lee buyers also buy Levi's and 22% buy Wrangler, while fewer than 7% "  # markdown-exec: hide
    "of Diesel buyers buy any other single brand."  # markdown-exec: hide
)  # markdown-exec: hide
```

Lee's customers already buy brands the store is keeping. 26% also bought Levi's this year and 22% bought Wrangler.
For comparison, fewer than 7% of Diesel's customers buy any other single brand. Most customers buy one pair of jeans a
year, so many of Lee's customers bought only Lee, and some of Lee's sales will be lost.

The table is not symmetric. 26% of Lee's customers buy Levi's, but only 9% of Levi's customers buy Lee, because Levi's
has far more customers (9,748 against Lee's 3,416). When assessing a brand, read that brand's row.

Each cell covers one other brand. To count exactly how many Lee customers bought neither Wrangler nor Levi's, use
[cross-shopping analysis](cross_shop.md) with those three brands.

## Wrangler, not Levi's, should take Lee's space

On raw overlap, Levi's looks like the natural home for Lee's customers. But raw overlap favours large brands, because
a large brand reaches a high share of every brand's customers. The affinity index corrects for brand size, with 100
meaning as likely as the average jeans customer.

```python exec="on" source="above" session="brand_affinity" workdir="docs/examples/data"
affinity = pairs.pivot(index="brand_name_1", columns="brand_name_2", values="uplift").mul(100)
heatmap.plot(
    affinity,
    cbar_label="Affinity index",
    cbar_format="{x:.0f}",
    eyebrow="Jeans customers, 2025",
    figsize=(9, 6),
    title="Lee and Wrangler share more customers than their size predicts",
    subtitle="Affinity index (100 = as likely as the average jeans customer)",
)
show_figure(  # markdown-exec: hide
    "Affinity index heatmap. Lee and Wrangler score 109, the only pair above 100, Lee and Levi's "  # markdown-exec: hide
    "score 53, and Diesel scores 17 or less with every brand."  # markdown-exec: hide
)  # markdown-exec: hide
```

```python exec="on" source="above" session="brand_affinity" workdir="docs/examples/data"
pairs[pairs["brand_name_1"] == "Lee"]
show_table(pairs[pairs["brand_name_1"] == "Lee"].round(3))  # markdown-exec: hide
```

Lee's customers lean towards Wrangler, not Levi's. Levi's reaches 26% of Lee's customers but 49% of all jeans
customers, so Lee's customers are about half as likely as average to buy it (index 53). Wrangler reaches 22% of Lee's
customers against 21% of all customers (index 109), and Lee and Wrangler are the only pair in the table above 100.

Most cells are below 100 because most customers buy one pair a year, and a customer who bought one brand has usually
not bought another. Compare the cells with each other rather than with 100.

Before acting on a pair, check how many customers it rests on. Lee and Wrangler share 763 customers, which is a solid
base. Lee and Diesel share only 21, so their index could change substantially from one year to the next.
`min_cooccurrences` excludes pairs below a minimum count.

!!! success "Recommendation"
    Drop Lee and give its space to Wrangler. Lee has the lowest sales, and its customers already buy brands the store
    is keeping. Levi's reaches more of them only because it is the largest brand, while Lee and Wrangler are the only
    pair that share more customers than their sizes predict.

## Adapting the analysis

The same approach applies to related questions:

- For a category that most customers buy less than once a year, use two years of sales so the table captures more of
  each customer's purchases.
- To compare products or categories instead of brands, set `value_col` to that column.
- To look at shopping trips instead of customers, pass `group_col="transaction_id"` to count brands bought in the same
  basket.

See also [Cross-Shopping Analysis](cross_shop.md), [Product Association](product_association.ipynb),
[Customer Decision Hierarchy](../gallery/plots/customer_decision_hierarchy.ipynb) and the
[`ProductAssociation` API reference](../api/analysis/product_association.md).
