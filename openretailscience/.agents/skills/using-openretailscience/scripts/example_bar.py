"""Example script demonstrating bar plot usage."""

import pandas as pd
from openretailscience.plots import bar

df = pd.DataFrame({
    "product": ["Product A", "Product B", "Product C", "Product D", "Product E"],
    "q1_sales": [25000, 18000, 32000, 15000, 28000],
    "q2_sales": [35000, 22000, 30000, 12000, 33000],
    "q3_sales": [30000, 25000, 35000, 18000, 31000],
    "q4_sales": [40000, 28000, 38000, 20000, 36000]
})

# Example 1: Basic vertical bar chart
bar.plot(
    df=df,
    value_col="q1_sales",
    x_col="product",
    title="Q1 Sales by Product",
    x_label="Product",
    y_label="Sales ($)",
    source_text="Source: Sales Data Q1 2024",
)

# Example 2: Sorted horizontal bars with data labels
bar.plot(
    df=df,
    value_col="q1_sales",
    x_col="product",
    orientation="horizontal",
    sort_order="descending",
    data_label_format="absolute",
    title="Q1 Sales Ranking",
    x_label="Sales ($)",
    y_label="Product",
)

# Example 3: Grouped bars (quarterly comparison)
bar.plot(
    df=df,
    value_col=["q1_sales", "q2_sales", "q3_sales", "q4_sales"],
    x_col="product",
    title="Quarterly Sales Comparison by Product",
    x_label="Product",
    y_label="Sales ($)",
    move_legend_outside=True,
    rot=0,
)

# Example 4: Stacked bars with percentage labels
bar.plot(
    df=df,
    value_col=["q1_sales", "q2_sales", "q3_sales", "q4_sales"],
    x_col="product",
    title="Annual Sales Breakdown by Quarter",
    y_label="Total Sales ($)",
    stacked=True,
    data_label_format="percentage_by_bar_group",
    move_legend_outside=True,
    num_digits=2,
)

# Example 5: With hatch patterns
bar.plot(
    df=df,
    value_col=["q1_sales", "q2_sales"],
    x_col="product",
    title="H1 Sales Comparison (Q1 vs Q2)",
    y_label="Sales ($)",
    use_hatch=True,
    data_label_format="absolute",
    move_legend_outside=True,
    num_digits=3,
)

# Example 6: Series plotting
total_sales = df.set_index("product")["q1_sales"]

bar.plot(
    df=total_sales,
    value_col=None,
    title="Q1 Sales (from Series)",
    y_label="Sales ($)",
    sort_order="ascending",
    data_label_format="absolute",
)

# Example 7: Percentage by series
bar.plot(
    df=df,
    value_col=["q1_sales", "q2_sales", "q3_sales"],
    x_col="product",
    title="Product Contribution % by Quarter",
    y_label="Sales ($)",
    data_label_format="percentage_by_series",
    move_legend_outside=True,
    num_digits=2,
)
