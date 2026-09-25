"""Module for analyzing customer purchase paths from transaction data.

This module defines the `purchase_path_analysis` function that tracks
customer journeys through product categories over time.
"""

import ibis
import pandas as pd

from pyretailscience.options import ColumnHelper


def _build_category_group_df(
    first_df: pd.DataFrame,
    category_column: str,
    sort_by_metric: bool,
    multi_category_handling: str,
) -> pd.DataFrame:
    """Creates a DataFrame mapping customers to concatenated or individual categories."""
    if multi_category_handling == "concatenate":
        sort_cols = ["customer_id", "first_basket_number"]
        if sort_by_metric:
            sort_cols.append("metric_value")
            ascending = [True, True, False]
        else:
            sort_cols.append(category_column)
            ascending = [True, True, True]

        return (
            first_df.sort_values(sort_cols, ascending=ascending)
            .groupby(["customer_id", "first_basket_number"])[category_column]
            .apply(lambda x: ",".join(str(v) for v in x))
            .reset_index()
            .rename(columns={category_column: "categories"})
        )
    return first_df[["customer_id", "first_basket_number", category_column]].rename(
        columns={category_column: "categories"},
    )


def _build_paths_df(category_groups_df: pd.DataFrame) -> pd.DataFrame:
    """Constructs a pivoted DataFrame representing customer purchase paths."""
    actual_baskets = sorted(category_groups_df["first_basket_number"].unique()) if not category_groups_df.empty else []
    paths_df = category_groups_df.pivot_table(
        index="customer_id",
        columns="first_basket_number",
        values="categories",
        aggfunc="first",
    ).reset_index()

    column_mapping = {"customer_id": "customer_id"}
    for i, basket_num in enumerate(sorted(actual_baskets), 1):
        if basket_num in paths_df.columns:
            column_mapping[basket_num] = f"basket_{i}"
    return paths_df.rename(columns=column_mapping).fillna("")


def purchase_path_analysis(
    transactions_df: pd.DataFrame,
    category_column: str = "product_category",
    min_transactions: int = 3,
    min_basket_size: int = 2,
    min_basket_value: float = 10.0,
    max_depth: int = 10,
    min_customers: int = 5,
    exclude_negative_revenue: bool = True,
    multi_category_handling: str = "concatenate",
    sort_by: str = "alphabetical",
    aggregation_column: str | None = None,
    aggregation_function: str = "sum",
) -> pd.DataFrame:
    """Analyzes customer purchase paths through product categories over time."""
    cols = ColumnHelper()
    required_cols = [cols.customer_id, cols.transaction_id, cols.transaction_date, "product_id", category_column]
    if exclude_negative_revenue:
        required_cols.append("revenue")
    missing_cols = set(required_cols) - set(transactions_df.columns)
    if missing_cols:
        msg = f"The following columns are required but missing: {missing_cols}"
        raise ValueError(msg)

    transactions_table = (
        ibis.memtable(transactions_df) if isinstance(transactions_df, pd.DataFrame) else transactions_df
    )
    if exclude_negative_revenue:
        transactions_table = transactions_table.filter(transactions_table.revenue > 0)

    customer_baskets = (
        transactions_table.group_by(["customer_id", "transaction_id", "transaction_date"])
        .aggregate(
            item_count=ibis._.product_id.nunique(),
            basket_value=ibis._.revenue.sum(),
        )
        .filter(
            (ibis._.item_count >= min_basket_size) & (ibis._.basket_value >= min_basket_value),
        )
        .mutate(
            basket_number=ibis.row_number().over(
                ibis.window(group_by="customer_id", order_by="transaction_date"),
            ),
        )
        .filter(ibis._.basket_number <= max_depth)
    )
    eligible_customers = (
        customer_baskets.group_by("customer_id")
        .aggregate(transaction_count=ibis._.basket_number.count())
        .filter(ibis._.transaction_count >= min_transactions)
        .select("customer_id")
    )

    transactions_with_baskets = transactions_table.inner_join(
        customer_baskets.inner_join(eligible_customers, "customer_id").select(
            ["customer_id", "transaction_id", "basket_number"],
        ),
        ["customer_id", "transaction_id"],
    )

    use_agg_sort = (
        multi_category_handling == "concatenate"
        and sort_by == "aggregation"
        and aggregation_column
        and aggregation_function
    )

    agg_func_map = {
        "sum": "sum",
        "max": "max",
        "min": "min",
        "avg": "mean",
    }

    if use_agg_sort:
        agg_method = agg_func_map.get(aggregation_function)
        if not agg_method:
            msg = f"Unsupported aggregation function: {aggregation_function}"
            raise ValueError(msg)
        agg_func = getattr(transactions_with_baskets[aggregation_column], agg_method)
        first_df = transactions_with_baskets.group_by(["customer_id", category_column]).aggregate(
            first_basket_number=ibis._.basket_number.min(),
            metric_value=agg_func(),
        )
    else:
        first_df = transactions_with_baskets.group_by(["customer_id", category_column]).aggregate(
            first_basket_number=ibis._.basket_number.min(),
        )
    first_df = first_df.execute()

    if first_df.empty:
        return pd.DataFrame(columns=["customer_count", "transition_probability"])

    category_groups_df = _build_category_group_df(first_df, category_column, use_agg_sort, multi_category_handling)
    paths_df = _build_paths_df(category_groups_df)

    basket_cols = sorted(
        [col for col in paths_df.columns if col.startswith("basket_")],
        key=lambda x: int(x.split("_")[1]),
    )
    paths_df = paths_df[paths_df[basket_cols].ne("").any(axis=1)]

    if paths_df.empty:
        return pd.DataFrame(columns=["customer_count", "transition_probability"])

    pattern_counts = paths_df.groupby(basket_cols).size().reset_index(name="customer_count")
    pattern_counts = pattern_counts[pattern_counts.customer_count >= min_customers]

    if not pattern_counts.empty:
        total_customers = pattern_counts.customer_count.sum()
        pattern_counts["transition_probability"] = (pattern_counts.customer_count / total_customers).round(3)
        return pattern_counts.sort_values("customer_count", ascending=False).reset_index(drop=True)

    return pd.DataFrame(columns=[*basket_cols, "customer_count", "transition_probability"])
