"""Purchase Path Analysis for category-progression insights.

## Business Context

Customers rarely buy from every category at once. They discover a retailer through one
category and broaden over time — a customer might start in women's clothing, add kids'
clothing on a later trip, then move into menswear. Purchase Path Analysis answers the
practical question this raises: given the categories customers have already bought, which
category do they tend to buy next?

## What the Output Means

The analysis records, for each customer, the basket in which every product category *first
appears*, then aggregates the consecutive acquisitions across all customers into first-order
category transitions. Each result row is a ``from_category`` → ``to_category`` transition
with ``transition_probability``: the share of customers who, having acquired
``from_category`` and gone on to buy something new next, acquired ``to_category``. This is an
aggregate tendency across the customer base ("women's buyers tend to buy menswear next"),
not a per-customer prediction.

Categories bought in the same basket are treated as acquired simultaneously — they impose no
order on each other and each transitions to whatever the customer acquires next — so the
analysis behaves sensibly whether baskets are category-rich or single-category.

## Business Applications

- **Cross-sell sequencing**: surface the category a customer is statistically likely to add next.
- **Onboarding**: understand which entry category leads customers to broaden fastest.
- **Category management**: plan adjacencies and promotions around natural progressions.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import ibis
import pandas as pd
from ibis import _

from openretailscience.core.validation import (
    ensure_data_has_columns,
    ensure_ibis_table,
    ensure_integer,
    ensure_number,
    ensure_positive,
)
from openretailscience.options import ColumnHelper

if TYPE_CHECKING:
    import networkx as nx

# transition_probability is a share in [0, 1]; four places preserves two-decimal percentage precision.
_ROUND_DECIMALS = 4
_TRANSITION_COLUMNS = ["from_category", "to_category", "customer_count", "transition_probability"]
_JOURNEY_COLUMNS = ["journey", "probability"]
# A journey needs at least one transition (two categories) to be worth reporting.
_MIN_JOURNEY_LENGTH = 2
_JOURNEY_SEPARATOR = " -> "


class PurchasePath:
    """Models the order in which customers first buy from each product category.

    For every customer the analysis orders their baskets in time and records the basket in
    which each category first appears. Consecutive acquisitions are then aggregated across all
    customers into first-order ``from_category`` → ``to_category`` transition probabilities,
    exposed via :attr:`df`.
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        category_col: str = "category",
        min_transactions: int = 2,
        min_basket_size: int = 1,
        min_basket_value: float = 0.0,
        max_depth: int = 10,
        min_customers: int = 1,
        exclude_returns: bool = True,
    ) -> None:
        """Compute category transitions from transaction line-item data.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction line items containing the customer,
                transaction, date, product, spend and category columns.
            category_col (str): Name of the product category column. Defaults to ``"category"``.
            min_transactions (int): Minimum number of qualifying baskets a customer must have
                to be included. Counted before ``max_depth`` truncation. Defaults to 2 (the
                minimum needed to observe a transition).
            min_basket_size (int): Minimum number of distinct products for a basket to qualify.
                Defaults to 1 (no size filtering).
            min_basket_value (float): Minimum total spend for a basket to qualify. Defaults to
                0.0 (no value filtering).
            max_depth (int): Maximum number of baskets per customer to analyse, ordered by date.
                Defaults to 10.
            min_customers (int): Minimum number of customers a transition must have to appear in
                the result. Defaults to 1.
            exclude_returns (bool): If True, drop line items with non-positive ``unit_spend``
                (returns/refunds) before building baskets. Defaults to True.

        Raises:
            ValueError: If a required column is missing or a filter parameter is non-positive.
            TypeError: If a filter parameter has the wrong type.
        """
        cols = ColumnHelper()

        for value, name in (
            (min_transactions, "min_transactions"),
            (min_basket_size, "min_basket_size"),
            (max_depth, "max_depth"),
            (min_customers, "min_customers"),
        ):
            ensure_integer(value, name)
            ensure_positive(value, name)
        ensure_number(min_basket_value, "min_basket_value")

        required_cols = [
            cols.customer_id,
            cols.transaction_id,
            cols.transaction_date,
            cols.product_id,
            cols.unit_spend,
            category_col,
        ]
        ensure_data_has_columns(df, required_cols)

        self._first_appearance_df = self._calc_first_appearances(
            df=df,
            cols=cols,
            category_col=category_col,
            min_transactions=min_transactions,
            min_basket_size=min_basket_size,
            min_basket_value=min_basket_value,
            max_depth=max_depth,
            exclude_returns=exclude_returns,
        )
        self._category_col = category_col
        self._min_customers = min_customers

    @staticmethod
    def _calc_first_appearances(
        df: pd.DataFrame | ibis.Table,
        cols: ColumnHelper,
        category_col: str,
        min_transactions: int,
        min_basket_size: int,
        min_basket_value: float,
        max_depth: int,
        exclude_returns: bool,
    ) -> pd.DataFrame:
        """Return, per eligible customer and category, the basket position of first appearance.

        The heavy filtering, sequencing and joining run in Ibis; the executed frame has one
        row per (customer, category) with the 1-based ``first_basket`` it first appeared in.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction line items.
            cols (ColumnHelper): Resolved column names.
            category_col (str): Product category column name.
            min_transactions (int): Minimum qualifying baskets per customer (pre-truncation).
            min_basket_size (int): Minimum distinct products per qualifying basket.
            min_basket_value (float): Minimum spend per qualifying basket.
            max_depth (int): Maximum baskets analysed per customer.
            exclude_returns (bool): Whether to drop non-positive spend line items first.

        Returns:
            pd.DataFrame: Columns ``[customer_id, category_col, first_basket]``.
        """
        table = ensure_ibis_table(df)
        if exclude_returns:
            table = table.filter(table[cols.unit_spend] > 0)

        baskets = (
            table.group_by([cols.customer_id, cols.transaction_id, cols.transaction_date])
            .aggregate(
                item_count=table[cols.product_id].nunique(),
                basket_value=table[cols.unit_spend].sum(),
            )
            .filter((_.item_count >= min_basket_size) & (_.basket_value >= min_basket_value))
            .mutate(
                basket_number=ibis.row_number().over(
                    ibis.window(group_by=cols.customer_id, order_by=[cols.transaction_date, cols.transaction_id]),
                )
                + 1,
            )
        )

        eligible_customers = (
            baskets.group_by(cols.customer_id)
            .aggregate(basket_count=_.basket_number.count())
            .filter(_.basket_count >= min_transactions)
            .select(cols.customer_id)
        )

        analysis_baskets = (
            baskets.filter(_.basket_number <= max_depth)
            .join(eligible_customers, cols.customer_id)
            .select([cols.customer_id, cols.transaction_id, "basket_number"])
        )

        line_items = table.join(analysis_baskets, [cols.customer_id, cols.transaction_id])
        first_appearances = line_items.group_by([cols.customer_id, category_col]).aggregate(
            first_basket=_.basket_number.min(),
        )
        return first_appearances.execute()

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """The aggregate category-transition table.

        Each row is a first-order transition between consecutive category acquisitions:
        ``transition_probability`` is the share of customers who, having acquired
        ``from_category`` and gone on to acquire at least one further new category, acquired
        ``to_category`` next. Probabilities for a given ``from_category`` need not sum to one
        because a customer's next basket can introduce several categories at once.

        Returns:
            pd.DataFrame: Columns ``["from_category", "to_category", "customer_count",
            "transition_probability"]``, sorted by source then descending probability. Empty
            (same columns) when no transition meets the criteria.
        """
        first_df = self._first_appearance_df
        if len(first_df) == 0:
            return pd.DataFrame(columns=_TRANSITION_COLUMNS)

        cid = ColumnHelper().customer_id

        # Collapse to one row per (customer, acquisition event), holding the categories that
        # first appeared together in that basket, then pair each event with the customer's next.
        events = (
            first_df.sort_values([cid, "first_basket"])
            .groupby([cid, "first_basket"])[self._category_col]
            .agg(list)
            .reset_index(name="from_categories")
        )
        events["to_categories"] = events.groupby(cid)["from_categories"].shift(-1)
        events = events.dropna(subset=["to_categories"])
        if len(events) == 0:
            return pd.DataFrame(columns=_TRANSITION_COLUMNS)

        pairs = (
            events[[cid, "from_categories", "to_categories"]]
            .explode("from_categories")
            .explode("to_categories")
            .rename(columns={"from_categories": "from_category", "to_categories": "to_category"})
        )
        pairs["from_category"] = pairs["from_category"].astype(str)
        pairs["to_category"] = pairs["to_category"].astype(str)

        # Denominator: customers who progressed past each source category (before support filtering).
        source_customers = pairs.groupby("from_category")[cid].nunique()

        transitions = pairs.groupby(["from_category", "to_category"])[cid].nunique().reset_index(name="customer_count")
        transitions = transitions[transitions["customer_count"] >= self._min_customers]
        if len(transitions) == 0:
            return pd.DataFrame(columns=_TRANSITION_COLUMNS)

        transitions["transition_probability"] = (
            transitions["customer_count"] / transitions["from_category"].map(source_customers)
        ).round(_ROUND_DECIMALS)
        return transitions.sort_values(
            ["from_category", "transition_probability", "to_category"],
            ascending=[True, False, True],
        ).reset_index(drop=True)[list(_TRANSITION_COLUMNS)]

    @functools.cached_property
    def _entry_categories(self) -> list[str]:
        """Categories customers acquire in their very first qualifying basket, sorted."""
        first_df = self._first_appearance_df
        cid = ColumnHelper().customer_id
        first_events = first_df[first_df["first_basket"] == first_df.groupby(cid)["first_basket"].transform("min")]
        return sorted(first_events[self._category_col].astype(str).unique())

    def dominant_journeys(self, max_length: int = 10) -> pd.DataFrame:
        """Trace the single most-likely onward journey from each entry category.

        Starting from every category customers begin with, the graph is walked greedily,
        always taking the highest-probability transition, until it reaches a category with no
        onward transition, revisits a category (cycle), or hits ``max_length``. The reported
        ``probability`` is the product of the transition probabilities along the walk — the
        first-order (Markov) likelihood of the journey, not the empirical frequency of that
        exact chain.

        Args:
            max_length (int): Maximum number of categories in a journey. Defaults to 10.

        Returns:
            pd.DataFrame: Columns ``["journey", "probability"]`` sorted by descending
            probability, where ``journey`` is an arrow-joined category sequence. Empty (same
            columns) when no journey of at least two categories exists.
        """
        ensure_integer(max_length, "max_length")
        ensure_positive(max_length, "max_length")

        edges = self.df
        if len(edges) == 0:
            return pd.DataFrame(columns=_JOURNEY_COLUMNS)

        best = edges.sort_values(
            ["from_category", "transition_probability", "to_category"],
            ascending=[True, False, True],
        ).drop_duplicates("from_category")
        next_category = dict(zip(best["from_category"], best["to_category"], strict=True))
        next_probability = dict(zip(best["from_category"], best["transition_probability"], strict=True))

        journeys = []
        for start in self._entry_categories:
            path = [start]
            probability = 1.0
            current = start
            while current in next_category and len(path) < max_length and next_category[current] not in path:
                probability *= next_probability[current]
                current = next_category[current]
                path.append(current)
            if len(path) >= _MIN_JOURNEY_LENGTH:
                journeys.append(
                    {"journey": _JOURNEY_SEPARATOR.join(path), "probability": round(probability, _ROUND_DECIMALS)},
                )

        if len(journeys) == 0:
            return pd.DataFrame(columns=_JOURNEY_COLUMNS)
        return (
            pd.DataFrame(journeys)
            .sort_values(["probability", "journey"], ascending=[False, True])
            .reset_index(drop=True)
        )

    def to_networkx(self) -> nx.DiGraph:
        """Return the transition table as a weighted directed graph.

        Nodes are categories and each edge carries ``transition_probability`` and
        ``customer_count`` attributes, ready for centrality, path-finding or drawing with the
        wider networkx ecosystem.

        Returns:
            nx.DiGraph: A directed graph of category transitions.

        Raises:
            ImportError: If networkx is not installed.
        """
        try:
            import networkx as nx  # noqa: PLC0415
        except ImportError as exc:
            msg = "to_networkx requires networkx. Install it with `pip install networkx`."
            raise ImportError(msg) from exc

        return nx.from_pandas_edgelist(
            self.df,
            source="from_category",
            target="to_category",
            edge_attr=["transition_probability", "customer_count"],
            create_using=nx.DiGraph,
        )
