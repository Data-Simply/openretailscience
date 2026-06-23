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

_TRANSITION_COLUMNS = ["from_category", "to_category", "customer_count", "transition_probability"]
_JOURNEY_COLUMNS = ["journey", "probability"]
# A journey needs at least one transition (two categories) to be worth reporting.
_MIN_JOURNEY_LENGTH = 2
_JOURNEY_SEPARATOR = " -> "


def _empty_journeys() -> pd.DataFrame:
    """Empty journeys table carrying the same column dtypes as a populated result."""
    return pd.DataFrame(
        {
            "journey": pd.Series(dtype="object"),
            "probability": pd.Series(dtype="float64"),
        },
    )


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
        category_col: str,
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
            category_col (str): Name of the product category column. Required, as the category
                column name is dataset-specific.
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

        self._category_col = category_col
        self._min_customers = min_customers
        self._customer_id = cols.customer_id
        # Kept as an unexecuted Ibis expression so the (customer, category) intermediate — which
        # can be in the billions of rows — is never pulled into pandas; only the small final
        # transition table (bounded by the number of category pairs) is executed in :attr:`df`.
        self._first_appearances = self._build_first_appearances(
            table=ensure_ibis_table(df),
            cols=cols,
            category_col=category_col,
            min_transactions=min_transactions,
            min_basket_size=min_basket_size,
            min_basket_value=min_basket_value,
            max_depth=max_depth,
            exclude_returns=exclude_returns,
        )

    @staticmethod
    def _build_first_appearances(
        table: ibis.Table,
        cols: ColumnHelper,
        category_col: str,
        min_transactions: int,
        min_basket_size: int,
        min_basket_value: float,
        max_depth: int,
        exclude_returns: bool,
    ) -> ibis.Table:
        """Build the (customer, category, first_basket) Ibis expression of first appearances.

        Returns an unexecuted Ibis table with one row per (customer, category) holding the
        1-based ``first_basket`` in which that category first appeared, for customers and
        baskets that pass the filters.

        Args:
            table (ibis.Table): Transaction line items.
            cols (ColumnHelper): Resolved column names.
            category_col (str): Product category column name.
            min_transactions (int): Minimum qualifying baskets per customer (pre-truncation).
            min_basket_size (int): Minimum distinct products per qualifying basket.
            min_basket_value (float): Minimum spend per qualifying basket.
            max_depth (int): Maximum baskets analysed per customer.
            exclude_returns (bool): Whether to drop non-positive spend line items first.

        Returns:
            ibis.Table: Columns ``[customer_id, category_col, first_basket]``.
        """
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

        # Uncategorised line items (null category) still count toward basket size and value
        # above, but they cannot anchor a category transition, so drop them here rather than
        # let them form a phantom "missing" category node.
        line_items = table.join(analysis_baskets, [cols.customer_id, cols.transaction_id]).filter(
            _[category_col].notnull(),  # noqa: PD004 (ibis API, not pandas)
        )
        return line_items.group_by([cols.customer_id, category_col]).aggregate(
            first_basket=_.basket_number.min(),
        )

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """The aggregate category-transition table.

        Each row is a first-order transition between consecutive category acquisitions:
        ``transition_probability`` is the share of customers who, having acquired
        ``from_category`` and gone on to acquire at least one further new category, acquired
        ``to_category`` next. Probabilities for a given ``from_category`` need not sum to one
        because a customer's next basket can introduce several categories at once.

        The aggregation runs entirely in Ibis; only this small result (bounded by the number of
        category pairs) is executed into pandas.

        Returns:
            pd.DataFrame: Columns ``["from_category", "to_category", "customer_count",
            "transition_probability"]``, sorted by source then descending probability. Empty
            (same columns) when no transition meets the criteria.
        """
        fa = self._first_appearances
        cid = self._customer_id
        order_window = ibis.window(group_by=cid, order_by="first_basket")

        # Distinct acquisition events per customer, each linked to the customer's next event.
        events = fa.select(cid, "first_basket").distinct()
        events = events.mutate(next_basket=events["first_basket"].lead(1).over(order_window))

        # Pair every category in an event (from) with every category in the next event (to).
        from_side = fa.rename(from_category=self._category_col).join(events, [cid, "first_basket"])
        to_side = fa.select(cid, to_category=fa[self._category_col], to_basket=fa["first_basket"])
        pairs = from_side.join(
            to_side,
            [from_side[cid] == to_side[cid], from_side["next_basket"] == to_side["to_basket"]],
        ).select(
            from_category=from_side["from_category"],
            to_category=to_side["to_category"],
            customer_key=from_side[cid],
        )

        # Each (from, to) pair is unique per customer, so a row count gives the customer count.
        # A source category reaches several targets, so its denominator needs distinct customers.
        transitions = pairs.group_by(["from_category", "to_category"]).aggregate(customer_count=pairs.count())
        source = pairs.group_by("from_category").aggregate(source_customers=pairs["customer_key"].nunique())

        result = (
            transitions.join(source, "from_category")
            .mutate(transition_probability=transitions["customer_count"] / source["source_customers"])
            .filter(transitions["customer_count"] >= self._min_customers)
            .order_by(["from_category", ibis.desc("transition_probability"), "to_category"])
            .select(_TRANSITION_COLUMNS)
        )
        return result.execute()

    @functools.cached_property
    def _entry_categories(self) -> list:
        """Categories customers acquire in their very first qualifying basket, sorted.

        The category values keep their native dtype (e.g. integer category codes), so the
        returned list matches the ``from_category`` keys used by :meth:`dominant_journeys`.
        """
        fa = self._first_appearances
        cid = self._customer_id
        with_min = fa.mutate(first_event=fa["first_basket"].min().over(ibis.window(group_by=cid)))
        entries = with_min.filter(with_min["first_basket"] == with_min["first_event"]).select(self._category_col)
        return sorted(entries.distinct().execute()[self._category_col])

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
            return _empty_journeys()

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
                journey = _JOURNEY_SEPARATOR.join(str(category) for category in path)
                journeys.append({"journey": journey, "probability": probability})

        if len(journeys) == 0:
            return _empty_journeys()
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
