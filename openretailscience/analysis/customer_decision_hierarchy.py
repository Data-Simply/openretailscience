"""Customer Decision Hierarchy Analysis for Product Substitutability and Range Optimization.

## Business Context

Customer Decision Hierarchy (CDH) analysis reveals how customers perceive products
as substitutes or complements. This critical intelligence informs range planning,
assortment optimization, and delisting decisions by understanding which products
customers view as interchangeable versus essential variety.

## The Business Problem

Retailers often struggle with range rationalization decisions:
- Which products can be delisted without losing customers?
- When does variety add value versus create confusion?
- Which products are true substitutes in customers' minds?
- How to optimize shelf space without sacrificing choice?

CDH analysis answers these questions by analyzing actual switching behavior rather
than relying on product attributes or manager intuition.

## How It Works

The analysis examines customer purchase patterns to identify substitutability:
- Products rarely bought by the same customer → likely substitutes
- Products often bought by the same customer → complements or variety-seeking
- Uses Yule's Q coefficient to measure substitutability strength
- Creates hierarchical clusters showing substitution relationships

## Real-World Applications

1. **Range Rationalization**
   - Identify safe delisting candidates within substitute clusters
   - Maintain one option per cluster to preserve choice
   - Reduce SKU count while maintaining customer satisfaction

2. **New Product Introduction**
   - Understand which existing products new items might cannibalize
   - Position new products to fill gaps rather than duplicate
   - Predict source of volume for new launches

3. **Private Label Strategy**
   - Identify national brand products suitable for PL alternatives
   - Understand where PL can substitute vs. complement
   - Optimize PL/NB mix within categories

4. **Space Optimization**
   - Allocate more space to non-substitutable products
   - Reduce facings for products within same substitute cluster
   - Optimize variety/productivity trade-off

5. **Markdown Strategy**
   - Clear substitute products sequentially, not simultaneously
   - Understand which products can drive category traffic
   - Identify products that won't cannibalize when promoted

## Business Value

- **Efficient Assortment**: Reduce complexity without losing sales
- **Better Space Productivity**: Allocate space based on true variety value
- **Improved Margins**: Replace duplicative SKUs with unique offerings
- **Customer Satisfaction**: Maintain perceived choice while reducing confusion
- **Strategic Clarity**: Data-driven approach to range decisions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import ibis
import matplotlib.pyplot as plt
import numpy as np
from ibis import _
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from openretailscience.core.validation import ensure_data_has_columns, ensure_ibis_table
from openretailscience.options import ColumnHelper
from openretailscience.plots.styles.styling_helpers import standard_graph_styles

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes, SubplotBase


class CustomerDecisionHierarchy:
    """Analyzes product substitutability patterns to optimize retail assortments.

    The CustomerDecisionHierarchy class identifies which products customers view as
    substitutes versus essential variety. This enables data-driven range planning
    decisions that maintain customer choice while improving operational efficiency.

    ## Business Insight

    Traditional range planning often assumes products in the same category are
    substitutes (e.g., all yogurts are interchangeable). However, customer behavior
    reveals the truth: some customers always buy both Greek and regular yogurt
    (complements), while others switch between strawberry and raspberry flavors
    (substitutes).

    ## Substitutability Logic

    The analysis identifies substitutes through purchase patterns:
    - **High substitutability**: Customers buy product A OR product B, rarely both
    - **Low substitutability**: Customers often buy both A AND B
    - **Exclusion logic**: Products bought in same transaction can't be substitutes

    ## Decision Framework

    The hierarchy output guides range decisions:
    - **Tight clusters**: Strong substitutes - keep best performer
    - **Loose clusters**: Weak substitutes - maintain variety
    - **Separate branches**: Different needs - preserve both
    - **Isolated products**: Unique value - protect from delisting

    ## Example Use Case

    A supermarket analyzing yogurt finds:
    - Cluster 1: Store brand vanilla, strawberry, raspberry (substitutes)
    - Cluster 2: Greek plain, Greek honey (substitutes)
    - Separate branch: Kids' squeezable yogurt (unique need)

    Decision: Can reduce flavor variety in Cluster 1, maintain Greek options,
    must keep kids' yogurt despite low sales.
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        product_col: str,
        exclude_same_transaction_products: bool = True,
        method: Literal["yules_q"] = "yules_q",
        random_state: int = 42,
    ) -> None:
        """Initialize customer decision hierarchy analysis for range optimization.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data with customer purchase history.
                Must contain: customer_id, transaction_id, and product identifier.
            product_col (str): Column containing products to analyze for substitutability
                (e.g., "product_name", "sku", "brand", "subcategory").
            exclude_same_transaction_products (bool, optional): Whether products bought
                together in one transaction should be considered non-substitutes.
                True = If customer buys milk and eggs together, they're not substitutes.
                False = Include all purchase patterns.
                Defaults to True (recommended for most retail contexts).
            method (Literal["yules_q"], optional): Statistical method for measuring
                substitutability. "yules_q" measures association strength between
                binary purchase patterns. Defaults to "yules_q". Additional distance
                methods (e.g. AIDS or Rotterdam-model demand systems converted to
                distances) are planned; each will be a new branch in
                ``_calculate_distances`` returning its own square distance matrix.
            random_state (int, optional): Seed for reproducible clustering results.
                Important for consistent range planning decisions. Defaults to 42.

        Raises:
            ValueError: If required columns are missing from the dataframe.

        Business Example:
            >>> # Analyze substitutability in coffee category
            >>> cdh = CustomerDecisionHierarchy(
            ...     df=transactions,
            ...     product_col="brand_flavor",  # e.g., "Folgers_Original"
            ...     exclude_same_transaction_products=True  # Bought together = not substitutes
            ... )
            >>> # Use results to identify which coffee SKUs can be delisted
        """
        cols = ColumnHelper()
        required_cols = [cols.customer_id, cols.transaction_id, product_col]
        ensure_data_has_columns(df, required_cols)

        self.random_state = random_state
        self.product_col = product_col
        pairs = self._get_pairs(ensure_ibis_table(df), exclude_same_transaction_products, product_col)
        self.distances, self.products = self._calculate_distances(pairs, method=method)

    @staticmethod
    def _get_pairs(df: ibis.Table, exclude_same_transaction_products: bool, product_col: str) -> ibis.Table:
        """Reduce transactions to the distinct customer/product pairs used for substitutability.

        Args:
            df (ibis.Table): Transaction data containing customer, transaction, and product columns.
            exclude_same_transaction_products (bool): When True, drop every customer/product pair
                belonging to a transaction that contained more than one distinct product, so
                products bought together are not treated as substitutes.
            product_col (str): Column identifying the products to analyze.

        Returns:
            ibis.Table: Distinct ``[customer_id, product_col]`` pairs.
        """
        cols = ColumnHelper()
        if exclude_same_transaction_products:
            # Deduplicate to distinct (transaction, product) so a plain count() per transaction
            # equals its distinct-product count -- avoids a COUNT DISTINCT over the base rows.
            multi_product_txns = (
                df.select(cols.transaction_id, product_col)
                .distinct()
                .group_by(cols.transaction_id)
                .aggregate(n_products=_.count())
                .filter(_.n_products > 1)
            )
            # A customer/product is excluded if it ever appeared in a multi-product transaction.
            excluded = (
                df.select(cols.customer_id, cols.transaction_id, product_col)
                .join(multi_product_txns, cols.transaction_id)
                .select(cols.customer_id, product_col)
                .distinct()
            )
            pairs = (
                df.select(cols.customer_id, product_col)
                .distinct()
                .anti_join(excluded, [cols.customer_id, product_col])
            )
        else:
            pairs = df.select(cols.customer_id, product_col).distinct()

        return pairs

    @staticmethod
    def _get_yules_q_distances(pairs: ibis.Table, product_col: str) -> tuple[np.ndarray, list[str]]:
        """Calculate the Yule's Q distance matrix from distinct customer/product pairs.

        Every entry of the 2x2 contingency table for a product pair is recoverable from
        per-product and per-pair customer counts, so the heavy customer-level work stays in the
        Ibis backend and only the small per-product-pair result (at most ``n_products`` squared
        rows) is materialized. The square matrix is then assembled with vectorized numpy.

        For products i and j with ``a`` customers buying both, ``occ_i``/``occ_j`` buying each,
        and ``N`` total customers: ``b = occ_i - a``, ``c = occ_j - a``, ``d = N - occ_i - occ_j
        + a``. Yule's Q is ``(ad - bc) / (ad + bc)``; where the denominator is zero (Q undefined,
        e.g. two products always bought together) Q is treated as 0 so the distance is well
        defined and scipy's ``linkage`` does not see a NaN.

        Scaling notes for billion-row inputs:
            - ``pairs`` is cached so the self-join and the customer count reuse one materialized
              intermediate instead of re-deriving it (including the exclusion anti-join) from the
              base table on every query.
            - A single ``<=`` self-join yields both occurrences (the ``product == product``
              diagonal) and co-occurrences (off-diagonal) in one pass.
            - Because each join side is distinct on ``(customer, product)``, every
              ``(product_1, product_2)`` cell counts each customer at most once, so a plain
              ``count()`` already equals the distinct-customer count -- no ``COUNT DISTINCT`` over
              the large pair table is needed.

        Args:
            pairs (ibis.Table): Distinct ``[customer_id, product_col]`` pairs.
            product_col (str): Column identifying the products.

        Returns:
            tuple[np.ndarray, list[str]]: A square ``[0, 1]`` distance matrix and the
                alphabetically sorted product labels indexing its rows/columns.
        """
        cols = ColumnHelper()
        cust = cols.customer_id

        pairs = pairs.cache()

        left = pairs.rename(product_1=product_col)
        right = pairs.rename(product_2=product_col)
        pair_counts = (
            left.join(right, [(left[cust] == right[cust]), (left.product_1 <= right.product_2)])
            .group_by(["product_1", "product_2"])
            .aggregate(n_customers=_.count())
        )

        counts_df = pair_counts.execute()
        n_customers = int(pairs[cust].nunique().execute())

        # The diagonal (product_1 == product_2) holds each product's occurrence count and, by
        # construction, every product that any customer bought; off-diagonal cells hold pairwise
        # co-occurrences.
        is_diagonal = counts_df["product_1"] == counts_df["product_2"]
        diagonal_df = counts_df[is_diagonal]
        cooccurrence_df = counts_df[~is_diagonal]

        products = sorted(diagonal_df["product_1"].tolist())
        index = {product: position for position, product in enumerate(products)}
        n_products = len(products)

        occ = np.zeros(n_products, dtype=float)
        occ[diagonal_df["product_1"].map(index).to_numpy()] = diagonal_df["n_customers"].to_numpy()

        both = np.zeros((n_products, n_products), dtype=float)
        if len(cooccurrence_df) > 0:
            rows = cooccurrence_df["product_1"].map(index).to_numpy()
            map_cols = cooccurrence_df["product_2"].map(index).to_numpy()
            both[rows, map_cols] = cooccurrence_df["n_customers"].to_numpy()
            both += both.T

        occ_i = occ[:, None]
        occ_j = occ[None, :]
        b = occ_i - both
        c = occ_j - both
        d = n_customers - occ_i - occ_j + both

        ad = both * d
        bc = b * c
        denominator = ad + bc
        # Undefined Q (denominator 0) -> 0.0 so distance is 0.5 rather than NaN.
        yules_q = np.divide(ad - bc, denominator, out=np.zeros_like(ad), where=denominator != 0)

        # 1 - Q lives in [0, 2]; halving rescales into [0, 1]. Force an exact zero diagonal so
        # the result is a valid distance matrix regardless of floating-point rounding.
        distances = (1.0 - yules_q) / 2.0
        np.fill_diagonal(distances, 0.0)
        return distances, products

    def _calculate_distances(
        self,
        pairs: ibis.Table,
        method: Literal["yules_q"],
    ) -> tuple[np.ndarray, list[str]]:
        """Calculates distances between items using the specified method.

        Args:
            pairs (ibis.Table): Distinct ``[customer_id, product_col]`` pairs.
            method (Literal["yules_q"]): The method to use for calculating distances.

        Raises:
            ValueError: If the method is not valid.

        Returns:
            tuple[np.ndarray, list[str]]: A square matrix of pairwise product distances and the
                sorted product labels indexing it.
        """
        if method == "yules_q":
            return self._get_yules_q_distances(pairs, self.product_col)
        raise ValueError("Method must be 'yules_q'")

    def _compute_linkage_matrix(self) -> np.ndarray:
        """Compute the hierarchical-clustering linkage matrix from precomputed distances.

        scipy's ``linkage`` infers input semantics from shape: a 1-D array is treated as a
        condensed distance vector; a 2-D array is treated as an observation matrix and scipy
        recomputes Euclidean distances between rows. ``squareform`` converts the square
        distance matrix to condensed form so scipy uses the precomputed distances directly.
        """
        return linkage(squareform(self.distances, checks=False), method="ward")

    def plot(
        self,
        title: str = "Customer Decision Hierarchy",
        x_label: str | None = None,
        y_label: str | None = None,
        ax: Axes | None = None,
        figsize: tuple[int, int] | None = None,
        eyebrow: str | None = None,
        subtitle: str | None = None,
        source_text: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> SubplotBase:
        """Plots the customer decision hierarchy dendrogram.

        Args:
            title (str, optional): The title of the plot. Defaults to "Customer Decision Hierarchy".
            x_label (str, optional): The label for the x-axis. Defaults to None.
            y_label (str, optional): The label for the y-axis. Defaults to None.
            ax (Axes, optional): The matplotlib Axes object to plot on. Defaults to None.
            figsize (tuple[int, int], optional): The figure size. Defaults to None.
            eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
            subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
            source_text (str, optional): The source text to annotate on the plot. Defaults to None.
            **kwargs (Any): Additional keyword arguments to pass to the dendrogram function.

        Returns:
            SubplotBase: The matplotlib SubplotBase object.
        """
        linkage_matrix = self._compute_linkage_matrix()
        labels = self.products

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        orientation = kwargs.get("orientation", "top")
        distance_is_y = orientation in ["top", "bottom"]
        default_x_label, default_y_label = (None, "Distance") if distance_is_y else ("Distance", None)

        dendrogram(linkage_matrix, labels=labels, ax=ax, **kwargs)

        # Move ticks/labels for orientations whose categorical axis sits on the
        # opposite side from the matplotlib default.
        if orientation == "left":
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        elif orientation == "bottom":
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")

        # Chrome runs last so its tight_layout sees the populated axes (rotated
        # category labels included) and reserves room for them — matching the
        # call order of every other plot module. _auto_rotate_categorical_x_ticks
        # inside standard_graph_styles owns x-tick rotation for top/bottom
        # orientations.
        standard_graph_styles(
            ax=ax,
            title=title,
            x_label=x_label if x_label is not None else default_x_label,
            y_label=y_label if y_label is not None else default_y_label,
            eyebrow=eyebrow,
            subtitle=subtitle,
            source_text=source_text,
            grid_axis="y" if distance_is_y else "x",
        )

        return ax
