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

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform

from openretailscience.options import ColumnHelper, get_option
from openretailscience.plots.styles.styling_helpers import standard_graph_styles


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
        df: pd.DataFrame,
        product_col: str,
        exclude_same_transaction_products: bool = True,
        method: Literal["yules_q"] = "yules_q",
        random_state: int = 42,
    ) -> None:
        """Initialize customer decision hierarchy analysis for range optimization.

        Args:
            df (pd.DataFrame): Transaction data with customer purchase history.
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
                binary purchase patterns. Defaults to "yules_q".
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
        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        self.random_state = random_state
        self.product_col = product_col
        self.pairs_df = self._get_pairs(df, exclude_same_transaction_products, product_col)
        self.distances = self._calculate_distances(method=method)

    @staticmethod
    def _get_pairs(df: pd.DataFrame, exclude_same_transaction_products: bool, product_col: str) -> pd.DataFrame:
        cols = ColumnHelper()
        if exclude_same_transaction_products:
            pairs_df = df[[cols.customer_id, cols.transaction_id, product_col]].drop_duplicates()
            pairs_to_exclude_df = (
                pairs_df.groupby(cols.transaction_id)
                .filter(lambda x: len(x) > 1)[[cols.customer_id, product_col]]
                .drop_duplicates()
            )
            # Drop all rows from pairs_df where customer_id and product_name are in pairs_to_exclude_df
            pairs_df = pairs_df.merge(
                pairs_to_exclude_df,
                on=[cols.customer_id, product_col],
                how="left",
                indicator=True,
            )
            pairs_df = pairs_df[pairs_df["_merge"] == "left_only"][[cols.customer_id, product_col]].drop_duplicates()
        else:
            pairs_df = df[[cols.customer_id, product_col]].drop_duplicates()

        return pairs_df.reset_index(drop=True).astype("category")

    @staticmethod
    def _calculate_yules_q(bought_product_1: np.array, bought_product_2: np.array) -> float:
        """Calculates the Yule's Q coefficient between two binary arrays.

        Args:
            bought_product_1 (np.array): Binary array representing the first bought product. Each element is 1 if the
                customer bought the product and 0 if they didn't.
            bought_product_2 (np.array): Binary array representing the second bought product. Each element is 1 if the
                customer bought the product and 0 if they didn't.

        Returns:
            float: The Yule's Q coefficient.

        Raises:
            ValueError: If the lengths of `bought_product_1` and `bought_product_2` are not the same.
            ValueError: If `bought_product_1` or `bought_product_2` is not a boolean array.

        """
        if len(bought_product_1) != len(bought_product_2):
            raise ValueError("The bought_product_1 and bought_product_2 must be the same length")
        if len(bought_product_1) == 0:
            return 0.0
        if bought_product_1.dtype != bool or bought_product_2.dtype != bool:
            raise ValueError("The bought_product_1 and bought_product_2 must be boolean arrays")

        a = np.count_nonzero(bought_product_1 & bought_product_2)
        b = np.count_nonzero(bought_product_1 & ~bought_product_2)
        c = np.count_nonzero(~bought_product_1 & bought_product_2)
        d = np.count_nonzero(~bought_product_1 & ~bought_product_2)

        # Calculate Yule's Q coefficient
        denominator = a * d + b * c
        if denominator == 0:
            # Both a*d and b*c are zero, making Q mathematically undefined (0/0).
            # Return 0.0 (no association) because NaN would break scipy's linkage() downstream.
            return 0.0

        return (a * d - b * c) / denominator

    def _get_yules_q_distances(self) -> np.ndarray:
        """Calculate the Yule's Q distances between pairs of products.

        Returns:
            np.ndarray: A square matrix of Yule's Q distances between pairs of products.
        """
        # Create a sparse matrix where the rows are the customers and the columns are the products
        # The values are True if the customer bought the product and False if they didn't
        product_matrix = csr_matrix(
            (
                [True] * len(self.pairs_df),
                (
                    self.pairs_df[self.product_col].cat.codes,
                    self.pairs_df[get_option("column.customer_id")].cat.codes,
                ),
            ),
            dtype=bool,
        )

        # Calculate the number of customers and products
        n_products = product_matrix.shape[0]

        # Create an empty matrix to store the yules q values
        yules_q_matrix = np.zeros((n_products, n_products), dtype=float)

        # Loop through each pair of products
        for i in range(n_products):
            arr_i = product_matrix[i].toarray()
            for j in range(i + 1, n_products):
                # Calculate the yules q value for the pair of products
                arr_j = product_matrix[j].toarray()
                yules_q_dist = 1 - self._calculate_yules_q(arr_i, arr_j)

                # Store the yules q value in the matrix
                yules_q_matrix[i, j] = yules_q_dist
                yules_q_matrix[j, i] = yules_q_dist

        # yules_q_dist = 1 - Q lives in [0, 2]; halving rescales it into [0, 1] while
        # preserving the zero diagonal so this is a valid distance matrix.
        return yules_q_matrix / 2

    def _calculate_distances(
        self,
        method: Literal["yules_q"],
    ) -> np.ndarray:
        """Calculates distances between items using the specified method.

        Args:
            method (Literal["yules_q"], optional): The method to use for calculating distances.

        Raises:
            ValueError: If the method is not valid.

        Returns:
            np.ndarray: A square matrix of pairwise product distances.
        """
        # Check method is valid
        if method == "yules_q":
            distances = self._get_yules_q_distances()
        else:
            raise ValueError("Method must be 'yules_q'")

        return distances

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
        labels = self.pairs_df[self.product_col].cat.categories

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
