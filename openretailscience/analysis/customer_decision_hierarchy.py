"""Product substitutability from customer co-purchase patterns via Yule's Q and ward hierarchical clustering.

Use to cluster products into substitute versus complement groups for range decisions.
For market-basket co-occurrence rules between products, use
:mod:`openretailscience.analysis.product_association` instead.
"""

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform

from openretailscience.core.validation import ensure_data_has_columns
from openretailscience.options import ColumnHelper, get_option
from openretailscience.plots.styles.styling_helpers import standard_graph_styles


class CustomerDecisionHierarchy:
    """Identifies which products customers treat as substitutes versus complements.

    Substitutes are rarely co-bought; complements are often co-bought. Input is
    transaction-level rows with the option-derived customer_id and transaction_id
    columns plus a product column (``product_col``); with the default
    ``exclude_same_transaction_products=True``, (customer, product) pairs bought
    together in the same transaction are excluded.

    Results, available after initialization:

    - ``pairs_df``: the deduplicated (customer, product) pairs used.
    - ``distances``: square product-distance matrix.
    - ``plot()``: dendrogram of the ward hierarchical clustering.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        product_col: str,
        exclude_same_transaction_products: bool = True,
        method: Literal["yules_q"] = "yules_q",
        random_state: int = 42,
    ) -> None:
        """Initialize the analysis; pairs, distances, and clustering inputs are computed immediately.

        Args:
            df (pd.DataFrame): Transaction-level rows; must contain the option-derived
                customer_id and transaction_id columns and ``product_col``.
            product_col (str): Column with the product identifiers to analyze.
            exclude_same_transaction_products (bool): When True (default), (customer,
                product) pairs bought together in the same transaction are dropped.
            method (Literal["yules_q"]): Distance method; only "yules_q" is accepted.
            random_state (int): Stored but never used; ward linkage on the deterministic
                distance matrix is already reproducible.

        Raises:
            ValueError: If the required columns are missing from the dataframe.
            ValueError: If ``method`` is not "yules_q".
        """
        cols = ColumnHelper()
        required_cols = [cols.customer_id, cols.transaction_id, product_col]
        ensure_data_has_columns(df, required_cols)

        self.random_state = random_state
        self.product_col = product_col
        self.pairs_df = self._get_pairs(df, exclude_same_transaction_products, product_col)
        self.distances = self._calculate_distances(method=method)

    @staticmethod
    def _get_pairs(df: pd.DataFrame, exclude_same_transaction_products: bool, product_col: str) -> pd.DataFrame:
        """Deduplicate (customer, product) pairs from transaction rows.

        With ``exclude_same_transaction_products``, any (customer, product) pair that
        appears in a multi-item transaction for that customer is removed entirely
        (pair-level left-anti-join, not row-level).

        Returns:
            pd.DataFrame: Deduplicated (customer_id, product) frame with categorical dtypes.
        """
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
        """Calculate Yule's Q coefficient between two boolean purchase vectors.

        Q = (ad - bc) / (ad + bc) on the 2x2 co-purchase table. Empty arrays, and
        degenerate tables where a*d + b*c == 0, return 0.0 because NaN would break
        scipy's ``linkage`` downstream.

        Args:
            bought_product_1 (np.array): Boolean vector, True where the customer bought the product.
            bought_product_2 (np.array): Boolean vector, True where the customer bought the product.

        Returns:
            float: The Yule's Q coefficient, in [-1, 1].

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
        """Calculate the square product-distance matrix.

        Distance is (1 - Yule's Q) / 2, rescaled from [0, 2] into [0, 1] so the
        result is a valid distance matrix. Rows and columns are ordered by
        ``pairs_df[product_col].cat.categories``, so ``distances[i, j]`` aligns
        with that category list (the order ``plot()`` uses for its labels).

        Returns:
            np.ndarray: Square matrix of pairwise product distances in [0, 1].
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
        """Calculate the product-distance matrix using the specified method.

        Args:
            method (Literal["yules_q"]): Only "yules_q" is implemented.

        Returns:
            np.ndarray: A square matrix of pairwise product distances.

        Raises:
            ValueError: If the method is not "yules_q".
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
        """Plot the customer decision hierarchy dendrogram.

        ``**kwargs`` are passed through to scipy's ``dendrogram`` (``orientation``
        defaults to "top"). A figure is created when ``ax`` is None.

        Args:
            title (str): The title of the plot.
            x_label (str | None): X-axis label; defaults to "Distance" for left/right orientations.
            y_label (str | None): Y-axis label; defaults to "Distance" for top/bottom orientations.
            ax (Axes | None): Axes to plot on; a new figure is created when None.
            figsize (tuple[int, int] | None): Figure size for the new figure.
            eyebrow (str | None): Small uppercase label rendered above the title.
            subtitle (str | None): Supporting copy rendered below the title.
            source_text (str | None): Source annotation.
            **kwargs (Any): Additional keyword arguments forwarded to scipy's ``dendrogram``.

        Returns:
            SubplotBase: The axes the dendrogram was drawn on.
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
