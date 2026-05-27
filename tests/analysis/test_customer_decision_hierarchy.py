"""Tests for the customer_decision_hierarchy module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

import openretailscience.analysis.customer_decision_hierarchy as rp
from openretailscience.options import ColumnHelper, option_context

cols = ColumnHelper()


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


class TestCustomerDecisionHierarchy:
    """Tests for the CustomerDecisionHierarchy class."""

    @pytest.mark.parametrize(
        ("bought_product_1", "bought_product_2", "expected_q"),
        [
            pytest.param(
                np.array([1, 0, 1, 0, 1], dtype=bool),
                np.array([1, 0, 1, 0, 1], dtype=bool),
                1.0,
                id="identical_arrays_return_one",
            ),
            pytest.param(
                np.array([1, 0, 1, 0, 1], dtype=bool),
                np.array([0, 1, 0, 1, 0], dtype=bool),
                -1.0,
                id="opposite_arrays_return_minus_one",
            ),
            pytest.param(
                np.array([], dtype=bool),
                np.array([], dtype=bool),
                0.0,
                id="empty_arrays_return_zero",
            ),
        ],
    )
    def test_calculate_yules_q_returns_expected_value(self, bought_product_1, bought_product_2, expected_q):
        """Test that Yule's Q returns the expected value for identical, opposite, and empty arrays."""
        assert rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2) == expected_q

    def test_calculate_yules_q_different_length_arrays(self):
        """Test that the function raises a ValueError when the arrays have different lengths."""
        bought_product_1 = np.array([1, 0, 1, 0, 1], dtype=bool)
        bought_product_2 = np.array([1, 0, 1, 0], dtype=bool)

        with pytest.raises(ValueError):
            rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2)

    @pytest.mark.parametrize(
        ("bought_product_1", "bought_product_2"),
        [
            pytest.param(
                np.array([True, True, True]),
                np.array([True, True, True]),
                id="all_customers_buy_both",
            ),
            pytest.param(
                np.array([True, True, True]),
                np.array([False, False, False]),
                id="no_overlap_no_neither",
            ),
        ],
    )
    def test_calculate_yules_q_zero_denominator_returns_zero(self, bought_product_1, bought_product_2):
        """Test that Yule's Q returns 0.0 when the denominator is zero."""
        result = rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2)

        assert result == 0.0

    def test_get_yules_q_distances(self):
        """Test that the function returns the correct Yules Q distances."""
        bought_product_1 = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=bool)
        bought_product_2 = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
        expected_q = -0.6363636363636364

        assert rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2) == expected_q

    def test_init_invalid_dataframe(self):
        """Test that the function raises a ValueError when the dataframe is invalid."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.transaction_id: [1, 2, 3],
                "product_name": ["Coke", "Pepsi", "Sprite"],
            },
        )
        exclude_same_transaction_products = True

        with pytest.raises(ValueError):
            rp.CustomerDecisionHierarchy(df, "invalid_product_col", exclude_same_transaction_products)

    def test_init_exclude_same_transaction_products_true(self):
        """Test that the function returns the correct pairs dataframe when exclude_same_transaction_products is True."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 2, 2, 2, 3, 3],
                cols.transaction_id: [1, 1, 2, 3, 3, 4, 5, 6],
                "product_name": ["Coke", "Pepsi", "Sprite", "Fanta", "Tonic", "Tonic", "Tonic", "Tonic"],
            },
        )
        exclude_same_transaction_products = True

        pairs_df = rp.CustomerDecisionHierarchy._get_pairs(
            df,
            exclude_same_transaction_products,
            product_col="product_name",
        )

        expected_pairs_df = pd.DataFrame(
            {cols.customer_id: [1, 3], "product_name": ["Sprite", "Tonic"]},
        ).astype("category")

        assert pairs_df.equals(expected_pairs_df)

    def test_init_exclude_same_transaction_products_false(self):
        """Test correct pairs dataframe when exclude_same_transaction_products is False."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 2, 2, 2, 3, 3],
                cols.transaction_id: [1, 1, 2, 3, 3, 4, 5, 6],
                "product_name": ["Coke", "Pepsi", "Sprite", "Fanta", "Tonic", "Tonic", "Tonic", "Tonic"],
            },
        )
        exclude_same_transaction_products = False

        pairs_df = rp.CustomerDecisionHierarchy._get_pairs(
            df,
            exclude_same_transaction_products,
            product_col="product_name",
        )

        expected_pairs_df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 2, 2, 3],
                "product_name": ["Coke", "Pepsi", "Sprite", "Fanta", "Tonic", "Tonic"],
            },
        ).astype("category")

        assert pairs_df.equals(expected_pairs_df)

    def test_with_custom_column_names(self):
        """Test CustomerDecisionHierarchy with custom column names to ensure column overrides work correctly."""
        # fmt: off
        custom_test_df = pd.DataFrame(
            {
                "cust_identifier": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
                "txn_identifier": [101, 102, 201, 202, 301, 302, 401, 402, 501, 502, 601, 602, 701, 702, 801, 802],
                "product_name": [
                    "Coke", "Pepsi", "Coke", "Sprite", "Pepsi", "Sprite",
                    "Coke", "Fanta", "Pepsi", "Fanta", "Sprite", "Fanta",
                    "Coke", "Pepsi", "Sprite", "Fanta",
                ],
            },
        )
        # fmt: on

        with option_context("column.customer_id", "cust_identifier", "column.transaction_id", "txn_identifier"):
            hierarchy = rp.CustomerDecisionHierarchy(
                df=custom_test_df,
                product_col="product_name",
                method="yules_q",
            )

            assert "cust_identifier" in hierarchy.pairs_df.columns, "Should handle custom customer_id column name"
            assert "product_name" in hierarchy.pairs_df.columns, "Should handle product column"


class TestPlot:
    """Tests for CustomerDecisionHierarchy.plot."""

    LONG_PRODUCT_NAMES = (
        "Premium Organic Whole Bean Coffee 1kg",
        "Single Origin Ethiopian Yirgacheffe 500g",
        "Decaffeinated Swiss Water Process Blend 250g",
        "Cold Brew Concentrate Vanilla Bean 1L",
        "Espresso Roast Italian Dark Blend 750g",
    )

    SHORT_PRODUCT_NAMES = (
        "Coke",
        "Pepsi",
        "Sprite",
        "Fanta",
        "Tonic",
    )

    @staticmethod
    def _build_cdh(product_names: tuple[str, ...]) -> rp.CustomerDecisionHierarchy:
        """Build a CDH with the deterministic 14-row purchase pattern over five products."""
        purchase_pattern = [
            (1, 1, product_names[0]),
            (1, 2, product_names[1]),
            (2, 3, product_names[0]),
            (2, 4, product_names[1]),
            (3, 5, product_names[2]),
            (3, 6, product_names[3]),
            (4, 7, product_names[2]),
            (4, 8, product_names[3]),
            (5, 9, product_names[4]),
            (6, 10, product_names[4]),
            (7, 11, product_names[0]),
            (7, 12, product_names[2]),
            (8, 13, product_names[1]),
            (8, 14, product_names[3]),
        ]
        df = pd.DataFrame(
            purchase_pattern,
            columns=[cols.customer_id, cols.transaction_id, "product_name"],
        )
        return rp.CustomerDecisionHierarchy(df=df, product_col="product_name")

    @pytest.fixture
    def long_label_cdh(self):
        """A CDH whose products have long names that would be clipped if chrome ran first."""
        return self._build_cdh(self.LONG_PRODUCT_NAMES)

    @pytest.fixture
    def short_label_cdh(self):
        """A CDH whose product names are short enough to fit horizontally at a generous figsize."""
        return self._build_cdh(self.SHORT_PRODUCT_NAMES)

    def test_chrome_layout_reserves_room_for_rotated_tick_labels(self, long_label_cdh):
        """Chrome must run after the dendrogram so its tight_layout sees the rotated x-tick labels.

        Regression: standard_graph_styles was previously called before dendrogram, so the
        chrome's tight_layout reserved no room for the rotated 45-degree product-name labels.
        With long category labels and orientation="top", they extended below the figure and
        were clipped.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        long_label_cdh.plot(title="Substitutability", ax=ax, orientation="top")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_y0_fig = [
            t.get_window_extent(renderer=renderer).y0 / fig.bbox.height for t in ax.get_xticklabels() if t.get_text()
        ]
        assert len(label_y0_fig) == len(self.LONG_PRODUCT_NAMES)
        assert min(label_y0_fig) >= 0.0, (
            f"rotated x-tick labels extend below the figure (min y0={min(label_y0_fig):.3f}); "
            "chrome layout must run after the dendrogram so tight_layout reserves space for them"
        )

    def test_short_labels_render_horizontally_via_auto_rotate(self, short_label_cdh):
        """Short product names should be left horizontal by _auto_rotate_categorical_x_ticks.

        With a manual ``plt.setp(rotation=45)`` before ``standard_graph_styles``, every
        dendrogram tilts to 45° regardless of available width, because the auto-rotate
        helper short-circuits when the current rotation is not 0 or 90. Letting the
        helper own rotation produces 0° here since five short names fit at this figsize.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        short_label_cdh.plot(title="Substitutability", ax=ax, orientation="top")
        fig.canvas.draw()
        rotations = {t.get_rotation() for t in ax.get_xticklabels() if t.get_text()}
        assert rotations == {0.0}, (
            f"expected horizontal labels for short product names at generous figsize, got {rotations}"
        )

    def test_distances_form_valid_distance_matrix(self, short_label_cdh):
        """The distance matrix must have a zero diagonal and values in [0, 1].

        Regression: ``_get_yules_q_distances`` previously rescaled with ``(matrix + 1) / 2``,
        which is the formula for converting raw Yule's Q from [-1, 1] to a [0, 1] similarity.
        Applied to the already-converted ``1 - Q`` distances, it shifted the diagonal to 0.5
        and compressed off-diagonals into [0.5, 1.5] — not a metric distance matrix.
        """
        distances = short_label_cdh.distances
        assert np.array_equal(np.diag(distances), np.zeros(distances.shape[0])), (
            f"distance matrix must have zero diagonal; got {np.diag(distances)}"
        )
        assert distances.min() >= 0.0, f"distance values must be non-negative; got min {distances.min()}"
        assert distances.max() <= 1.0, f"distance values must be at most 1.0; got max {distances.max()}"

    def test_plot_treats_distances_as_precomputed_not_observations(self, short_label_cdh):
        """``plot`` must pass distances to ``linkage`` in condensed form.

        Otherwise scipy treats the NxN distance matrix as an observation matrix and
        recomputes Euclidean distances between rows, collapsing similar products to
        zero and pinning unrelated ones to large values.
        """
        expected_linkage = linkage(squareform(short_label_cdh.distances, checks=False), method="ward")
        actual_linkage = short_label_cdh._compute_linkage_matrix()
        np.testing.assert_allclose(actual_linkage, expected_linkage)
