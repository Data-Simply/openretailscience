"""Tests for the customer_decision_hierarchy module."""

import ibis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
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

    def test_distances_match_known_yules_q_matrix(self):
        """The full distance matrix must match hand-computed Yule's Q distances for a known pattern.

        Worked from the deduplicated customer/product pairs over a five-customer population:
        customers 1-4 buy one product each (A, B, C, D) and customer 5 buys both A and B. So
        occ_A=2, occ_B=2, occ_C=occ_D=1, N=5. For A/B the 2x2 contingency table is a=1 (both),
        b=1 (A only), c=1 (B only), d=2 (neither) -> Q=(2-1)/(2+1)=1/3, distance=(1-Q)/2=1/3.
        Every pair with zero co-occurrence (e.g. A/C, C/D) has a=0 so Q=-1 and distance 1.0.
        This pins the actual off-diagonal values, not just the matrix structure, so a transposed
        or mislabeled reimplementation is caught.
        """
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 5],
                cols.transaction_id: [1, 2, 3, 4, 5, 6],
                "product_name": ["A", "B", "C", "D", "A", "B"],
            },
        )

        cdh = rp.CustomerDecisionHierarchy(df=df, product_col="product_name", exclude_same_transaction_products=False)

        # Categories sort alphabetically: A, B, C, D.
        one_third = 1.0 / 3.0
        expected = np.array(
            [
                [0.0, one_third, 1.0, 1.0],
                [one_third, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
        )
        assert list(cdh.products) == ["A", "B", "C", "D"]
        np.testing.assert_allclose(cdh.distances, expected)

    def test_undefined_yules_q_maps_to_half_distance(self):
        """Pairs with a zero contingency denominator must yield distance 0.5, not NaN.

        When two products are always bought together by every customer, the 2x2 table has
        b=c=d=0, so a*d + b*c = 0 and Yule's Q is mathematically undefined (0/0). The module
        substitutes Q=0 (no association) to keep scipy's linkage from breaking, giving a
        distance of (1 - 0) / 2 = 0.5. A NaN here would propagate into the dendrogram.
        """
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 3, 3],
                cols.transaction_id: [1, 1, 2, 2, 3, 3],
                "product_name": ["A", "B", "A", "B", "A", "B"],
            },
        )

        cdh = rp.CustomerDecisionHierarchy(df=df, product_col="product_name", exclude_same_transaction_products=False)

        assert not np.isnan(cdh.distances).any()
        np.testing.assert_allclose(cdh.distances, np.array([[0.0, 0.5], [0.5, 0.0]]))

    def test_distances_identical_for_pandas_and_ibis_input(self):
        """Passing an ibis.Table must produce the same distance matrix as the equivalent DataFrame.

        Guards the native-Ibis path: the constructor accepts an ibis.Table directly, and the
        result must match the pandas input exactly (same product ordering, same distances).
        """
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 3, 3, 4, 5, 6, 7],
                cols.transaction_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "product_name": [
                    "Coke",
                    "Pepsi",
                    "Coke",
                    "Pepsi",
                    "Sprite",
                    "Fanta",
                    "Sprite",
                    "Fanta",
                    "Coke",
                    "Sprite",
                ],
            },
        )

        from_pandas = rp.CustomerDecisionHierarchy(df=df, product_col="product_name")
        from_ibis = rp.CustomerDecisionHierarchy(df=ibis.memtable(df), product_col="product_name")

        assert list(from_pandas.products) == list(from_ibis.products)
        np.testing.assert_allclose(from_ibis.distances, from_pandas.distances)

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

    @pytest.mark.parametrize(
        ("exclude_same_transaction_products", "expected_pairs"),
        [
            pytest.param(
                True,
                {cols.customer_id: [1, 3], "product_name": ["Sprite", "Tonic"]},
                id="exclude_same_transaction_products_true",
            ),
            pytest.param(
                False,
                {
                    cols.customer_id: [1, 1, 1, 2, 2, 3],
                    "product_name": ["Coke", "Pepsi", "Sprite", "Fanta", "Tonic", "Tonic"],
                },
                id="exclude_same_transaction_products_false",
            ),
        ],
    )
    def test_get_pairs_applies_same_transaction_exclusion(self, exclude_same_transaction_products, expected_pairs):
        """`_get_pairs` keeps only customer/product pairs that survive the same-transaction rule.

        With exclusion on, any customer/product whose product co-occurred with another product in
        a single transaction is dropped entirely for that customer. Customer 1 buys Coke and Pepsi
        together (txn 1), so only their solo Sprite survives; customer 2's Fanta/Tonic share txn 3,
        leaving nothing; customer 3 only ever buys Tonic alone.
        """
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 2, 2, 2, 3, 3],
                cols.transaction_id: [1, 1, 2, 3, 3, 4, 5, 6],
                "product_name": ["Coke", "Pepsi", "Sprite", "Fanta", "Tonic", "Tonic", "Tonic", "Tonic"],
            },
        )

        pairs_table = rp.CustomerDecisionHierarchy._get_pairs(
            ibis.memtable(df),
            exclude_same_transaction_products,
            product_col="product_name",
        )

        actual = (
            pairs_table.execute()
            .sort_values([cols.customer_id, "product_name"])
            .reset_index(drop=True)
            .astype({cols.customer_id: "int64", "product_name": "object"})
        )
        expected = pd.DataFrame(expected_pairs).astype({cols.customer_id: "int64", "product_name": "object"})
        assert_frame_equal(actual, expected)

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

            # The custom customer_id/transaction_id options must be honored end-to-end: the analysis
            # discovers the four products and produces a 4x4 distance matrix without error.
            assert list(hierarchy.products) == ["Coke", "Fanta", "Pepsi", "Sprite"]
            assert hierarchy.distances.shape == (4, 4)


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
