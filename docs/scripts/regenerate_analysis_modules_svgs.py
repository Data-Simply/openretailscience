"""Regenerate the SVGs used on ``docs/analysis_modules.md``.

Each chart on the analysis modules page is a curated artefact: the eyebrow,
narrative title, and subtitle are hand-crafted so the headline is literally
true at a glance for the rendered data. The synthetic data here is sized to
produce those headlines — it is not the same as the inline example code in
``analysis_modules.md``, which is kept minimal for readers.

Run from the repo root::

    uv run python docs/scripts/regenerate_analysis_modules_svgs.py

Outputs are written to ``docs/assets/images/analysis_modules/`` and the
maintainer is expected to commit them alongside any plot-styling or data
changes that motivated the regeneration.

This script is intentionally not referenced from the public docs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

mpl.use("Agg")

from openretailscience.analysis.cross_shop import CrossShop
from openretailscience.analysis.customer import (
    DaysBetweenPurchases,
    PurchasesPerCustomer,
    TransactionChurn,
)
from openretailscience.analysis.customer_decision_hierarchy import CustomerDecisionHierarchy
from openretailscience.analysis.gain_loss import GainLoss
from openretailscience.plots import area, bar, histogram
from openretailscience.plots.styles.graph_utils import set_axis_percent, set_axis_shorthand
from openretailscience.segmentation.hml import HMLSegmentation
from openretailscience.segmentation.threshold import ThresholdSegmentation

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "images" / "analysis_modules"


def _save(fig: Figure, name: str) -> None:
    path = OUT_DIR / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(Path.cwd())}")


def build_transaction_panel() -> pd.DataFrame:
    """Synthetic 12-month panel shared by HML, Threshold, and the customer-analysis charts."""
    rng = np.random.default_rng(7)
    n_customers = 2_500
    customer_ids = np.arange(1, n_customers + 1)
    lifetime_spend = rng.lognormal(mean=4.0, sigma=1.0, size=n_customers).round(2)
    n_visits = np.clip(rng.lognormal(mean=1.5, sigma=0.6, size=n_customers).round().astype(int), 1, 25)

    rows: list[tuple[int, int, pd.Timestamp, float, int]] = []
    tid = 100_000
    for cid, total_spend, visits in zip(customer_ids, lifetime_spend, n_visits, strict=True):
        splits = rng.dirichlet(np.ones(visits))
        spends = (splits * total_spend).round(2)
        visit_offsets = np.sort(rng.integers(0, 365, size=visits))
        visit_dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(visit_offsets, unit="D")
        for spend, dt in zip(spends, visit_dates, strict=True):
            rows.append((int(cid), tid, dt, float(spend), int(rng.integers(1, 5))))
            tid += 1

    # pandas-stubs rejects list[str] for the columns parameter (its SequenceNotStr protocol
    # does not accept list); pd.Index is an accepted member of the columns type union.
    return pd.DataFrame(
        rows,
        columns=pd.Index(["customer_id", "transaction_id", "transaction_date", "unit_spend", "unit_quantity"]),
    )


def regenerate_hml_segmentation(transactions: pd.DataFrame) -> None:
    hml = HMLSegmentation(transactions, zero_value_customers="include_with_light")
    # groupby(...).sum().reindex(...) on a single column returns a Series, but pandas-stubs
    # widens it to DataFrame | NDFrame | Series. Narrow back to Series for the plot call.
    segment_spend = cast(
        "pd.Series",
        hml.df.groupby("segment_name", observed=True)["unit_spend"].sum().reindex(["Heavy", "Medium", "Light"]),
    )
    ratio = segment_spend["Heavy"] / segment_spend["Light"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    bar.plot(
        segment_spend,
        value_col="unit_spend",
        ax=ax,
        eyebrow="HML SEGMENTATION",
        title=f"Heavy customers drive {ratio:.1f}× the spend of light customers",
        subtitle="Total segment spend across the 20/30/50 Heavy / Medium / Light split",
        source_text="Source: synthetic 12-month customer panel",
        sort_order="descending",
        x_label="",
        y_label="Segment spend ($)",
        rot=0,
    )
    set_axis_shorthand(ax.yaxis)
    _save(fig, "hml_segmentation")


def regenerate_threshold_segmentation(transactions: pd.DataFrame) -> None:
    thresh = ThresholdSegmentation(
        df=transactions,
        thresholds=[0.25, 0.50, 0.75, 1.0],
        segments=["Bronze", "Silver", "Gold", "Platinum"],
        zero_value_customers="separate_segment",
    )
    # groupby(...).sum().reindex(...) on a single column returns a Series, but pandas-stubs
    # widens it to DataFrame | NDFrame | Series. Narrow back to Series for the plot call.
    quartile_spend = cast(
        "pd.Series",
        thresh.df.groupby("segment_name", observed=True)["unit_spend"]
        .sum()
        .reindex(["Platinum", "Gold", "Silver", "Bronze"]),
    )
    platinum_share = quartile_spend["Platinum"] / quartile_spend.sum()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    bar.plot(
        quartile_spend,
        value_col="unit_spend",
        ax=ax,
        eyebrow="THRESHOLD SEGMENTATION",
        title=f"Platinum customers concentrate {platinum_share:.0%} of total spend",
        subtitle="Quartile-based segmentation: Bronze / Silver / Gold / Platinum",
        source_text="Source: synthetic 12-month customer panel",
        x_label="",
        y_label="Segment spend ($)",
        rot=0,
    )
    set_axis_shorthand(ax.yaxis)
    _save(fig, "threshold_segmentation")


def regenerate_purchases_per_customer(transactions: pd.DataFrame) -> None:
    ppc = PurchasesPerCustomer(transactions)
    p80 = ppc.purchases_percentile(0.8)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    histogram.plot(
        df=ppc.cust_purchases_s,
        ax=ax,
        eyebrow="PURCHASE FREQUENCY",
        title=f"80% of customers make {int(p80)} purchases or fewer",
        subtitle="Number of lifetime purchases per customer; dashed line marks the churn-window cut-off",
        x_label="Number of purchases",
        y_label="Number of customers",
        source_text="Source: synthetic 12-month customer panel",
    )
    ax.axvline(x=p80, color="black", linestyle="--", lw=2)
    set_axis_shorthand(ax.yaxis)
    _save(fig, "purchases_per_customer")


def regenerate_days_between_purchases(transactions: pd.DataFrame) -> None:
    dbp = DaysBetweenPurchases(transactions)
    median_days = dbp.purchases_percentile(0.5)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # Trim the long right tail (a handful of customers with averages >200 days) so the
    # body of the distribution and the median line are easier to read.
    # Boolean-mask indexing on a Series returns a Series, but pandas-stubs widens the result
    # type; narrow it back for the plot call.
    series = cast("pd.Series", dbp.purchase_dist_s[dbp.purchase_dist_s <= 200])
    histogram.plot(
        df=series,
        ax=ax,
        bins=25,
        eyebrow="REPURCHASE CADENCE",
        title=f"Half of customers come back within {round(median_days)} days",
        subtitle="Average days between purchases per customer; dashed line marks the median",
        x_label="Average days between purchases",
        y_label="Number of customers",
        source_text="Source: synthetic 12-month customer panel",
    )
    ax.axvline(x=median_days, color="black", linestyle="--", lw=2)
    set_axis_shorthand(ax.yaxis)
    _save(fig, "days_between_purchases")


def regenerate_transaction_churn(transactions: pd.DataFrame) -> None:
    churn_window_days = 60
    tc = TransactionChurn(transactions, churn_period=churn_window_days)
    # cumsum on `churned` propagates NaN when a purchase index has no churners,
    # leaving holes in the cumulative line. Fill them forward so the area renders
    # as a single connected shape, and clip to the first 15 purchases — beyond that
    # the per-bin customer count drops below 20 and the line is noisy.
    cumulative = tc.purchase_dist_df["churned"].cumsum().ffill() / tc.n_unique_customers
    cumulative = cumulative.loc[:15].to_frame(name="cumulative_churn_rate")
    plateau_rate = cumulative["cumulative_churn_rate"].iloc[-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    area.plot(
        df=cumulative,
        value_col="cumulative_churn_rate",
        ax=ax,
        eyebrow="TRANSACTION CHURN",
        title="Churn climbs sharply through the 5th purchase, then plateaus",
        subtitle=(
            f"Cumulative churn share by lifetime purchase count, {churn_window_days}-day window; "
            f"plateau ≈ {plateau_rate:.0%}"
        ),
        x_label="Number of purchases",
        y_label="% of customers churned (cumulative)",
        source_text="Source: synthetic 12-month customer panel",
    )
    set_axis_percent(ax.yaxis)
    _save(fig, "transaction_churn")


def regenerate_cross_shop() -> None:
    rng = np.random.default_rng(11)
    n_customers = 800
    cust_ids = np.arange(1, n_customers + 1)
    modes = rng.choice(
        ["E_only", "C_only", "H_only", "E+C", "E+H", "C+H", "E+C+H"],
        size=n_customers,
        p=[0.08, 0.10, 0.12, 0.18, 0.16, 0.14, 0.22],
    )
    mode_to_cats = {
        "E_only": ["Electronics"],
        "C_only": ["Clothing"],
        "H_only": ["Home"],
        "E+C": ["Electronics", "Clothing"],
        "E+H": ["Electronics", "Home"],
        "C+H": ["Clothing", "Home"],
        "E+C+H": ["Electronics", "Clothing", "Home"],
    }
    rows: list[tuple[int, str, float]] = []
    for cid, mode in zip(cust_ids, modes, strict=True):
        for cat in mode_to_cats[mode]:
            rows.append((int(cid), cat, float(rng.integers(20, 300))))
    cs_df = pd.DataFrame(rows, columns=pd.Index(["customer_id", "category_name", "unit_spend"]))

    cs = CrossShop(
        cs_df,
        group_1_col="category_name",
        group_1_val="Electronics",
        group_2_val="Clothing",
        group_3_val="Home",
        labels=["Electronics", "Clothing", "Home"],
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    cs.plot(
        ax=ax,
        eyebrow="CROSS-SHOP",
        title="One in three shoppers buys across all three categories",
        subtitle="Customer overlap across Electronics, Clothing, and Home",
        source_text="Source: synthetic 800-customer panel",
    )
    _save(fig, "cross_shop")


def regenerate_gain_loss() -> None:
    rng = np.random.default_rng(13)
    n_customers = 300
    customers = [f"C{i:04d}" for i in range(n_customers)]
    # Skew the patterns so Brand A is clearly winning — loyal A shoppers also
    # spend meaningfully more in P2 so the "Increased Brand A" band is visible
    # rather than a thin sliver.
    patterns = rng.choice(
        ["loyal_A", "loyal_B", "lost_A", "new_A", "switch_AtoB", "switch_BtoA"],
        size=n_customers,
        p=[0.28, 0.18, 0.05, 0.18, 0.06, 0.25],
    )
    rows: list[tuple[str, float, str, str]] = []
    for cid, pat in zip(customers, patterns, strict=True):
        if pat == "loyal_A":
            rows.append((cid, float(rng.integers(30, 80)), "Brand A", "p1"))
            rows.append((cid, float(rng.integers(70, 160)), "Brand A", "p2"))
        elif pat == "loyal_B":
            rows.append((cid, float(rng.integers(60, 140)), "Brand B", "p1"))
            rows.append((cid, float(rng.integers(50, 120)), "Brand B", "p2"))
        elif pat == "lost_A":
            rows.append((cid, float(rng.integers(40, 120)), "Brand A", "p1"))
        elif pat == "new_A":
            rows.append((cid, float(rng.integers(60, 160)), "Brand A", "p2"))
        elif pat == "switch_AtoB":
            rows.append((cid, float(rng.integers(40, 100)), "Brand A", "p1"))
            rows.append((cid, float(rng.integers(40, 100)), "Brand B", "p2"))
        elif pat == "switch_BtoA":
            rows.append((cid, float(rng.integers(60, 160)), "Brand B", "p1"))
            rows.append((cid, float(rng.integers(70, 180)), "Brand A", "p2"))
    gl_df = pd.DataFrame(rows, columns=pd.Index(["customer_id", "unit_spend", "brand", "period"]))

    gain_loss = GainLoss(
        df=gl_df,
        p1_index=gl_df["period"] == "p1",
        p2_index=gl_df["period"] == "p2",
        focus_group_index=gl_df["brand"] == "Brand A",
        focus_group_name="Brand A",
        comparison_group_index=gl_df["brand"] == "Brand B",
        comparison_group_name="Brand B",
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    gain_loss.plot(
        ax=ax,
        eyebrow="GAIN / LOSS",
        title="Brand A nets revenue gains from switchers and new shoppers",
        subtitle="Where Brand A's revenue change came from, period-over-period",
        x_label="Revenue change ($)",
        source_text="Source: synthetic 250-customer panel, P1 vs P2",
        move_legend_outside=True,
    )
    set_axis_shorthand(ax.xaxis)
    _save(fig, "gain_loss")


def regenerate_customer_decision_hierarchy() -> None:
    rng = np.random.default_rng(42)
    clusters = [
        ["Dark Chocolate Bar", "Caramel Chocolate Bar", "Milk Chocolate Bar"],
        ["Wheat Crackers", "Peanut Butter Crackers", "Cheese Crackers"],
        ["Sour Cream Chips", "BBQ Chips", "Salted Chips"],
        ["Salted Pretzels", "Honey Mustard Pretzels"],
    ]
    n_per_cluster = 60
    # Within-cluster brackets and between-cluster joins are both balanced by giving most
    # shoppers a primary cluster *and* an occasional dip into an adjacent one. Without the
    # adjacency dips, brackets pin to 0 and joins fly out to 1.5+, leaving the chart as
    # short stubs joined by long blue rails.
    adjacency = {
        0: [1],  # chocolate → crackers
        1: [0, 2],  # crackers → chocolate, chips
        2: [1, 3],  # chips → crackers, pretzels
        3: [2],  # pretzels → chips
    }
    rows: list[tuple[int, int, str]] = []
    next_customer_id = 1_000
    next_transaction_id = 5_000
    for primary_idx, cluster in enumerate(clusters):
        for _ in range(n_per_cluster):
            # 1, 2, or 3 SKUs from the favoured cluster — weighted toward 2 so within-cluster
            # Yule's Q is strong but variable enough that bracket heights are nonzero.
            cluster_size = len(cluster)
            if cluster_size >= 3:
                n_skus = int(rng.choice([1, 2, 3], p=[0.25, 0.45, 0.30]))
            else:
                n_skus = int(rng.choice([1, 2], p=[0.35, 0.65]))
            chosen = rng.choice(cluster, size=n_skus, replace=False)
            for sku in chosen:
                rows.append((next_customer_id, next_transaction_id, str(sku)))
                next_transaction_id += 1
            # Roughly 35% of shoppers also pick up one SKU from an adjacent cluster, which
            # gives inter-cluster joins a meaningful (rather than maxed-out) distance.
            if rng.random() < 0.35:
                neighbour_idx = int(rng.choice(adjacency[primary_idx]))
                neighbour_sku = str(rng.choice(clusters[neighbour_idx]))
                rows.append((next_customer_id, next_transaction_id, neighbour_sku))
                next_transaction_id += 1
            next_customer_id += 1

    snack_transactions = pd.DataFrame(rows, columns=pd.Index(["customer_id", "transaction_id", "product_name"]))
    cdh = CustomerDecisionHierarchy(snack_transactions, product_col="product_name")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    cdh.plot(
        ax=ax,
        orientation="right",
        eyebrow="SNACK ASSORTMENT",
        title="Shoppers cluster around four snack archetypes",
        subtitle="Substitutability distance across 11 SKUs; closer joins mean tighter cross-purchase",
        source_text="Source: synthetic snack panel, ~240 shoppers",
    )
    _save(fig, "customer_decision_hierarchy")


def main() -> None:
    transactions = build_transaction_panel()
    print(f"built panel: {len(transactions):,} transactions, {transactions['customer_id'].nunique():,} customers")
    regenerate_hml_segmentation(transactions)
    regenerate_threshold_segmentation(transactions)
    regenerate_purchases_per_customer(transactions)
    regenerate_days_between_purchases(transactions)
    regenerate_transaction_churn(transactions)
    regenerate_cross_shop()
    regenerate_gain_loss()
    regenerate_customer_decision_hierarchy()
    print("done")


if __name__ == "__main__":
    main()
