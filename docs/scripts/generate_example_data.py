"""Regenerate the synthetic transaction datasets used by the example pages in ``docs/examples/``.

The example pages quote numbers computed from these files, so check the affected pages after any change here.

Run from the repo root::

    uv run python docs/scripts/generate_example_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OUT_DIR = Path(__file__).resolve().parent.parent / "examples" / "data"
YEAR_START = pd.Timestamp("2025-01-01")
DAYS_IN_YEAR = 365
ZSTD_LEVEL = 19

# Coffee: some machine buyers buy their pods elsewhere, and some pod buyers own a machine bought elsewhere.
COFFEE_SEED = 7
COFFEE_CUSTOMERS = 5_000
P_MACHINE = 0.16
P_PODS_WITH_MACHINE = 0.48
P_PODS_WITHOUT_MACHINE = 0.12
MEAN_EXTRA_POD_ORDERS = 11
MEAN_OTHER_ITEMS = 2.5
MACHINE_PRICES = [99.0, 149.0, 199.0, 249.0]
POD_PRICE = 6.5
OTHER_COFFEE_ITEMS = {"Ground Coffee": 7.0, "Tea": 4.0, "Kettles": 35.0, "Mugs": 8.0}

# Jeans: (share of customers, mean extra pairs a year beyond the first, brand mix) per shopper type. The affinity page
# relies on Levi's leading Lee's raw overlap through size alone, while Wrangler shares more of Lee's customers than its
# size predicts.
JEANS_SEED = 42
JEANS_CUSTOMERS = 20_000
JEANS_SEGMENTS = {
    "value": (0.35, 0.55, {"Wrangler": 0.32, "Lee": 0.30, "Levi's": 0.32, "Calvin Klein Jeans": 0.05, "Diesel": 0.01}),
    "mainstream": (
        0.53,
        0.30,
        {"Levi's": 0.55, "Calvin Klein Jeans": 0.30, "Wrangler": 0.08, "Lee": 0.04, "Diesel": 0.03},
    ),
    "premium": (
        0.12,
        0.12,
        {"Diesel": 0.88, "Levi's": 0.07, "Calvin Klein Jeans": 0.04, "Wrangler": 0.005, "Lee": 0.005},
    ),
}
JEANS_PRICES = {"Levi's": 79.0, "Wrangler": 49.0, "Lee": 45.0, "Calvin Klein Jeans": 89.0, "Diesel": 149.0}


def _coffee(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for customer_id in range(1, COFFEE_CUSTOMERS + 1):
        machine = rng.random() < P_MACHINE
        pods = rng.random() < (P_PODS_WITH_MACHINE if machine else P_PODS_WITHOUT_MACHINE)
        if machine:
            rows.append((customer_id, "Coffee Machines", float(rng.choice(MACHINE_PRICES))))
        if pods:
            rows += [(customer_id, "Coffee Pods", POD_PRICE)] * (1 + rng.poisson(MEAN_EXTRA_POD_ORDERS))
        n_other = rng.poisson(MEAN_OTHER_ITEMS) + (0 if machine or pods else 1)
        rows += [
            (customer_id, str(item), OTHER_COFFEE_ITEMS[item]) for item in rng.choice(list(OTHER_COFFEE_ITEMS), n_other)
        ]
    return pd.DataFrame(rows, columns=["customer_id", "category_name", "unit_spend"])


def _jeans(rng: np.random.Generator) -> pd.DataFrame:
    names = list(JEANS_SEGMENTS)
    segment = rng.choice(names, size=JEANS_CUSTOMERS, p=[JEANS_SEGMENTS[s][0] for s in names])
    n_pairs = 1 + rng.poisson([JEANS_SEGMENTS[s][1] for s in segment])
    customer_segment = np.repeat(segment, n_pairs)
    brands = np.empty(len(customer_segment), dtype=object)
    for name in names:
        mask = customer_segment == name
        mix = JEANS_SEGMENTS[name][2]
        brands[mask] = rng.choice(list(mix), size=mask.sum(), p=list(mix.values()))
    return pd.DataFrame(
        {
            "customer_id": np.repeat(np.arange(1, JEANS_CUSTOMERS + 1), n_pairs),
            "brand_name": brands,
            "unit_spend": pd.Series(brands).map(JEANS_PRICES).to_numpy(),
        },
    )


def _write(df: pd.DataFrame, label_col: str, rng: np.random.Generator, path: Path) -> None:
    df = df.assign(transaction_date=YEAR_START + pd.to_timedelta(rng.integers(0, DAYS_IN_YEAR, len(df)), unit="D"))
    df = df.sort_values(["customer_id", "transaction_date"], kind="stable").reset_index(drop=True)
    df.insert(0, "transaction_id", np.arange(1, len(df) + 1, dtype="int32"))
    df = df[["transaction_id", "transaction_date", "customer_id", label_col, "unit_spend"]]
    df = df.astype({"customer_id": "int32", "unit_spend": "float32"})
    # Delta encoding shrinks the sorted ID columns to almost nothing; the default dictionary encoding doesn't, and
    # makes up most of the file.
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        path,
        compression="zstd",
        compression_level=ZSTD_LEVEL,
        use_dictionary=["transaction_date", label_col, "unit_spend"],
        column_encoding={"transaction_id": "DELTA_BINARY_PACKED", "customer_id": "DELTA_BINARY_PACKED"},
    )


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    coffee_rng = np.random.default_rng(COFFEE_SEED)
    _write(_coffee(coffee_rng), "category_name", coffee_rng, OUT_DIR / "coffee_transactions.parquet")
    jeans_rng = np.random.default_rng(JEANS_SEED)
    _write(_jeans(jeans_rng), "brand_name", jeans_rng, OUT_DIR / "jeans_transactions.parquet")
