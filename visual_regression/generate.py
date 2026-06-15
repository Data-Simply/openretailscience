"""Generate the labelled visual-regression dataset.

Renders ``--count`` retail charts. A configurable fraction are left clean; the rest have exactly one
defect from :mod:`visual_regression.defects` injected. Writes the PNGs plus a single ``manifest.json``
(the authoritative record of every plot and its issue) and a human-readable ``manifest.csv``.

Run ``python -m visual_regression.generate --help`` for options.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from visual_regression import defects as defect_mod
from visual_regression.retail_plots import BUILDERS

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.random import Generator

    from visual_regression.defects import Defect

DEFAULT_COUNT = 300
DEFAULT_SEED = 42
DEFAULT_CLEAN_FRACTION = 0.25
DEFAULT_DPI = 110


def _render(fig: Figure, ax: Axes, out_path: Path, defect: Defect | None, rng: Generator, dpi: int) -> None:
    """Save ``fig`` to ``out_path``, optionally injecting ``defect`` first.

    For clean charts the chrome layout engine runs normally during ``savefig``. For defective charts we
    draw once (so the engine lays the chrome out at the final size), freeze the engine, mutate, then
    save — the breakage would otherwise be undone by the engine's pre-draw reflow. ``savefig`` is called
    without ``bbox_inches="tight"`` on purpose: a tight bbox would grow the canvas to reveal anything
    pushed off the edge, silently "fixing" the clipping defects.
    """
    if defect is None:
        fig.savefig(out_path, dpi=dpi)
        return
    fig.canvas.draw()
    fig.set_layout_engine("none")
    defect.apply(fig, ax, rng)
    fig.savefig(out_path, dpi=dpi)


def generate_dataset(
    count: int = DEFAULT_COUNT,
    out_dir: Path | str = "visual_regression/output",
    seed: int = DEFAULT_SEED,
    clean_fraction: float = DEFAULT_CLEAN_FRACTION,
    dpi: int = DEFAULT_DPI,
) -> dict:
    """Render ``count`` labelled charts into ``out_dir`` and return the manifest dict.

    Args:
        count: Number of charts to render.
        out_dir: Destination directory; images land in ``<out_dir>/images``.
        seed: Base RNG seed; chart ``i`` uses ``seed + i`` so the dataset is fully reproducible.
        clean_fraction: Fraction of charts left defect-free (for measuring false positives).
        dpi: Render resolution.

    Returns:
        The manifest dict with ``taxonomy`` and ``plots`` keys (also written to ``manifest.json``).

    Raises:
        ValueError: If ``count`` is not positive or ``clean_fraction`` is outside ``[0, 1]``.
    """
    if count <= 0:
        msg = f"count must be positive, got {count}"
        raise ValueError(msg)
    if not 0.0 <= clean_fraction <= 1.0:
        msg = f"clean_fraction must be in [0, 1], got {clean_fraction}"
        raise ValueError(msg)

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    plots: list[dict] = []
    for i in range(count):
        rng = np.random.default_rng(seed + i)
        builder = BUILDERS[int(rng.integers(len(BUILDERS)))]
        fig, ax, tags = builder.build(rng)

        defect: Defect | None = None
        if rng.random() >= clean_fraction:
            candidates = defect_mod.applicable_defects(tags)
            defect = candidates[int(rng.integers(len(candidates)))]

        file_name = f"plot_{i:04d}.png"
        _render(fig, ax, images_dir / file_name, defect, rng, dpi)
        plt.close(fig)

        plots.append(
            {
                "file": f"images/{file_name}",
                "chart_type": builder.name,
                "has_defect": defect is not None,
                "defects": []
                if defect is None
                else [{"name": defect.name, "description": defect.description, "severity": defect.severity}],
            },
        )

    manifest = {"taxonomy": defect_mod.taxonomy(), "plots": plots}
    _write_manifest(out_dir, manifest)
    return manifest


def _write_manifest(out_dir: Path, manifest: dict) -> None:
    """Write ``manifest.json`` (authoritative) and ``manifest.csv`` (human-readable)."""
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "chart_type", "has_defect", "defect_names", "issue"])
        for plot in manifest["plots"]:
            names = ";".join(d["name"] for d in plot["defects"])
            issues = " ".join(d["description"] for d in plot["defects"])
            writer.writerow([plot["file"], plot["chart_type"], plot["has_defect"], names, issues])


def _summarise(manifest: dict) -> str:
    """Return a one-line summary of the generated dataset."""
    plots = manifest["plots"]
    defective = sum(1 for p in plots if p["has_defect"])
    return f"{len(plots)} plots ({defective} defective, {len(plots) - defective} clean)"


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate a labelled visual-regression dataset of retail charts.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of charts to render.")
    parser.add_argument("--out", default="visual_regression/output", help="Output directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base RNG seed.")
    parser.add_argument(
        "--clean-fraction",
        type=float,
        default=DEFAULT_CLEAN_FRACTION,
        help="Fraction of charts left defect-free.",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Render resolution.")
    args = parser.parse_args(argv)

    manifest = generate_dataset(
        count=args.count,
        out_dir=args.out,
        seed=args.seed,
        clean_fraction=args.clean_fraction,
        dpi=args.dpi,
    )
    print(f"Generated {_summarise(manifest)} in {args.out}")
    print(f"Manifest: {Path(args.out) / 'manifest.json'}")


if __name__ == "__main__":
    main()
