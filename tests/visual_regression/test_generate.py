"""Tests for dataset generation."""

import json

import matplotlib as mpl
import pytest

mpl.use("Agg")

from visual_regression.defects import DEFECTS
from visual_regression.generate import generate_dataset

_VALID_DEFECT_NAMES = {d.name for d in DEFECTS}
_DATASET_SIZE = 16
_MIN_PNG_BYTES = 1000


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    """Generate a small reproducible dataset once for the module."""
    out_dir = tmp_path_factory.mktemp("vr_dataset")
    result = generate_dataset(count=_DATASET_SIZE, out_dir=out_dir, seed=7, clean_fraction=0.5, dpi=80)
    return out_dir, result


class TestGenerateDataset:
    """Behaviour of ``generate_dataset``."""

    def test_renders_requested_number_of_plots(self, manifest):
        """One manifest entry and one non-trivial PNG per requested plot."""
        out_dir, result = manifest
        assert len(result["plots"]) == _DATASET_SIZE
        for plot in result["plots"]:
            image_path = out_dir / plot["file"]
            assert image_path.exists()
            assert image_path.stat().st_size > _MIN_PNG_BYTES

    def test_defect_labels_come_from_the_catalogue(self, manifest):
        """Every recorded defect name is a known catalogue entry; clean plots record none."""
        _, result = manifest
        for plot in result["plots"]:
            names = [d["name"] for d in plot["defects"]]
            assert all(name in _VALID_DEFECT_NAMES for name in names)
            if plot["has_defect"]:
                assert len(names) == 1
            else:
                assert names == []

    def test_dataset_contains_both_clean_and_defective_plots(self, manifest):
        """A 50% clean fraction yields both kinds, so the binary task is non-degenerate."""
        _, result = manifest
        assert any(plot["has_defect"] for plot in result["plots"])
        assert any(not plot["has_defect"] for plot in result["plots"])

    def test_manifest_files_are_written(self, manifest):
        """``manifest.json`` round-trips and matches the returned dict."""
        out_dir, result = manifest
        on_disk = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert on_disk == result
        assert (out_dir / "manifest.csv").exists()

    def test_generation_is_reproducible_for_a_fixed_seed(self, tmp_path):
        """The same seed produces the same chart types and defect assignments."""
        first = generate_dataset(count=8, out_dir=tmp_path / "a", seed=99, clean_fraction=0.5, dpi=80)
        second = generate_dataset(count=8, out_dir=tmp_path / "b", seed=99, clean_fraction=0.5, dpi=80)
        assert first["plots"] == second["plots"]

    @pytest.mark.parametrize(
        ("count", "clean_fraction"),
        [(0, 0.25), (-3, 0.25), (10, 1.5), (10, -0.1)],
    )
    def test_invalid_arguments_raise(self, tmp_path, count, clean_fraction):
        """Out-of-range count or clean_fraction is rejected at the public boundary."""
        with pytest.raises(ValueError, match=r"count|clean_fraction"):
            generate_dataset(count=count, out_dir=tmp_path, clean_fraction=clean_fraction)
