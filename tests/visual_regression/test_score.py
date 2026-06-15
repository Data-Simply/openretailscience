"""Tests for the scoring logic."""

import pytest
from visual_regression.score import evaluate

_TAXONOMY = [
    {"name": "title_clipped", "description": "title cut off", "severity": "high"},
    {"name": "legend_overlaps_data", "description": "legend on data", "severity": "high"},
]


def _plots():
    """Three ground-truth plots: two defective, one clean."""
    return [
        {
            "file": "images/a.png",
            "chart_type": "vertical_bar",
            "has_defect": True,
            "defects": [{"name": "title_clipped"}],
        },
        {"file": "images/b.png", "chart_type": "line", "has_defect": False, "defects": []},
        {
            "file": "images/c.png",
            "chart_type": "grouped_bar",
            "has_defect": True,
            "defects": [{"name": "legend_overlaps_data"}],
        },
    ]


class TestEvaluate:
    """``evaluate`` against hand-computed confusion counts."""

    @pytest.fixture
    def report(self):
        """Detections: a correct, b false-positive, c missed."""
        detections = [
            {"file": "images/a.png", "detected": ["title_clipped"]},  # true positive
            {"file": "images/b.png", "detected": ["title_clipped"]},  # false positive (clean plot)
            {"file": "images/c.png", "detected": []},  # false negative (missed)
        ]
        return evaluate(_plots(), detections, _TAXONOMY)

    def test_binary_confusion_counts(self, report):
        """a=TP, b=FP, c=FN, no TN -> precision/recall both 0.5, accuracy 1/3."""
        binary = report["binary"]
        assert (binary["tp"], binary["fp"], binary["fn"], binary["tn"]) == (1, 1, 1, 0)
        assert binary["precision"] == pytest.approx(0.5)
        assert binary["recall"] == pytest.approx(0.5)
        assert binary["accuracy"] == pytest.approx(1 / 3)

    def test_per_category_title_clipped(self, report):
        """title_clipped: hit on a, wrongly fired on b -> precision 0.5, recall 1.0."""
        stats = report["per_category"]["title_clipped"]
        assert (stats["tp"], stats["fp"], stats["fn"]) == (1, 1, 0)
        assert stats["precision"] == pytest.approx(0.5)
        assert stats["recall"] == pytest.approx(1.0)

    def test_per_category_legend_missed(self, report):
        """legend_overlaps_data was present on c but never detected -> recall 0.0."""
        stats = report["per_category"]["legend_overlaps_data"]
        assert (stats["tp"], stats["fp"], stats["fn"]) == (0, 0, 1)
        assert stats["recall"] == pytest.approx(0.0)
        assert stats["support"] == 1

    def test_perfect_detection_scores_one(self):
        """Identical predictions and ground truth give precision = recall = f1 = 1."""
        detections = [
            {"file": "images/a.png", "detected": ["title_clipped"]},
            {"file": "images/b.png", "detected": []},
            {"file": "images/c.png", "detected": ["legend_overlaps_data"]},
        ]
        report = evaluate(_plots(), detections, _TAXONOMY)
        assert report["binary"]["f1"] == pytest.approx(1.0)
        assert report["micro"]["f1"] == pytest.approx(1.0)
        assert report["macro"]["f1"] == pytest.approx(1.0)

    def test_no_overlapping_files_raises(self):
        """Scoring requires the detections to reference the manifest's plots."""
        detections = [{"file": "images/unknown.png", "detected": []}]
        with pytest.raises(ValueError, match="No overlapping plot files"):
            evaluate(_plots(), detections, _TAXONOMY)
