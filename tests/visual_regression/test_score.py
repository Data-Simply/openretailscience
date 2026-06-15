"""Tests for the scoring logic."""

import pytest
from visual_regression.score import (
    _parse_judge,
    build_judge_prompt,
    evaluate,
    evaluate_freetext,
)

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


def _keyword_judge(is_clean, _truth_descriptions, model_description):
    """A deterministic stand-in judge: 'correct' means it identified the issue, 'problem' means it flagged one."""
    text = model_description.lower()
    identifies = (not is_clean) and "correct" in text
    return {"identifies_real_issue": identifies, "claims_problem": ("problem" in text) or identifies}


class TestEvaluateFreetext:
    """Free-text scoring turns judge verdicts into binary detection metrics."""

    @pytest.fixture
    def report(self):
        """Two defective and two clean plots: one of each judged right, one of each wrong."""
        plots = [
            {"file": "a.png", "defects": [{"name": "title_clipped", "description": "title cut off"}]},
            {"file": "b.png", "defects": []},
            {"file": "c.png", "defects": [{"name": "legend_overlaps_data", "description": "legend on data"}]},
            {"file": "d.png", "defects": []},
        ]
        detections = [
            {"file": "a.png", "description": "correctly: the title is cut off at the edge"},  # TP
            {"file": "b.png", "description": "looks fine"},  # TN
            {"file": "c.png", "description": "I think there is a problem with spacing"},  # FN (wrong issue)
            {"file": "d.png", "description": "there is a problem with the legend"},  # FP
        ]
        return evaluate_freetext(plots, detections, _keyword_judge)

    def test_binary_confusion_counts(self, report):
        """One TP, one FP, one FN, one TN -> precision/recall/accuracy all 0.5."""
        binary = report["binary"]
        assert (binary["tp"], binary["fp"], binary["fn"], binary["tn"]) == (1, 1, 1, 1)
        assert binary["precision"] == pytest.approx(0.5)
        assert binary["recall"] == pytest.approx(0.5)
        assert binary["accuracy"] == pytest.approx(0.5)

    def test_per_plot_outcomes_recorded(self, report):
        """Each plot's verdict is recorded so a human can audit the judge."""
        outcomes = {v["file"]: v["outcome"] for v in report["verdicts"]}
        assert outcomes == {"a.png": "tp", "b.png": "tn", "c.png": "fn", "d.png": "fp"}

    def test_errored_detections_are_skipped(self):
        """A detection that failed (carries an error) is excluded from scoring."""
        plots = [{"file": "a.png", "defects": [{"name": "title_clipped", "description": "title cut off"}]}]
        detections = [{"file": "a.png", "description": "", "error": "model unavailable"}]
        with pytest.raises(ValueError, match="No overlapping"):
            evaluate_freetext(plots, detections, _keyword_judge)


class TestJudge:
    """The judge prompt and verdict parsing."""

    def test_prompt_states_known_defect_for_defective_chart(self):
        """A defective chart's prompt names the injected issue and asks for the two booleans."""
        prompt = build_judge_prompt(False, ["title cut off"], "the title is cut off")
        assert "title cut off" in prompt
        assert "identifies_real_issue" in prompt
        assert "claims_problem" in prompt

    def test_prompt_marks_clean_chart_as_clean(self):
        """A clean chart's prompt tells the judge no defect was injected."""
        prompt = build_judge_prompt(True, [], "looks fine")
        assert "clean" in prompt.lower()

    @pytest.mark.parametrize(
        ("raw", "identifies", "claims"),
        [
            ('{"identifies_real_issue": true, "claims_problem": true}', True, True),
            ('{"identifies_real_issue": false, "claims_problem": true}', False, True),
            ("not json", False, False),
            ('{"claims_problem": true}', False, True),  # missing field defaults to False
        ],
    )
    def test_parse_judge_verdict(self, raw, identifies, claims):
        """The judge verdict is parsed robustly with False defaults for missing/garbled fields."""
        verdict = _parse_judge(raw)
        assert verdict["identifies_real_issue"] is identifies
        assert verdict["claims_problem"] is claims
