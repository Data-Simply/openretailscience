"""Tests for the detection harness (parsing and orchestration, no network/subprocess)."""

import json

import pytest
from visual_regression.detect import (
    _extract_stream_json_result,
    _parse_detections,
    build_prompt,
    detect_all,
)

_TAXONOMY = [
    {"name": "title_clipped", "description": "title cut off", "severity": "high"},
    {"name": "legend_overlaps_data", "description": "legend on data", "severity": "high"},
]
_VALID = {d["name"] for d in _TAXONOMY}

# Module-scope stub backends (the style guide forbids nested function definitions).
_STUB_RESPONSES = {
    "p0.png": '{"defects": ["title_clipped", "bogus"]}',
    "p1.png": "looks clean",
}


def _stub_backend(image_path, _prompt):
    """Return a canned response keyed on the image file name."""
    key = "p0.png" if image_path.endswith("p0.png") else "p1.png"
    return _STUB_RESPONSES[key]


def _exploding_backend(_image_path, _prompt):
    """A backend that always fails, to exercise per-plot error handling."""
    msg = "model unavailable"
    raise RuntimeError(msg)


class TestParseDetections:
    """Robust parsing of varied model responses into known category names."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('{"defects": ["title_clipped"]}', ["title_clipped"]),
            ('```json\n{"defects": ["legend_overlaps_data"]}\n```', ["legend_overlaps_data"]),
            ('Sure! Here you go: {"defects": ["title_clipped"]}', ["title_clipped"]),
            ('{"defects": ["Title_Clipped"]}', ["title_clipped"]),  # case-insensitive
            ('{"defects": [{"name": "title_clipped"}]}', ["title_clipped"]),  # object form
            ('{"defects": ["title_clipped", "not_a_category"]}', ["title_clipped"]),  # unknown dropped
            ('{"defects": []}', []),
            ("the chart looks fine to me", []),  # no JSON at all
            ('{"defects": "title_clipped"}', []),  # malformed (not a list)
        ],
    )
    def test_parse_variants(self, raw, expected):
        """Detected categories are extracted and filtered to the known taxonomy."""
        assert _parse_detections(raw, _VALID) == expected


class TestStreamJsonResult:
    """Parsing the assistant text out of ``claude --output-format stream-json`` output."""

    def test_extracts_result_event_ignoring_other_lines(self):
        """The final ``type=result`` event's ``result`` field is returned, other events ignored."""
        events = [
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": "thinking"}},
            {"type": "result", "subtype": "success", "result": '{"defects": ["title_clipped"]}'},
        ]
        stdout = "\n".join(json.dumps(e) for e in events[:2]) + "\nnot json at all\n" + json.dumps(events[2])
        assert json.loads(_extract_stream_json_result(stdout)) == {"defects": ["title_clipped"]}

    def test_missing_result_event_raises(self):
        """A stream without a result event is an error, not a silent empty string."""
        stdout = '{"type": "system", "subtype": "init"}\n{"type": "assistant"}'
        with pytest.raises(RuntimeError, match="no result event"):
            _extract_stream_json_result(stdout)


class TestBuildPrompt:
    """The prompt advertises the closed category set and demands JSON."""

    def test_prompt_lists_every_category(self):
        """Each taxonomy name and the required JSON shape appear in the prompt."""
        prompt = build_prompt(_TAXONOMY)
        assert "title_clipped" in prompt
        assert "legend_overlaps_data" in prompt
        assert '{"defects":' in prompt


class TestDetectAll:
    """Orchestration over a manifest with a stubbed backend."""

    @pytest.fixture
    def dataset(self, tmp_path):
        """Two dummy image files referenced by a minimal plots list."""
        (tmp_path / "images").mkdir()
        for name in ("p0.png", "p1.png"):
            (tmp_path / "images" / name).write_bytes(b"not-a-real-png")
        plots = [
            {"file": "images/p0.png", "defects": []},
            {"file": "images/p1.png", "defects": []},
        ]
        return tmp_path, plots

    def test_detects_parses_and_records_per_plot(self, dataset):
        """Known categories are kept, unknowns dropped, has_defect derived from detections."""
        base_dir, plots = dataset
        results = detect_all(plots, base_dir, _stub_backend, _TAXONOMY)

        by_file = {r["file"]: r for r in results}
        assert by_file["images/p0.png"]["detected"] == ["title_clipped"]
        assert by_file["images/p0.png"]["has_defect"] is True
        assert by_file["images/p1.png"]["detected"] == []
        assert by_file["images/p1.png"]["has_defect"] is False

    def test_backend_error_is_recorded_not_raised(self, dataset):
        """A backend failure on one plot is captured in the record, not propagated."""
        base_dir, plots = dataset
        results = detect_all(plots, base_dir, _exploding_backend, _TAXONOMY)
        assert len(results) == len(plots)
        assert all(r["error"] == "model unavailable" for r in results)
        assert all(r["detected"] == [] for r in results)
