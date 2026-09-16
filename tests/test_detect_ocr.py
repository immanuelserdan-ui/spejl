"""Detection gates from build plan §10, run against the golden fixture.

Slow (loads ONNX models, three OCR passes), so it runs at one resolution
and shares a session-scoped backend. Mark: ``pytest -m "not slow"`` skips.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from spejl.detect.ocr import RapidOcrBackend
from spejl.detect.rotations import detect_all_orientations
from spejl.lexicon.snap import snap
from spejl.qa import fixture_gen
from spejl.qa.metrics import load_ground_truth, score

pytestmark = pytest.mark.slow

DPI = 150


@pytest.fixture(scope="session")
def backend() -> RapidOcrBackend:
    return RapidOcrBackend()


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    """Regenerate the fixture rather than trusting the checked-in copy —
    ground truth is only trustworthy if it comes from the same run."""
    out = tmp_path_factory.mktemp("golden")
    fixture_gen.generate(out, dpi=DPI)
    return out


@pytest.fixture(scope="session")
def report(backend, fixture_dir):
    truth = load_ground_truth(fixture_dir / f"ground_truth_{DPI}.json")
    image = cv2.imread(str(fixture_dir / truth["image"]))
    detections = detect_all_orientations(image, backend)
    corrections = {i: (snap(d.text).text, snap(d.text).warning)
                   for i, d in enumerate(detections)}
    return score(truth, detections, corrections)


def test_the_fixture_has_the_runs_we_expect(fixture_dir):
    truth = load_ground_truth(fixture_dir / f"ground_truth_{DPI}.json")
    texts = [r["text"] for r in truth["runs"]]
    # The real sheet's vocabulary, including the diacritics OCR will fight.
    for expected in ("Køkken", "Stue", "Vær. 1", "Bad", "Toilet", "Entre", "H*"):
        assert expected in texts
    for dim in ("2900", "3200", "4045", "5155", "4060", "2105", "1680", "1400", "870"):
        assert dim in texts
    # Five vertical dimension runs — the case recognisers handle worst.
    assert sum(r["angle_deg"] == 90 for r in truth["runs"]) == 5


def test_text_recall_gate(report):
    assert report.recall == 1.0, [r.truth for r in report.runs if not r.found]


def test_character_accuracy_gate(report):
    """§10 gate: >= 99% after the lexicon stage."""
    assert report.char_accuracy(corrected=True) >= 0.99


def test_lexicon_measurably_improves_raw_output(report):
    """The justification for stage S3 existing at all."""
    assert report.char_accuracy(True) > report.char_accuracy(False)


def test_anchor_error_gate(report):
    """§10 gate: <= 3 px at 150 dpi."""
    assert report.mean_anchor_error <= 3.0


def test_orientation_gate(report):
    """Vertical dimension runs must be identified as vertical, or they get
    re-drawn horizontally across the wall they annotate."""
    assert report.angle_accuracy >= 0.95


def test_every_vertical_dimension_is_found_and_read(report):
    verticals = [r for r in report.runs if r.angle_truth == 90]
    assert len(verticals) == 5
    for run in verticals:
        assert run.found, f"{run.truth} not detected"
        assert run.exact_corrected, f"{run.truth} read as {run.corrected!r}"


def test_an_ocr_engine_failure_is_normalised_to_a_runtime_error(backend):
    """Regression: RapidOcrBackend.detect_and_recognise had no exception
    handling around the engine call at all. onnxruntime's own exception
    family (Fail, InvalidArgument, ...) inherits directly from Exception
    — confirmed by inspecting the hierarchy directly — sharing no base
    with ValueError/OSError/RuntimeError, the three types cli.py's own
    top-level handler catches. Left unwrapped, a real engine failure on
    an adversarial or merely unusual scan would reach the CLI as a raw,
    uncaught exception — a Python traceback instead of the clean,
    actionable message every other failure path in this project shows.

    The real ``self._ocr`` callable is swapped out, not the whole
    backend: this only needs to prove `detect_and_recognise` catches and
    re-wraps whatever the engine raises, not that a specific onnxruntime
    exception is reachable from a specific bad image — restored
    afterwards since `backend` is session-scoped and shared by every
    other test in this file.
    """
    import numpy as np

    class _EngineExplodes:
        def __call__(self, image):
            raise RuntimeError("simulated onnxruntime.capi.onnxruntime_pybind11_state.Fail")

    original = backend._ocr
    backend._ocr = _EngineExplodes()
    try:
        with pytest.raises(RuntimeError, match="OCR engine failed"):
            backend.detect_and_recognise(np.zeros((10, 10, 3), dtype=np.uint8))
    finally:
        backend._ocr = original
