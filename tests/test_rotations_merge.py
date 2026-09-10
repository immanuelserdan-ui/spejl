"""NMS merge logic in detect/rotations.py, tested directly against
synthetic Detection objects — no OCR, no image, just the geometry.
"""

from __future__ import annotations

from spejl.detect.ocr import Detection
from spejl.detect.rotations import _merge


def _det(text: str, box: tuple[float, float, float, float], conf: float) -> Detection:
    x0, y0, x1, y1 = box
    return Detection(text=text, quad=((x0, y0), (x1, y0), (x1, y1), (x0, y1)), conf=conf)


def test_truncated_run_is_replaced_by_the_fuller_read():
    """The designed case: one overlapping incumbent, longer text wins."""
    candidates = [
        _det("105", (100, 0, 130, 20), 0.99),   # truncated, kept first (sorted by conf then len)
        _det("2105", (95, 0, 135, 20), 0.95),    # full read, overlaps the truncated one
    ]
    kept = _merge(candidates, iou_threshold=0.3, containment_threshold=0.6)
    assert [d.text for d in kept] == ["2105"]


def test_a_candidate_spanning_two_real_runs_is_rejected_not_a_partial_replace():
    """Regression: a merged/garbled read overlapping TWO already-kept,
    genuinely distinct runs used to replace only the longer-text one,
    leaving the other still present as a duplicate beside the bogus
    merged read. It must now be rejected outright, preserving both
    real runs untouched.
    """
    candidates = [
        _det("600", (0, 0, 40, 20), 0.97),
        _det("300", (60, 0, 100, 20), 0.96),
        # Spans both boxes above — a plausible merged-rotation misread.
        _det("600300", (0, 0, 100, 20), 0.93),
    ]
    kept = _merge(candidates, iou_threshold=0.3, containment_threshold=0.6)
    texts = sorted(d.text for d in kept)
    assert texts == ["300", "600"], texts


def test_non_overlapping_runs_are_all_kept():
    candidates = [
        _det("2900", (0, 0, 40, 20), 0.99),
        _det("4045", (200, 0, 240, 20), 0.99),
        _det("Stue", (0, 100, 60, 130), 0.99),
    ]
    kept = _merge(candidates, iou_threshold=0.3, containment_threshold=0.6)
    assert sorted(d.text for d in kept) == ["2900", "4045", "Stue"]


def test_a_fragment_fully_inside_a_kept_run_is_suppressed():
    """A single-character fragment ('B' inside 'Bad') has low IoU with
    the larger box but high containment — the case containment was
    added to catch."""
    candidates = [
        _det("Bad", (0, 0, 60, 30), 0.999),
        _det("B", (2, 2, 22, 28), 0.996),  # inside 'Bad', low IoU, high containment
    ]
    kept = _merge(candidates, iou_threshold=0.3, containment_threshold=0.6)
    assert [d.text for d in kept] == ["Bad"]
