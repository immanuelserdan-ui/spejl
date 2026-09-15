"""NMS merge logic in detect/rotations.py, tested directly against
synthetic Detection objects — no OCR, no image, just the geometry.
"""

from __future__ import annotations

import numpy as np

from spejl.detect.ocr import Detection
from spejl.detect.rotations import _merge, detect_all_orientations


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


def test_a_correct_full_read_is_not_dropped_for_a_perfect_confidence_fragment():
    """Regression: on a real plan, a rotated pass misread the trailing
    '2' of 'Vaer. 2' as a lone, trivially easy digit at PERFECT 1.000
    confidence -- kept ahead of the correct 'Vaer. 2' read (0.949, an
    honest score for a much longer, harder read). The old tie-break
    (`cand.conf > best.conf - 0.05`) required 'Vaer. 2' to clear 0.95
    to reclaim its own territory from the fragment; it missed by one
    thousandth, so the ENTIRE correct run was silently dropped.

    That's a materially worse failure than a wrong label: a dropped run
    is never added to text_boxes, so it's never erased -- its raw
    source pixels pass straight through the mirror flip as ordinary
    geometry, genuinely mirrored text on the output, the one failure
    mode this whole application exists to prevent.

    A single-digit fragment scoring 1.000 is not more trustworthy than
    a 7-character run scoring 0.949 -- it's just an easier read. When
    the shorter incumbent's box sits almost entirely inside the longer
    candidate's own box (near-total containment, not just overlap),
    that's near-unambiguous evidence it's a fragment of the SAME text,
    so the longer read only needs to clear a sane absolute confidence
    floor, not out-score the fragment.
    """
    candidates = [
        _det("Vaer. 2", (1136, 1298, 1303, 1349), 0.949),
        _det("2", (1268, 1301, 1300, 1344), 1.000),  # inside 'Vaer. 2', a rotated-pass misread
    ]
    kept = _merge(candidates, iou_threshold=0.3, containment_threshold=0.6)
    assert [d.text for d in kept] == ["Vaer. 2"]


def test_a_low_confidence_long_candidate_still_loses_to_a_good_short_fragment():
    """The new containment-based override requires the longer candidate
    to clear an absolute confidence floor (0.85) -- it must not let a
    genuinely bad long misread steal territory just because its box
    happens to contain a good short read."""
    candidates = [
        _det("2", (10, 0, 30, 20), 0.99),
        _det("Zgarbled9", (0, 0, 60, 20), 0.60),  # contains '2's box, but a poor read
    ]
    kept = _merge(candidates, iou_threshold=0.3, containment_threshold=0.6)
    assert [d.text for d in kept] == ["2"]


class _FirstPassOnlyBackend:
    """Returns ``first_pass`` detections only on the very first call
    (the un-rotated, angle=0 pass) and nothing on the other two — good
    enough for a test that only cares what detect_all_orientations does
    with one pass's raw output, not full multi-pass behaviour."""

    def __init__(self, first_pass: list[Detection]) -> None:
        self._first_pass = first_pass
        self._calls = 0

    def detect_and_recognise(self, image: np.ndarray) -> list[Detection]:
        self._calls += 1
        return self._first_pass if self._calls == 1 else []


def test_a_zero_area_degenerate_quad_never_reaches_nms():
    """Regression: _area, _iou and _containment all short-circuit to
    0.0 whenever either input box has no area — there's no "smaller" or
    "union" to measure — so a hairline-scratch/hallucinated-sliver
    detection with a truly zero-width or zero-height quad was INVISIBLE
    to every overlap check _merge relies on and survived unconditionally,
    however completely it sat on top of a real, legitimate detection —
    the opposite of what NMS is for. Filtered out before it can ever
    become a candidate at all, upstream of _merge entirely.
    """
    image = np.full((200, 300, 3), 255, np.uint8)
    degenerate = Detection(
        text="l", quad=((10.0, 10.0), (10.0, 10.0), (10.0, 40.0), (10.0, 40.0)), conf=0.9
    )
    real = Detection(
        text="Stue", quad=((0.0, 0.0), (60.0, 0.0), (60.0, 30.0), (0.0, 30.0)), conf=0.99
    )
    backend = _FirstPassOnlyBackend([degenerate, real])

    detections = detect_all_orientations(image, backend)
    texts = [d.text for d in detections]
    assert "l" not in texts
    assert "Stue" in texts


def test_an_orientation_implausible_reject_is_routed_into_dropped_out():
    """Regression: an _orientation_is_plausible rejection used to vanish
    silently — never passed to dropped_out, so raster/pipeline.py's own
    coverage-check safety net (_find_likely_missed_text) could never see
    it. Fine for the common case (a redundant re-read of text another
    pass already got right, still covered by the real kept detection),
    but a real loss for a short, genuinely-horizontal lexicon word (e.g.
    'WC') whose tight bbox happens to read very slightly taller than
    wide — ordinary detector padding noise, not a real orientation
    problem — on every pass. Confirmed via detect_all_orientations
    itself, not the plausibility check in isolation: a rejected
    candidate must actually reach ``dropped_out`` for the safety net to
    have any chance of catching it.
    """
    image = np.full((200, 300, 3), 255, np.uint8)
    # Pass angle 0 (claims horizontal); box taller than wide -> rejected
    # by _orientation_is_plausible's own shape check.
    implausible = Detection(
        text="WC", quad=((10.0, 10.0), (30.0, 10.0), (30.0, 40.0), (10.0, 40.0)), conf=0.95
    )
    backend = _FirstPassOnlyBackend([implausible])

    dropped: list[Detection] = []
    detections = detect_all_orientations(image, backend, dropped_out=dropped)
    assert detections == []
    assert [d.text for d in dropped] == ["WC"]
