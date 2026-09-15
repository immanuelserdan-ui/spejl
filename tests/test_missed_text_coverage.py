"""raster/pipeline.py's _find_likely_missed_text — the coverage check
that catches a valid OCR read getting silently dropped before it ever
reaches snap()/render_run(), tested directly against synthetic
Detection objects. No OCR, no image, just the geometry.
"""

from __future__ import annotations

from spejl.detect.ocr import Detection
from spejl.raster.pipeline import _find_likely_missed_text


def _det(text: str, box: tuple[float, float, float, float], conf: float) -> Detection:
    x0, y0, x1, y1 = box
    return Detection(text=text, quad=((x0, y0), (x1, y0), (x1, y1), (x0, y1)), conf=conf)


def test_a_confident_long_read_dropped_for_a_small_fragment_is_flagged():
    """Regression: the exact real-plan numbers from the 'Vaer. 2' bug
    that motivated this whole function. Before that fix, _merge's own
    NMS kept a stray single-digit fragment ('2', a rotated-pass misread
    at PERFECT 1.000 confidence) and silently dropped the correct,
    confident seven-character 'Vaer. 2' (0.949) — never erased, never
    re-rendered, its raw pixels passed straight through the mirror flip
    as ordinary geometry: genuinely mirrored text on the output.

    This is the general safety net for that whole CLASS of bug, not a
    check for this one fixed instance — confirmed here by directly
    reproducing what _merge actually produced before its own fix, to
    prove this function would have caught it.
    """
    dropped = [_det("Vaer. 2", (1136, 1298, 1303, 1349), 0.949)]
    kept = [_det("2", (1268, 1301, 1300, 1344), 1.000)]  # what the buggy merge kept instead

    flags = _find_likely_missed_text(dropped, kept)
    assert len(flags) == 1
    assert flags[0].code == "possible-missed-text"
    assert "Vaer. 2" in flags[0].message


def test_a_short_dropped_fragment_is_not_flagged():
    """Most drops are correct and routine (a stray letter, a truncated
    duplicate) — the whole point of _merge's NMS. Flagging every one of
    those would bury the rare genuine loss in noise."""
    dropped = [_det("B", (2, 2, 22, 28), 0.996)]  # correctly-suppressed fragment of 'Bad'
    kept = [_det("Bad", (0, 0, 60, 30), 0.999)]
    assert _find_likely_missed_text(dropped, kept) == []


def test_a_low_confidence_dropped_candidate_is_not_flagged():
    """Long but unreliable ('it was noise anyway' is a plausible
    explanation here) — not worth a human's attention."""
    dropped = [_det("Garbled?", (0, 0, 80, 30), 0.40)]
    kept = [_det("Stue", (200, 200, 260, 230), 0.99)]  # unrelated, elsewhere on the sheet
    assert _find_likely_missed_text(dropped, kept) == []


def test_a_dropped_candidate_genuinely_covered_by_a_kept_run_is_not_flagged():
    """A dropped read that IS substantially the same physical text as
    something that WAS kept (e.g. a duplicate read from another
    rotation pass, later superseded by a fuller/better one at nearly
    the same location) is not a loss -- the NMS pass did its job."""
    dropped = [_det("Vaerelse", (100, 100, 220, 140), 0.90)]
    kept = [_det("Værelse", (98, 99, 222, 141), 0.97)]  # near-identical box, the winning read
    assert _find_likely_missed_text(dropped, kept) == []


def test_containment_direction_matters_a_small_kept_run_inside_a_big_dropped_box_is_not_coverage():
    """Regression in the coverage check itself: an earlier version used
    _merge's own _containment() helper, which normalises by the
    SMALLER box's area -- built for "is this fragment a piece of that
    run," backwards for this question. A kept run that is itself small
    and sits inside a small corner of a much bigger dropped candidate's
    box scores _containment() near 1.0 (the kept box is nearly 100%
    inside the dropped one) while covering almost none of the dropped
    text -- exactly the shape of the real 'Vaer. 2' case, where the
    first version of this function failed to catch its own motivating
    bug for precisely this reason. Coverage must be directional: what
    fraction of the DROPPED candidate's own area a kept run covers.
    """
    dropped = [_det("Soveværelse", (0, 0, 220, 40), 0.93)]  # a big box
    kept = [_det("2", (190, 5, 215, 35), 1.0)]  # small, sits in one corner of it
    flags = _find_likely_missed_text(dropped, kept)
    assert len(flags) == 1  # NOT considered covered -- most of 'Soveværelse' is unaccounted for
