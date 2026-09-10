"""qa/metrics.py's truth-to-detection matcher, tested directly."""

from __future__ import annotations

from spejl.detect.ocr import Detection
from spejl.qa.metrics import _match_truth_to_detections


def _det(box: tuple[float, float, float, float]) -> Detection:
    x0, y0, x1, y1 = box
    return Detection(text="x", quad=((x0, y0), (x1, y0), (x1, y1), (x0, y1)), conf=0.9)


def test_earlier_truth_run_does_not_steal_a_later_runs_better_match():
    """Regression: truth run A (list index 0) had only a marginal IoU
    (0.2) with detection D, while truth run B (index 1) would have
    matched D almost perfectly (IoU 0.9) and had no other candidate at
    all. The old per-truth-in-list-order greedy locked in A<->D on A's
    turn, before B ever got a chance — reporting B as a false negative
    purely because of iteration order, not a real detector failure.
    """
    truth_runs = [
        {"bbox_px": [0, 0, 50, 20]},        # A: only overlaps D marginally
        {"bbox_px": [100, 100, 140, 120]},  # B: near-exact match for D
    ]
    detections = [_det((98, 98, 142, 122))]  # D: matches B almost perfectly

    matches = _match_truth_to_detections(truth_runs, detections, min_iou=0.15)
    assert matches == {1: 0}, matches  # B <-> D, A left unmatched


def test_each_detection_is_used_at_most_once():
    truth_runs = [
        {"bbox_px": [0, 0, 50, 20]},
        {"bbox_px": [5, 2, 52, 22]},  # overlaps the same detection heavily
    ]
    detections = [_det((0, 0, 50, 20))]
    matches = _match_truth_to_detections(truth_runs, detections, min_iou=0.15)
    assert len(set(matches.values())) == len(matches)  # no detection double-claimed


def test_no_match_below_threshold_is_left_unmatched():
    truth_runs = [{"bbox_px": [0, 0, 10, 10]}]
    detections = [_det((500, 500, 510, 510))]
    matches = _match_truth_to_detections(truth_runs, detections, min_iou=0.15)
    assert matches == {}


def test_disjoint_pairs_all_match_independently():
    truth_runs = [
        {"bbox_px": [0, 0, 20, 20]},
        {"bbox_px": [100, 0, 120, 20]},
        {"bbox_px": [200, 0, 220, 20]},
    ]
    detections = [_det((0, 0, 20, 20)), _det((100, 0, 120, 20)), _det((200, 0, 220, 20))]
    matches = _match_truth_to_detections(truth_runs, detections, min_iou=0.15)
    assert matches == {0: 0, 1: 1, 2: 2}
