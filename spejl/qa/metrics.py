"""Score detection output against ground truth.

Implements the measurable half of build plan §10: text recall, character
accuracy, and anchor error, computed by matching each detection to a
ground-truth run by box overlap. Reported per-run as well as in
aggregate, because "97% character accuracy" hides whether the missing 3%
is a dropped diacritic or a dimension that lost a digit — and those two
have very different consequences on a construction drawing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from spejl.detect.ocr import Detection
from spejl.detect.rotations import _iou


@dataclass
class RunScore:
    """One ground-truth run and whatever was matched to it."""

    truth: str
    detected: str | None
    corrected: str | None
    kind: str
    angle_truth: float
    angle_detected: float | None
    conf: float | None
    iou: float
    anchor_error_px: float | None
    cap_height_px: float
    warning: str | None = None

    @property
    def found(self) -> bool:
        return self.detected is not None

    @property
    def exact_raw(self) -> bool:
        return self.detected == self.truth

    @property
    def exact_corrected(self) -> bool:
        return self.corrected == self.truth

    @property
    def char_errors(self) -> int:
        best = self.corrected if self.corrected is not None else self.detected
        if best is None:
            return len(self.truth)
        return Levenshtein.distance(self.truth, best)


@dataclass
class Report:
    runs: list[RunScore] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return sum(r.found for r in self.runs) / len(self.runs) if self.runs else 0.0

    def char_accuracy(self, corrected: bool = True) -> float:
        total = sum(len(r.truth) for r in self.runs)
        if not total:
            return 0.0
        errors = 0
        for r in self.runs:
            best = (r.corrected if corrected else r.detected)
            if best is None:
                errors += len(r.truth)
            else:
                errors += Levenshtein.distance(r.truth, best)
        return max(0.0, 1.0 - errors / total)

    @property
    def exact_raw_rate(self) -> float:
        return sum(r.exact_raw for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def exact_corrected_rate(self) -> float:
        return sum(r.exact_corrected for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def mean_anchor_error(self) -> float:
        errs = [r.anchor_error_px for r in self.runs if r.anchor_error_px is not None]
        return sum(errs) / len(errs) if errs else float("nan")

    @property
    def angle_accuracy(self) -> float:
        ok = [
            r for r in self.runs
            if r.angle_detected is not None
            and abs(((r.angle_truth - r.angle_detected + 180) % 360) - 180) < 15
        ]
        return len(ok) / len(self.runs) if self.runs else 0.0

    @property
    def failures(self) -> list[RunScore]:
        return [r for r in self.runs if not r.exact_corrected]


def load_ground_truth(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _match_truth_to_detections(
    truth_runs: list[dict], detections: list[Detection], min_iou: float
) -> dict[int, int]:
    """Truth-index -> detection-index, one-to-one, by global best-IoU-first.

    Not the previous per-truth-run-in-list-order greedy: matching truth
    run A (list order: first) to whatever detection scores best for A,
    even at IoU 0.16, before truth run B (list order: second) gets a
    turn — even when B would have matched that same detection at IoU
    0.9 and has no other candidate above threshold — understated recall
    and character accuracy for B as a pure scoring artifact, not a real
    detector failure.

    Sorting every (truth, detection) candidate pair by IoU first and
    assigning greedily is the standard approximate solution to this
    (a true Hungarian assignment would be optimal but is unwarranted
    complexity here — at most a few dozen runs per plan, and near-ties
    this reorders are rare enough that "the same globally-best match
    every time" is what actually matters, not exact optimality).
    """
    candidates: list[tuple[float, int, int]] = []  # (iou, truth_idx, det_idx)
    for ti, run in enumerate(truth_runs):
        t_box = tuple(run["bbox_px"])
        for di, det in enumerate(detections):
            iou = _iou(t_box, det.bbox)
            if iou >= min_iou:
                candidates.append((iou, ti, di))
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_truth: set[int] = set()
    matched_det: set[int] = set()
    result: dict[int, int] = {}
    for iou, ti, di in candidates:
        if ti in matched_truth or di in matched_det:
            continue
        result[ti] = di
        matched_truth.add(ti)
        matched_det.add(di)
    return result


def score(
    truth_payload: dict,
    detections: list[Detection],
    corrections: dict[int, tuple[str, str | None]] | None = None,
    min_iou: float = 0.15,
) -> Report:
    """Match detections to ground truth and score.

    ``corrections`` maps a detection's index to its (corrected_text,
    warning) after the lexicon stage, so raw and corrected accuracy can be
    reported side by side — the number that justifies stage S3 existing.
    """
    corrections = corrections or {}
    report = Report()
    truth_runs = truth_payload["runs"]
    matches = _match_truth_to_detections(truth_runs, detections, min_iou)

    for ti, run in enumerate(truth_runs):
        di = matches.get(ti)
        if di is None:
            report.runs.append(
                RunScore(
                    truth=run["text"], detected=None, corrected=None, kind=run["kind"],
                    angle_truth=run["angle_deg"], angle_detected=None, conf=None,
                    iou=0.0, anchor_error_px=None, cap_height_px=run["cap_height_px"],
                )
            )
            continue

        det = detections[di]
        t_box = tuple(run["bbox_px"])
        corrected, warning = corrections.get(di, (det.text, None))

        tcx, tcy = (t_box[0] + t_box[2]) / 2, (t_box[1] + t_box[3]) / 2
        dcx, dcy = det.center
        report.runs.append(
            RunScore(
                truth=run["text"], detected=det.text, corrected=corrected,
                kind=run["kind"], angle_truth=run["angle_deg"],
                angle_detected=det.angle_deg, conf=det.conf, iou=_iou(t_box, det.bbox),
                anchor_error_px=math.hypot(tcx - dcx, tcy - dcy),
                cap_height_px=run["cap_height_px"], warning=warning,
            )
        )

    return report
