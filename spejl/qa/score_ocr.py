"""Run the detection spike and print the numbers.

    python -m spejl.qa.score_ocr --dpi 150 300

This is the Phase-2 go/no-go instrument: it answers whether OCR can read
a plan well enough for the rest of Route B to be worth building, and it
separates raw recognition from lexicon-corrected output so the value of
stage S3 is visible rather than asserted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from spejl.detect.ocr import RapidOcrBackend
from spejl.detect.rotations import detect_all_orientations
from spejl.lexicon.snap import snap
from spejl.qa.metrics import Report, load_ground_truth, score

GOLDEN = Path("spejl/qa/golden")


def run_one(dpi: int, backend: RapidOcrBackend, triple_pass: bool = True) -> Report:
    truth = load_ground_truth(GOLDEN / f"ground_truth_{dpi}.json")
    image = cv2.imread(str(GOLDEN / truth["image"]))
    if image is None:
        raise SystemExit(f"missing fixture image: {GOLDEN / truth['image']}")

    if triple_pass:
        detections = detect_all_orientations(image, backend)
    else:
        detections = backend.detect_and_recognise(image)

    corrections: dict[int, tuple[str, str | None]] = {}
    for idx, det in enumerate(detections):
        result = snap(det.text)
        corrections[idx] = (result.text, result.warning)

    return score(truth, detections, corrections)


def print_report(label: str, report: Report) -> None:
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    print(f"  text recall            {report.recall:6.1%}   "
          f"({sum(r.found for r in report.runs)}/{len(report.runs)} runs found)")
    print(f"  char accuracy (raw)    {report.char_accuracy(corrected=False):6.1%}")
    print(f"  char accuracy (snapped){report.char_accuracy(corrected=True):6.1%}   <- after lexicon")
    print(f"  exact strings (raw)    {report.exact_raw_rate:6.1%}")
    print(f"  exact strings (snapped){report.exact_corrected_rate:6.1%}")
    print(f"  orientation correct    {report.angle_accuracy:6.1%}")
    print(f"  mean anchor error      {report.mean_anchor_error:6.1f} px")

    failures = report.failures
    if failures:
        print(f"\n  string errors ({len(failures)}):")
        for r in failures:
            got = r.detected if r.detected is not None else "<not found>"
            arrow = f"{got!r}" + (f" -> {r.corrected!r}" if r.corrected != r.detected else "")
            print(f"    {r.truth!r:<12} {r.kind:<10} cap {r.cap_height_px:4.0f}px  "
                  f"conf {r.conf if r.conf is not None else 0:.3f}  got {arrow}")
            if r.warning:
                print(f"      ! {r.warning}")
    else:
        print("\n  no string errors.")

    # Orientation is scored separately from the string, because a run can
    # be read perfectly and still be re-drawn rotated the wrong way —
    # which is a visible defect on the sheet, not a silent one.
    wrong_angle = [
        r for r in report.runs
        if r.angle_detected is None
        or abs(((r.angle_truth - r.angle_detected + 180) % 360) - 180) >= 15
    ]
    if wrong_angle:
        print(f"\n  orientation errors ({len(wrong_angle)}):")
        for r in wrong_angle:
            got = r.angle_detected if r.angle_detected is not None else float("nan")
            print(f"    {r.truth!r:<12} {r.kind:<10} truth {r.angle_truth:+.0f}deg  "
                  f"got {got:+.0f}deg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, nargs="+", default=[150])
    ap.add_argument("--single-pass", action="store_true",
                    help="Skip the rotated passes, to measure what they buy.")
    args = ap.parse_args()

    backend = RapidOcrBackend()
    for dpi in args.dpi:
        if args.single_pass:
            print_report(f"dpi {dpi} — single pass", run_one(dpi, backend, triple_pass=False))
        else:
            print_report(f"dpi {dpi} — single pass", run_one(dpi, backend, triple_pass=False))
            print_report(f"dpi {dpi} — triple pass (0/+90/-90)", run_one(dpi, backend, True))


if __name__ == "__main__":
    main()
