"""Triple-pass detection: read the page, then read it turned both ways.

Recognition models are trained on horizontal lines. A vertical dimension
label is the case they handle worst — and on an architectural plan every
wall run parallel to the sheet's left edge carries one. So the page is
OCR'd three times (as-is, turned clockwise, turned anticlockwise), every
detection is mapped back to source coordinates, and the passes are merged
keeping the most confident claim per region.

Which pass found a run also *tells you its orientation*, for free:
rotating the image clockwise maps an original direction (dx, dy) to
(-dy, dx), so original bottom-to-top text (0, -1) becomes horizontal
(1, 0) and is legible in the clockwise pass. Hence clockwise ⇒ 90°
(bottom-to-top), anticlockwise ⇒ -90° (top-to-bottom).
"""

from __future__ import annotations

import cv2
import numpy as np

from spejl.detect.ocr import Detection, OcrBackend

# Rotation passes: (cv2 rotate code, the source-space angle it reveals)
_PASSES = (
    (None, 0.0),
    (cv2.ROTATE_90_CLOCKWISE, 90.0),
    (cv2.ROTATE_90_COUNTERCLOCKWISE, -90.0),
)


def _unrotate_point(
    x: float, y: float, src_w: int, src_h: int, rotate_code: int | None
) -> tuple[float, float]:
    """Map a point from a rotated frame back to source coordinates.

    Forward transforms (source -> rotated), for a source of ``src_w`` x ``src_h``:
        clockwise:      (x, y) -> (src_h - 1 - y, x)
        anticlockwise:  (x, y) -> (y, src_w - 1 - x)
    These are their inverses.
    """
    if rotate_code is None:
        return (x, y)
    if rotate_code == cv2.ROTATE_90_CLOCKWISE:
        return (y, src_h - 1 - x)
    return (src_w - 1 - y, x)  # ROTATE_90_COUNTERCLOCKWISE


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _orientation_is_plausible(det: Detection) -> bool:
    """Does this detection's claimed angle match the shape of its box?

    A rotated pass will happily re-read *horizontal* text too (often at
    high confidence, since the recogniser's angle classifier straightens
    the crop before reading it) and that read arrives tagged 90°. Left
    alone it wins NMS and the run gets re-drawn rotated — which is how
    orientation accuracy ends up *worse* with three passes than with one.

    Source-space geometry settles it: a genuine vertical run's box is
    taller than it is wide. Single glyphs are exempt, their aspect ratio
    carries no orientation information.
    """
    if len(det.text) < 2:
        return True
    x0, y0, x1, y1 = det.bbox
    w, h = x1 - x0, y1 - y0
    claims_horizontal = abs(det.angle_deg) < 45
    if claims_horizontal:
        return not (h > w * 1.15)
    return not (w > h * 1.15)


def detect_all_orientations(
    image: np.ndarray, backend: OcrBackend, iou_threshold: float = 0.3
) -> list[Detection]:
    """Run every rotation pass and merge into one set of source-space runs."""
    src_h, src_w = image.shape[:2]
    candidates: list[Detection] = []

    for rotate_code, angle in _PASSES:
        frame = image if rotate_code is None else cv2.rotate(image, rotate_code)
        for det in backend.detect_and_recognise(frame):
            quad = tuple(
                _unrotate_point(px, py, src_w, src_h, rotate_code) for px, py in det.quad
            )
            candidate = Detection(text=det.text, quad=quad, conf=det.conf, angle_deg=angle)
            if _orientation_is_plausible(candidate):
                candidates.append(candidate)

    return [_canonicalise_vertical(d) for d in _merge(candidates, iou_threshold)]


def _canonicalise_vertical(det: Detection) -> Detection:
    """Collapse -90° to +90°: the detector reports *whether* a run is
    vertical, not which way it was read.

    The recogniser's angle classifier straightens each crop before
    reading it, so a bottom-to-top run is read confidently by *both*
    rotated passes and the winning sign is close to a coin flip. That
    sign is also information the pipeline never consumes: the renderer
    normalises vertical text to bottom-to-top regardless (ISO dimension
    convention — see vector/pdf_mirror._is_canonical_direction), so a
    -90° tag can only ever disagree with the drawing that gets produced.

    Genuinely top-to-bottom vertical annotation would need the angle
    classifier disabled to be distinguished here; it does not occur on
    dimensioned architectural plans, and the review step (§06) is the
    backstop if it ever does.
    """
    if det.angle_deg == -90.0:
        return Detection(text=det.text, quad=det.quad, conf=det.conf, angle_deg=90.0)
    return det


def _merge(candidates: list[Detection], iou_threshold: float) -> list[Detection]:
    """Confidence-ranked NMS.

    A tie-break matters here: the same run is often found by two passes
    with near-identical confidence, and the winner decides the recorded
    angle. Longer text wins ties, because the failure mode this whole
    triple pass exists to fix is a rotated run coming back *truncated*
    ('2105' read as '105') — and the truncated read is not reliably the
    less confident one.
    """
    ordered = sorted(candidates, key=lambda d: (d.conf, len(d.text)), reverse=True)
    kept: list[Detection] = []
    for cand in ordered:
        overlapping = [k for k in kept if _iou(cand.bbox, k.bbox) > iou_threshold]
        if not overlapping:
            kept.append(cand)
            continue
        # Same region already claimed. Replace the incumbent only if this
        # candidate reads *more* characters at comparable confidence.
        best = max(overlapping, key=lambda k: len(k.text))
        if len(cand.text) > len(best.text) and cand.conf > best.conf - 0.05:
            kept.remove(best)
            kept.append(cand)
    return sorted(kept, key=lambda d: (d.bbox[1], d.bbox[0]))
