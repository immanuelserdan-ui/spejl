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


def _intersection(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    return iw * ih


def _area(box: tuple[float, ...]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    inter = _intersection(a, b)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _containment(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over the *smaller* box's area.

    IoU is the wrong measure for a fragment sitting inside a run: a lone
    'B' detected inside 'Bad' scores only 0.28 IoU, because the union is
    dominated by the larger box — so it survives NMS and gets drawn a
    second time, as a ghost glyph over the real label. Containment
    catches it at 1.0.
    """
    inter = _intersection(a, b)
    if inter <= 0:
        return 0.0
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0 else 0.0


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


def _merge(
    candidates: list[Detection],
    iou_threshold: float,
    containment_threshold: float = 0.6,
) -> list[Detection]:
    """Confidence-ranked NMS over both overlap measures.

    Two tie-breaks matter here:

    * **Longer text wins.** The failure this triple pass exists to fix is
      a rotated run coming back *truncated* ('2105' read as '105'), and
      the truncated read is not reliably the less confident one.
    * **Containment suppresses.** A detection mostly inside an already
      kept run is a fragment of it, never a new run — see
      :func:`_containment`.

    A candidate overlapping *more than one* already-kept run is handled
    as neither of the above: it is rejected outright, never used to
    replace anything. Two adjacent real dimensions ('600', '300') kept
    separately, followed by a merged rotated-pass misread spanning both
    ('600300'), overlaps both — replacing only the longer-text incumbent
    would drop '600' from the output while leaving '300' beside a bogus
    '600300', a duplicate exactly like this NMS pass exists to prevent.
    Single-run truncation correction (the designed case, e.g. '105' ->
    '2105') only ever has one overlapping incumbent, so this leaves that
    path unchanged.

    The ``cand.conf > best.conf - 0.05`` margin on "longer text wins" is
    itself a trap for the exact case it exists to fix, confirmed on a
    real plan: a rotated pass misread the trailing digit of 'Vaer. 2' as
    a lone, trivially easy '2' at PERFECT 1.000 confidence, kept ahead
    of the correct seven-character 'Vaer. 2' (0.949 — an honest score
    for a longer, harder read, not a sign of anything wrong with it).
    0.949 misses a 0.05 margin below 1.000 by one thousandth, so the
    entire correct run was silently dropped — never erased, never
    re-rendered, its raw source pixels left to pass straight through
    the mirror flip as ordinary geometry: genuinely mirrored text, the
    one failure mode this whole application exists to prevent. A
    single-digit fragment scoring 1.000 is not meaningfully MORE
    trustworthy than a 7-character run scoring 0.949; it is just an
    easier read, and comparing raw confidence across such different
    string lengths at a fixed few-point margin doesn't hold up. When
    the shorter incumbent's box sits almost ENTIRELY inside the longer
    candidate's own box (containment > 0.9 — a near-total subset
    relationship, not just heavy overlap), that is near-unambiguous
    evidence the short one is a fragment of the SAME physical text the
    long one read more completely, not independent competing evidence
    about different content — so the longer candidate only has to clear
    a sensible absolute confidence bar (the same 0.85 LOW_CONFIDENCE
    threshold raster/pipeline.py already flags a run at), not out-score
    the fragment it is replacing.
    """
    ordered = sorted(candidates, key=lambda d: (d.conf, len(d.text)), reverse=True)
    kept: list[Detection] = []
    for cand in ordered:
        overlapping = [
            k
            for k in kept
            if _iou(cand.bbox, k.bbox) > iou_threshold
            or _containment(cand.bbox, k.bbox) > containment_threshold
        ]
        if not overlapping:
            kept.append(cand)
            continue
        if len(overlapping) > 1:
            continue  # spans multiple real runs — a merged misread, drop it
        best = overlapping[0]
        if len(cand.text) <= len(best.text):
            continue
        beats_margin = cand.conf > best.conf - 0.05
        is_fuller_read_of_a_fragment = (
            _containment(best.bbox, cand.bbox) > 0.9 and cand.conf >= 0.85
        )
        if beats_margin or is_fuller_read_of_a_fragment:
            kept.remove(best)
            kept.append(cand)
    return sorted(kept, key=lambda d: (d.bbox[1], d.bbox[0]))
