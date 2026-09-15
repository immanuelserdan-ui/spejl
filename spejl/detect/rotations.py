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

import math

import cv2
import numpy as np

from spejl.detect.ocr import Detection, OcrBackend
from spejl.transform import mirror as M

# Rotation passes: (cv2 rotate code, the source-space angle it reveals)
_PASSES = (
    (None, 0.0),
    (cv2.ROTATE_90_CLOCKWISE, 90.0),
    (cv2.ROTATE_90_COUNTERCLOCKWISE, -90.0),
)

# Below this quad elongation (along-edge / cross-edge), the detection is
# too close to square for its own tilt to mean anything — a 2-character
# annotation like 'H*' can read its quad angle as several degrees off
# true (confirmed on a real plan: -9.3°) purely from how little geometry
# there is to anchor an angle estimate, not because it is actually
# drawn crooked. Every genuinely tilted dimension number on that same
# plan (4+ digits, following a diagonal wall) measured 1.9-2.4 —
# comfortably clear of this floor — while 'H*' measured 1.25.
_MIN_ASPECT_FOR_QUAD_ANGLE = 1.5

# A quad angle within this many degrees of a cardinal (0/90/-90/180) is
# snapped to it outright, no further questions asked: real architectural
# text is drawn EXACTLY horizontal or vertical unless it deliberately
# follows a sloped wall, and a sloped wall's own tilt is a real,
# physical, consistent angle, not a fraction of a degree — every
# axis-aligned multi-character label sampled across several real plans
# measured under 2.5° of ordinary detection noise.
#
# A single fixed floor covering the WHOLE gap up to a genuine tilt
# turned out not to hold, though: one specific word ('Gang', of all
# things — recurring across three separate real files, so a property
# of how that word's own quad measures, not a fluke of one detection)
# measured as high as 5.96° of pure noise, overlapping the low end of
# confirmed REAL tilts elsewhere (6.1°+) closely enough that no single
# threshold in between reliably tells them apart. Beyond this floor,
# _resolve_ambiguous_tilts below decides using a stronger signal than
# magnitude alone: whether another, independent detection on the SAME
# sheet corroborates a closely matching tilt — see its own docstring.
_ALWAYS_SNAP_DEG = 3.0

# Above this, a deviation from cardinal is treated as unambiguously a
# real tilt without needing corroboration — matches
# style/metrics._ROTATED_CROP_MARGIN_DEG, the point past which a run is
# far enough from every cardinal that confusing it with ordinary
# detection noise was never plausible to begin with.
_UNAMBIGUOUS_TILT_DEG = 20.0

# Two independent detections' deviations from the SAME cardinal must
# agree within this many degrees of EACH OTHER to corroborate one
# another.
#
# 2.0° was the first calibration, and it was wrong: 'Gang' and 'Stue'
# (see _ALWAYS_SNAP_DEG's own comment — the same two noisy, unrelated
# room labels) agree with EACH OTHER to within 1.28°, close enough to
# pass a 2.0° tolerance and corroborate one another into looking like a
# real shared tilt neither of them has. Every genuine connection in the
# confirmed real cluster is tighter than that: the loosest link needed
# to keep the whole three-run vertical family connected (3921 to 4381,
# bridging the 1.20° gap from 3921 to 4504 directly) is 0.80°; the
# horizontal family's own pair agrees to 0.60°. 1.0° sits with margin
# on both sides of the ACTUAL worst cases on both sides (0.2° above the
# tightest bridge a real cluster has needed, 0.28° below the noise
# pair's own coincidental agreement) rather than a round number picked
# without checking either boundary.
_CORROBORATION_TOLERANCE_DEG = 1.0


def _quad_angle_deg(quad: tuple[tuple[float, float], ...], fallback: float) -> float:
    """The run's true reading-direction angle, read directly from its
    own detected quad, in Spejl's angle convention.

    RapidOCR's detector is not limited to axis-aligned boxes — it
    already reports where a run's own four corners actually sit, tilt
    included, regardless of which of the three rotation passes found
    it (confirmed on a real plan: a single UN-rotated pass alone
    reported a -4.9° tilt for a dimension number running along a
    diagonal wall). The triple-pass rotation exists to get a confident
    READ of hard-to-recognise vertical text, not to discover its
    geometry — the geometry was sitting in the quad the whole time.
    Snapping every candidate to whichever of 0°/90°/-90° its pass
    corresponds to, as this function replaces, discards that and
    forces a genuinely diagonal run into an axis-aligned bounding box
    well over twice its actual footprint — the direct cause of a real,
    user-reported bug: several dimension numbers following a sloped
    partition wall measured wildly oversized and rendered broken.

    Two guards keep this from trusting geometry that doesn't deserve
    it (see the two module constants' own comments for the real
    numbers behind each):

    * Below ``_MIN_ASPECT_FOR_QUAD_ANGLE`` elongation, the quad is too
      close to square for an angle to mean anything — falls back to
      ``fallback`` (the detecting pass's own canonical angle).
    * Within ``_ALWAYS_SNAP_DEG`` of a cardinal, snapped to it outright
      — ordinary detection noise around a genuinely axis-aligned run,
      not a real fractional-degree tilt. This alone isn't the whole
      story for angles beyond that floor, though — see
      :func:`_resolve_ambiguous_tilts`, applied afterwards in
      :func:`detect_all_orientations`, for why a moderate deviation
      still isn't trusted on its magnitude alone.

    The direction along the long edge is resolved with an
    axis-DOMINANT reading convention, deliberately NOT
    :func:`transform.mirror.is_canonical_direction`: that helper
    always defers to the horizontal component's sign whenever it is
    non-zero, which is right for mirroring an ALREADY-canonical
    direction (there, non-zero horizontal only ever shows up on
    genuinely horizontal text) but wrong here — a mostly-vertical raw
    edge from a tilted quad has a small but very much non-zero
    horizontal component purely from the tilt, and that component's
    sign is essentially arbitrary (an artefact of which corner the
    detector happened to label first), not a signal about reading
    direction. So: whichever axis actually dominates this direction
    decides which convention applies — left-to-right if the edge is
    more horizontal than vertical, bottom-to-top (ISO dimension-text
    convention, matching :func:`_canonicalise_vertical`) if more
    vertical than horizontal.
    """
    (x0, y0), (x1, y1), _, (x3, y3) = quad
    top_dx, top_dy = x1 - x0, y1 - y0
    left_dx, left_dy = x3 - x0, y3 - y0
    top_len = math.hypot(top_dx, top_dy)
    left_len = math.hypot(left_dx, left_dy)
    if top_len >= left_len:
        dx, dy, along, across = top_dx, top_dy, top_len, left_len
    else:
        dx, dy, along, across = left_dx, left_dy, left_len, top_len

    if along / max(1.0, across) < _MIN_ASPECT_FOR_QUAD_ANGLE:
        return fallback

    if abs(dx) >= abs(dy):
        if dx < 0:  # more horizontal than vertical: canonical is left-to-right
            dx, dy = -dx, -dy
    elif dy > 0:  # more vertical than horizontal: canonical is bottom-to-top (y-down coords)
        dx, dy = -dx, -dy
    angle = M.angle_from_direction(dx, dy)

    for cardinal in (-180.0, -90.0, 0.0, 90.0, 180.0):
        if abs(angle - cardinal) < _ALWAYS_SNAP_DEG:
            return cardinal
    return angle


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
    image: np.ndarray,
    backend: OcrBackend,
    iou_threshold: float = 0.3,
    dropped_out: list[Detection] | None = None,
) -> list[Detection]:
    """Run every rotation pass and merge into one set of source-space runs.

    ``dropped_out``, if given, is passed straight through to
    :func:`_merge` — see its own docstring for what ends up in it and
    why. Not populated with anything :func:`_orientation_is_plausible`
    rejects before ``_merge`` ever sees it: that filter exists to catch
    a rotated pass re-reading text that's ALREADY correctly read in its
    own proper-angle pass (a false orientation claim, not lost content),
    a different kind of noise than a genuine drop inside ``_merge``.

    Each surviving candidate's angle is then refined by
    :func:`_quad_angle_deg` — the detection's own quad geometry — in
    place of the pass's fixed 0°/90°/-90°; see that function's own
    docstring for why the pass angle alone used to force genuinely
    diagonal text (a dimension number following a sloped wall) into an
    axis-aligned box more than twice its real size.

    That refinement happens AFTER :func:`_orientation_is_plausible`,
    deliberately, not before: plausibility still checks the PASS's own
    canonical angle against the box shape, exactly as it always has.
    Confirmed as load-bearing on a real plan, not just cautious
    layering: a 90°-rotated pass misread 'Vær. 2' as the truncated
    'Vær.' (missing its '2', at HIGHER confidence than the correct
    full read) with a box far wider than tall — that mismatch against
    the pass's forced 90° claim is exactly what plausibility is built
    to catch, and it does, filtering the truncated read out before it
    can ever compete in the merge below. Computing the quad's own
    angle first and checking THAT against the box instead would remove
    this protection: the truncated read's own geometry genuinely IS
    near-horizontal, so it would pass a shape check against ITS OWN
    angle even though it is still a truncated, worse read of the same
    text a different pass got right in full.

    :func:`_resolve_ambiguous_tilts` runs last, on the final merged
    set — see its own docstring for why a single per-run magnitude
    threshold isn't the whole story for a moderate tilt.
    """
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
                resolved_angle = _quad_angle_deg(quad, fallback=angle)
                candidates.append(
                    Detection(text=det.text, quad=quad, conf=det.conf, angle_deg=resolved_angle)
                )

    merged = _merge(candidates, iou_threshold, dropped_out=dropped_out)
    canonicalised = [_canonicalise_vertical(d) for d in merged]
    return _resolve_ambiguous_tilts(canonicalised)


def _nearest_cardinal_and_deviation(angle_deg: float) -> tuple[float, float]:
    """The closest cardinal (0/90/-90/180) to ``angle_deg`` and the
    SIGNED deviation (``angle_deg - cardinal``) from it."""
    best_cardinal, best_dev = 0.0, angle_deg
    for cardinal in (-180.0, -90.0, 0.0, 90.0, 180.0):
        dev = angle_deg - cardinal
        if abs(dev) < abs(best_dev):
            best_cardinal, best_dev = cardinal, dev
    return best_cardinal, best_dev


def _resolve_ambiguous_tilts(detections: list[Detection]) -> list[Detection]:
    """A moderate deviation from cardinal (between ``_ALWAYS_SNAP_DEG``
    and ``_UNAMBIGUOUS_TILT_DEG`` — see both constants' own comments)
    is trusted only when at least one OTHER detection on the SAME
    sheet independently shows a closely matching deviation from the
    SAME cardinal — not on its own magnitude clearing some fixed floor.

    Why magnitude alone isn't enough: confirmed on three separate real
    plans, one specific word ('Gang') measured its own quad angle as
    high as 5.96° off true purely from detection noise — no diagonal
    wall anywhere near it on any of the three files — overlapping the
    low end of confirmed REAL tilts elsewhere (6.1°+) closely enough
    that no single threshold reliably tells the two apart. But a
    genuine tilt is never just one detection's opinion: every
    dimension number actually following a sloped wall agreed closely
    with its neighbours doing the same (three runs within 6.1-7.3° of
    each other, two more within 6.8-7.4°) — because they are all
    measuring the SAME physical slope. 'Gang' agreed with nothing
    else on its own sheet at anything close to its own deviation,
    because there was nothing else to agree with. Corroboration is
    the signal magnitude alone can't provide: cheap to check (this
    function runs once, on the small final per-page set, not per
    candidate), and it only ever makes MORE detections un-ambiguous,
    never fewer — a real tilt with no corroborating neighbour on this
    sheet still falls back to the safe default (snapped to cardinal,
    matching this run's own pre-quad-angle-fix behaviour) rather than
    guessing.
    """
    ambiguous: list[tuple[int, float, float]] = []  # (index, cardinal, deviation)
    for i, det in enumerate(detections):
        cardinal, dev = _nearest_cardinal_and_deviation(det.angle_deg)
        if _ALWAYS_SNAP_DEG <= abs(dev) < _UNAMBIGUOUS_TILT_DEG:
            ambiguous.append((i, cardinal, dev))

    corroborated: set[int] = set()
    for a in range(len(ambiguous)):
        i, cardinal_i, dev_i = ambiguous[a]
        for b in range(a + 1, len(ambiguous)):
            j, cardinal_j, dev_j = ambiguous[b]
            if cardinal_i == cardinal_j and abs(dev_i - dev_j) <= _CORROBORATION_TOLERANCE_DEG:
                corroborated.add(i)
                corroborated.add(j)

    resolved = list(detections)
    for i, cardinal, _dev in ambiguous:
        if i not in corroborated:
            det = detections[i]
            resolved[i] = Detection(text=det.text, quad=det.quad, conf=det.conf, angle_deg=cardinal)
    return resolved


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
    dropped_out: list[Detection] | None = None,
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

    ``dropped_out``, if given, is appended with every candidate that did
    NOT make it into the returned list — not for this function's own
    use, but so a caller can run a completeness check afterwards (see
    raster/pipeline.py's coverage flag): among everything OCR actually
    read, was anything substantial and confident left out that no kept
    run adequately covers? Most drops here are correct and expected (a
    truncated duplicate, a fragment inside a real run) — it's the
    caller's job to tell those apart from a genuine miss like the
    'Vaer. 2' case above, not this function's.
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
            if dropped_out is not None:
                dropped_out.append(cand)
            continue  # spans multiple real runs — a merged misread, drop it
        best = overlapping[0]
        if len(cand.text) <= len(best.text):
            if dropped_out is not None:
                dropped_out.append(cand)
            continue
        beats_margin = cand.conf > best.conf - 0.05
        is_fuller_read_of_a_fragment = (
            _containment(best.bbox, cand.bbox) > 0.9 and cand.conf >= 0.85
        )
        if beats_margin or is_fuller_read_of_a_fragment:
            kept.remove(best)
            kept.append(cand)
            if dropped_out is not None:
                dropped_out.append(best)
        elif dropped_out is not None:
            dropped_out.append(cand)
    return sorted(kept, key=lambda d: (d.bbox[1], d.bbox[0]))
