"""Stage S4 — measure the type so it can be re-rendered invisibly.

Four numbers per run, all *measured* from the crop rather than guessed:
ink colour, paper colour, font size, and tracking. Tracking matters more
than it looks: CAD text is frequently letter-spaced, and re-rendering at
the right size but natural spacing is what makes a label look subtly
wrong next to untouched linework.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

# Font resolution order. Arial is metric-compatible with the Helvetica
# that CAD exporters emit, which is why re-rendered runs land on the same
# advance widths. Packaging note (build plan §11): arial.ttf is NOT
# redistributable — a frozen build ships Liberation Sans instead, which is
# metric-compatible with both.
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)
_BOLD_CANDIDATES = (
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


@dataclass(frozen=True)
class TextStyle:
    """Everything the renderer needs to redraw one run."""

    font_path: str
    px_size: int
    tracking: float          # extra per-glyph spacing, as a fraction of px_size (em-relative — see solve_tracking)
    ink: tuple[int, int, int]      # RGB
    paper: tuple[int, int, int]    # RGB
    cap_height_px: float           # measured ink extent across the baseline
    ink_along_px: float = 0.0      # measured ink extent along the baseline
    width_scale: float = 1.0       # horizontal stretch/compress, applied after tracking — see solve_horizontal_scale


def resolve_font(bold: bool = False) -> str:
    candidates = _BOLD_CANDIDATES + _FONT_CANDIDATES if bold else _FONT_CANDIDATES
    for path in candidates:
        if Path(path).exists():
            return path
    raise RuntimeError(
        "No usable sans-serif TTF found. Install Liberation Sans, or pass an "
        "explicit font_path — Spejl will not silently fall back to a bitmap "
        "font, because that would visibly differ from the untouched sheet."
    )


@lru_cache(maxsize=512)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def measure_ink_and_paper(
    image: np.ndarray, bbox: tuple[float, float, float, float]
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Ink = median of the darkest decile, paper = median of the lightest.

    Deciles rather than min/max: an antialiased glyph edge produces a
    continuum, and the extremes are outliers. This also handles grey
    annotation text and tinted backgrounds with no special case.
    """
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    h, w = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return ((0, 0, 0), (255, 255, 255))

    crop = image[y0:y1, x0:x1]
    if crop.ndim == 2:
        crop = np.dstack([crop] * 3)
    flat = crop.reshape(-1, 3).astype(np.float32)
    luma = flat @ np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR weights

    dark_cut = np.percentile(luma, 10)
    light_cut = np.percentile(luma, 90)
    dark = flat[luma <= dark_cut]
    light = flat[luma >= light_cut]
    if len(dark) == 0 or len(light) == 0:
        return ((0, 0, 0), (255, 255, 255))

    ink_bgr = np.median(dark, axis=0)
    paper_bgr = np.median(light, axis=0)
    to_rgb = lambda c: (int(round(c[2])), int(round(c[1])), int(round(c[0])))  # noqa: E731
    return to_rgb(ink_bgr), to_rgb(paper_bgr)


_PAPER_TOLERANCE = 14  # matches erase.clean.PAPER_TOLERANCE


def measure_ink_extent(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    angle_deg: float,
    other_boxes: list[tuple[float, float, float, float]] | None = None,
    expected_glyphs: int = 0,
) -> tuple[float, float]:
    """Measured (along-baseline, across-baseline) extent of actual glyph
    ink inside ``bbox``, in pixels.

    Measuring rather than inferring, for a concrete reason: a detector's
    box is *padded*, so scaling it by any fixed ink-to-box constant
    over-estimates cap height — which fits an over-large font, which is
    how re-rendered labels end up ~20% wider than the ones beside them.

    Linework is excluded by shape. A dimension line crossing the box is
    a single component that is both extremely elongated and spans nearly
    the whole box; glyphs are compact. Without that filter, the run
    ``4060`` — which sits directly on its own dimension line — would
    measure as tall as the line is long.

    ``other_boxes``, if given, are every OTHER run's own detection box —
    the same exclusion erase.clean.build_text_mask already applies for
    the same reason (see its docstring): real plans crowd runs only a
    few pixels apart, close enough that one run's box clips the edge of
    a neighbour's glyphs. Left in, that foreign ink becomes its own
    connected component inside THIS crop and can be swept into the
    glyph cluster below — confirmed on a real plan where a vertical
    '4381' dimension's tight box clipped a corner of the 'Entre' label
    sitting right next to it.

    Sparse architectural linework inside the box is excluded too (see
    _drop_sparse_linework): a door-jamb symbol next to 'Entre' on that
    same real plan spans the crop's full height but is mostly empty
    space between its two thin strokes — not shaped like the dimension
    line the check above already catches, but just as much not a glyph.
    """
    raw_x0, raw_y0, raw_x1, raw_y1 = (int(round(v)) for v in bbox)
    vertical = abs(angle_deg) > 45

    def _fallback() -> tuple[float, float]:
        # Use the ORIGINAL (unclamped) box, not the clamped one — a box
        # that lies partly or wholly outside the image would otherwise
        # collapse toward zero regardless of the box's real size, which
        # used to hand fit_font_size a bogus ~1px target and produce an
        # unreadably tiny font with no error raised.
        bw = max(1.0, float(raw_x1 - raw_x0))
        bh = max(1.0, float(raw_y1 - raw_y0))
        return (bh, bw) if vertical else (bw, bh)

    cluster = _measure_ink_cluster_bbox(image, bbox, angle_deg, other_boxes, expected_glyphs)
    if cluster is None:
        return _fallback()
    gx0, gy0, gx1, gy1 = cluster
    ink_w = max(1.0, gx1 - gx0)
    ink_h = max(1.0, gy1 - gy0)
    return (ink_h, ink_w) if vertical else (ink_w, ink_h)


def measure_ink_center(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    angle_deg: float,
    other_boxes: list[tuple[float, float, float, float]] | None = None,
    expected_glyphs: int = 0,
) -> tuple[float, float]:
    """The actual glyph ink's centre point, in absolute image
    coordinates — not the raw detection box's own centre.

    A detector's box is padded, and that padding is not always even on
    every side: dash-noise from a crossing reference line prepended to
    a room label's raw OCR read ('---Entre') pads the box's LEFT edge
    much further out than its right, so the box's own geometric centre
    sits measurably left of where the word 'Entre' actually visually
    centres. Confirmed on a real plan: anchoring the re-render on that
    raw centre reproduces the same lopsided offset — MIRRORED, which
    turns a small leftward bias in the source into a rightward one in
    the output, visibly off-centre in its own room relative to where
    the word sat on the original drawing. Anchoring on the actual ink
    cluster's own centre instead means the re-rendered word sits
    exactly where the word itself was, regardless of what else the
    detection box happened to pad around it.

    Shares :func:`_measure_ink_cluster_bbox` with :func:`measure_ink_extent`
    — same filtered, clustered ink extent, just reported as a centre
    point instead of a size — and falls back to the raw box's own
    centre under the same condition that function falls back to the raw
    box's own size (no usable ink found).
    """
    cluster = _measure_ink_cluster_bbox(image, bbox, angle_deg, other_boxes, expected_glyphs)
    if cluster is None:
        x0, y0, x1, y1 = bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    gx0, gy0, gx1, gy1 = cluster
    return ((gx0 + gx1) / 2.0, (gy0 + gy1) / 2.0)


def _measure_ink_cluster_bbox(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    angle_deg: float,
    other_boxes: list[tuple[float, float, float, float]] | None,
    expected_glyphs: int,
) -> tuple[float, float, float, float] | None:
    """The actual glyph ink's bounding box inside ``bbox``, in ABSOLUTE
    image coordinates — or None when no usable ink was found (empty or
    entirely out-of-bounds crop), leaving the caller to fall back to
    the raw detection box.

    Shared by :func:`measure_ink_extent` and :func:`measure_ink_center`:
    both need the identical filtered, clustered ink extent — one
    reports its size, the other its centre — so the connected-component
    analysis and its three contamination filters live here once rather
    than as two copies to keep in sync.
    """
    raw_x0, raw_y0, raw_x1, raw_y1 = (int(round(v)) for v in bbox)
    h, w = image.shape[:2]
    x0, y0 = max(0, raw_x0), max(0, raw_y0)
    x1, y1 = min(w, raw_x1), min(h, raw_y1)
    vertical = abs(angle_deg) > 45

    if x1 <= x0 or y1 <= y0:
        return None

    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    paper = float(np.percentile(gray, 90))
    ink = (gray < paper - _PAPER_TOLERANCE).astype(np.uint8)
    if other_boxes:
        _mask_out_other_boxes(ink, x0, y0, other_boxes)
    if not ink.any():
        return None

    ch, cw = ink.shape
    count, _labels, stats, _cent = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes: list[tuple[int, int, int, int, int]] = []
    for i in range(1, count):
        cx, cy, cwid, chgt, area = stats[i]
        if area < 2:
            continue  # single-pixel speckle
        spans_box = cwid > cw * 0.8 or chgt > ch * 0.8
        elongated = cwid > chgt * 8 or chgt > cwid * 8
        if spans_box and elongated:
            continue  # a rule or dimension line, not a glyph
        boxes.append((cx, cy, cx + cwid, cy + chgt, int(area)))

    if not boxes:
        return None

    # Only safe when at least as many components survived as the string
    # has characters — i.e. nothing suggests a real glyph is already
    # fused with something else into one component. When a component
    # count is short (see _drop_foreign_strokes's docstring for the
    # '3306' case: a wall fused with two of its own digits, leaving only
    # 3 components for 4 characters), dropping the largest one would
    # throw away real glyph ink with no way to recover it — worse than
    # leaving the measurement inflated, which is at least visible and
    # recoverable via the render-time shrink-to-fit guard.
    if expected_glyphs and len(boxes) >= expected_glyphs:
        boxes = _drop_foreign_strokes(boxes, vertical)
        boxes = _drop_sparse_linework(boxes)
        boxes = _drop_edge_touching_intrusions(boxes, ch, cw, vertical)
        if not boxes:
            return None

    boxes = _main_glyph_cluster(boxes, ch, cw, vertical)

    gx0 = min(b[0] for b in boxes)
    gy0 = min(b[1] for b in boxes)
    gx1 = max(b[2] for b in boxes)
    gy1 = max(b[3] for b in boxes)
    return (float(x0 + gx0), float(y0 + gy0), float(x0 + gx1), float(y0 + gy1))


def _mask_out_other_boxes(
    ink: np.ndarray,
    crop_x0: int,
    crop_y0: int,
    other_boxes: list[tuple[float, float, float, float]],
) -> None:
    """Zero out, in place, any ``ink`` pixels that actually belong to
    another run's own detection box (see measure_ink_extent's docstring
    for why this matters — same failure mode erase.clean.build_text_mask
    already guards against for erasure)."""
    ch, cw = ink.shape
    for ox0, oy0, ox1, oy1 in other_boxes:
        lx0 = max(0, int(np.floor(ox0)) - crop_x0)
        ly0 = max(0, int(np.floor(oy0)) - crop_y0)
        lx1 = min(cw, int(np.ceil(ox1)) - crop_x0)
        ly1 = min(ch, int(np.ceil(oy1)) - crop_y0)
        if lx1 > lx0 and ly1 > ly0:
            ink[ly0:ly1, lx0:lx1] = 0


def _core_candidates(
    boxes: list[tuple[int, int, int, int, int]],
) -> list[tuple[int, int, int, int, int]]:
    """The subset of ``boxes`` big enough, by true pixel count, to
    plausibly be one of the run's own glyphs rather than a stray dash,
    dot, or reference-line fragment mixed in among them.

    Used as the reference POOL for a median by both outlier filters
    below — not as a final answer on its own. A handful of small
    contaminants can otherwise drag a median far enough off-centre that
    an ordinary, correctly-sized real glyph looks like the outlier
    instead (confirmed on a real plan's '--Vr.3': 5 tiny dash/dot
    fragments outnumbered its 4 real letter-ish components, and taking
    the median across all 9 wrongly flagged one of the real letters in
    both filters below — see each one's docstring for which direction).

    Same 8%-of-the-largest-component threshold :func:`_main_glyph_cluster`
    already uses for the same "sliver vs. real letter part" judgment.
    """
    if not boxes:
        return boxes
    anchor_area = max(b[4] for b in boxes)
    min_area = max(1, round(anchor_area * 0.08))
    core = [b for b in boxes if b[4] >= min_area]
    return core or boxes


def _drop_foreign_strokes(
    boxes: list[tuple[int, int, int, int, int]], vertical: bool
) -> list[tuple[int, int, int, int, int]]:
    """Drop a component whose extent ALONG the reading direction dwarfs
    its siblings' — the signature of a foreign stroke (a wall or
    reference line clipping the edge of the detection box) rather than
    one of the run's own glyphs.

    Every glyph in one OCR-detected run is drawn at the same font size,
    so their along-baseline extents cluster tightly — digits in a
    vertical dimension are all roughly the same height, letters in a
    horizontal word are all roughly the same height too. A linework
    intrusion has no such constraint. The existing elongation filter in
    :func:`measure_ink_extent` already drops an axis-aligned rule or
    dimension line, but a DIAGONAL wall clipping a rotated detection box
    is neither long enough relative to the crop nor extreme enough in
    aspect ratio to trip it — confirmed on a real plan, where a diagonal
    partition wall entering a vertical '4381' dimension's tight box
    produced a component 3x taller (along the reading direction) than
    any of the run's own digits. Being the single largest component by
    pixel count, it became :func:`_main_glyph_cluster`'s anchor and
    dragged the whole cluster's measured extent out to match it, fitting
    a font roughly double the size of every neighbouring dimension.

    Comparing against the group's own median rather than a fixed pixel
    threshold keeps this self-calibrating to whatever size this
    particular run happens to be drawn at, instead of a magic number
    tuned to one sheet's scale.

    Only called when ``len(boxes) >= len(text)`` (see the call site in
    measure_ink_extent): a component count that already falls short of
    the character count is itself evidence that a real glyph is fused
    with something else into one component — confirmed on the SAME
    project's '3306', where a diagonal wall touching two of its own
    digits left only 3 components for 4 characters. Dropping the
    largest component there would discard real digit ink with no way
    to recover it, which is strictly worse than the over-measurement
    this function exists to fix.

    The median is taken from :func:`_core_candidates`, not every
    surviving box: confirmed on the SAME project's '--Vr.3' (a garbled
    OCR read crossed by the same kind of dashed reference line as
    'Entre' — see _drop_sparse_linework), where 5 tiny dash/dot
    fragments outnumbered the 4 real letter-ish components. Taking the
    median across all 9 dragged it down to the dashes' own width, which
    made the real (and merely average-width) 'r' glyph look like the
    oversized outlier and get wrongly dropped — under-measuring the run
    and nearly breaking its render, the same shape of regression '3306'
    caught for the sibling filter below.
    """
    if len(boxes) <= 2:
        return boxes  # too few siblings for "dwarfs the others" to mean anything

    def along_span(b: tuple[int, int, int, int, int]) -> int:
        return (b[3] - b[1]) if vertical else (b[2] - b[0])

    spans = sorted(along_span(b) for b in _core_candidates(boxes))
    median = spans[len(spans) // 2]
    if median <= 0:
        return boxes
    kept = [b for b in boxes if along_span(b) <= median * 2.5]
    return kept or boxes  # never discard every candidate outright


def _drop_sparse_linework(
    boxes: list[tuple[int, int, int, int, int]],
) -> list[tuple[int, int, int, int, int]]:
    """Drop a component whose FILL RATIO — actual ink pixels divided by
    its own bounding-box area — is far sparser than its siblings'.

    A real glyph is a comparatively solid shape: even a hollow letter
    like 'O' or a thin stroke like '1' fills a substantial fraction of
    its own tight bounding box (confirmed across this run's own siblings
    below, all 35-61%). A door-jamb symbol — two thin parallel strokes
    with open space between them, exactly the shape architectural plans
    draw next to a doorway — is nothing like that: mostly empty box.

    Confirmed on a real plan: a door-jamb symbol sitting right next to
    'Entre' had a bounding box tall enough to span the ENTIRE detection
    crop (fill ratio 9.5%, against 35-61% for 'Entre's own five
    letters). Neither existing filter catches it — it's not elongated
    enough on one axis to trip measure_ink_extent's own rule-line check,
    and unlike a wall clipping a ROTATED (vertical) run, this sits
    beside ordinary HORIZONTAL text, so _drop_foreign_strokes's
    along-baseline (here: X-axis, the reading direction) comparison
    never sees it as an outlier — the jamb symbol's problem is its
    HEIGHT (the cross-baseline axis), which that filter deliberately
    leaves alone since letters legitimately vary there (ascenders,
    descenders). Being the tallest component, it single-handedly set
    the cluster's measured cap height to the full crop height, fitting
    a font roughly a third larger than every sibling room label on the
    same sheet.

    Same median-relative, self-calibrating shape as
    :func:`_drop_foreign_strokes`, and gated by the same
    ``expected_glyphs`` precondition at the call site for the same
    reason: a fused component's fill ratio can't be trusted either.

    The median comes from :func:`_core_candidates`, not every surviving
    box — and here the direction of the failure this guards against is
    the OPPOSITE of _drop_foreign_strokes's: a handful of solid, fully-
    filled dash fragments (fill ratio 1.0, being nothing but ink in a
    box the same size as themselves) can outnumber the run's own
    letters and drag the median UP rather than down, making a
    perfectly ordinary letter look sparse by comparison and get wrongly
    dropped — confirmed on the SAME '--Vr.3' case _drop_foreign_strokes
    documents: without this exclusion, its own 'V' (fill 0.36) and
    another real letter (fill 0.37) both fell under a dash-inflated
    threshold and were dropped alongside the dashes.
    """
    if len(boxes) <= 2:
        return boxes

    def fill_ratio(b: tuple[int, int, int, int, int]) -> float:
        box_area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return b[4] / box_area

    fills = sorted(fill_ratio(b) for b in _core_candidates(boxes))
    median = fills[len(fills) // 2]
    if median <= 0:
        return boxes
    kept = [b for b in boxes if fill_ratio(b) >= median * 0.4]
    return kept or boxes  # never discard every candidate outright


def _drop_edge_touching_intrusions(
    boxes: list[tuple[int, int, int, int, int]], crop_height: int, crop_width: int, vertical: bool
) -> list[tuple[int, int, int, int, int]]:
    """Drop — or, when it's too big to safely discard, CLIP — a
    component that touches the crop boundary on the CROSS-baseline
    axis: the top or bottom edge for horizontal text, the left or
    right edge for vertical text.

    measure_ink_extent's own docstring states the assumption this relies
    on: a detector's box is *padded* around its own text, specifically
    so a fixed ink-to-box constant would over-estimate cap height. Real
    glyph ink therefore has room on every side and should never reach
    the crop's edge exactly; something that DOES touch an edge is, by
    that same assumption, linework that continues beyond the box rather
    than a self-contained glyph — confirmed on a real plan's 'Depot',
    where a wall stroke touched both the top and bottom of the crop
    and, being the single largest component by bounding-box area, set
    the cluster's measured cap height to the full crop height.

    Dropping it outright is only safe when it's genuinely small
    (comparable to noise, not to a letter): on that SAME 'Depot', the
    wall isn't just adjacent to the 'D' — it's physically TOUCHING it,
    fused into one connected component neither erosion nor any
    component-level shape test can cleanly split. Dropping that fused
    component wholesale (an earlier version of this fix did) discarded
    'D' along with the wall, under-measuring the run's WIDTH and
    forcing an unrelated render-time shrink nobody could see the cause
    of. The area check below recognises when that's happening — a
    component too big to be pure linework — and CLIPS its cross-axis
    extent to match the run's own unambiguous letters instead of
    discarding it: the wall's excess height/width is cut away, but
    whatever of its along-baseline extent might be a real, fused-in
    glyph is kept. A genuinely small edge-toucher (a stray mark, not a
    fusion) still gets dropped outright, same as before.

    Deliberately axis-specific, the same way _drop_foreign_strokes is:
    a real glyph's ALONG-baseline edges (left/right of a horizontal
    word, top/bottom of a vertical dimension) legitimately sit close to
    the box edge — that's just where the first or last character is —
    so only the cross-baseline edges are checked.
    """
    if len(boxes) <= 2:
        return boxes

    def touches_cross_edge(b: tuple[int, int, int, int, int]) -> bool:
        return (b[0] <= 0 or b[2] >= crop_width) if vertical else (b[1] <= 0 or b[3] >= crop_height)

    interior = [b for b in boxes if not touches_cross_edge(b)]
    if not interior:
        return boxes  # every candidate touches an edge -- nothing trustworthy to clip against

    if vertical:
        interior_lo = min(b[0] for b in interior)
        interior_hi = max(b[2] for b in interior)
    else:
        interior_lo = min(b[1] for b in interior)
        interior_hi = max(b[3] for b in interior)

    areas = sorted(b[4] for b in interior)
    median_area = areas[len(areas) // 2]

    kept = list(interior)
    for b in boxes:
        if not touches_cross_edge(b):
            continue
        if median_area > 0 and b[4] > median_area * 1.5:
            if vertical:
                kept.append((max(b[0], interior_lo), b[1], min(b[2], interior_hi), b[3], b[4]))
            else:
                kept.append((b[0], max(b[1], interior_lo), b[2], min(b[3], interior_hi), b[4]))
        # else: small enough to be pure linework or stray noise -- drop it.
    return kept or boxes  # never discard every candidate outright


def _main_glyph_cluster(
    boxes: list[tuple[int, int, int, int, int]], crop_height: int, crop_width: int, vertical: bool
) -> list[tuple[int, int, int, int, int]]:
    """Keep only the components that form one word's own glyphs, and
    drop anything sitting in a detection box that isn't actually part
    of it.

    A detector's box is not always airtight around just its own text —
    a nearby door-swing arc or a dimension tick can fall just inside
    it. That single extra component is compact (so the elongation
    filter above lets it through), which is what makes it dangerous:
    the ONE thing that reliably separates it from real letter parts
    (an 'i's dot, a broken serif, disconnected pixels of a stroke) is
    that a real letter part is never a sliver of the main glyph mass —
    it is a meaningful fraction of it. Confirmed on this project's own
    golden fixture: a stray 2x7px, 10-pixel-area fragment sitting just
    above 'Bad's own 726-pixel-area text mass inflated its measured cap
    height 14% over its true size (identical to 'Toilet' right next to
    it, verified against this fixture's own ground truth).

    Filtering happens on the CROSS-baseline axis only — Y for
    horizontal text, X for vertical — never on the along-baseline axis,
    which is the reading direction and where real, sometimes-generous
    gaps between characters or words are completely normal (','Vær. 1'
    has a genuine gap before its '1'; a vertical dimension's digits
    are legitimately spaced apart top-to-bottom). An early version of
    this filter checked gaps on the crop's Y-axis unconditionally,
    which is correct for horizontal words but wrong for a vertical run:
    it read the ordinary spacing between a vertical string's own
    stacked digits as "this might be a stray mark," discarded every
    digit but the single largest, and roughly halved '5155's measured
    length — a 37% OVERSIZE once that truncated measurement was used
    as the target width for re-rendering the full string.
    """
    if len(boxes) <= 1:
        return boxes

    def area(b: tuple[int, int, int, int]) -> int:
        return (b[2] - b[0]) * (b[3] - b[1])

    def cross_span(b: tuple[int, int, int, int]) -> tuple[int, int]:
        return (b[0], b[2]) if vertical else (b[1], b[3])

    ordered = sorted(boxes, key=area, reverse=True)
    anchor_area = area(ordered[0])
    min_area = max(1, round(anchor_area * 0.08))
    gap_tolerance = max(1, round((crop_width if vertical else crop_height) * 0.15))

    cluster = [ordered.pop(0)]
    remaining = [b for b in ordered if area(b) >= min_area]
    band0, band1 = cross_span(cluster[0])

    changed = True
    while changed and remaining:
        changed = False
        for box in list(remaining):
            b0, b1 = cross_span(box)
            if b0 <= band1 + gap_tolerance and b1 >= band0 - gap_tolerance:
                cluster.append(box)
                remaining.remove(box)
                band0, band1 = min(band0, b0), max(band1, b1)
                changed = True

    return cluster


def fit_font_size(
    text: str, target_cap_height: float, font_path: str, lo: int = 4, hi: int = 400
) -> int:
    """Binary-search the point size whose rendered cap height matches.

    Measures the actual ink box of the string, not the font's nominal
    metrics, because cap height varies with which glyphs are present
    ('870' has no descender; 'Køkken' does).

    ``text or "0"`` only guards a truly empty string — a whitespace-only
    string (`" "`) is truthy, so it reached ``getbbox()`` unguarded and
    produced a zero-height box at *every* candidate size. With every
    iteration's error then identical, the strict ``err < best_err``
    comparison only ever fires once, on the very first midpoint probed —
    the search silently freezes there, at a size with no relation to
    ``target_cap_height``, rather than failing loudly or converging.
    Probing with ``"0"`` instead whenever the string has no visible
    glyphs keeps the search meaningful even for text nobody should ever
    end up passing here.
    """
    target = max(1.0, target_cap_height)
    probe_text = text if text.strip() else "0"
    best, best_err = lo, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        bbox = _font(font_path, mid).getbbox(probe_text)
        height = (bbox[3] - bbox[1]) if bbox else 0
        err = abs(height - target)
        if err < best_err:
            best, best_err = mid, err
        if height < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return max(1, best)


def solve_tracking(text: str, font_path: str, px_size: int, target_width: float) -> float:
    """Extra per-gap spacing that makes the run occupy its measured width,
    returned as an **em-relative fraction of px_size** — not absolute
    pixels.

    That unit choice is load-bearing, not stylistic. Tracking is computed
    once, at the size the run is first fitted at — but the fit-to-box
    guard (render/text.py) may shrink ``px_size`` afterwards. A fixed
    pixel gap, carried unchanged into a smaller font, becomes a
    proportionally *larger* — eventually overlapping — gap between
    increasingly small glyphs. That was a real defect found in this
    build: '1680' rendered with a clamped -12%-of-original-size pixel
    gap read back, after a second mirroring pass, as '11680'. An
    em-relative fraction scales down together with the size, so it
    cannot run away like that.
    """
    if len(text) < 2:
        return 0.0
    natural = font_measure_width(text, font_path, px_size, 0.0)
    gaps = len(text) - 1
    tracking_px = (target_width - natural) / gaps
    tracking_em = tracking_px / px_size
    # Asymmetric on purpose, and the asymmetry is load-bearing, found
    # by a real regression this build caught before it shipped:
    # negative (compressing) tracking risks real illegibility — two
    # glyphs squeezed together can read as a different letter, which is
    # exactly how a first, symmetrically-tightened version of this
    # clamp turned 'Bad' into something RapidOCR read back as 'Stue' on
    # this project's own golden fixture. Positive (expanding) tracking
    # has no such failure mode — spaced-out glyphs stay individually
    # legible, they just look loose — so it can be bounded far more
    # aggressively without that risk. The compression side keeps the
    # original, already-proven -8% bound; the expansion side is what a
    # real plan actually needed tightening (Arial substituting for a
    # narrower house font needed enough width correction that the old
    # +50% bound rendered short room names as visibly gapped-out
    # letters) — solve_horizontal_scale picks up the rest of an
    # expansion-direction gap this tighter bound leaves behind.
    return float(np.clip(tracking_em, -0.08, 0.15))


def solve_horizontal_scale(
    natural_width: float, target_width: float, min_scale: float = 0.96, max_scale: float = 1.25
) -> float:
    """How much to horizontally stretch or compress a naturally-tracked
    render to close whatever gap remains to the measured target width.

    Also asymmetric, for the same reason ``solve_tracking``'s clamp is
    (see its docstring): compressing risks squeezing glyphs into
    illegibility, expanding does not. ``min_scale`` stays close to 1 —
    a bounded amount of compression is safe, much is not — while
    ``max_scale`` allows real room for a substitute font that's
    genuinely narrower than the original to expand toward it, reading
    as a font's own proportions (the same idea as a "Condensed" or
    "Expanded" variant) rather than as stretched inter-letter gaps.
    """
    if natural_width <= 0:
        return 1.0
    return float(np.clip(target_width / natural_width, min_scale, max_scale))


def font_measure_width(text: str, font_path: str, px_size: int, tracking_px: float) -> float:
    """Advance width of ``text`` at ``tracking_px`` **absolute pixels**
    of extra per-gap spacing — the renderer's own measurement, so fitting
    and drawing can never disagree. (Absolute pixels here, deliberately
    different from :class:`TextStyle`'s em-relative ``tracking`` — see
    :func:`solve_tracking`; every caller of this function either passes
    ``0.0``, where the unit is moot, or converts explicitly.)"""
    font = _font(font_path, px_size)
    if not text:
        return 0.0
    width = sum(font.getlength(ch) for ch in text)
    return float(width + tracking_px * (len(text) - 1))


def fit_style(
    image: np.ndarray,
    text: str,
    bbox: tuple[float, float, float, float],
    angle_deg: float,
    bold: bool = False,
    other_boxes: list[tuple[float, float, float, float]] | None = None,
) -> TextStyle:
    """Measure every number for one run — none are assumed.

    ``other_boxes`` — every OTHER run's own detection box on this page —
    is forwarded to :func:`measure_ink_extent` so a neighbour's ink
    close enough to clip this run's box can't inflate its measured
    extent. See that function's docstring for the real-plan case that
    motivated it.
    """
    font_path = resolve_font(bold=bold)
    ink, paper = measure_ink_and_paper(image, bbox)
    along, across = measure_ink_extent(
        image, bbox, angle_deg, other_boxes=other_boxes, expected_glyphs=len(text)
    )

    px_size = fit_font_size(text, across, font_path)
    tracking = solve_tracking(text, font_path, px_size, along)
    natural_tracked = font_measure_width(text, font_path, px_size, tracking * px_size)
    width_scale = solve_horizontal_scale(natural_tracked, along)

    return TextStyle(
        font_path=font_path,
        px_size=px_size,
        tracking=tracking,
        ink=ink,
        paper=paper,
        cap_height_px=across,
        ink_along_px=along,
        width_scale=width_scale,
    )
