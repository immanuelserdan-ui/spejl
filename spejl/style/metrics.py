"""Stage S4 — measure the type so it can be re-rendered invisibly.

Four numbers per run, all *measured* from the crop rather than guessed:
ink colour, paper colour, font size, and tracking. Tracking matters more
than it looks: CAD text is frequently letter-spaced, and re-rendering at
the right size but natural spacing is what makes a label look subtly
wrong next to untouched linework.
"""

from __future__ import annotations

import math
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

# A condensed fallback for the case every regular candidate above
# shares: none of them is genuinely narrow, so a short, dense string
# (a 4-digit dimension is the common real case) can need MORE
# horizontal room than its own measured box allows even after
# solve_tracking and solve_horizontal_scale both hit their own
# evidence-based compression floors (see their docstrings — both
# floors exist to prevent glyphs visually merging into a different,
# misread character, a real regression this project already found and
# fixed once; loosening them to chase width is not on the table).
# When that happens, render/text.py's own shrink-to-fit loop still has
# to intervene, cutting px_size — and therefore the run's CAP HEIGHT —
# purely to recover width room that was never really a height problem.
# Confirmed on three real dimension numbers from an actual project
# file, each independently: Arial needed 118-154% of the numbers' own
# measured width even at both compression floors, which is what
# pushed their rendered height down to 89-96% of target.
#
# Bahnschrift ships as a standard font on every Windows 10/1709+
# install (same trust level this module already extends to Arial —
# see _FONT_CANDIDATES' own note: referenced by system path, NEVER
# bundled into this app's own distribution, since neither is a font
# this project holds redistribution rights to) and carries a genuine
# width AXIS as a variable font, with "Condensed" (75% width) as one
# of its own named instances — not a synthetic squeeze applied after
# the fact, an actual narrower cut of the same typeface. Measured
# against every classic AutoCAD/Revit shape-font TrueType conversion
# actually installed on this project's own machine too (isocp, romans,
# simplex, txt, monos — decorative title fonts, it turns out, wider
# than Arial for plain digits, not narrower): Bahnschrift Condensed
# was the only candidate that measured close to the real numbers' own
# width at full, uncompressed height — 100.5% on average, against
# 133-179% for every other font tried, Arial included.
#
# Deliberately still a per-run FALLBACK, kept only when it measurably
# beats the regular candidate for THAT run — not fit_style's default
# for every run, which was tried and reverted. Applied everywhere, it
# does fix height for every dimension number, exactly as the numbers
# above predict — and ALSO makes '870' undetectable by OCR outright on
# this project's own golden fixture, a run that never had a width
# problem to begin with. Condensed's lighter default weight and
# narrower strokes are exactly what makes it fit better, and exactly
# what an OCR model already working with a small, low-contrast run
# needs least. Losing a run outright is a strictly worse outcome than
# the height shortfall this fallback exists to fix, so it stays scoped
# to runs that actually need it. Absent on non-Windows systems;
# _resolve_condensed_font returns None there and fit_style falls back
# to today's behaviour unchanged.
_CONDENSED_FONT_CANDIDATES = ("C:/Windows/Fonts/bahnschrift.ttf",)
_CONDENSED_VARIATION = "Condensed"
_CONDENSED_VARIATION_BOLD = "Bold Condensed"

# Mirrors render/text.py's own OVERFLOW_TOLERANCE (1.04) — kept as a
# separate constant here, not imported, for the same reason
# lexicon/snap.py's _CONFUSABLE_CONFIDENCE_CEILING mirrors
# raster/pipeline.py's LOW_CONFIDENCE rather than importing it:
# render/text.py already imports FROM this module (TextStyle,
# font_measure_width), so the reverse import would be circular.
_WIDTH_OVERFLOW_TOLERANCE = 1.04

# ITU-R BT.601 luma weights, BGR order. The one formula every
# ink/paper or darkest-colour sampler in this project needs against a
# flat array of pixels — shared here (see bgr_luma) rather than
# hand-rolled per call site, which had drifted into two independent
# copies (this module's own ink/paper sampler, erase/clean.py's
# _darkest_colour) before this fix.
_BGR_LUMA_WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def bgr_luma(flat_pixels: np.ndarray) -> np.ndarray:
    """Perceptual luma of a flat ``(N, 3)`` array of BGR pixels."""
    return flat_pixels.astype(np.float32) @ _BGR_LUMA_WEIGHTS


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
    # Named instance to select on a VARIABLE font (e.g. "Condensed" on
    # Bahnschrift — see _CONDENSED_FONT_CANDIDATES) — None for every
    # ordinary, non-variable candidate in _FONT_CANDIDATES/
    # _BOLD_CANDIDATES, which is all of them except the condensed
    # fallback. A font swap that changes font_path (see
    # self_correct.py's _correct_text_fidelity, font-coverage path)
    # must clear this back to None: a newly-chosen font has no reason
    # to share a variation name with whatever this run was on before.
    font_variation: str | None = None


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


def _resolve_condensed_font() -> str | None:
    """The one condensed candidate, if this system has it — None,
    gracefully, everywhere else (non-Windows, or Windows older than
    1709). Never raises: unlike resolve_font's own regular candidates,
    this one is an optimisation fit_style can do without, not a
    requirement rendering depends on."""
    for path in _CONDENSED_FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


@lru_cache(maxsize=512)
def _font(path: str, size: int, variation: str | None = None) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(path, size)
    if variation is not None:
        font.set_variation_by_name(variation)
    return font


def measure_ink_and_paper(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    angle_deg: float = 0.0,
    expected_glyphs: int = 0,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Ink = median of the darkest decile, paper = median of the lightest.

    Deciles rather than min/max: an antialiased glyph edge produces a
    continuum, and the extremes are outliers. This also handles grey
    annotation text and tinted backgrounds with no special case.

    ``angle_deg``, when far enough from a cardinal (see
    :func:`_needs_rotated_measurement`), switches the sample source to
    a crop TIGHT around a straightened run's own found ink — see
    :func:`_sample_extreme_angle_ink_paper`'s own docstring for a real,
    confirmed bug this exact distinction (tight cluster crop, not the
    whole generously padded straightened square) fixes: a dimension
    number tilted ~45° had so little of its own (much larger, padded)
    straightened crop covered by ink that the darkest-decile cut never
    dipped below near-white, measuring ink colour as (255, 255, 255) —
    the SAME as paper. The text was being positioned and sized
    correctly; it was invisible, rendered in white ink on white paper.
    """
    if _needs_rotated_measurement(angle_deg):
        result = _sample_extreme_angle_ink_paper(image, bbox, angle_deg, expected_glyphs)
        if result is not None:
            return result

    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    h, w = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return ((0, 0, 0), (255, 255, 255))

    result = _sample_ink_and_paper(image[y0:y1, x0:x1])
    return result if result is not None else ((0, 0, 0), (255, 255, 255))


def _sample_ink_and_paper(
    crop: np.ndarray,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """The percentile sampling :func:`measure_ink_and_paper` does,
    factored out so both the plain crop and the straightened-crop path
    can share it. None if the crop has no usable dark/light split.
    """
    if crop.ndim == 2:
        crop = np.dstack([crop] * 3)
    flat = crop.reshape(-1, 3).astype(np.float32)
    luma = bgr_luma(flat)

    dark_cut = np.percentile(luma, 10)
    light_cut = np.percentile(luma, 90)
    dark = flat[luma <= dark_cut]
    light = flat[luma >= light_cut]
    if len(dark) == 0 or len(light) == 0:
        return None

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
    vertical = _reads_vertically(angle_deg)

    def _fallback() -> tuple[float, float]:
        # Use the ORIGINAL (unclamped) box, not the clamped one — a box
        # that lies partly or wholly outside the image would otherwise
        # collapse toward zero regardless of the box's real size, which
        # used to hand fit_font_size a bogus ~1px target and produce an
        # unreadably tiny font with no error raised.
        bw = max(1.0, float(raw_x1 - raw_x0))
        bh = max(1.0, float(raw_y1 - raw_y0))
        return (bh, bw) if vertical else (bw, bh)

    if _needs_rotated_measurement(angle_deg):
        extreme = _measure_extreme_angle(image, bbox, angle_deg, expected_glyphs)
        if extreme is not None:
            # Already correctly oriented -- measured directly along/across
            # the run's OWN reading direction in the straightened frame,
            # unlike the plain-crop path below, which measures in image
            # axes and needs `vertical` to know which axis is which.
            along, across, _cx, _cy = extreme
            return along, across

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

    For a run tilted far enough to need :func:`_measure_extreme_angle`
    (see that function and :func:`_needs_rotated_measurement`), the
    centre comes from there instead — mapped back through the SAME
    rotation used to straighten the crop, so it lands at the run's own
    true centre in absolute image coordinates either way.
    """
    if _needs_rotated_measurement(angle_deg):
        extreme = _measure_extreme_angle(image, bbox, angle_deg, expected_glyphs)
        if extreme is not None:
            _along, _across, cx, cy = extreme
            return cx, cy

    cluster = _measure_ink_cluster_bbox(image, bbox, angle_deg, other_boxes, expected_glyphs)
    if cluster is None:
        x0, y0, x1, y1 = bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    gx0, gy0, gx1, gy1 = cluster
    return ((gx0 + gx1) / 2.0, (gy0 + gy1) / 2.0)


# How far a run's angle must sit from every cardinal (0/90/-90/180)
# before an axis-aligned crop is abandoned in favour of a straightened
# one. Below this, the existing axis-aligned measurement already works
# well (every gentle wall-following tilt confirmed on a real plan
# measured 5-9° and fit its sibling labels' size correctly once the
# angle itself was resolved) — an axis-aligned box's excess padding
# over the true rotated footprint grows with the tilt and only becomes
# a real problem approaching 45°, not at a few degrees. Two real
# dimension numbers at ~47° and ~131° confirmed the failure mode this
# guards against: an axis-aligned crop at those angles is dominated by
# empty corner padding, and _measure_ink_cluster_bbox's own filters —
# built to reject a handful of stray marks, not a crop that is MOSTLY
# not the run — can't rescue a measurement that starts from a crop this
# inefficient. One of the two even measured ink colour as pure white
# (identical to paper): with under 5% of the crop being real ink, the
# darkest-decile percentile cut used to find ink colour never dipped
# below near-white, so the text was correctly sized... in a colour
# indistinguishable from the empty page around it.
_ROTATED_CROP_MARGIN_DEG = 20.0


def _needs_rotated_measurement(angle_deg: float) -> bool:
    return all(
        abs(angle_deg - cardinal) >= _ROTATED_CROP_MARGIN_DEG
        for cardinal in (-180.0, -90.0, 0.0, 90.0, 180.0)
    )


# Which cardinal (from the same five-cardinal set _needs_rotated_
# measurement already uses) a run's own reading direction is closest to
# determines whether it reads along the box's width or its height.
_CARDINAL_IS_VERTICAL = {0.0: False, 90.0: True, -90.0: True, 180.0: False, -180.0: False}


def _reads_vertically(angle_deg: float) -> bool:
    """True if ``angle_deg`` reads along the box's height, not its width.

    A plain ``abs(angle_deg) > 45`` test — used here until this fix —
    mislabels anything near 180 deg/-180 deg (a HORIZONTAL direction,
    same as 0 deg) as vertical: raw magnitude has no notion that 180
    and -180 are the same nearby direction, so upside-down horizontal
    text (e.g. 170 deg — well outside +-45 deg of 0, but genuinely
    horizontal) got its width and height swapped everywhere this
    decision is used, handing the font-size solver the box's WIDTH as
    its target cap height. Resolved the same way detect/rotations.py's
    own wraparound fix resolves it: find the nearest of the same five
    cardinals _needs_rotated_measurement already treats as equivalent,
    with a proper wrapped distance, not raw subtraction.
    """
    nearest = min(
        _CARDINAL_IS_VERTICAL,
        key=lambda c: abs(((angle_deg - c + 180.0) % 360.0) - 180.0),
    )
    return _CARDINAL_IS_VERTICAL[nearest]


def _straighten_crop(
    image: np.ndarray, bbox: tuple[float, float, float, float], angle_deg: float
) -> tuple[np.ndarray, float, float, np.ndarray] | None:
    """A generously padded crop around ``bbox``, rotated so the run
    reads left-to-right — straightened by the SAME rotation
    (``180 - angle_deg``, not the more obvious ``-angle_deg``) that
    :func:`spejl.render.text.render_run` implicitly undoes when it
    later rotates a freshly-drawn horizontal tile BY ``angle_deg`` to
    reproduce this same orientation. Confirmed empirically against two
    independent real runs (one already axis-aligned-measurable at ~95°,
    one only measurable through this function at ~131°) rather than
    derived from first principles and trusted — ``-angle_deg`` reliably
    produced upside-down, backwards text for both; ``180 - angle_deg``
    reliably produced correct, upright text for both, in both PIL's
    ``Image.rotate`` and cv2's ``getRotationMatrix2D`` (same sign
    convention in both libraries here, once the correct base formula
    was found).

    Returns ``(straightened_image, origin_x, origin_y, rotation_matrix)``:
    ``origin_x``/``origin_y`` is the padded crop's own top-left corner
    in ORIGINAL image coordinates, BEFORE rotation — together with
    ``rotation_matrix`` (the exact affine matrix applied), enough for
    :func:`_map_point_from_straightened` to send a point found in the
    straightened frame back to absolute image coordinates. None if
    ``bbox`` is degenerate or lies entirely outside the image.
    """
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return None
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = x1 - x0, y1 - y0
    ih, iw = image.shape[:2]
    # Half the box's own diagonal, plus a fixed margin: generous enough
    # that rotating a bw x bh box about its own centre never clips a
    # corner, at any angle. Capped against the SOURCE image's own
    # diagonal: a real detected run is always small relative to the
    # sheet it's on (a dimension number is tens of pixels on a
    # thousand-plus-pixel sheet), so this cap never engages for any
    # legitimate bbox — it exists only for a malformed or wrongly-merged
    # box (two labels fused into one detection, say), which would
    # otherwise allocate a canvas whose area scales with the diagonal
    # SQUARED, with no upper bound, for a single run.
    half = min(math.hypot(bw, bh) / 2.0 + 12.0, math.hypot(iw, ih))
    side = max(1, int(round(2 * half)))

    origin_x, origin_y = cx - half, cy - half
    sx0, sy0 = int(round(origin_x)), int(round(origin_y))
    cx0, cy0 = max(0, sx0), max(0, sy0)
    cx1, cy1 = min(iw, sx0 + side), min(ih, sy0 + side)
    if cx1 <= cx0 or cy1 <= cy0:
        return None

    extra_dims = image.shape[2:]
    canvas = np.full((side, side, *extra_dims), 255, dtype=image.dtype)
    canvas[cy0 - sy0 : cy1 - sy0, cx0 - sx0 : cx1 - sx0] = image[cy0:cy1, cx0:cx1]

    rotation = 180.0 - angle_deg
    matrix = cv2.getRotationMatrix2D((side / 2.0, side / 2.0), rotation, 1.0)
    border = 255 if not extra_dims else (255,) * extra_dims[0]
    straightened = cv2.warpAffine(
        canvas, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=border
    )
    return straightened, origin_x, origin_y, matrix


def _map_point_from_straightened(
    local_x: float, local_y: float, matrix: np.ndarray, origin_x: float, origin_y: float
) -> tuple[float, float]:
    """Send a point found in a straightened crop (from
    :func:`_straighten_crop`) back to absolute image coordinates."""
    inv = cv2.invertAffineTransform(matrix)
    ux = inv[0, 0] * local_x + inv[0, 1] * local_y + inv[0, 2]
    uy = inv[1, 0] * local_x + inv[1, 1] * local_y + inv[1, 2]
    return origin_x + ux, origin_y + uy


def _measure_extreme_angle(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    angle_deg: float,
    expected_glyphs: int,
) -> tuple[float, float, float, float] | None:
    """Measure a run tilted too far from any cardinal for the plain
    axis-aligned path (see :func:`_needs_rotated_measurement`) by
    straightening its crop first, instead of measuring in a crop
    dominated by empty corner padding.

    Returns ``(along_px, across_px, centre_x, centre_y)`` — the first
    two already correctly oriented along the run's own reading
    direction (no ``vertical`` swap needed, unlike the plain-crop
    path), the last two in absolute image coordinates. None if nothing
    usable was found, leaving the caller to fall back to the plain
    axis-aligned measurement.

    Reuses :func:`_measure_ink_cluster_bbox` — the SAME connected-
    component analysis and contamination filters every other run gets
    — by handing it the straightened crop as if it were an ordinary
    ``angle_deg=0`` detection spanning the whole crop; the straightening
    is what makes that crop tight around the real glyphs again, the
    same way it already is for a genuinely horizontal run.

    Neighbouring-box exclusion (``other_boxes`` elsewhere in this
    module) is deliberately not threaded through here: a dimension
    number tilted this far is reading along an isolated dimension
    line, not crowded against another label the way a room name can
    be, and transforming those boxes into the straightened crop's own
    rotated frame for a case that has not shown a need for it is
    complexity this fix does not need to carry yet.
    """
    result = _straightened_cluster(image, bbox, angle_deg, expected_glyphs)
    if result is None:
        return None
    _crop, (gx0, gy0, gx1, gy1), origin_x, origin_y, matrix = result
    along = max(1.0, gx1 - gx0)
    across = max(1.0, gy1 - gy0)
    local_cx, local_cy = (gx0 + gx1) / 2.0, (gy0 + gy1) / 2.0
    center_x, center_y = _map_point_from_straightened(local_cx, local_cy, matrix, origin_x, origin_y)
    return along, across, center_x, center_y


_ANGLE_REFINEMENT_SEARCH_DEG = 5.0
_ANGLE_REFINEMENT_STEP_DEG = 0.5
# The quad-derived angle feeding this whole rotated-crop path can itself
# be off by a couple of degrees for a short, steeply-angled run — the
# aspect-ratio floor (_MIN_ASPECT_FOR_QUAD_ANGLE, detect/rotations.py)
# only guards against a quad too SQUARE to trust; it says nothing about
# ordinary sub-degree noise in an elongated quad's own edges, and at
# these angles that noise translates into a real, visible size error.
# Confirmed on two separate real dimension numbers on two separate real
# files: straightening '1383' at its own detected 46.9° left its four
# digits on a baseline that drifts 2px top-to-bottom across the run —
# each digit individually still measures full height, but the drift
# alone inflates the UNION bounding box from a true ~25px to 27px.
# Straightening at 44.9° instead (a 2° correction) levels the baseline
# completely and drops the measurement back to 25px. The best angle is
# knowable directly from what's already being measured: the CORRECT
# straightening angle is whichever one packs this run's own glyphs into
# the tightest cross-baseline (across) extent, since baseline drift can
# only ever inflate that box, never shrink it. A small local search
# around the detected angle, keeping whichever candidate ties or beats
# every other on tightness (without losing real glyph content — a
# candidate that shrinks the ALONG extent by more than 10% dropped a
# character, not drift, and is rejected regardless of its own across
# measurement) finds it directly, with no separate calibration constant
# to get wrong. Confirmed on a second, independent real case ('1346',
# a different file): the same search found a corrected angle that
# additionally recovered a 4th digit component two fused digits had
# been hiding at the originally detected angle.
def _refine_extreme_angle(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    angle_deg: float,
    expected_glyphs: int,
) -> float:
    """The straightening angle, refined to minimise baseline-drift
    inflation — see the module-level comment above this function for
    the real data behind it. Falls back to ``angle_deg`` unchanged if
    nothing in the search range does better, or if there isn't enough
    ink to judge (a degenerate/empty crop) at all.
    """
    orig = _straighten_and_measure(image, bbox, angle_deg, expected_glyphs)
    if orig is None:
        return angle_deg
    orig_along, orig_across = orig
    if orig_along <= 0:
        return angle_deg

    best_angle, best_across = angle_deg, orig_across
    delta = -_ANGLE_REFINEMENT_SEARCH_DEG
    while delta <= _ANGLE_REFINEMENT_SEARCH_DEG + 1e-9:
        if delta != 0.0:
            candidate_angle = angle_deg + delta
            measured = _straighten_and_measure(image, bbox, candidate_angle, expected_glyphs)
            if measured is not None:
                cand_along, cand_across = measured
                if cand_along >= orig_along * 0.9 and cand_across < best_across:
                    best_angle, best_across = candidate_angle, cand_across
        delta += _ANGLE_REFINEMENT_STEP_DEG
    return best_angle


def _straighten_and_measure(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    angle_deg: float,
    expected_glyphs: int,
) -> tuple[float, float] | None:
    """Straighten once and report the measured cluster's own (along,
    across) extent — the raw building block :func:`_refine_extreme_angle`
    searches over. Deliberately NOT :func:`_straightened_cluster`
    itself, which calls this function's refinement first: that would
    recurse.
    """
    straightened = _straighten_crop(image, bbox, angle_deg)
    if straightened is None:
        return None
    crop, _origin_x, _origin_y, _matrix = straightened
    side = crop.shape[0]
    cluster = _measure_ink_cluster_bbox(
        crop, (0.0, 0.0, float(side), float(side)), 0.0, None, expected_glyphs
    )
    if cluster is None:
        return None
    x0, y0, x1, y1 = cluster
    return x1 - x0, y1 - y0


def _straightened_cluster(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    angle_deg: float,
    expected_glyphs: int,
) -> tuple[np.ndarray, tuple[float, float, float, float], float, float, np.ndarray] | None:
    """Straighten the crop, then run the SAME connected-component
    cluster analysis every other run gets — shared by
    :func:`_measure_extreme_angle` (needs the size and centre) and
    :func:`_sample_extreme_angle_ink_paper` (needs a crop TIGHT around
    just the real ink, not the whole generously padded straightened
    square — see that function's own docstring for why sampling ink
    colour from the padded square itself is a real, confirmed bug of
    its own, distinct from the sizing problem this whole file exists
    to fix).

    ``angle_deg`` is refined first (see :func:`_refine_extreme_angle`)
    to correct for a possibly-imprecise upstream angle estimate before
    ever cropping or measuring.

    Returns ``(straightened_crop, local_cluster_bbox, origin_x,
    origin_y, rotation_matrix)`` — ``local_cluster_bbox`` in the
    straightened crop's OWN local pixel coordinates, the rest as
    :func:`_straighten_crop` returns them, for mapping a point back to
    absolute image coordinates.
    """
    angle_deg = _refine_extreme_angle(image, bbox, angle_deg, expected_glyphs)
    straightened = _straighten_crop(image, bbox, angle_deg)
    if straightened is None:
        return None
    crop, origin_x, origin_y, matrix = straightened
    side = crop.shape[0]
    cluster = _measure_ink_cluster_bbox(
        crop, (0.0, 0.0, float(side), float(side)), 0.0, None, expected_glyphs
    )
    if cluster is None:
        return None
    return crop, cluster, origin_x, origin_y, matrix


def _sample_extreme_angle_ink_paper(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    angle_deg: float,
    expected_glyphs: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """Ink/paper colour for a run tilted too far for the plain crop
    (see :func:`_needs_rotated_measurement`), sampled from a crop TIGHT
    around the found ink cluster — not the whole straightened square
    :func:`_straighten_crop` returns.

    That distinction is load-bearing, confirmed as a second, separate
    real bug: the straightened square is deliberately padded generously
    (half the run's own diagonal, so rotating it never clips a corner
    at any angle) — appropriate for FINDING the ink, wrong for
    SAMPLING its colour. On a real plan, well under 10% of that padded
    square's pixels were ink; :func:`_sample_ink_and_paper`'s darkest-
    decile cut landed in the white background instead of the actual
    ink, measuring ink colour as pure white — identical to paper. The
    run was being sized and positioned correctly by then; it was
    rendering in a colour indistinguishable from the empty page around
    it. A small margin around the cluster's own tight bounding box
    keeps the crop this function samples from close to what a
    genuinely axis-aligned run's own (already-tight) detection box
    would give :func:`measure_ink_and_paper` directly.
    """
    result = _straightened_cluster(image, bbox, angle_deg, expected_glyphs)
    if result is None:
        return None
    crop, (gx0, gy0, gx1, gy1), _origin_x, _origin_y, _matrix = result
    ch, cw = crop.shape[:2]
    margin = 4
    lx0, ly0 = max(0, int(gx0) - margin), max(0, int(gy0) - margin)
    lx1, ly1 = min(cw, int(gx1) + margin), min(ch, int(gy1) + margin)
    if lx1 <= lx0 or ly1 <= ly0:
        return None
    return _sample_ink_and_paper(crop[ly0:ly1, lx0:lx1])


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

    The initial rule/dimension-line rejection (below) also catches a
    component that's SPARSE — not just the classically thin-and-long
    ``elongated`` shape — for a real case the original check missed:
    straightening a dimension line running through :func:`_measure_extreme_angle`'s
    crop turns it, arrowheads included, into a component that spans
    most of the crop's width but is no longer thin enough to read as
    "elongated" (the arrowhead triangles at each end make it locally
    tall) — confirmed on a real plan, where such a component (fill
    ratio 0.14) escaped the elongation check entirely and, with too few
    OTHER components surviving to satisfy the ``expected_glyphs`` gate
    the three filters below share, was never caught by any of them
    either, inflating a dimension number's measured size by nearly 3x.
    A genuine glyph never drops this low — every component measured
    across every real case this file's fixes are built on stayed above
    0.3 fill, even a fused wall-and-digit blob (0.51, real ink raises
    it) — so 0.2 stays a safe, unconditional floor un-gated by
    ``expected_glyphs``, the same as the elongation check beside it.

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
    vertical = _reads_vertically(angle_deg)

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
        sparse = (area / max(1, cwid * chgt)) < 0.2
        # `sparse` alone, with no span requirement, is enough on its own:
        # this file's own confirmed real data (see this function's
        # docstring, and _drop_sparse_linework's) is that a genuine glyph
        # never measures below ~0.3 fill even in a fused, contaminated
        # case — 0.2 already sits with margin below every real letter
        # this project's fixes are built on, so requiring it to ALSO span
        # most of the crop (spans_box) before trusting it only reopens a
        # gap the span-gated version already fixed one instance of: a
        # rule/dimension line spanning a more modest majority of the crop
        # (short of the 80% spans_box cutoff, not just short of the
        # elongation ratio) is still definitively not a glyph by the fill
        # signal alone. `elongated` keeps its own spans_box requirement —
        # unlike sparse, a thin shape with no span requirement at all
        # could legitimately be a narrow real glyph ('1', 'l').
        if sparse or (spans_box and elongated):
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
        boxes = _drop_foreign_strokes(boxes, vertical, expected_glyphs)
        boxes = _drop_sparse_linework(boxes, expected_glyphs)
        boxes = _drop_edge_touching_intrusions(boxes, ch, cw, vertical, expected_glyphs)
        if not boxes:
            return None

    boxes = _main_glyph_cluster(boxes, ch, cw, vertical, expected_glyphs)

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
    boxes: list[tuple[int, int, int, int, int]], vertical: bool, expected_glyphs: int = 0
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
    if expected_glyphs and len(kept) < expected_glyphs:
        # Dropping would leave fewer components than the text has
        # characters — the same "a real glyph might be the one being
        # discarded, with no way to recover it" reasoning the call
        # site's own outer gate already applies before calling this
        # function at all, extended to the case where the OUTLIER
        # itself turns out to be real (a genuinely wide glyph mixed
        # with narrower siblings, not a foreign stroke).
        return boxes
    return kept or boxes  # never discard every candidate outright


def _drop_sparse_linework(
    boxes: list[tuple[int, int, int, int, int]],
    expected_glyphs: int = 0,
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
    if expected_glyphs and len(kept) < expected_glyphs:
        return boxes  # see _drop_foreign_strokes's own equivalent guard
    return kept or boxes  # never discard every candidate outright


def _drop_edge_touching_intrusions(
    boxes: list[tuple[int, int, int, int, int]],
    crop_height: int,
    crop_width: int,
    vertical: bool,
    expected_glyphs: int = 0,
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

    def _clipped(b: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
        if vertical:
            return (max(b[0], interior_lo), b[1], min(b[2], interior_hi), b[3], b[4])
        return (b[0], max(b[1], interior_lo), b[2], min(b[3], interior_hi), b[4])

    kept = list(interior)
    small_edge_touchers: list[tuple[int, int, int, int, int]] = []
    for b in boxes:
        if not touches_cross_edge(b):
            continue
        if median_area > 0 and b[4] > median_area * 1.5:
            kept.append(_clipped(b))
        else:
            small_edge_touchers.append(b)  # pure linework or noise, presumed droppable

    if expected_glyphs and len(kept) < expected_glyphs:
        # Not enough survived without them — rather than lose a real,
        # small glyph fused with edge-touching linework entirely (with
        # no way to recover it), clip it the same way a large intrusion
        # is already clipped above, instead of discarding it outright.
        kept.extend(_clipped(b) for b in small_edge_touchers)

    return kept or boxes  # never discard every candidate outright


def _main_glyph_cluster(
    boxes: list[tuple[int, int, int, int, int]],
    crop_height: int,
    crop_width: int,
    vertical: bool,
    expected_glyphs: int = 0,
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
    if expected_glyphs and len(remaining) + 1 < expected_glyphs:
        # The 8% floor left fewer CANDIDATES than the text has
        # characters — the same risk _core_candidates' own docstring
        # already documents for the sibling filters above (a real but
        # small letter part, like an 'i' dot, could fall under a fixed
        # fraction of a much larger anchor in the same run): admit every
        # component as a growth candidate rather than let a real one be
        # permanently excluded from ever joining the cluster. This can
        # only ADD candidates the region-growing pass below still has to
        # earn its way in by proximity — never force one to join.
        remaining = list(ordered)
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
    text: str, target_cap_height: float, font_path: str, lo: int = 4, hi: int = 400,
    font_variation: str | None = None,
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
        bbox = _font(font_path, mid, font_variation).getbbox(probe_text)
        height = (bbox[3] - bbox[1]) if bbox else 0
        err = abs(height - target)
        if err < best_err:
            best, best_err = mid, err
        if height < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return max(1, best)


def solve_tracking(
    text: str, font_path: str, px_size: int, target_width: float,
    font_variation: str | None = None,
) -> float:
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
    natural = font_measure_width(text, font_path, px_size, 0.0, font_variation)
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


def font_measure_width(
    text: str, font_path: str, px_size: int, tracking_px: float,
    font_variation: str | None = None,
) -> float:
    """Advance width of ``text`` at ``tracking_px`` **absolute pixels**
    of extra per-gap spacing — the renderer's own measurement, so fitting
    and drawing can never disagree. (Absolute pixels here, deliberately
    different from :class:`TextStyle`'s em-relative ``tracking`` — see
    :func:`solve_tracking`; every caller of this function either passes
    ``0.0``, where the unit is moot, or converts explicitly.)"""
    font = _font(font_path, px_size, font_variation)
    if not text:
        return 0.0
    width = sum(font.getlength(ch) for ch in text)
    return float(width + tracking_px * (len(text) - 1))


def _fit_dimensions(
    text: str, along: float, across: float, font_path: str, font_variation: str | None = None,
) -> tuple[int, float, float, float]:
    """px_size/tracking/width_scale for ``font_path`` against one run's
    own measured (along, across) — plus how much of ``along`` the
    result still overflows by. fit_style itself only ever needs the
    first three (a single font decided up front, not compared against
    an alternative per run — see its own comment); the overflow figure
    is what let this module's OWN candidate search compare fonts
    against each other in the first place, and what its test suite
    still checks against real measured cases."""
    px_size = fit_font_size(text, across, font_path, font_variation=font_variation)
    tracking = solve_tracking(text, font_path, px_size, along, font_variation)
    natural_tracked = font_measure_width(text, font_path, px_size, tracking * px_size, font_variation)
    width_scale = solve_horizontal_scale(natural_tracked, along)
    overflow = (natural_tracked * width_scale) / max(1.0, along)
    return px_size, tracking, width_scale, overflow


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
    ink, paper = measure_ink_and_paper(image, bbox, angle_deg, expected_glyphs=len(text))
    along, across = measure_ink_extent(
        image, bbox, angle_deg, other_boxes=other_boxes, expected_glyphs=len(text)
    )

    px_size, tracking, width_scale, overflow = _fit_dimensions(text, along, across, font_path)
    font_variation = None

    # Condensed was tried as the PRIMARY font for every run, not just
    # this fallback — reverted, confirmed wrong by the same standard
    # everything else in this module is held to: real evidence, not
    # reasoning alone. Measured against the same real corpus this
    # whole feature was built from, universal Condensed genuinely does
    # fix height for every dimension number (matches the field
    # comparison in this constant's own docstring) — and ALSO makes
    # '870' undetectable by OCR outright on the project's own golden
    # fixture, a run this module never touched before and had no
    # problem to begin with. Losing a run entirely is a strictly worse
    # outcome than the 4-11% height shortfall this whole feature exists
    # to fix, so "every run, always" is off the table — Condensed's
    # lighter default weight and narrower strokes are exactly what
    # makes it fit better, and exactly what an OCR model already
    # struggling with a small run's low contrast needs least.
    #
    # Kept as the narrow, evidence-scoped fix it started as: tried only
    # when the regular candidate genuinely doesn't fit even after
    # solve_tracking and solve_horizontal_scale both hit their own
    # compression floors (see their docstrings — both exist to prevent
    # glyphs visually merging into a different, misread character, a
    # real regression this project already found and fixed once), and
    # kept only if it's a measured improvement for THAT run — never
    # applied to a run with no width problem to solve in the first
    # place, which is exactly the class '870' turned out to belong to.
    if overflow > _WIDTH_OVERFLOW_TOLERANCE:
        condensed_path = _resolve_condensed_font()
        if condensed_path is not None:
            c_variation = _CONDENSED_VARIATION_BOLD if bold else _CONDENSED_VARIATION
            c_px_size, c_tracking, c_width_scale, c_overflow = _fit_dimensions(
                text, along, across, condensed_path, c_variation
            )
            if c_overflow < overflow:
                font_path, px_size, tracking, width_scale = (
                    condensed_path, c_px_size, c_tracking, c_width_scale
                )
                font_variation = c_variation

    return TextStyle(
        font_path=font_path,
        px_size=px_size,
        tracking=tracking,
        ink=ink,
        paper=paper,
        cap_height_px=across,
        ink_along_px=along,
        width_scale=width_scale,
        font_variation=font_variation,
    )
