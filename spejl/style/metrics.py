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
    image: np.ndarray, bbox: tuple[float, float, float, float], angle_deg: float
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
    """
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    h, w = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    vertical = abs(angle_deg) > 45

    def _fallback() -> tuple[float, float]:
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
        return (bh, bw) if vertical else (bw, bh)

    if x1 <= x0 or y1 <= y0:
        return _fallback()

    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    paper = float(np.percentile(gray, 90))
    ink = (gray < paper - _PAPER_TOLERANCE).astype(np.uint8)
    if not ink.any():
        return _fallback()

    ch, cw = ink.shape
    count, _labels, stats, _cent = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, count):
        cx, cy, cwid, chgt, area = stats[i]
        if area < 2:
            continue  # single-pixel speckle
        spans_box = cwid > cw * 0.8 or chgt > ch * 0.8
        elongated = cwid > chgt * 8 or chgt > cwid * 8
        if spans_box and elongated:
            continue  # a rule or dimension line, not a glyph
        boxes.append((cx, cy, cx + cwid, cy + chgt))

    if not boxes:
        return _fallback()

    gx0 = min(b[0] for b in boxes)
    gy0 = min(b[1] for b in boxes)
    gx1 = max(b[2] for b in boxes)
    gy1 = max(b[3] for b in boxes)
    ink_w = max(1.0, float(gx1 - gx0))
    ink_h = max(1.0, float(gy1 - gy0))
    return (ink_h, ink_w) if vertical else (ink_w, ink_h)


def fit_font_size(
    text: str, target_cap_height: float, font_path: str, lo: int = 4, hi: int = 400
) -> int:
    """Binary-search the point size whose rendered cap height matches.

    Measures the actual ink box of the string, not the font's nominal
    metrics, because cap height varies with which glyphs are present
    ('870' has no descender; 'Køkken' does).
    """
    target = max(1.0, target_cap_height)
    best, best_err = lo, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        bbox = _font(font_path, mid).getbbox(text or "0")
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
    # Tightened from a naive -12%: digits have tight side bearings, and
    # tracking past roughly -8% of the em starts to visibly touch.
    return float(np.clip(tracking_em, -0.08, 0.5))


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
) -> TextStyle:
    """Measure every number for one run — none are assumed."""
    font_path = resolve_font(bold=bold)
    ink, paper = measure_ink_and_paper(image, bbox)
    along, across = measure_ink_extent(image, bbox, angle_deg)

    px_size = fit_font_size(text, across, font_path)
    tracking = solve_tracking(text, font_path, px_size, along)

    return TextStyle(
        font_path=font_path,
        px_size=px_size,
        tracking=tracking,
        ink=ink,
        paper=paper,
        cap_height_px=across,
        ink_along_px=along,
    )
