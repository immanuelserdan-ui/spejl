"""The mirror itself — geometry for both routes.

This is the one module both routes share, and the reason the tool works:
reflect an *anchor*, reflect a baseline *direction*, then pick the
traversal of that reflected line which still reads the way a drafter
expects. Glyphs are never reflected; they are re-emitted.

Angle convention throughout Spejl (screen space, x right, y **down**,
matching image and PDF-page coordinates alike):

    0°   text reads left-to-right      direction ( 1,  0)
    90°  text reads bottom-to-top      direction ( 0, -1)   ISO vertical
    180° text reads right-to-left      direction (-1,  0)   never emitted
    -90° text reads top-to-bottom      direction ( 0,  1)

so ``direction = (cos θ, -sin θ)`` and ``θ = -atan2(dy, dx)``.
"""

from __future__ import annotations

import math

import numpy as np

from spejl.models import Axis

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Points and boxes
# ---------------------------------------------------------------------------


def mirror_point(
    x: float, y: float, width: float, height: float, axis: Axis
) -> tuple[float, float]:
    """Reflect a single point about the sheet's centre line.

    Note ``width - x``, not ``width - 1 - x``: callers work in continuous
    coordinates (PDF points, sub-pixel OCR quads). The pixel-grid version
    belongs to :func:`flip_image`, which delegates to OpenCV.
    """
    if axis is Axis.VERTICAL:
        return (width - x, y)
    if axis is Axis.HORIZONTAL:
        return (x, height - y)
    return (width - x, height - y)  # BOTH — point reflection


def mirror_bbox(
    bbox: tuple[float, float, float, float], width: float, height: float, axis: Axis
) -> tuple[float, float, float, float]:
    """Reflect an axis-aligned box, re-normalising the corners.

    Reflection swaps which corner is minimal, so the result must be
    rebuilt from min/max rather than mapped corner-for-corner — mapping
    in place yields an inverted box that silently reads as empty.
    """
    x0, y0, x1, y1 = bbox
    ax, ay = mirror_point(x0, y0, width, height, axis)
    bx, by = mirror_point(x1, y1, width, height, axis)
    return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


# ---------------------------------------------------------------------------
# Baseline direction and angle
# ---------------------------------------------------------------------------


def direction_from_angle(angle_deg: float) -> tuple[float, float]:
    r = math.radians(angle_deg)
    return (math.cos(r), -math.sin(r))


def angle_from_direction(dx: float, dy: float) -> float:
    return -math.degrees(math.atan2(dy, dx))


def is_canonical_direction(dx: float, dy: float) -> bool:
    """True if a baseline pointing (dx, dy) reads the way a drafter expects:
    left-to-right when it has any horizontal component, bottom-to-top when
    it is purely vertical (ISO dimension-text convention).

    This is a fixed convention, deliberately independent of which axis is
    being mirrored: a vertical dimension always ends up reading bottom-to-
    top, whether the mirror was left/right or top/bottom. The alternative
    — letting a top/bottom mirror flip vertical text upside down — would
    make output depend on an axis choice in a way no drafting standard
    does. Only the *position* of a run should move.
    """
    if dx > _EPS:
        return True
    if dx < -_EPS:
        return False
    return dy < _EPS  # dx ~ 0: canonical iff bottom-to-top (or degenerate)


def mirror_direction(dx: float, dy: float, axis: Axis) -> tuple[float, float]:
    """Reflect a baseline direction, then choose the readable traversal.

    Safe for diagonals as well as cardinals: negating a direction vector
    names the *same infinite line* (identical slope), so flipping
    traversal to restore left-to-right reading never changes which line
    the text sits on — only which end it starts from.
    """
    if axis is Axis.VERTICAL:
        dxf, dyf = -dx, dy
    elif axis is Axis.HORIZONTAL:
        dxf, dyf = dx, -dy
    else:
        dxf, dyf = -dx, -dy
    if not is_canonical_direction(dxf, dyf):
        dxf, dyf = -dxf, -dyf
    return dxf, dyf


def mirror_angle(angle_deg: float, axis: Axis) -> float:
    """Angle-space equivalent of :func:`mirror_direction`.

    A purely vertical run is invariant under a vertical-axis mirror
    (its direction has no x-component to flip), which is why ``5155``
    keeps reading bottom-to-top after crossing to the other side of the
    sheet — the worked example in the build plan's Figure 2.
    """
    dx, dy = direction_from_angle(angle_deg)
    dxf, dyf = mirror_direction(dx, dy, axis)
    mirrored = angle_from_direction(dxf, dyf)
    # Snap microscopic float error to exact cardinals; renderers and
    # tests both compare these, and 89.99999° is noise, not information.
    for cardinal in (-180.0, -90.0, 0.0, 90.0, 180.0):
        if abs(mirrored - cardinal) < 1e-6:
            return cardinal
    return mirrored


# ---------------------------------------------------------------------------
# Raster
# ---------------------------------------------------------------------------


def flip_image(image: np.ndarray, axis: Axis) -> np.ndarray:
    """Reflect the pixels. The cheap step — and the one that must be
    provably lossless, which ``test_geometry_integrity`` checks by
    comparing non-text regions against this exact call."""
    import cv2

    if axis is Axis.VERTICAL:
        return cv2.flip(image, 1)
    if axis is Axis.HORIZONTAL:
        return cv2.flip(image, 0)
    return cv2.flip(image, -1)
