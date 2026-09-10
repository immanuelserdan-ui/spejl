"""Stage S5 — erase the type, and repair the linework it was sitting on.

General inpainting is the wrong default for a floor plan. The background
is flat white, so a background-colour fill is cleaner *and* faster than
Telea. The hard part is not the fill: it is that dimension lines,
witness lines and wall edges run under and beside labels, and a naive
mask-and-fill leaves visible gaps in them once the sheet is mirrored.

So the order matters: measure the lines **before** erasing, erase, then
redraw any line whose path crossed an erased region.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from spejl.models import Flag


@dataclass(frozen=True)
class EraseResult:
    image: np.ndarray
    mask: np.ndarray
    repaired_px: int
    flags: tuple[Flag, ...] = ()


# How far a pixel's luma may sit below local paper and still count as
# paper. Antialiased glyph fringes run the whole way from ink to paper,
# so a global "is it dark?" threshold leaves the light half of every
# fringe behind as a grey halo — visible as ghost outlines on the
# mirrored sheet. Measuring against *local* paper instead catches the
# whole fringe, and any linework caught with it is restored by
# :func:`line_pixel_mask` afterwards.
PAPER_TOLERANCE = 14


def _luma(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.int16)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16)


def build_text_mask(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    dilate_px: int = 3,
) -> np.ndarray:
    """Mask = dilated text boxes ∩ everything that isn't local paper.

    Per-box rather than global: a plan may be tinted, and a label may sit
    on a lighter or darker patch than the sheet average. Intersecting
    with the box (rather than filling the rectangle outright) keeps the
    erase from wiping a wall edge that merely passes through a corner —
    and what it does catch of the linework, line repair puts back.
    """
    gray = _luma(image)
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for x0, y0, x1, y1 in boxes:
        bx0 = max(0, int(np.floor(x0)) - dilate_px)
        by0 = max(0, int(np.floor(y0)) - dilate_px)
        bx1 = min(w, int(np.ceil(x1)) + dilate_px)
        by1 = min(h, int(np.ceil(y1)) + dilate_px)
        if bx1 <= bx0 or by1 <= by0:
            continue
        patch = gray[by0:by1, bx0:bx1]
        # Local paper = the light end of this patch, not its mean: a box
        # containing mostly ink would otherwise set a paper level so low
        # that nothing gets erased.
        paper = float(np.percentile(patch, 90))
        mask[by0:by1, bx0:bx1] |= (patch < paper - PAPER_TOLERANCE).astype(np.uint8) * 255

    return mask


def _ring_modal_colour(
    image: np.ndarray, box: tuple[int, int, int, int], ring_px: int = 6
) -> np.ndarray:
    """The dominant colour immediately around a box — the local paper."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    ox0, oy0 = max(0, x0 - ring_px), max(0, y0 - ring_px)
    ox1, oy1 = min(w, x1 + ring_px), min(h, y1 + ring_px)
    outer = image[oy0:oy1, ox0:ox1]
    if outer.size == 0:
        return np.array([255, 255, 255], dtype=np.uint8)

    ring = np.ones(outer.shape[:2], dtype=bool)
    ix0, iy0 = x0 - ox0, y0 - oy0
    ix1, iy1 = ix0 + (x1 - x0), iy0 + (y1 - y0)
    ring[max(0, iy0):max(0, iy1), max(0, ix0):max(0, ix1)] = False
    pixels = outer[ring]
    if pixels.size == 0:
        return np.array([255, 255, 255], dtype=np.uint8)

    # Modal, not mean: a ring that clips a black wall would otherwise
    # produce grey paper.
    if pixels.ndim == 1:
        pixels = pixels.reshape(-1, 1)
    quantised = (pixels // 8 * 8).astype(np.uint8)
    colours, counts = np.unique(quantised, axis=0, return_counts=True)
    return colours[int(np.argmax(counts))]


def line_pixel_mask(image: np.ndarray, min_length: int = 40) -> np.ndarray:
    """Pixels belonging to long axis-aligned runs — walls, dimension and
    witness lines — as a mask, not as fitted primitives.

    Morphological opening with long thin kernels keeps exactly the
    structural runs a plan is made of and drops glyph strokes, which are
    short in both axes.

    Deliberately *not* reduced to line segments. Fitting one primitive
    per connected component looks tidier and is wrong: the black wall
    between the ``1680`` and ``870`` dimension lines joins those two
    collinear runs into a single component that is both tall and wide, so
    its bounding-box centre and height describe a thick stroke through
    the middle of nothing — which then gets painted straight through the
    text. The mask needs no fitting and cannot invent geometry.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    out = np.zeros(binary.shape, dtype=np.uint8)
    for kernel_size in ((min_length, 1), (1, min_length)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        out = cv2.bitwise_or(out, cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel))
    return out


def erase_text(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    dilate_px: int = 3,
    repair_lines: bool = True,
) -> EraseResult:
    """Remove every text run and restore the geometry underneath."""
    work = image.copy()
    if work.ndim == 2:
        work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)

    # Measure the linework BEFORE erasing anything — afterwards the
    # evidence is gone.
    lines_before = line_pixel_mask(work) if repair_lines else None
    mask = build_text_mask(work, boxes, dilate_px=dilate_px)
    flags: list[Flag] = []

    # Fill each box's masked pixels with its own local paper colour, so a
    # label on a tinted plan or over a light hatch still disappears.
    for x0, y0, x1, y1 in boxes:
        box = (
            max(0, int(np.floor(x0)) - dilate_px),
            max(0, int(np.floor(y0)) - dilate_px),
            min(work.shape[1], int(np.ceil(x1)) + dilate_px),
            min(work.shape[0], int(np.ceil(y1)) + dilate_px),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        colour = _ring_modal_colour(work, box)
        sub_mask = mask[box[1]:box[3], box[0]:box[2]]
        region = work[box[1]:box[3], box[0]:box[2]]
        region[sub_mask > 0] = colour

        if _looks_patterned(work, box):
            flags.append(
                Flag(
                    code="erase-over-pattern",
                    message=f"Text at {box} sat on a patterned or shaded "
                    "background; flat fill may be visible.",
                    severity="warn",
                )
            )

    repaired = 0
    if lines_before is not None:
        repaired = _restore_line_pixels(work, lines_before, mask)

    return EraseResult(image=work, mask=mask, repaired_px=repaired, flags=tuple(flags))


def _looks_patterned(image: np.ndarray, box: tuple[int, int, int, int], ring_px: int = 6) -> bool:
    """Is the surround varied enough that a flat fill will show?"""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    ox0, oy0 = max(0, x0 - ring_px), max(0, y0 - ring_px)
    ox1, oy1 = min(w, x1 + ring_px), min(h, y1 + ring_px)
    outer = image[oy0:oy1, ox0:ox1]
    if outer.size == 0:
        return False
    gray = cv2.cvtColor(outer, cv2.COLOR_BGR2GRAY) if outer.ndim == 3 else outer
    light = gray[gray > 128]
    return bool(light.size > 0 and light.std() > 18)


def _restore_line_pixels(
    image: np.ndarray, lines_before: np.ndarray, mask: np.ndarray
) -> int:
    """Repaint exactly the long-line pixels that the erase removed.

    Only the intersection of (was linework) and (was erased) is touched,
    so a repair can never thicken, shift, or invent linework — it can
    only put back what was demonstrably there a moment ago.
    """
    restore = cv2.bitwise_and(lines_before, mask)
    count = int(np.count_nonzero(restore))
    if count:
        image[restore > 0] = _darkest_colour(image)
    return count


def _darkest_colour(image: np.ndarray) -> np.ndarray:
    """The sheet's ink colour, taken globally — linework on a plan is one
    colour, and sampling globally avoids inheriting a local artefact."""
    flat = image.reshape(-1, 3).astype(np.float32)
    luma = flat @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
    dark = flat[luma <= np.percentile(luma, 1)]
    if dark.size == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    return np.median(dark, axis=0).astype(np.uint8)
