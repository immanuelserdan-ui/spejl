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
from spejl.style.metrics import bgr_luma


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


_RING_PX = 6  # matches _ring_modal_colour's own default — see build_text_mask


def build_text_mask(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    dilate_px: int = 3,
) -> np.ndarray:
    """Mask = dilated text boxes ∩ everything that differs from local paper.

    Per-box rather than global: a plan may be tinted, and a label may sit
    on a lighter or darker patch than the sheet average. Intersecting
    with the box (rather than filling the rectangle outright) keeps the
    erase from wiping a wall edge that merely passes through a corner —
    and what it does catch of the linework, line repair puts back.

    "Local paper" is read from the ring immediately OUTSIDE each box,
    not the box's own interior. That is a deliberate change from reading
    the interior's own light end: the interior is exactly the one region
    guaranteed to contain the glyph, so a box that happens to be MOSTLY
    ink, or — the case this fixes — reverse-out text lighter than its
    own background (white room-name lettering knocked out of a filled
    panel), makes the interior's own brightness distribution describe
    the glyph, not the paper. A reverse-out box's light end is the glyph
    itself, so the old "darker than the light end" test never fired: the
    panel got erased as "not paper" and the white lettering was judged
    paper and left standing, confirmed surviving whole and unmirrored on
    a synthetic reverse-out label. The ring just outside the box has no
    such ambiguity — it is real surrounding material almost by
    definition, whichever kind of material that is.

    Every OTHER box's pixels are excluded from the ring sample. Real
    plans crowd runs a few pixels apart (a room's number beside its
    door-swing radius label, e.g. 'Bad' next to '1400') — close enough
    that a 6px ring reaches a neighbour. Without this exclusion, a
    neighbour's dark ink corrupted *this* box's paper reading, dragging
    it toward grey until the neighbour's own light antialiased fringe
    read as paper and survived the fill — a visible grey ghost of both
    labels, confirmed on this exact 'Bad'/'1400' cluster in the golden
    fixture.

    Masked by absolute deviation from that reference, not "darker than
    paper": ink lighter than its surround needs exactly the same erase
    as ink darker than it, so one symmetric test replaces what would
    otherwise be two directions to get right — and get wrong.
    """
    gray = _luma(image)
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    rects: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in boxes:
        rects.append(
            (
                max(0, int(np.floor(x0)) - dilate_px),
                max(0, int(np.floor(y0)) - dilate_px),
                min(w, int(np.ceil(x1)) + dilate_px),
                min(h, int(np.ceil(y1)) + dilate_px),
            )
        )

    for i, (bx0, by0, bx1, by1) in enumerate(rects):
        if bx1 <= bx0 or by1 <= by0:
            continue

        rx0, ry0 = max(0, bx0 - _RING_PX), max(0, by0 - _RING_PX)
        rx1, ry1 = min(w, bx1 + _RING_PX), min(h, by1 + _RING_PX)
        ring = gray[ry0:ry1, rx0:rx1]

        # This box's own interior, plus every OTHER box's footprint that
        # reaches into the ring band, excluded from the sample — neither
        # is real surrounding material.
        exclude = np.zeros(ring.shape, dtype=bool)
        ix0, iy0 = bx0 - rx0, by0 - ry0
        ix1, iy1 = ix0 + (bx1 - bx0), iy0 + (by1 - by0)
        exclude[max(0, iy0):max(0, iy1), max(0, ix0):max(0, ix1)] = True
        for j, (ox0, oy0, ox1, oy1) in enumerate(rects):
            if j == i:
                continue
            jx0, jy0 = max(ox0, rx0), max(oy0, ry0)
            jx1, jy1 = min(ox1, rx1), min(oy1, ry1)
            if jx1 > jx0 and jy1 > jy0:
                exclude[jy0 - ry0 : jy1 - ry0, jx0 - rx0 : jx1 - rx0] = True

        sample = ring[~exclude] if exclude.any() else ring.reshape(-1)
        if sample.size == 0:  # ring fully claimed — fall back to the raw band
            sample = ring.reshape(-1)
        if sample.size == 0:  # box touches the image edge on every side
            continue

        # Modal, not mean or a percentile: a ring that clips a black
        # wall would otherwise read as grey paper on one box and true
        # paper on its neighbour a few pixels along — see
        # _ring_modal_colour's own docstring for the confirmed instance
        # of exactly this failure.
        quantised = (sample // 8) * 8
        values, counts = np.unique(quantised, return_counts=True)
        winning = values[int(np.argmax(counts))]
        paper = float(sample[quantised == winning].mean())

        patch = gray[by0:by1, bx0:bx1]
        mask[by0:by1, bx0:bx1] |= (
            np.abs(patch.astype(np.float32) - paper) > PAPER_TOLERANCE
        ).astype(np.uint8) * 255

    return mask


def _ring_pixels(
    image: np.ndarray, box: tuple[int, int, int, int], ring_px: int
) -> np.ndarray:
    """Pixels in a ``ring_px``-wide band around ``box``, box interior
    excluded — the shared "just the surround" sampling both
    :func:`_ring_modal_colour` and :func:`_looks_patterned` need.
    """
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    ox0, oy0 = max(0, x0 - ring_px), max(0, y0 - ring_px)
    ox1, oy1 = min(w, x1 + ring_px), min(h, y1 + ring_px)
    outer = image[oy0:oy1, ox0:ox1]
    if outer.size == 0:
        return outer

    ring = np.ones(outer.shape[:2], dtype=bool)
    ix0, iy0 = x0 - ox0, y0 - oy0
    ix1, iy1 = ix0 + (x1 - x0), iy0 + (y1 - y0)
    ring[max(0, iy0):max(0, iy1), max(0, ix0):max(0, ix1)] = False
    return outer[ring]


def _ring_modal_colour(
    image: np.ndarray, box: tuple[int, int, int, int], ring_px: int = 6
) -> np.ndarray:
    """The dominant colour immediately around a box — the local paper."""
    pixels = _ring_pixels(image, box, ring_px)
    if pixels.size == 0:
        return np.array([255, 255, 255], dtype=np.uint8)

    # Modal, not mean: a ring that clips a black wall would otherwise
    # produce grey paper. But the modal *bucket* found by quantising is
    # not itself a colour to fill with — `255 // 8 * 8 == 248`, so a
    # ring that is 100% pure white used to return (248,248,248) as
    # "paper", every single time (the floor of the bucket, never the
    # actual value). That is not a rare edge case: it silently
    # darkened EVERY erase fill on EVERY run by the same ~3%, visible at
    # zoom as a faint ghost with the exact silhouette of whatever was
    # erased — confirmed on the 'Bad'/'1400' cluster in the golden
    # fixture. The bucket only identifies *which* pixels are the
    # majority; the colour returned must be their own true average.
    if pixels.ndim == 1:
        pixels = pixels.reshape(-1, 1)
    quantised = (pixels // 8 * 8).astype(np.uint8)
    colours, counts = np.unique(quantised, axis=0, return_counts=True)
    winning_bucket = colours[int(np.argmax(counts))]
    in_bucket = np.all(quantised == winning_bucket, axis=1)
    true_colour = pixels[in_bucket].astype(np.float32).mean(axis=0)
    return np.round(true_colour).astype(np.uint8)


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

        # Checked against the ORIGINAL `image`, not the progressively-
        # mutated `work`: this loop fills one box per iteration, so by
        # the time a LATER box is checked, `work` already carries every
        # EARLIER box's fresh flat fill — and real plans crowd runs only
        # a few pixels apart (a room name beside its own area figure),
        # close enough that one box's 6px ring can dip into a
        # neighbour's box that was just filled a moment ago. Sampling
        # that fill instead of the neighbour's true original texture
        # dilutes the measured variance below the pattern threshold,
        # silently suppressing this warning for exactly the crowded-
        # label case it exists to catch.
        if _looks_patterned(image, box):
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
        repaired = _restore_line_pixels(work, image, lines_before, mask, boxes, dilate_px)

    return EraseResult(image=work, mask=mask, repaired_px=repaired, flags=tuple(flags))


def _looks_patterned(image: np.ndarray, box: tuple[int, int, int, int], ring_px: int = 6) -> bool:
    """Is the surround varied enough that a flat fill will show?

    Deliberately reuses :func:`_ring_pixels` rather than sampling
    ``box``'s own rectangle plus its ring: this runs *after* the box
    interior has already been flat-filled (see ``erase_text``), and for
    any normal-width text run that flat interior dwarfs the thin 6px
    ring of real surrounding texture — which drags the measured
    variance toward zero regardless of how patterned the surround
    actually is, and the warning this function exists to raise never
    fires. Sampling only the ring is what the docstring ("the surround")
    already promised.
    """
    pixels = _ring_pixels(image, box, ring_px)
    if pixels.size == 0:
        return False
    gray = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY) if pixels.ndim == 2 else pixels
    light = gray[gray > 128]
    return bool(light.size > 0 and light.std() > 18)


# A genuine wall or dimension line that happens to run under a label
# continues into the surrounding sheet well past that one label — real
# cases confirmed on 722-0553-0006-1016-T22-S extend 55 to 1428px beyond
# the text box whose erasure exposed them (the outer wall outline, a
# witness line under a dimension number). But a room label's OWN letter
# strokes can independently satisfy line_pixel_mask's >=40px straight-run
# detector once the sheet's own cap height clears it — confirmed on the
# same file: 'Køkken' (cap height 54px) produced three separate 4-5px-
# wide, 40px-tall components, one per straight vertical in 'K', 'k', 'k';
# 'Bad' produced two, for 'B' and 'd'. Every one of them sits ENTIRELY
# inside its own text box (worst case still 9px short of even reaching
# the box edge) — before this fix, _restore_line_pixels painted them
# straight back onto the erased canvas regardless, a ghost stroke under
# every re-rendered glyph tall enough to trigger it ('Køkken' rendered as
# a smeared 'Kølkken!', 'Bad' as 'Badl'). The two cases are cleanly
# separable on this one signal — a real line already extends hundreds of
# pixels past any single label; a letter stroke, by construction, never
# leaves its own glyph's box at all — so 20px sits with wide margin on
# both sides of the real data (29px above the worst false positive's own
# containment, 35px below the smallest genuine extension observed).
_MIN_LINE_EXTENSION_PX = 20.0

# A disk-opening pre-filter (drop any locally-thick material before the
# extension test below, so a filled room panel or a reverse-out label's
# own dark background can't stand in as "genuine linework" the way a
# large fill otherwise can — see build_text_mask's own docstring for the
# reverse-out case this was chasing) was tried here and reverted. It
# worked on the golden fixture in isolation — a kernel size existed
# (31px) that reproduced the unfiltered flank recovery exactly while
# fully closing a synthetic panel's ghost-outline leak — but that window
# was only 31–33px wide, and re-mirroring an already-mirrored sheet
# redraws every line with very slightly different antialiasing, which
# was enough to push a real wall segment near '1680'/'870' below the
# same threshold on the second pass: `test_mirroring_twice_returns_
# close_to_the_source` started failing deterministically, with a
# visible notch cut into both dimension lines that was not there
# without the filter. A fix whose safety margin doesn't survive the
# pipeline's own round trip is worse than the narrower cosmetic issue
# (a thin antialiased outline surviving around reverse-out text — see
# build_text_mask) it was trying to close. Left as a known, documented
# residual rather than shipped.


def _restore_line_pixels(
    image: np.ndarray,
    original: np.ndarray,
    lines_before: np.ndarray,
    mask: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    dilate_px: int,
) -> int:
    """Repaint the long-line pixels that the erase removed.

    Only the intersection of (was linework) and (was erased) is touched,
    so a repair can never shift or invent linework — it can only put
    back what was demonstrably there a moment ago.

    Restricted to connected components of ``lines_before`` that actually
    extend past at least one text box they overlap (see
    ``_MIN_LINE_EXTENSION_PX``) — a component fully contained inside the
    box it was found in is a glyph stroke line_pixel_mask mistook for
    linework, not a real line the erase needs to repair.

    A large 2D fill (a filled room panel, or a reverse-out label's own
    dark background) can ALSO pass this test — a solid rectangle
    trivially contains 40px+ runs everywhere, so line_pixel_mask cannot
    tell "large fill" from "real line" on its own — and a disk-opening
    pre-filter meant to make that distinction was tried and reverted;
    see the comment above ``_MIN_LINE_EXTENSION_PX`` for why. A
    reverse-out label sitting inside such a fill can therefore still
    show a faint antialiased outline of itself surviving through the
    flank below — see build_text_mask's own docstring for the much more
    severe bug (the whole label surviving unmirrored) this module does
    fix, and for why the residual outline was judged the lesser risk.

    A line is repaired in two bands, because its core and its edge want
    different answers:

    * **The core** — the component itself, what Otsu called linework —
      is repainted flat, in the sheet's own ink colour. Faithfully
      restoring each pixel's ORIGINAL value here looks more honest and
      is not: where a glyph sat ON a line, those original pixels are
      part glyph, so restoring them paints the old, pre-mirror string
      back onto the line it crossed. Invisible for the ordinary case of
      black type on black linework, and a clearly legible ghost the
      moment the two differ in tone — a grey annotation over a black
      wall leaves its own silhouette sitting in the wall. A flat fill
      cannot reproduce a glyph shape at all.
    * **The flank** — one pixel beyond the core — is restored from
      ``original``, because there is no flat value that would be right.
      ``lines_before`` is thresholded by Otsu, whereas
      :func:`build_text_mask` erases everything below *local* paper, a
      deliberately lower bar (see ``PAPER_TOLERANCE``). So the erase
      consistently takes one pixel more of every line than Otsu ever
      labelled as line: its antialiased edge. Leaving that pixel out
      thinned every repaired line down both sides for exactly the width
      of the label crossing it — 190 of the 319 pixels of real linework
      the golden fixture was losing. Filling it with solid ink instead
      would thicken the line, which is the one thing the paragraph
      above promises never happens; the original grey is the line's own
      edge ramp and is the only value that restores the weight it had.

    One dilation step, not more: on the golden fixture it recovers those
    190 pixels while touching no glyph pixel at all, where a second step
    gains 3 more and starts eating glyphs.
    """
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        lines_before, connectivity=8
    )
    if num_labels <= 1:
        return 0

    h, w = mask.shape[:2]
    genuine: set[int] = set()
    for x0, y0, x1, y1 in boxes:
        bx0 = max(0, int(np.floor(x0)) - dilate_px)
        by0 = max(0, int(np.floor(y0)) - dilate_px)
        bx1 = min(w, int(np.ceil(x1)) + dilate_px)
        by1 = min(h, int(np.ceil(y1)) + dilate_px)
        if bx1 <= bx0 or by1 <= by0:
            continue
        for lbl in np.unique(labels[by0:by1, bx0:bx1]):
            if lbl == 0 or int(lbl) in genuine:
                continue
            lx, ly, lw, lh, _area = stats[lbl]
            extension = max(
                bx0 - lx, (lx + lw) - bx1, by0 - ly, (ly + lh) - by1
            )
            if extension > _MIN_LINE_EXTENSION_PX:
                genuine.add(int(lbl))

    if not genuine:
        return 0

    component_mask = np.isin(labels, list(genuine)).astype(np.uint8) * 255
    band = cv2.dilate(component_mask, np.ones((3, 3), np.uint8))
    # Intersected with `mask` last, so neither band can reach a pixel the
    # erase itself did not take — a repair still cannot paint outside the
    # footprint it is repairing.
    core = cv2.bitwise_and(component_mask, mask) > 0
    flank = (cv2.bitwise_and(band, mask) > 0) & ~core

    if core.any():
        image[core] = _darkest_colour(image)
    if flank.any():
        image[flank] = original[flank]
    return int(np.count_nonzero(core) + np.count_nonzero(flank))


def _darkest_colour(image: np.ndarray) -> np.ndarray:
    """The sheet's ink colour, taken globally — linework on a plan is one
    colour, and sampling globally avoids inheriting a local artefact."""
    flat = image.reshape(-1, 3).astype(np.float32)
    luma = bgr_luma(flat)
    dark = flat[luma <= np.percentile(luma, 1)]
    if dark.size == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    return np.median(dark, axis=0).astype(np.uint8)
