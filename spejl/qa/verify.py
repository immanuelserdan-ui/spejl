"""Post-hoc verification — independently confirm a SAVED mirrored file
still holds every pixel of the source drawing, outside wherever a text
label was deliberately erased and redrawn.

Distinct from qa/self_correct.py's own Rule.SPATIAL_INTEGRITY, which
this borrows its exact reasoning from: that check runs INSIDE
mirror_raster, against the in-memory canvas, before a file is ever
written — and by the time a human wants to double-check a save, that
in-memory state is long gone. This module re-derives the same
comparison from two files already on disk, on demand, without
re-running the whole pipeline. It is the thing this project's own
investigations were doing by hand, script after script, for most of a
session — built once, properly, instead of once more, ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from spejl.detect.ocr import OcrBackend, RapidOcrBackend
from spejl.detect.rotations import detect_all_orientations
from spejl.models import Axis
from spejl.transform import mirror as M

# Padding around a detected text box before excluding it from the
# comparison. Wider than self_correct.py's own _FOOTPRINT_PAD_PX (12px)
# deliberately: that value is tuned against the tighter MEASURED-ink
# footprint the live pipeline computes per run (style/metrics.py's own
# fit, target size and all) — none of which exists here. All this
# function has, after the fact, is the raw OCR detection box, which is
# already less precise than a measured-ink footprint. Erring wider
# costs only a slightly less precise "clean" region right at a label's
# own edge; erring narrow risks reading the label's own re-rendered
# ink, in a different font at a different weight, as a missing-geometry
# false alarm — the one thing this check exists to rule out correctly.
_TEXT_EXCLUSION_PAD_PX = 20

# A pixel intensity difference below this reads as ordinary
# antialiasing or lossy-JPEG noise, not a genuine missing- or
# added-pixel — matches self_correct.py's check_spatial_integrity own
# plain luma split (< 128 = dark) in spirit, tuned instead against the
# DIFFERENCE between two images rather than one image's own darkness.
_DIFF_THRESHOLD = 60


@dataclass(frozen=True)
class VerifyRegion:
    """One connected cluster of differing pixels that sits outside
    every detected text label's own (padded) box — a genuine candidate
    for lost or altered drawing geometry, not an expected text-shape
    difference."""

    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1), output-image pixel space
    area_px: int


@dataclass(frozen=True)
class VerifyReport:
    passed: bool
    total_px: int
    differing_px: int
    text_regions_checked: int
    outside_text_regions: tuple[VerifyRegion, ...] = ()

    @property
    def differing_pct(self) -> float:
        return 100.0 * self.differing_px / self.total_px if self.total_px else 0.0

    @property
    def outside_text_px(self) -> int:
        return sum(r.area_px for r in self.outside_text_regions)


def verify_mirror(
    source_path: Path,
    output_path: Path,
    axis: Axis,
    backend: OcrBackend | None = None,
) -> VerifyReport:
    """Independently confirm ``output_path`` preserves every pixel of
    ``source_path``'s own drawing, outside wherever a text label was
    legitimately erased and redrawn.

    Text is detected on the SOURCE, not the output, deliberately: the
    output's own re-rendered glyphs are a different shape, font and
    size than whatever originally sat there — exactly the difference
    this function exists to ignore. Detecting on the source and
    mirroring those boxes into output space describes where a label
    WAS on the sheet, which is the region this check is actually
    supposed to stay out of, regardless of what got redrawn there.

    Raises ``ValueError`` if either file can't be read, or if the two
    don't even agree on size — this function reports what differs, but
    a size mismatch isn't a "difference" it can meaningfully localise,
    so it fails loudly rather than compare misaligned arrays.
    """
    source = cv2.imread(str(source_path))
    output = cv2.imread(str(output_path))
    if source is None:
        raise ValueError(f"Could not read source image: {source_path}")
    if output is None:
        raise ValueError(f"Could not read output image: {output_path}")

    flipped_source = M.flip_image(source, axis)
    if flipped_source.shape[:2] != output.shape[:2]:
        raise ValueError(
            f"Source/output size mismatch ({flipped_source.shape[:2]} vs "
            f"{output.shape[:2]}) — cannot compare pixel-for-pixel."
        )
    h, w = output.shape[:2]

    backend = backend or RapidOcrBackend()
    detections = detect_all_orientations(source, backend)
    boxes: list[tuple[int, int, int, int]] = []
    p = _TEXT_EXCLUSION_PAD_PX
    for det in detections:
        x0, y0, x1, y1 = M.mirror_bbox(det.bbox, w, h, axis)
        boxes.append((
            max(0, int(x0 - p)), max(0, int(y0 - p)),
            min(w, int(x1 + p)), min(h, int(y1 + p)),
        ))

    source_gray = cv2.cvtColor(flipped_source, cv2.COLOR_BGR2GRAY)
    output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(source_gray, output_gray)
    _t, diff_mask = cv2.threshold(diff, _DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        diff_mask, connectivity=8
    )
    outside: list[VerifyRegion] = []
    for lbl in range(1, num_labels):
        lx, ly, lw, lh, area = stats[lbl]
        cx, cy = lx + lw / 2, ly + lh / 2
        # A component's CENTROID deciding membership, not its full
        # bbox — matches how a text label's own re-rendered ink can
        # legitimately brush the padded exclusion box's edge without
        # the whole differing blob sitting inside it; the centroid is
        # the more honest single answer to "is this really the text,
        # or something else that happens to touch it".
        inside_any = any(bx0 <= cx <= bx1 and by0 <= cy <= by1 for bx0, by0, bx1, by1 in boxes)
        if not inside_any:
            outside.append(VerifyRegion(
                bbox=(int(lx), int(ly), int(lx + lw), int(ly + lh)), area_px=int(area),
            ))

    total_px = int(diff_mask.size)
    differing_px = int(np.count_nonzero(diff_mask))
    return VerifyReport(
        passed=not outside,
        total_px=total_px,
        differing_px=differing_px,
        text_regions_checked=len(boxes),
        outside_text_regions=tuple(outside),
    )


def render_diff_overlay(
    source_path: Path,
    output_path: Path,
    axis: Axis,
    dest_path: Path,
) -> None:
    """Save a visualisation: the source's own geometry in black and
    white, every pixel that differs from the mirrored output painted
    red — the same evidence this project's own investigations were
    built on throughout, generated on demand instead of a one-off
    script each time. Marks every differing pixel, not just the ones
    verify_mirror flags as outside text — seeing the (expected) text
    shape differences alongside any (unexpected) geometry ones is what
    makes the picture legible as a whole, the same way every diff
    image shown during this project's own debugging did.
    """
    source = cv2.imread(str(source_path))
    output = cv2.imread(str(output_path))
    if source is None or output is None:
        raise ValueError("Could not read one of the two images to compare.")

    flipped_source = M.flip_image(source, axis)
    if flipped_source.shape[:2] != output.shape[:2]:
        raise ValueError("Source/output size mismatch — cannot render a diff overlay.")

    source_gray = cv2.cvtColor(flipped_source, cv2.COLOR_BGR2GRAY)
    output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(source_gray, output_gray)
    _t, diff_mask = cv2.threshold(diff, _DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    vis = cv2.cvtColor(source_gray, cv2.COLOR_GRAY2BGR)
    vis[diff_mask > 0] = (0, 0, 255)  # BGR red
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest_path), vis)
