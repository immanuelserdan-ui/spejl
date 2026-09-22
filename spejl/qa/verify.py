"""Post-save verification of geometry outside text exclusions and output wording.

Pixel exclusions bound the geometry guarantee. A separate output OCR pass checks
wording against source OCR; agreement is not independent ground truth. Missing,
different or uncertain readings require review instead of an unconditional pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from spejl.detect.ocr import OcrBackend, RapidOcrBackend
from spejl.detect.rotations import detect_all_orientations
from spejl.models import Axis
from spejl.image_io import write_image
from spejl.lexicon.snap import snap
from spejl.detect.rotations import _iou
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
    text_issues: tuple[str, ...] = ()
    text_runs_checked: int = 0

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
    """Compare geometry above tolerance and cross-check output wording.

    Geometry exclusions are detected on the SOURCE deliberately: the
    output's own re-rendered glyphs are a different shape, font and
    size than whatever originally sat there — exactly the difference
    this function exists to ignore. Detecting on the source and
    mirroring those boxes into output space describes where a label
    WAS on the sheet, which is the region this check is actually
    supposed to stay out of, regardless of what got redrawn there.

    A second OCR pass checks the output wording. Agreement with source OCR
    does not certify the source reading or excluded geometry. The pipeline's
    exact 2x upscaling is supported; other size mismatches raise ValueError.
    """
    source = cv2.imread(str(source_path))
    output = cv2.imread(str(output_path))
    if source is None:
        raise ValueError(f"Could not read source image: {source_path}")
    if output is None:
        raise ValueError(f"Could not read output image: {output_path}")

    source = _align_source_scale(source, output)
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

    outside_mask = diff_mask.copy()
    for bx0, by0, bx1, by1 in boxes:
        outside_mask[by0:by1, bx0:bx1] = 0
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        outside_mask, connectivity=8
    )
    outside: list[VerifyRegion] = []
    for lbl in range(1, num_labels):
        lx, ly, lw, lh, area = stats[lbl]
        outside.append(VerifyRegion(
            bbox=(int(lx), int(ly), int(lx + lw), int(ly + lh)), area_px=int(area),
        ))

    expected = [(snap(d.text, conf=d.conf).text,
                 M.mirror_bbox(d.bbox, w, h, axis)) for d in detections]
    text_issues = check_rendered_text(output, expected, backend)
    text_issues.extend(f"{d.text!r}: source OCR confidence {d.conf:.2f}; review required."
                       for d in detections if d.conf < .85)

    total_px = int(diff_mask.size)
    differing_px = int(np.count_nonzero(diff_mask))
    return VerifyReport(
        passed=not outside and not text_issues,
        total_px=total_px,
        differing_px=differing_px,
        text_regions_checked=len(boxes),
        outside_text_regions=tuple(outside),
        text_issues=tuple(text_issues),
        text_runs_checked=len(expected),
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

    source = _align_source_scale(source, output)
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
    write_image(dest_path, vis)


def check_rendered_text(image, expected, backend):
    """Read output pixels and match each expected string spatially, one-to-one.

    Agreement is an OCR cross-check, not ground truth. Missing, conflicting,
    and low-confidence readings remain explicit review items.
    """
    observed = detect_all_orientations(image, backend)
    candidates = sorted(
        [(_iou(box, d.bbox), i, j) for i, (_text, box) in enumerate(expected)
         for j, d in enumerate(observed) if _iou(box, d.bbox) >= .15], reverse=True)
    matches, used = {}, set()
    for _overlap, i, j in candidates:
        if i not in matches and j not in used:
            matches[i] = j
            used.add(j)
    issues = []
    for i, (text, _box) in enumerate(expected):
        if i not in matches:
            issues.append(f"{text!r}: no output OCR match; wording unconfirmed.")
            continue
        det = observed[matches[i]]
        read = snap(det.text, conf=det.conf).text
        if read != text or det.conf < .85:
            issues.append(f"{text!r}: output OCR read {read!r} at {det.conf:.2f}; review required.")
    return issues


def _align_source_scale(source, output):
    """Reproduce the pipeline's documented 2x small-text upscaling only.

    Do not arbitrarily resize unrelated files to manufacture alignment.
    """
    h, w = source.shape[:2]
    if output.shape[:2] == (2*h, 2*w):
        return cv2.resize(source, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    return source
