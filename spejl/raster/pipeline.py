"""Route B — the raster pipeline, S1 through S7.

    detect (S2) -> snap (S3) -> style (S4) -> erase+repair (S5)
      -> flip (S6) -> mirror anchors -> re-render (S7)

The load-bearing property, and the reason the stages are in this order:
text attributes are extracted *before* the flip and applied *after* it,
so no glyph ever passes through the reflection. The flip sees a plate
with no type on it at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from spejl.detect.ocr import Detection, OcrBackend, RapidOcrBackend
from spejl.detect.rotations import detect_all_orientations
from spejl.erase.clean import erase_text
from spejl.lexicon.snap import snap
from spejl.models import Axis, Document, Flag, PageResult, Route
from spejl.render.text import linework_mask_for, render_run
from spejl.style.metrics import TextStyle, fit_style
from spejl.transform import mirror as M

MIN_CAP_HEIGHT_PX = 14.0   # below this, upscale before OCR (build plan S1)
REFUSE_CAP_HEIGHT_PX = 8.0  # below this, refuse rather than mangle
LOW_CONFIDENCE = 0.85


@dataclass
class MirroredRun:
    """One run, tracked from detection through to where it was drawn."""

    text: str
    text_raw: str
    kind: str
    conf: float
    angle_src: float
    angle_out: float
    center_src: tuple[float, float]
    center_out: tuple[float, float]
    bbox_src: tuple[float, float, float, float]
    style: TextStyle
    flags: list[Flag] = field(default_factory=list)


@dataclass
class RasterResult:
    image: np.ndarray
    runs: list[MirroredRun]
    document: Document
    upscale: float = 1.0


def _upscale_factor(detections: list[Detection]) -> float:
    """Decide whether the sheet needs enlarging before OCR is trusted."""
    if not detections:
        return 1.0
    caps = []
    for det in detections:
        x0, y0, x1, y1 = det.bbox
        across = (x1 - x0) if abs(det.angle_deg) > 45 else (y1 - y0)
        caps.append(across * 0.72)
    median_cap = float(np.median(caps))
    if median_cap < REFUSE_CAP_HEIGHT_PX:
        raise ValueError(
            f"Median glyph cap height is {median_cap:.1f}px — too low to mirror "
            "reliably. Re-export the plan at a higher resolution; Spejl refuses "
            "rather than produce a plausible-looking wrong drawing."
        )
    return 2.0 if median_cap < MIN_CAP_HEIGHT_PX else 1.0


def mirror_raster(
    input_path: Path,
    output_path: Path,
    axis: Axis = Axis.VERTICAL,
    backend: OcrBackend | None = None,
    protected: list[tuple[float, float, float, float]] | None = None,
) -> RasterResult:
    """Mirror a raster plan, keeping every string readable.

    ``protected`` regions (north arrow, scale bar, title block, logo) are
    lifted out before the flip and composited back unmirrored at their
    mirrored anchor — a mirrored north arrow is a false statement about
    the building, so this is a correctness feature, not a nicety.
    """
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    backend = backend or RapidOcrBackend()
    document = Document(
        source=input_path, output=output_path, axis=axis, route=Route.RASTER
    )
    flags: list[Flag] = []

    # ---- S2: detect -------------------------------------------------------
    detections = detect_all_orientations(image, backend)
    upscale = _upscale_factor(detections)
    if upscale != 1.0:
        image = cv2.resize(
            image, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4
        )
        detections = detect_all_orientations(image, backend)
        flags.append(
            Flag("upscaled", f"Sheet upscaled {upscale:g}x before OCR (small type).", "info")
        )

    h, w = image.shape[:2]
    protected = protected or []

    # ---- S3 + S4: correct the strings, measure the type -------------------
    runs: list[MirroredRun] = []
    for det in detections:
        if _inside_any(det.bbox, protected):
            continue  # handled with the protected regions, not as text

        result = snap(det.text)
        run_flags: list[Flag] = []
        if det.conf < LOW_CONFIDENCE:
            run_flags.append(
                Flag("low-confidence", f"OCR confidence {det.conf:.2f} for {det.text!r}.", "warn")
            )
        if result.changed:
            run_flags.append(
                Flag("lexicon-snap", f"{result.raw!r} -> {result.text!r}", "info")
            )
        if result.warning:
            run_flags.append(Flag("implausible", result.warning, "warn"))

        style = fit_style(image, result.text, det.bbox, det.angle_deg)
        out_bbox = M.mirror_bbox(det.bbox, w, h, axis)
        runs.append(
            MirroredRun(
                text=result.text,
                text_raw=det.text,
                kind=result.kind,
                conf=det.conf,
                angle_src=det.angle_deg,
                angle_out=M.mirror_angle(det.angle_deg, axis),
                center_src=M.bbox_center(det.bbox),
                center_out=M.bbox_center(out_bbox),
                bbox_src=det.bbox,
                style=style,
                flags=run_flags,
            )
        )

    # ---- S5: erase the type, repair the linework it covered ---------------
    erased = erase_text(image, [d.bbox for d in detections if not _inside_any(d.bbox, protected)])
    flags.extend(erased.flags)

    # ---- S6: flip the type-free plate ------------------------------------
    flipped = M.flip_image(erased.image, axis)
    flipped_text_mask = M.flip_image(erased.mask, axis)

    # Protected regions: mirror the anchor, not the pixels.
    for region in protected:
        _replace_unmirrored(flipped, image, region, w, h, axis)

    # ---- S7: re-render every run upright at its mirrored anchor -----------
    lines = linework_mask_for(flipped, flipped_text_mask)
    for run in runs:
        rendered = render_run(
            canvas=flipped,
            text=run.text,
            center=run.center_out,
            angle_deg=run.angle_out,
            style=run.style,
            target_size=_target_size(run),
            linework_mask=lines,
        )
        if rendered.shrunk:
            run.flags.append(
                Flag("fit-shrunk", f"{run.text!r} reduced to fit its original box.", "info")
            )
        if rendered.collided:
            run.flags.append(
                Flag("collision", f"{run.text!r} overlaps linework after mirroring.", "warn")
            )

    cv2.imwrite(str(output_path), flipped)

    document.pages.append(
        PageResult(
            index=0,
            width=float(w),
            height=float(h),
            text_runs_mirrored=len(runs),
            flags=flags + [f for r in runs for f in r.flags if f.severity == "warn"],
        )
    )
    return RasterResult(image=flipped, runs=runs, document=document, upscale=upscale)


def _target_size(run: MirroredRun) -> tuple[float, float]:
    """The original run's measured (along, across) *ink* extent, in px.

    Measured ink, not the detection box: the box is padded, so comparing
    drawn ink against it would leave ~20% of slack before the guard ever
    fired — which is exactly how re-rendered labels came out visibly
    larger than their neighbours.
    """
    if run.style.ink_along_px > 0:
        return (run.style.ink_along_px, run.style.cap_height_px)
    x0, y0, x1, y1 = run.bbox_src
    if abs(run.angle_src) > 45:  # vertical: baseline runs down the box
        return (y1 - y0, x1 - x0)
    return (x1 - x0, y1 - y0)


def _inside_any(
    bbox: tuple[float, float, float, float],
    regions: list[tuple[float, float, float, float]],
) -> bool:
    cx, cy = M.bbox_center(bbox)
    return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in regions)


def _replace_unmirrored(
    canvas: np.ndarray,
    source: np.ndarray,
    region: tuple[float, float, float, float],
    w: float,
    h: float,
    axis: Axis,
) -> None:
    """Paste a protected region unflipped, at its mirrored position."""
    sx0, sy0, sx1, sy1 = (int(round(v)) for v in region)
    patch = source[max(0, sy0):sy1, max(0, sx0):sx1]
    if patch.size == 0:
        return
    dx0, dy0, dx1, dy1 = (int(round(v)) for v in M.mirror_bbox(region, w, h, axis))
    dy1 = min(canvas.shape[0], dy0 + patch.shape[0])
    dx1 = min(canvas.shape[1], dx0 + patch.shape[1])
    if dx1 <= dx0 or dy1 <= dy0:
        return
    canvas[dy0:dy1, dx0:dx1] = patch[: dy1 - dy0, : dx1 - dx0]
