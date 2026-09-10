"""Route B — the raster pipeline, S1 through S7.

    detect (S2) -> snap (S3) -> style (S4) -> erase+repair (S5)
      -> flip (S6) -> mirror anchors -> re-render (S7)

The load-bearing property, and the reason the stages are in this order:
text attributes are extracted *before* the flip and applied *after* it,
so no glyph ever passes through the reflection. The flip sees a plate
with no type on it at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from spejl.detect.ocr import Detection, OcrBackend, RapidOcrBackend
from spejl.detect.rotations import detect_all_orientations
from spejl.erase.clean import erase_text
from spejl.lexicon.snap import snap
from spejl.models import Axis, Document, Flag, PageResult, Route
from spejl.render.text import linework_mask_for, render_run
from spejl.style.metrics import (
    TextStyle,
    fit_style,
    font_measure_width,
    measure_ink_center,
    solve_horizontal_scale,
    solve_tracking,
)
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
    protected = protected or []
    if upscale != 1.0:
        image = cv2.resize(
            image, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4
        )
        detections = detect_all_orientations(image, backend)
        # `protected` regions arrive in the ORIGINAL image's coordinates —
        # the only space the caller can have known before this function
        # decided (internally) to upscale. Every detection and the image
        # itself just moved into upscaled-pixel space; `protected` must
        # follow, or a north arrow at the edge of its region silently
        # stops being recognised as protected and gets erased and
        # re-rendered as ordinary text — exactly the "mirrored north
        # arrow" correctness bug this whole feature exists to prevent.
        protected = [tuple(v * upscale for v in region) for region in protected]
        flags.append(
            Flag("upscaled", f"Sheet upscaled {upscale:g}x before OCR (small type).", "info")
        )

    h, w = image.shape[:2]

    # ---- S3 + S4: correct the strings, measure the type -------------------
    runs: list[MirroredRun] = []
    text_boxes: list[tuple[float, float, float, float]] = []
    non_text_flags: list[Flag] = []
    for det in detections:
        if _inside_any(det.bbox, protected):
            continue  # handled with the protected regions, not as text

        if _looks_like_a_graphical_symbol(det.text) or _looks_like_the_wrong_script(det.text):
            # Either signal means the same thing downstream: this is
            # almost certainly OCR mis-firing, not real text, whether
            # that is a drafting symbol read as characters (no letters
            # or digits at all — a '→' direction arrow detected at 0.50
            # confidence, no lexicon match, then erased and re-rendered
            # as garbled text overlapping a real room label) or a
            # hallucinated character from the wrong script entirely (a
            # stray mark read as a CJK ideograph, same confidence range,
            # same failure mode). Left out of both the erase list and
            # `runs` entirely, so it passes through as ordinary
            # geometry — mirrored correctly along with every other line
            # on the sheet, the same as a north arrow that HASN'T been
            # explicitly protected, rather than being destroyed and
            # replaced with a nonsense string. Flagged so a human can
            # confirm nothing meaningful was actually lost.
            non_text_flags.append(
                Flag(
                    "non-text-symbol",
                    f"{det.text!r} (OCR confidence {det.conf:.2f}) looks like a "
                    "drafting symbol, not text — left as geometry, not re-rendered.",
                    "info",
                )
            )
            continue

        text_boxes.append(det.bbox)
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

        other_boxes = [d.bbox for d in detections if d is not det]
        style = fit_style(image, result.text, det.bbox, det.angle_deg, other_boxes=other_boxes)
        # The raw detection box's own centre is not necessarily where the
        # WORD itself centres — dash-noise from a crossing reference line
        # (or any other one-sided contamination measure_ink_extent already
        # excludes when fitting size) pads one edge of the box further
        # than the other, and mirroring that off-centre raw box turns a
        # small bias in the source into a visible one in the output.
        # measure_ink_center reuses the same filtered ink-cluster analysis
        # fit_style just used for sizing, so the anchor reflects the same
        # ink the render is actually fitted to.
        center_src = measure_ink_center(
            image, det.bbox, det.angle_deg, other_boxes=other_boxes,
            expected_glyphs=len(result.text),
        )
        center_out = M.mirror_point(center_src[0], center_src[1], w, h, axis)
        runs.append(
            MirroredRun(
                text=result.text,
                text_raw=det.text,
                kind=result.kind,
                conf=det.conf,
                angle_src=det.angle_deg,
                angle_out=M.mirror_angle(det.angle_deg, axis),
                center_src=center_src,
                center_out=center_out,
                bbox_src=det.bbox,
                style=style,
                flags=run_flags,
            )
        )

    _snap_consistent_sizes(runs)

    # ---- S5: erase the type, repair the linework it covered ---------------
    erased = erase_text(image, text_boxes)
    flags.extend(erased.flags)
    flags.extend(non_text_flags)

    # ---- S6: flip the type-free plate ------------------------------------
    flipped = M.flip_image(erased.image, axis)
    flipped_text_mask = M.flip_image(erased.mask, axis)

    # Protected regions: mirror the anchor, not the pixels.
    for region in protected:
        _replace_unmirrored(flipped, image, region, w, h, axis)

    # ---- S7: re-render every run upright at its mirrored anchor -----------
    # Passed by reference to every render_run() call below and mutated by
    # each one (render/text.py stamps its own ink in after checking for a
    # collision) — so run N is checked against the geometry AND every run
    # 1..N-1 already drawn on this same canvas, not just the static
    # linework it started as.
    lines = linework_mask_for(flipped, flipped_text_mask)
    for run in runs:
        run_target_size = _target_size(run)
        target_along = run_target_size[0]
        rendered = render_run(
            canvas=flipped,
            text=run.text,
            center=run.center_out,
            angle_deg=run.angle_out,
            style=run.style,
            target_size=run_target_size,
            linework_mask=lines,
        )
        if rendered.shrunk:
            # The shrink-to-fit loop runs a bounded number of attempts
            # (render/text.py) and can legitimately give up still over
            # target — most often a lexicon-corrected string is longer
            # than the raw OCR read it replaced ('Vaer.' -> 'Værelse').
            # A silent "fit-shrunk" info flag looked identical whether
            # the loop converged to a 1% overshoot or gave up at 25% —
            # the second case is worth a human's attention, the first
            # is not, so the flag now says which one happened.
            overflow = rendered.ink_width / max(1.0, target_along)
            if overflow > 1.15:
                run.flags.append(
                    Flag(
                        "fit-shrink-incomplete",
                        f"{run.text!r} is still {overflow:.0%} of its original width "
                        "after the shrink-to-fit limit — may overlap neighbouring content.",
                        "warn",
                    )
                )
            else:
                run.flags.append(
                    Flag("fit-shrunk", f"{run.text!r} reduced to fit its original box.", "info")
                )
        if rendered.collided:
            run.flags.append(
                Flag(
                    "collision",
                    f"{run.text!r} overlaps linework or another label after mirroring.",
                    "warn",
                )
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


# Same-kind labels whose independently fitted sizes are this close are
# treated as measurement noise around one intended size, not a real
# difference — see _snap_consistent_sizes.
_SIZE_CLUSTER_TOLERANCE = 0.12


def _snap_consistent_sizes(runs: list[MirroredRun]) -> None:
    """Real technical drawings draft room labels — and separately,
    dimensions — at one of a small number of DISCRETE sizes, a drafting
    convention, not a continuum. Every run's size is fit independently
    from its own detected ink, necessarily, since nothing else can know
    a label's true size without assuming one — but that independence
    means ordinary OCR measurement noise (a pixel or two of difference
    in one detected box's height) can make two labels that were
    IDENTICAL on the original drawing come out at two visibly different
    sizes after mirroring.

    This groups same-KIND runs whose fitted sizes land close enough
    together that the gap reads as noise rather than intent, and snaps
    each such cluster to its own median — never merging clusters that
    are genuinely far apart, which is exactly the case a real drawing
    also uses on purpose (a large room's label bigger than a small
    fixture room's). Tracking and width-scale are re-derived at the
    snapped size rather than just overwritten alongside it, so a run
    nudged to its cluster's size still hits its own measured target
    width, not a stale one computed for its original size.
    """
    by_kind: dict[str, list[MirroredRun]] = {}
    for run in runs:
        if run.kind in ("room", "dimension"):
            by_kind.setdefault(run.kind, []).append(run)

    for group in by_kind.values():
        ordered = sorted(group, key=lambda r: r.style.px_size)
        cluster: list[MirroredRun] = []
        for run in ordered:
            if cluster and (run.style.px_size - cluster[-1].style.px_size) > (
                _SIZE_CLUSTER_TOLERANCE * cluster[-1].style.px_size
            ):
                _snap_cluster_to_median(cluster)
                cluster = []
            cluster.append(run)
        _snap_cluster_to_median(cluster)


_SNAP_MAX_OVERSHOOT = 1.20  # matches the size-fidelity gate's own worst-case ceiling


def _snap_cluster_to_median(cluster: list[MirroredRun]) -> None:
    if len(cluster) < 2:
        return
    sizes = sorted(r.style.px_size for r in cluster)
    median = sizes[len(sizes) // 2]
    for run in cluster:
        if run.style.px_size == median:
            continue
        old = run.style
        tracking = solve_tracking(run.text, old.font_path, median, old.ink_along_px)
        natural_tracked = font_measure_width(run.text, old.font_path, median, tracking * median)
        width_scale = solve_horizontal_scale(natural_tracked, old.ink_along_px)

        # Guard against exactly the failure this feature's own regression
        # test caught: a short string (a 4-digit dimension) has little
        # room to absorb a size bump proportionally, so snapping it to a
        # cluster-mate's larger size — appropriate for most members —
        # can overshoot ITS OWN measured width well past what any other
        # stage of this pipeline would accept. Skip the snap rather than
        # render a run wider than the fit-to-box guard elsewhere in this
        # same pipeline would ever let through.
        rendered_width = natural_tracked * width_scale
        if rendered_width > old.ink_along_px * _SNAP_MAX_OVERSHOOT:
            continue

        run.style = replace(old, px_size=median, tracking=tracking, width_scale=width_scale)


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


def _looks_like_a_graphical_symbol(text: str) -> bool:
    """True for a detection with no letters or digits at all.

    That's the signature of an OCR engine mis-firing on a drafting
    symbol — an arrow, a bullet, a stray dash — rather than reading
    real text. Deliberately permissive otherwise: any run with even one
    alphanumeric character (including a real annotation glyph like
    'H*', which has one) is treated as text and goes through the normal
    lexicon/erase/render path.
    """
    return not any(ch.isalnum() for ch in text)


# CJK Unified Ideographs, Hiragana/Katakana, Hangul, Cyrillic, Hebrew,
# Arabic — ranges no Danish plan's own text ever legitimately uses.
_NON_LATIN_RANGES = (
    (0x0400, 0x04FF),  # Cyrillic
    (0x0590, 0x08FF),  # Hebrew, Arabic
    (0x2E80, 0x9FFF),  # CJK radicals through CJK Unified Ideographs
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0xAC00, 0xD7A3),  # Hangul syllables
)


def _looks_like_the_wrong_script(text: str) -> bool:
    """True if any character falls in a script no Danish plan's own
    text ever legitimately uses.

    RapidOCR's recognition model is trained on mixed Latin+CJK data
    (its own model filename says as much: ch_PP-OCRv4), and it can
    hallucinate a single CJK character from an ambiguous, ink-adjacent
    mark — confirmed on a real project drawing: a short vertical stroke
    near 'Bad' was read as U+4E00 ('one') at 0.50 confidence. Python's
    `str.isalnum()` — what :func:`_looks_like_a_graphical_symbol` checks
    — classifies CJK ideographs as alphanumeric, so that filter alone
    does not catch this; the wrong-script check is a separate,
    deliberately narrow net for exactly this failure mode, not a
    broader language guess.
    """
    return any(
        any(lo <= ord(ch) <= hi for lo, hi in _NON_LATIN_RANGES) for ch in text
    )


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
    """Paste a protected region unflipped, at its mirrored position.

    Both ends are clamped into bounds before slicing — the destination
    as carefully as the source already was. A region near the mirrored
    edge of the sheet can put a raw ``dx0`` below 0 (e.g. a box that
    extended past the source's right edge mirrors to one that starts
    left of the canvas's left edge); Python/numpy slicing treats a
    negative start as counting from the *end* of the array rather than
    clipping to it, so an unclamped ``canvas[dy0:dy1, -50:-20]`` silently
    pastes the patch near the opposite edge of the image instead of
    raising or clipping — corruption with no exception and no flag.
    """
    sx0, sy0, sx1, sy1 = (int(round(v)) for v in region)
    sx0, sy0 = max(0, sx0), max(0, sy0)
    sx1, sy1 = min(source.shape[1], sx1), min(source.shape[0], sy1)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    patch = source[sy0:sy1, sx0:sx1]

    dx0, dy0, dx1, dy1 = (int(round(v)) for v in M.mirror_bbox(region, w, h, axis))
    # Clip the destination to the canvas FIRST, then shrink the patch by
    # exactly what was clipped off each side — this is what keeps a
    # region that runs off one edge from wrapping onto the other.
    clip_left = max(0, -dx0)
    clip_top = max(0, -dy0)
    cdx0, cdy0 = max(0, dx0), max(0, dy0)
    cdx1 = min(canvas.shape[1], dx0 + patch.shape[1])
    cdy1 = min(canvas.shape[0], dy0 + patch.shape[0])
    if cdx1 <= cdx0 or cdy1 <= cdy0:
        return
    canvas[cdy0:cdy1, cdx0:cdx1] = patch[
        clip_top : clip_top + (cdy1 - cdy0), clip_left : clip_left + (cdx1 - cdx0)
    ]
