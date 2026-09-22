"""Route B — the raster pipeline, S1 through S8.

    detect (S2) -> snap (S3) -> style (S4) -> erase+repair (S5)
      -> flip (S6) -> mirror anchors -> re-render (S7) -> QA gate (S8)

The load-bearing property, and the reason the stages are in this order:
text attributes are extracted *before* the flip and applied *after* it,
so no glyph ever passes through the reflection. The flip sees a plate
with no type on it at all.

S7 then composites in layers rather than simply painting: paper, the
re-rendered runs, and the sheet's own geometry over the top. The two
kinds of content are not equally recoverable — a run can be redrawn
from its string and measured style at any time, whereas linework a
glyph painted over is gone from the output — so where they compete for
a pixel, the drawing wins. See render/text.py.

S8, immediately before the file is written, is the one stage that does
not trust S1-S7 to have done its own job correctly — it checks the
FINISHED sheet's actual bytes against spatial integrity, layer
ordering and text fidelity, corrects what it safely can, and refuses
the save outright rather than write output any of the three are still
wrong on. See qa/self_correct.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re

import cv2
import numpy as np

from spejl.detect.ocr import Detection, OcrBackend, RapidOcrBackend
from spejl.detect.rotations import _intersection, detect_all_orientations
from spejl.erase.clean import erase_text
from spejl.lexicon.snap import SnapResult, snap
from spejl.models import Axis, Document, Flag, PageResult, Route
from spejl.image_io import write_image
from spejl.qa.self_correct import QAContext, QAGateFailure, run_qa_gate
from spejl.render.text import RenderedRun, drawing_alpha_for, linework_mask_for, render_run
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

# Share of a run's own ink the drawing may cover before the run counts as
# buried rather than merely crossed. A dimension number drawn on its own
# dimension line — drafting convention, not a fault — loses only the few
# percent of its ink the line actually runs through, so a threshold near
# zero would fire on most correctly-placed numbers on a real sheet (the
# same false-positive trap documented at length in render/text.py's
# reverted proximity check). Half the run's ink gone is past any amount
# a single crossing line can account for, and means the label has landed
# on poché or a fixture.
_MOSTLY_HIDDEN = 0.5


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
    # Graphical symbols and wrong-script hallucinations (see
    # _looks_like_a_graphical_symbol/_looks_like_the_wrong_script below)
    # are excluded before the median is taken: this runs BEFORE the S3
    # loop that would otherwise filter them out of `runs`, and every
    # confirmed real case (a spot-elevation marker, an I-beam/lintel
    # cross-section, two valve/knob icons) measured noticeably SMALLER
    # than genuine dimension or room-label text — left in, enough of
    # them on an icon-dense sheet could drag the median down enough to
    # trigger an unneeded upscale, or even the REFUSE_CAP_HEIGHT_PX
    # refusal, for a sheet whose actual TEXT is perfectly legible.
    # (Deliberately not the full snap()-based short-unmatched-symbol
    # check too: that needs the lexicon and runs again per-detection in
    # the S3 loop regardless — duplicating it here for this coarse,
    # sheet-wide size estimate isn't worth the extra cost.)
    text_like = [
        det
        for det in detections
        if not _looks_like_a_graphical_symbol(det.text)
        and not _looks_like_the_wrong_script(det.text)
    ]
    detections = text_like or detections  # never let filtering empty the estimate entirely
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


# Below this, a dropped candidate is routine noise this NMS pass is
# SUPPOSED to discard (a stray letter, a truncated duplicate) — not
# worth a human's attention. Matches _merge's own containment-override
# floor: a run this short can't clear that path either, so nothing
# below this length is ever the 'Vaer. 2' shape of loss in the first
# place.
_MISSED_TEXT_MIN_LEN = 3


def _find_likely_missed_text(
    dropped: list[Detection], kept: list[Detection]
) -> list[Flag]:
    """Coverage check: among everything OCR actually read on a pass,
    was anything substantial and confident left out that nothing in the
    final, kept detection set adequately covers?

    Exists because of a real, severe bug this exact question would have
    caught before a user ever saw it: 'Vaer. 2', read correctly and
    confidently, lost too strict an NMS tie-break to a stray single-
    digit fragment and was silently dropped — never erased, never
    re-rendered, its own source pixels passed straight through the
    mirror flip as ordinary geometry, genuinely mirrored text on the
    output. That specific tie-break is fixed (_merge's own docstring),
    but this is the general defence: whatever else might someday cause
    OCR's own correct read of something to not survive to the final
    run list, on this file or a completely different one, this is what
    would surface it — a flag a human sees, not a silent gap.

    Most drops _merge produces are correct and expected — a genuine
    duplicate, a fragment properly absorbed into the run it's part of
    — so this applies two filters before ever calling one worth
    surfacing: long and confident enough that "it was noise anyway"
    isn't a plausible explanation (_MISSED_TEXT_MIN_LEN characters,
    LOW_CONFIDENCE or better — the literal conditions the real
    'Vaer. 2' case met), AND not already substantially covered by
    anything that WAS kept (a real duplicate of a kept run is not a
    loss, it's the NMS pass working as intended).

    "Covered" is deliberately NOT _merge's own _containment(): that
    helper normalises by the SMALLER box's area, built for "is this
    small fragment a piece of that other run" — exactly backwards here,
    where the dropped candidate is typically the BIGGER box and a small
    kept detection sitting inside a small corner of it would score
    _containment() near 1.0 while covering almost none of the text.
    (Confirmed against the real 'Vaer. 2' case: _containment() of its
    box against the kept '2' fragment's box scores ~1.0 — '2' fully
    inside 'Vaer. 2' — which is exactly why the first version of this
    function using it failed to catch the very bug it was written for.)
    What matters here is directional: what fraction of the DROPPED
    candidate's OWN area does a kept detection actually cover.
    """
    flags: list[Flag] = []
    for cand in dropped:
        if len(cand.text) < _MISSED_TEXT_MIN_LEN or cand.conf < LOW_CONFIDENCE:
            continue
        cand_area = max(1.0, (cand.bbox[2] - cand.bbox[0]) * (cand.bbox[3] - cand.bbox[1]))
        covered = any(
            _intersection(cand.bbox, k.bbox) / cand_area > 0.5 for k in kept
        )
        if covered:
            continue
        cx, cy = M.bbox_center(cand.bbox)
        flags.append(
            Flag(
                "possible-missed-text",
                f"OCR read {cand.text!r} (confidence {cand.conf:.2f}) near "
                f"({cx:.0f}, {cy:.0f}) but it did not survive to the final result "
                "— confirm nothing was lost here.",
                "warn",
            )
        )
    return flags


def mirror_raster(
    input_path: Path,
    output_path: Path,
    axis: Axis = Axis.VERTICAL,
    backend: OcrBackend | None = None,
    protected: list[tuple[float, float, float, float]] | None = None,
    qa_gate: bool = True,
) -> RasterResult:
    """Mirror a raster plan, keeping every string readable.

    ``protected`` regions (north arrow, scale bar, title block, logo) are
    lifted out before the flip and composited back unmirrored at their
    mirrored anchor — a mirrored north arrow is a false statement about
    the building, so this is a correctness feature, not a nicety.

    ``qa_gate`` runs the S8 pre-save check (qa/self_correct.py) before
    ``output_path`` is written — spatial integrity, layer ordering, and
    text fidelity, corrected where a correction is safe and refused
    where it isn't (see that module's own docstring for exactly which
    is which). Off switch exists for the test suite and for callers
    doing their own gating (fixture generation, golden-fixture
    round-trip scoring), which have their own reasons to see the raw,
    un-gated output — not a knob a normal caller should reach for.
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
    dropped: list[Detection] = []
    detections = detect_all_orientations(image, backend, dropped_out=dropped)
    upscale = _upscale_factor(detections)
    protected = protected or []
    if upscale != 1.0:
        image = cv2.resize(
            image, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4
        )
        dropped = []
        detections = detect_all_orientations(image, backend, dropped_out=dropped)
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
    flags.extend(_find_likely_missed_text(dropped, detections))

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

        result = snap(det.text, conf=det.conf)
        if _looks_like_a_short_unmatched_symbol(det.text, result):
            # The same failure mode as the graphical-symbol check above,
            # for the case it structurally cannot catch: a misread that
            # happens to spell real alphanumeric characters, so
            # `_looks_like_a_graphical_symbol`'s own "no letters or
            # digits at all" test lets it straight through. Confirmed on
            # three separate real files: a spot-elevation marker read as
            # 'O', an I-beam/lintel cross-section read as 'H' at 1.00
            # confidence (visual resemblance alone, not OCR being
            # unsure), two valve/knob icons read as 'GO' — none of them
            # real text, all three erased and re-rendered as garbage
            # letters over real drafting geometry before this check.
            # See _looks_like_a_short_unmatched_symbol's own docstring
            # for why this is safe to leave unmatched-and-short rather
            # than a new annotation this project hasn't catalogued yet.
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
            # A confusable-annotation correction (snap.py's own fallback,
            # gated on low OCR confidence) is not the same situation as
            # "nothing in the lexicon accounts for this at all" or an
            # out-of-range dimension — it found an answer, just one this
            # module can't verify independently — so it gets its own flag
            # code rather than being lumped in under "implausible", which
            # would read as "still unresolved" when it isn't.
            code = "annotation-confusable" if result.kind == "annotation" and result.changed else "implausible"
            run_flags.append(Flag(code, result.warning, "warn"))

        other_boxes = [d.bbox for d in detections if d is not det]
        style = fit_style(
            image,
            result.text,
            det.bbox,
            det.angle_deg,
            other_boxes=other_boxes,
            allow_condensed=not bool(re.fullmatch(r"[0-9][0-9 .,:/\\-]*", result.text.strip())),
        )
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
    # The drawing layer every run below is composited UNDERNEATH, so no
    # re-rendered label can paint over a wall, an arc or a dimension
    # line (see render/text.py's module docstring for why the drawing
    # outranks the type). Captured once, here, before a single run is
    # drawn — not rebuilt per run — so that it describes the plan alone.
    # Rebuilding it inside the loop would fold each run's own fresh ink
    # into the "drawing" the next run hides behind, and labels would
    # start occluding each other in page order, which is neither what
    # this layer means nor stable under a reordering of `runs`.
    drawing = drawing_alpha_for(flipped)
    # The S8 QA gate's own reference for "what did this sheet look like
    # before any text was drawn" — a copy, not a view: `flipped` is
    # mutated in place by every render_run() call in the loop below, so
    # this has to be taken now or it would just be `flipped` again by
    # the time the gate runs.
    plate = flipped.copy()
    targets: list[tuple[float, float]] = []
    rendered_runs: list[RenderedRun] = []
    for run in runs:
        run_target_size = _target_size(run)
        targets.append(run_target_size)
        target_along, target_across = run_target_size
        rendered = render_run(
            canvas=flipped,
            text=run.text,
            center=run.center_out,
            angle_deg=run.angle_out,
            style=run.style,
            target_size=run_target_size,
            linework_mask=lines,
            drawing_alpha=drawing,
        )
        if rendered.shrunk:
            # The shrink-to-fit loop runs a bounded number of attempts
            # (render/text.py) and can legitimately give up still over
            # target — most often a lexicon-corrected string is longer
            # than the raw OCR read it replaced ('Vaer.' -> 'Værelse').
            # A silent "fit-shrunk" info flag looked identical whether
            # the loop converged to a 1% overshoot or gave up at 25% —
            # the second case is worth a human's attention, the first
            # is not, so the flag now says which one happened. Checked
            # on whichever axis overflows worse, not just width: the
            # loop this mirrors (render/text.py) now gives up on either
            # axis, so a run still oversized only vertically (a taller
            # font substitution, say) must be caught here too, not just
            # the width axis this flag originally shipped with.
            width_overflow = rendered.ink_width / max(1.0, target_along)
            height_overflow = rendered.ink_height / max(1.0, target_across)
            overflow = max(width_overflow, height_overflow)
            if overflow > 1.15:
                axis_word = "width" if width_overflow >= height_overflow else "height"
                run.flags.append(
                    Flag(
                        "fit-shrink-incomplete",
                        f"{run.text!r} is still {overflow:.0%} of its original {axis_word} "
                        "after the shrink-to-fit limit — may overlap neighbouring content.",
                        "warn",
                    )
                )
            else:
                run.flags.append(
                    Flag("fit-shrunk", f"{run.text!r} reduced to fit its original box.", "info")
                )
        if rendered.hidden > _MOSTLY_HIDDEN:
            # Drawing-over-type costs exactly one thing, and this is it:
            # a run whose mirrored anchor lands on solid geometry is now
            # *under* that geometry rather than punched through it. That
            # is the right trade — the plan stays intact and the label
            # is still recoverable — but it is not something to let pass
            # silently, because the label is unreadable on the output
            # sheet until a human moves it. Reported INSTEAD of
            # `collision`, not alongside it: being mostly buried
            # strictly implies touching, and two warnings about one
            # event would just make the review list harder to read.
            run.flags.append(
                Flag(
                    "hidden-behind-linework",
                    f"{run.text!r} lands on geometry that covers {rendered.hidden:.0%} of it — "
                    "the drawing is kept in front, so the label reads poorly or not at all.",
                    "warn",
                )
            )
        elif rendered.collided:
            run.flags.append(
                Flag(
                    "collision",
                    f"{run.text!r} touches linework or another label after mirroring.",
                    "warn",
                )
            )
        rendered_runs.append(rendered)

    # ---- S8: pre-save QA gate ----------------------------------------------
    if qa_gate:
        report = run_qa_gate(QAContext(
            canvas=flipped, plate=plate, source=image, axis=axis,
            runs=runs, targets=targets, rendered=rendered_runs,
            drawing_alpha=drawing, linework_mask=lines, protected=protected,
        ))
        flags.extend(v.as_flag() for v in report.corrected)
        if not report.passed:
            # Refuse rather than save — matches REFUSE_CAP_HEIGHT_PX's own
            # precedent above. No sidecar, no output file: a caller that
            # wants the diagnostics anyway has them on exc.report, and the
            # partially-rendered `flipped` array is never written to disk.
            raise QAGateFailure(report)

    if qa_gate:
        from spejl.qa.verify import check_rendered_text
        expected = []
        for run in runs:
            x0, y0, x1, y1 = M.mirror_bbox(run.bbox_src, w, h, axis)
            expected.append((run.text, (x0, y0, x1, y1)))
        for issue in check_rendered_text(flipped, expected, backend):
            flags.append(Flag("rendered-text-unconfirmed", issue, "warn"))
    write_image(output_path, flipped)

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

    Each candidate is compared against the PREVIOUS one added
    (``cluster[-1]``), not the cluster's first/smallest member —
    deliberately: anchoring to the first member was tried and reverted,
    confirmed WRONG against this project's own golden fixture, not just
    theoretically reconsidered. A long chain of individually-under-
    tolerance steps CAN drift a cluster's own min and max past the
    tolerance measured end-to-end — but on the real fixture
    ``_snap_consistent_sizes`` was built for, 11 same-kind dimension
    runs that are ALL the same size on the original drawing measure
    independently across a real, confirmed 25-30px spread (noise that
    happens to distribute smoothly across the range, not in tight,
    separated clumps) — anchoring to the first member refused to bridge
    that real 20% total spread in one step, left it as two sizes
    instead of one, and measurably broke this project's own round-trip
    fidelity test on the actual golden fixture (char_accuracy fell from
    a clean pass to 0.973, below its own 0.99 floor). Pairwise chaining
    against the immediately preceding member is what correctly re-
    collapses that real, continuously-distributed noise back to one
    size — the reason not to "fix" this into a smaller
    theoretical gap without new real evidence for that specific shape
    of failure, since this project's own real data contradicts it.
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
        tracking = solve_tracking(run.text, old.font_path, median, old.ink_along_px, old.font_variation)
        natural_tracked = font_measure_width(
            run.text, old.font_path, median, tracking * median, old.font_variation
        )
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


# Every real annotation this project has ever confirmed (H*, T, HS, EL,
# VVB, GA, ST) is a short, curated, EXACT lexicon entry. A read at or
# below this length that still comes back kind=="unknown" after the full
# closed vocabulary — rooms, dimensions, areas, annotations,
# abbreviations, even the confusable-correction fallback — has nothing
# to say about it is therefore not a new, not-yet-catalogued annotation;
# see _looks_like_a_short_unmatched_symbol's own docstring for the real
# evidence. A longer unmatched string (the existing 'Qzxwv' case) stays
# outside this net — likelier a genuinely garbled read of real text a
# human should still see rendered, not geometry to leave untouched.
_SHORT_UNKNOWN_MAX_LEN = 2


def _looks_like_a_short_unmatched_symbol(text: str, result: SnapResult) -> bool:
    """True for a short OCR read the entire lexicon has nothing to say
    about — the same failure mode :func:`_looks_like_a_graphical_symbol`
    exists to catch, for the case that check structurally cannot: a
    misread that happens to spell real alphanumeric characters, so "no
    letters or digits at all" lets it straight through.

    Confirmed on three separate real files, none of them real text at
    all: a spot-elevation marker (a small circle under a short stroke)
    read as 'O' at 0.56 confidence; an I-beam/lintel cross-section read
    as 'H' at 1.00 confidence — visual resemblance alone, not OCR being
    unsure, produced that one; two valve/knob icons read as 'GO'. Every
    one of them was, before this check, erased and re-rendered as
    garbage letters directly over real drafting geometry.
    """
    return result.kind == "unknown" and len(text) <= _SHORT_UNKNOWN_MAX_LEN


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

    A region that overflows the SOURCE's own edge (not just the
    destination's) needs one more thing: the patch is placed UNFLIPPED
    — its own internal pixel order is never reversed, only the block's
    overall position moves to the mirrored side — so a pixel's
    destination is the mirrored NOMINAL region's own start plus that
    pixel's offset from the NOMINAL region's own start, not simply "the
    mirrored box's start, unshifted" (which silently drops however much
    was trimmed off the source's leading edge and mis-registers the
    whole block by exactly that amount — confirmed: a 5px source-edge
    clamp reproducibly shifted content nowhere near either edge of the
    patch itself by that same 5px on the canvas) and not "re-mirror the
    already-clamped patch's own box" either (loses the same information
    a different way — the clamp amount is invisible to a box that has
    already been clamped).
    """
    rx0, ry0, rx1, ry1 = (int(round(v)) for v in region)
    sx0, sy0 = max(0, rx0), max(0, ry0)
    sx1, sy1 = min(source.shape[1], rx1), min(source.shape[0], ry1)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    patch = source[sy0:sy1, sx0:sx1]

    dx0, dy0, _dx1, _dy1 = (int(round(v)) for v in M.mirror_bbox(region, w, h, axis))
    dest_x0, dest_x1 = dx0 + (sx0 - rx0), dx0 + (sx1 - rx0)
    dest_y0, dest_y1 = dy0 + (sy0 - ry0), dy0 + (sy1 - ry0)

    # Clip the destination to the canvas FIRST, then shrink the patch by
    # exactly what was clipped off each side — this is what keeps a
    # region that runs off one edge from wrapping onto the other.
    cdx0, cdy0 = max(0, dest_x0), max(0, dest_y0)
    cdx1 = min(canvas.shape[1], dest_x1)
    cdy1 = min(canvas.shape[0], dest_y1)
    if cdx1 <= cdx0 or cdy1 <= cdy0:
        return
    clip_left = cdx0 - dest_x0
    clip_top = cdy0 - dest_y0
    canvas[cdy0:cdy1, cdx0:cdx1] = patch[
        clip_top : clip_top + (cdy1 - cdy0), clip_left : clip_left + (cdx1 - cdx0)
    ]
