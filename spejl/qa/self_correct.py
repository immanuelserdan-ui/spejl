"""Stage S8 — the pre-save gate: check the finished sheet against Spejl's
non-negotiable invariants, correct what can be corrected locally, and
refuse to save what can't.

Every other stage in this pipeline already enforces its own piece of
these rules AT THE POINT of doing the work — S7 anchors on measured ink,
shrinks a run that overflows its box, composites the drawing in front of
type. This stage does not re-decide any of that. It is the thing that
looks at the FINISHED, fully-composited sheet and asks, independently of
whether every upstream stage did its job correctly: does the actual
output, as bytes, satisfy the invariant? That distinction matters for
two reasons. A future change to an upstream stage that quietly breaks
one of these guarantees is caught here even if nobody remembers to
update this module to match. And some things are only checkable once
every run has been drawn — a page-wide geometry comparison against the
source, or a "does layer ordering hold everywhere" sweep, aren't
naturally expressed as a per-run check inside the S7 loop.

The rules, and how each is actually decided rather than asserted:

* **Spatial integrity** — every dark pixel belonging to the source
  drawing (not text) must survive, at its mirrored position, in the
  output. Checked by flipping the source and diffing it against the
  output outside every run's own footprint. This rule has no corrector
  (see ``_CORRECTORS``): if drawing geometry is missing from a rendered
  sheet, the one thing this stage KNOWS about what should be there is
  already gone from every input it has — there is no "correct" value to
  invent, only a source to escalate back to render/erase and re-run.
  Attempting to synthesize missing geometry here would be the QA gate
  quietly failing DIFFERENTLY — a wrong picture instead of a caught bug.
* **Layer ordering** — every fully-solid drawing pixel (per
  ``render.text.drawing_alpha_for``) must equal what the plate held
  before any text was drawn there. This IS correctable, and cheaply: the
  plate is the authoritative pre-text geometry, so a violation is fixed
  by restoring the offending pixels from it — a repair that, by
  construction, can only ever put geometry back, never invent any.
* **Text fidelity** — two independently checked things. Size lock: a
  run's rendered cap height must land within a tolerance BAND of its
  measured target, not just under a ceiling (S7's own shrink-to-fit
  guard only stops a run from growing past its box; nothing upstream
  stops the SAME loop from legitimately shrinking a run considerably to
  make it fit, which is a successful shrink by S7's own rule and still
  a real fidelity loss by this one). Font coverage: every character in
  a run's own corrected text must have a real glyph in the resolved
  font — checked against the font's own cmap via fontTools, not by
  rendering and looking for ink (see ``_font_covers`` for why that
  naive approach was tried and is wrong). Both are correctable within
  bounded, local retries — see ``_correct_text_fidelity``.
* **Clearance** — a ROOM NAME must never overlap solid drawing geometry
  at all (any nonzero ``RenderedRun.hidden``, not just the >50%
  ``_MOSTLY_HIDDEN`` threshold render/text.py's own soft warning uses).
  Deliberately scoped to ``kind == "room"`` and nothing else. This is
  the second attempt at a proximity-style rule in this codebase — the
  first (see render/text.py's own "proximity-based collision check"
  comment, near its module docstring) was reverted after a fixed
  pixel-distance margin flagged the clear majority of correctly-placed
  DIMENSION numbers, which routinely and legitimately sit on or beside
  their own dimension/witness line by drafting convention.

  Checked directly against the real project corpus before shipping
  this one, not just reasoned about: a dimension number can
  legitimately sit directly on solid wall poché too — '3306' (three
  separate real files) labels a diagonal wall's own length by sitting
  squarely on it, 25-31% of its own ink hidden, an apparently common
  convention for a wall segment with no room for an offset witness
  line. A room name has no equivalent convention. Every one of
  hundreds of correctly-placed room labels across that same corpus
  measured exactly zero on this signal, and the one genuine exception
  (a real, shipped user file — see ``_correct_clearance``) was a label
  sitting close enough to its own wall for a corner to clip it, which
  room-name placement never has a legitimate reason to do. Correctable
  within a few bounded nudges away from the geometry it overlaps.
* **OCR confidence** — no run may carry an unresolved low-confidence,
  implausible, or confusable-substitution flag (S3/S4's own
  ``"low-confidence"``, ``"implausible"``, ``"annotation-confusable"``
  codes — see raster/pipeline.py and lexicon/snap.py). Split two ways,
  deliberately, not treated as one block-or-don't decision:

  ``"low-confidence"`` and ``"annotation-confusable"`` ARE auto-
  accepted (see ``_correct_ocr_confidence``) — in both cases S3/S4 has
  already settled on the one string it's going to render, this stage
  is only deciding whether to trust that already-made call, and there
  is no second candidate to weigh it against. Accepting it is accepting
  the pipeline's own best (and only) answer, the same category of move
  every other corrector in this module makes — not a guess invented
  here. ``"implausible"`` stays refused: it means NO lexicon entry, no
  fold, no fuzzy hit, no confusable substitution matched AT ALL — there
  is no "pipeline's own answer" behind that flag to accept, only the
  raw OCR string with nothing grounding it. That is a materially
  different, riskier act than the other two, on what may be a real
  construction document, so it keeps escalating to a human the same as
  spatial integrity does: there is no geometry to nudge or box to widen
  that fixes not knowing what a character actually was, and this stage
  will not invent an answer it has no basis for. An accepted flag is
  renamed, not deleted, so it stays visible on the saved file's own
  Review panel — accepted is not the same as never having been a close
  call, and a human should still be able to see which reads were.

What this stage deliberately does NOT do: re-run OCR against the
rendered output to confirm a string reads back correctly. That check is
real and already exists — it is exactly what
``tests/test_raster_pipeline.py``'s round-trip test does — but it loads
an OCR model and runs three detection passes, which is a fine cost for
a test suite and a bad one to pay on every single save. This gate
checks what can be verified from geometry and font data alone, in
milliseconds, on every mirror; the expensive full-fidelity check stays
where it already lives.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from spejl.models import Axis, Flag
from spejl.render.text import RenderedRun, render_run
from spejl.style.metrics import _BOLD_CANDIDATES, _FONT_CANDIDATES, TextStyle
from spejl.transform import mirror as M


class Rule(str, Enum):
    SPATIAL_INTEGRITY = "spatial-integrity"
    LAYER_ORDERING = "layer-ordering"
    TEXT_FIDELITY = "text-fidelity"
    CLEARANCE = "clearance"
    OCR_CONFIDENCE = "ocr-confidence"


@dataclass(frozen=True)
class Violation:
    """One failed check. ``measured``/``threshold`` are whatever number
    the check actually compared, and ``kind`` whichever specific test
    within the rule fired — kept as real fields, not folded into the
    message string, so a corrector or a test can act on them directly
    rather than re-parsing English out of a sentence meant for a human.
    ``kind`` matters concretely for TEXT_FIDELITY, which packs two
    independent checks (size lock, font coverage) under one rule —
    without it, picking the right correction would mean pattern-matching
    the message text, which breaks silently the moment the wording
    changes without both call sites being updated together."""

    rule: Rule
    run_index: int | None  # None = a page-level violation, not one run's fault
    kind: str  # e.g. "geometry", "layer-order", "size-lock", "font-coverage"
    message: str
    measured: float
    threshold: float
    correctable: bool

    def as_flag(self, severity: str = "warn") -> Flag:
        return Flag(code=f"qa-{self.rule.value}", message=self.message, severity=severity)


@dataclass
class QAReport:
    passed: bool
    iterations: int
    violations: list[Violation]  # the state of things on the LAST pass
    corrected: list[Violation] = field(default_factory=list)  # fixed along the way
    history: list[list[Violation]] = field(default_factory=list)  # every pass, for audit


class QAGateFailure(ValueError):
    """Raised instead of saving — matches raster/pipeline.py's own
    REFUSE_CAP_HEIGHT_PX precedent: refuse rather than produce a
    plausible-looking wrong drawing."""

    def __init__(self, report: QAReport) -> None:
        self.report = report
        lines = "\n".join(f"  - [{v.rule.value}] {v.message}" for v in report.violations)
        super().__init__(
            f"QA gate failed after {report.iterations} pass(es) with "
            f"{len(report.violations)} unresolved violation(s):\n{lines}"
        )


class RunLike(Protocol):
    """Structural type for raster/pipeline.py's MirroredRun — named here
    rather than imported, to keep this module free of a circular import
    (pipeline.py is the caller of this one)."""

    text: str
    kind: str  # "room" | "dimension" | "annotation" | "area" | "unknown" — see lexicon/snap.py
    center_out: tuple[float, float]
    angle_out: float
    bbox_src: tuple[float, float, float, float]
    style: TextStyle
    flags: list[Flag]  # S3/S4's own OCR-confidence flags — see check_ocr_confidence


@dataclass
class QAContext:
    """Everything a check or a corrector needs, gathered once. Mutated
    in place by correctors — ``canvas`` and ``rendered`` are the live
    state the loop re-checks on every pass.

    Two different coordinate spaces travel together here, and every
    field belongs to exactly one of them:

    * ``canvas``, ``plate``, ``drawing_alpha`` and ``linework_mask`` are
      all in MIRRORED (output) space — the same space every run's own
      ``center_out`` is anchored in. ``canvas`` is ``plate`` plus every
      run's own ink; restoring a pixel of ``canvas`` FROM ``plate`` (see
      ``_correct_layer_ordering``) is therefore a same-space copy, no
      flip involved.
    * ``source`` is the ONE field in pre-mirror space — the original
      image, untouched. Comparing it against anything in mirrored space
      (see ``check_spatial_integrity``) needs ``transform.mirror.
      flip_image`` first. Building ``source`` FROM an already-corrupted
      ``plate`` (instead of from an independent, still-correct copy)
      quietly turns a spatial-integrity loss into something that reads
      as agreeing with itself — get this backwards and the rule that is
      supposed to have no corrector stops firing at all, silently.
    """

    canvas: np.ndarray  # the fully-composited sheet — mutated by correctors
    plate: np.ndarray  # the type-free, flipped canvas, BEFORE any S7 run was drawn
    source: np.ndarray  # the original, pre-mirror image — ground truth for geometry
    axis: Axis
    runs: list[RunLike]
    targets: list[tuple[float, float]]  # (along, across) target ink size, one per run
    rendered: list[RenderedRun]  # last render_run() result, one per run
    drawing_alpha: np.ndarray
    linework_mask: np.ndarray
    # Same pre-mirror (source) coordinate space and scale as `source`
    # itself — north arrows, scale bars, title blocks — pasted back
    # UNFLIPPED at their mirrored position (raster/pipeline.py's own
    # _replace_unmirrored), which is a deliberate, sanctioned exception
    # to "the output is a plain mirror of the source" and must be
    # excluded from check_spatial_integrity the same way a text run's
    # footprint is, or a correctly-protected region reads as 100%
    # missing geometry on every single mirror that has one.
    protected: list[tuple[float, float, float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule 1 — spatial integrity
# ---------------------------------------------------------------------------

# Padding around a run's own mirrored footprint excluded from the
# geometry comparison. Not zero: a label's box is a detection estimate,
# and the erase/re-render boundary sits a few px outside it (see
# erase/clean.py's own dilate_px) — comparing right up to the box's exact
# edge would flag the ORDINARY, expected difference at that boundary
# (old glyph pixels replaced by new ones, or paper replaced by ink) as
# missing drawing. 12px is comfortably past erase/clean.py's own 3px
# dilation plus a few px of antialiasing margin either side.
_FOOTPRINT_PAD_PX = 12


def _run_footprint(run: RunLike, target: tuple[float, float]) -> tuple[int, int, int, int]:
    """The RE-RENDERED run's own footprint, in output space — what a
    correction needs to know to erase-and-redraw (see _redraw_run).
    Deliberately NOT what check_spatial_integrity excludes on its own;
    see _erased_source_footprint for why those are two different boxes.
    """
    cx, cy = run.center_out
    along, across = target
    half_w, half_h = (along, across) if abs(run.angle_out) <= 45 else (across, along)
    p = _FOOTPRINT_PAD_PX
    return (
        int(cx - half_w / 2 - p), int(cy - half_h / 2 - p),
        int(cx + half_w / 2 + p), int(cy + half_h / 2 + p),
    )


# Matches erase.clean.erase_text's own default dilate_px — the box the
# S5 erase stage actually wipes is the raw OCR detection box (bbox_src),
# floor/ceil'd and padded by this many pixels, NOT the tighter measured-
# ink extent _run_footprint above is built from. The two routinely
# differ: fit_style/measure_ink_extent deliberately filter OUT padding,
# dash-noise and non-glyph content from a raw detection box to get an
# accurate SIZE to re-render at (see style/metrics.py), which is exactly
# right for that purpose and exactly wrong as a stand-in for "what did
# S5 actually erase" — the erased area is bigger, by design and by a
# confirmed amount, not a rounding difference.
_ERASE_DILATE_PX = 3


def _erased_source_footprint(run: RunLike, w: int, h: int, axis: Axis) -> tuple[int, int, int, int]:
    """The box S5 actually wiped, mapped into output space — what
    check_spatial_integrity needs to exclude, and _run_footprint above
    does not cover.

    Confirmed as a real, reachable false positive, not a theoretical
    one: on a genuine real plan (not a synthetic fixture), erasing
    '4504' correctly removed a couple of source pixels that sat inside
    its own raw, dilated OCR box but outside its measured-ink footprint
    — line_pixel_mask's own repair (erase/clean.py) only restores a
    component that extends past the box by _MIN_LINE_EXTENSION_PX,
    deliberately, so a short mark entirely inside the box (here: a
    stray couple of source pixels, not real structural linework) is not
    restored, by design. Checking the erased sheet against the
    measured-ink-only exclusion then read the correctly-erased pixels
    as "missing drawing geometry" — a false alarm about the CHECK's own
    footprint, not a real loss: the S5 stage behaved exactly as
    documented, it was the S8 gate that hadn't been told what S5
    actually touches.
    """
    mirrored = M.mirror_bbox(run.bbox_src, w, h, axis)
    x0, y0, x1, y1 = mirrored
    p = _ERASE_DILATE_PX + 4  # + antialiasing margin, matching _FOOTPRINT_PAD_PX's own reasoning
    return (int(x0 - p), int(y0 - p), int(x1 + p), int(y1 + p))


def check_spatial_integrity(ctx: QAContext) -> list[Violation]:
    """Every drawing pixel on the source must survive, mirrored, in the
    output — nothing added, moved, or lost that a plain reflection
    wouldn't already explain."""
    h, w = ctx.canvas.shape[:2]
    flipped_source = M.flip_image(ctx.source, ctx.axis)
    if flipped_source.shape[:2] != (h, w):
        # An upscale happened between source and canvas (raster/pipeline.py's
        # own S1) — this check needs matching scales, and pipeline.py is
        # responsible for calling it with a source already resized to
        # match; a mismatch here means it wasn't, which is a caller bug,
        # not a mirroring one, so surface it plainly rather than silently
        # comparing misaligned arrays.
        return [Violation(
            Rule.SPATIAL_INTEGRITY, None, "scale-mismatch",
            f"source/canvas size mismatch ({flipped_source.shape[:2]} vs {(h, w)}) — "
            "spatial integrity cannot be checked against an unscaled source.",
            0.0, 0.0, correctable=False,
        )]

    source_gray = cv2.cvtColor(flipped_source, cv2.COLOR_BGR2GRAY)
    canvas_gray = cv2.cvtColor(ctx.canvas, cv2.COLOR_BGR2GRAY)

    exclude = np.zeros((h, w), bool)
    for run, target in zip(ctx.runs, ctx.targets):
        # The union of both: what got RE-RENDERED on top (_run_footprint)
        # and what S5 actually ERASED from the source to make room for it
        # (_erased_source_footprint) — two different boxes (see the
        # latter's own docstring for a confirmed, reachable case where
        # they disagree), and either one alone under-excludes.
        for x0, y0, x1, y1 in (
            _run_footprint(run, target),
            _erased_source_footprint(run, w, h, ctx.axis),
        ):
            exclude[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    for region in ctx.protected:
        x0, y0, x1, y1 = (int(v) for v in M.mirror_bbox(region, w, h, ctx.axis))
        p = _FOOTPRINT_PAD_PX
        exclude[max(0, y0 - p):min(h, y1 + p), max(0, x0 - p):min(w, x1 + p)] = True

    was_dark = (source_gray < 128) & ~exclude
    still_dark = canvas_gray < 128
    missing = was_dark & ~still_dark
    missing_px = int(np.count_nonzero(missing))
    if missing_px == 0:
        return []
    return [Violation(
        Rule.SPATIAL_INTEGRITY, None, "geometry",
        f"{missing_px}px of source drawing geometry is missing from the mirrored sheet "
        "outside any text run's own footprint.",
        float(missing_px), 0.0, correctable=False,
    )]


# ---------------------------------------------------------------------------
# Rule 2 — layer ordering
# ---------------------------------------------------------------------------


def check_layer_ordering(ctx: QAContext) -> list[Violation]:
    """Every fully-solid drawing pixel must still hold the plate's own
    value — render.text.render_run already guarantees this at the point
    each run is drawn (see its own module docstring); this is the
    independent, whole-page audit that the guarantee actually held,
    which is what catches a future code path that bypasses it."""
    solid = ctx.drawing_alpha >= 0.999
    if not solid.any():
        return []
    mismatch = np.any(ctx.canvas[solid] != ctx.plate[solid], axis=-1)
    bad_px = int(np.count_nonzero(mismatch))
    if bad_px == 0:
        return []
    return [Violation(
        Rule.LAYER_ORDERING, None, "layer-order",
        f"{bad_px}px of solid drawing geometry was overwritten by re-rendered text.",
        float(bad_px), 0.0, correctable=True,
    )]


def _correct_layer_ordering(ctx: QAContext, violation: Violation) -> bool:
    """Restore every offending pixel from the plate — the plate is
    authoritative pre-text geometry, so this can only ever put drawing
    back, never invent any."""
    solid = ctx.drawing_alpha >= 0.999
    mismatch = solid & np.any(ctx.canvas != ctx.plate, axis=-1)
    if not mismatch.any():
        return False
    ctx.canvas[mismatch] = ctx.plate[mismatch]
    return True


# ---------------------------------------------------------------------------
# Rule 3 — text fidelity
# ---------------------------------------------------------------------------

# A run's rendered cap height must land within this band of its own
# measured target.
#
# The ceiling was FIRST set to render/text.py's own OVERFLOW_TOLERANCE
# (1.04), on the reasoning that the guard already stops growth past it
# so re-stating it here would just be a belt-and-braces audit — wrong,
# confirmed immediately against the golden fixture: '2900' and '3200'
# legitimately render at 105% with no shrink even attempted, because
# 1.04 is render_run's threshold for WHEN TO START shrinking, not a
# ceiling real output never crosses — a render landing at 104.6% is
# unremarkable, not a defect, and OVERFLOW_TOLERANCE was never meant to
# bound it. raster/pipeline.py already draws the line this stage should
# reuse instead: the SAME overflow measurement escalates to a human
# past 115%, not 104% (see its own "fit-shrink-incomplete" vs
# "fit-shrunk" split) — matching that is what "past this ceiling is
# genuinely a problem" means elsewhere in this codebase, and this
# stage's ceiling now matches it exactly rather than inventing a
# stricter, untested one of its own.
#
# The FLOOR is the genuinely new check: S7's own shrink-to-fit loop has
# no lower bound on how far it compresses a run to make it fit (only a
# STILL-OVERSIZED run past 115% escalates today; a run successfully
# squeezed down to, say, 60% of its target height is "fit-shrunk", not
# flagged at all) — which is exactly the "squishing" the height-lock
# rule this stage enforces is about. Unlike the ceiling, 0.85 is NOT yet
# backed by a confirmed real-plan case the way _MIN_LINE_EXTENSION_PX is
# — there is no golden-fixture failure yet where legitimate shrink-to-
# fit crosses it. Treat it as a reasoned starting point pending real
# evidence, not a proven boundary.
_FIDELITY_HEIGHT_BAND = (0.85, 1.15)


@lru_cache(maxsize=32)
def _font_cmap(font_path: str) -> frozenset[int]:
    """Codepoints this font actually maps to a glyph.

    Deliberately NOT "render the character and check for ink" — tried
    first, and wrong: PIL/FreeType renders a missing glyph as a visible
    fallback box (the classic "tofu" glyph), which has the SAME nonzero
    pixel count as a real glyph would. Confirmed on the resolved system
    font: a CJK and a Thai codepoint neither Arial nor Liberation Sans
    actually supports both rendered a 104-pixel box, indistinguishable
    by ink alone from any real short glyph — a render-and-measure check
    would have silently reported both as "covered". The font's own cmap
    table is the one place "does this font have a real glyph for this
    codepoint" is actually recorded.

    ``fontTools`` is imported HERE, not at module level, deliberately:
    this is the only function in the whole S8 gate that needs it. An
    unconditional top-of-file import made it a hard dependency of
    `spejl.raster.pipeline` itself — confirmed by actually simulating a
    missing install: the entire raster pipeline (detect, erase, render,
    every stage that has nothing to do with the QA gate) failed to
    IMPORT, not just to run the font check, and `qa_gate=False` gave no
    way around it either, since the failure happened before any caller
    code ever runs. Deferred here, only `check_text_fidelity` (and
    therefore only a `qa_gate=True` run) ever needs `fontTools`
    installed at all.
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError as exc:
        raise RuntimeError(
            "Font glyph-coverage checking needs the 'fonttools' package "
            "(pip install fonttools, or the 'raster' extra) — Spejl will "
            "not silently skip the check and call unverified text fidelity "
            "clean."
        ) from exc

    tt = TTFont(font_path, fontNumber=0, lazy=True)
    try:
        return frozenset(tt.getBestCmap() or {})
    finally:
        tt.close()


def _font_covers(font_path: str, text: str) -> str:
    """Characters in ``text`` the font at ``font_path`` cannot render —
    empty if every one is covered."""
    cmap = _font_cmap(font_path)
    return "".join(sorted({ch for ch in text if not ch.isspace() and ord(ch) not in cmap}))


def check_text_fidelity(ctx: QAContext) -> list[Violation]:
    violations: list[Violation] = []
    for i, (run, target, rendered) in enumerate(zip(ctx.runs, ctx.targets, ctx.rendered)):
        _along, across = target
        if across > 0:
            ratio = rendered.ink_height / across
            lo, hi = _FIDELITY_HEIGHT_BAND
            if not (lo <= ratio <= hi):
                violations.append(Violation(
                    Rule.TEXT_FIDELITY, i, "size-lock",
                    f"{run.text!r} rendered at {ratio:.0%} of its measured cap height "
                    f"(band is {lo:.0%}-{hi:.0%}) — reads as squished or oversized.",
                    ratio, lo if ratio < lo else hi, correctable=True,
                ))

        missing = _font_covers(run.style.font_path, run.text)
        if missing:
            violations.append(Violation(
                Rule.TEXT_FIDELITY, i, "font-coverage",
                f"{run.text!r} has no glyph for {missing!r} in the resolved font "
                f"({run.style.font_path}) — would render as a missing-glyph box.",
                0.0, 1.0, correctable=True,
            ))
    return violations


def _candidate_fonts(bold: bool) -> tuple[str, ...]:
    return _BOLD_CANDIDATES + _FONT_CANDIDATES if bold else _FONT_CANDIDATES


def _redraw_run(
    ctx: QAContext, index: int, style: TextStyle, target_size: tuple[float, float]
) -> RenderedRun:
    """Erase run ``index``'s own current footprint back to bare plate,
    then draw it again with ``style`` against ``target_size`` — the one
    place every text-fidelity correction below actually touches the
    canvas.

    The linework/collision mask is NOT similarly rewound: render_run
    mutates it in place by design (see its own docstring), stamping
    this run's ink in at whatever footprint it drew at, and a
    correction can move or resize that footprint. The mask is used here
    only to avoid drawing fresh ink over already-placed ink, so a stale
    stamp at the run's PREVIOUS footprint is a narrow, acknowledged
    imprecision — it can only make a later run's own collision check
    slightly more conservative, never less, and collision is not one of
    the three invariants this gate enforces.
    """
    run = ctx.runs[index]
    x0, y0, x1, y1 = _run_footprint(run, ctx.targets[index])
    h, w = ctx.canvas.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        ctx.canvas[y0:y1, x0:x1] = ctx.plate[y0:y1, x0:x1]

    run.style = style  # MirroredRun is a plain mutable dataclass — see raster/pipeline.py
    rendered = render_run(
        canvas=ctx.canvas,
        text=run.text,
        center=run.center_out,
        angle_deg=run.angle_out,
        style=style,
        target_size=target_size,
        linework_mask=ctx.linework_mask,
        drawing_alpha=ctx.drawing_alpha,
    )
    ctx.rendered[index] = rendered
    return rendered


def _correct_text_fidelity(ctx: QAContext, violation: Violation) -> bool:
    """Two distinct, bounded local retries, picked by which check
    actually failed. Neither loops indefinitely: each tries a handful
    of concrete alternatives and reports failure rather than spin if
    none of them clears the check — matching resolve_font's own refusal
    to silently fall back to something worse."""
    i = violation.run_index
    if i is None:
        return False
    run = ctx.runs[i]

    if violation.kind == "font-coverage":
        # Cycle to the next candidate font that actually covers this
        # run's own text. A real font swap, not a style tweak — a run
        # that needs a glyph the current font lacks has no fix that
        # keeps the current font.
        bold = run.style.font_path in _BOLD_CANDIDATES
        for candidate in _candidate_fonts(bold):
            if not Path(candidate).exists() or _font_covers(candidate, run.text):
                continue
            # font_variation cleared, not carried over: it names a
            # variable-font instance specific to whatever font_path
            # this run was on before (e.g. Bahnschrift's "Condensed" —
            # see style/metrics.py's _CONDENSED_FONT_CANDIDATES), and
            # `candidate` here is a completely different font with no
            # reason to share that instance name — PIL's own
            # set_variation_by_name would raise on a font that doesn't
            # have it.
            new_style = replace(run.style, font_path=candidate, font_variation=None)
            _redraw_run(ctx, i, new_style, ctx.targets[i])
            return True
        return False  # no available font covers this text — not locally fixable

    # Otherwise: a size-lock violation. Re-render against a taller
    # effective target so render_run's own shrink-to-fit loop has less
    # reason to compress — bounded to a few widened attempts, not an
    # open search.
    _along, across = ctx.targets[i]
    lo, hi = _FIDELITY_HEIGHT_BAND
    for relax in (1.08, 1.16, 1.25):
        relaxed = (ctx.targets[i][0] * relax, ctx.targets[i][1] * relax)
        rendered = _redraw_run(ctx, i, run.style, relaxed)
        ratio = rendered.ink_height / across if across > 0 else 1.0
        if lo <= ratio <= hi:
            return True
    return False  # every relaxation tried and the run still won't sit in-band


# ---------------------------------------------------------------------------
# Rule 4 — clearance (room names only — see the module docstring for why
# this is scoped this tightly, and what was tried and rejected first)
# ---------------------------------------------------------------------------


def check_clearance(ctx: QAContext) -> list[Violation]:
    violations: list[Violation] = []
    for i, (run, rendered) in enumerate(zip(ctx.runs, ctx.rendered)):
        if run.kind == "room" and rendered.hidden > 0:
            violations.append(Violation(
                Rule.CLEARANCE, i, "room-overlap",
                f"{run.text!r} overlaps drawing geometry ({rendered.hidden:.0%} of its own ink "
                "sits behind solid linework) — a room name should never touch a wall.",
                rendered.hidden, 0.0, correctable=True,
            ))
    return violations


# A room label's own centre nudged this many pixels, perpendicular to its
# reading direction first (the axis a wall it's brushing is most likely
# to sit across), then along it, each tried in both directions — bounded,
# not a search: four small, mechanical steps, matching the same
# "deterministic retry over a fixed set of alternatives" discipline every
# other corrector in this module already follows. Confirmed sufficient
# on the one real, shipped case this rule was built from (a 0.12%
# overlap — a hairline clip, not a deep intrusion) without needing a
# targeted, overlap-centroid-aware nudge direction; a future case this
# doesn't resolve escalates rather than searching further, the same as
# every other bounded corrector here.
_CLEARANCE_NUDGE_PX = (4, 8, 14, 22)


def _ink_footprint(run: RunLike, rendered: RenderedRun) -> tuple[int, int, int, int]:
    """A run's own actual rendered ink extent, in output space — the
    real thing a NEIGHBOUR's nudge candidate must stay clear of.

    Deliberately tighter than ``_run_footprint``'s padded erase box:
    that padding is this run's OWN erase margin (room for
    _FOOTPRINT_PAD_PX of antialiasing around ITS ink), not extra
    territory a neighbour is barred from approaching. Reusing the
    padded box for both sides made the clearance corrector's own
    bounding-box check reject every one of the 16 bounded nudges tried
    for the golden fixture's own 'Bad' — it sits roughly 40-49px from
    its nearest neighbours ('1400', '870'), comfortably clear of their
    actual ink, but well inside the ~54px two padded 12px-margin boxes
    would each need to stay apart, so the check was rejecting nudges
    that never touched anything.
    """
    cx, cy = run.center_out
    w, h = rendered.ink_width, rendered.ink_height
    half_w, half_h = (w, h) if abs(run.angle_out) <= 45 else (h, w)
    p = 2  # antialiasing margin only — matches _erased_source_footprint's own
    return (
        int(cx - half_w / 2 - p), int(cy - half_h / 2 - p),
        int(cx + half_w / 2 + p), int(cy + half_h / 2 + p),
    )


def _correct_clearance(ctx: QAContext, violation: Violation) -> bool:
    """Try every bounded nudge, keep whichever left the LEAST overlap —
    not the first that reaches exactly zero, and never the original,
    worst position.

    That distinction is not cosmetic: confirmed on the golden fixture's
    own 'Bad', a real case this rule's own first version got wrong. Its
    best reachable nudge cut overlap from 9% to 0.65% (still nonzero —
    a corner brushing a doorway threshold, not a clean miss) — a real,
    worthwhile improvement no candidate reduced to a clean 0. Returning
    False on "didn't reach exactly zero" alone made every one of those
    16 genuinely-improving renders get thrown away for the ORIGINAL,
    worst one, on the reasoning that if it's still going to report a
    violation, it might as well be un-nudged — exactly backwards: a
    label picked to minimise real overlap is a strictly better thing to
    have saved, or to show a human for review, than the one picked to
    maximise it.

    Rejecting a candidate on ``RenderedRun.collided`` alone was tried
    next and is ALSO wrong, confirmed the same way: ``collided`` is any
    overlap with ``linework_mask``, and that mask is seeded with the
    sheet's own real geometry before a single run is ever drawn onto it
    (see ``render_run``'s own docstring) — so it is True the instant a
    label so much as brushes a wall, which for a room name pinned in a
    tight corner is every candidate in the bounded set, 'Bad' included.
    Gating on it made the search reject its own best-known improvement
    (0.65% hidden) right along with the genuinely bad ones, landing back
    on the original 9% with nothing tried. `hidden` (measured against
    ``drawing_alpha``, the sheet's real geometry alone, never mutated by
    any run) is the wall signal this corrector is trying to shrink, and
    is allowed to land anywhere from 9% down to 0 — a hairline clip is
    exactly the kind of partial win this function exists to keep.

    What genuinely needs rejecting is a candidate whose footprint reaches
    another run's own territory at all — not just an ink-level overlap.
    Confirmed on the same fixture, a DIFFERENT bug than the one above:
    even after gating on ink collision alone (checked against a mask of
    other runs' already-stamped ink, excluding real geometry),
    'Entre'/'1400' still broke on the golden fixture's own round-trip
    ('1400' read back as '400'). The cause is `_redraw_run` itself, not
    this search: it erases its run's CURRENT footprint back to bare
    plate before drawing (see its own docstring), unconditionally, over
    the whole padded box — so a nudge that lands close enough for its
    own footprint to overlap a neighbour's, even without either run's
    actual ink ever touching, silently wipes the neighbour's ink where
    the two boxes intersect. A subsequent pixel-collision check can't
    catch this after the fact, because by the time it runs the damage
    (erasing bytes of a neighbour's glyph) is already done. The only
    safe fix is to never call `_redraw_run` for a candidate whose
    footprint reaches another run's footprint in the first place —
    `_footprint_clear_of_other_runs` below is a plain bounding-box
    check, computed BEFORE any canvas mutation, exactly for that.
    """
    i = violation.run_index
    if i is None:
        return False
    run = ctx.runs[i]
    cx, cy = run.center_out
    perpendicular = abs(run.angle_out) <= 45  # horizontal reading -> nudge vertically first

    # Snapshots, restored before EVERY trial candidate, not just before
    # the final pick — two separate accumulation bugs this guards
    # against, both confirmed while building this corrector, not
    # theoretical:
    #
    # * render_run stamps each attempt's own ink into linework_mask (by
    #   design — see its own docstring), so without restoring it, an
    #   EARLIER trial this same search already rejected leaves ghost
    #   ink behind that can wrongly fail a LATER, genuinely clear
    #   candidate's own collision check.
    # * _redraw_run only clears the footprint at the run's CURRENT
    #   center back to plate before drawing — never wherever a PREVIOUS
    #   trial in this same search left its own ink, since that sits at
    #   a different position now that center_out has moved again.
    #   Without restoring the canvas too, every rejected trial position
    #   stays visible on the final saved sheet as a stray, uncleared
    #   copy of the glyph — confirmed: exactly this scattered a handful
    #   of ghost 'Bad's across the golden fixture before this fix.
    mask_before_search = ctx.linework_mask.copy()
    canvas_before_search = ctx.canvas.copy()

    # This run's OWN starting footprint, erased to bare plate in the
    # restore baseline itself — not just left for whichever candidate's
    # own _redraw_run happens to erase. _redraw_run only clears the box
    # at the position it is ABOUT to draw (see its own docstring): a
    # nudge that only partly overlaps where this run started (any step
    # smaller than its own footprint) leaves the non-overlapping sliver
    # of the ORIGINAL glyph sitting on the canvas, uncleared, right next
    # to the freshly-drawn one. Confirmed as a real, reachable bug, not
    # a theoretical one: on the golden fixture, nudging 'Entre' 8px left
    # left an 8px-wide strip of its own original 'e' behind at the old
    # right edge, which OCR read back as a stray extra letter
    # ('Entre' -> 'Entrel' -> mis-snapped to 'Entré').
    h, w = ctx.canvas.shape[:2]
    sx0, sy0, sx1, sy1 = _run_footprint(run, ctx.targets[i])
    sx0, sy0 = max(0, sx0), max(0, sy0)
    sx1, sy1 = min(w, sx1), min(h, sy1)
    canvas_before_search[sy0:sy1, sx0:sx1] = ctx.plate[sy0:sy1, sx0:sx1]

    def _restore() -> None:
        ctx.linework_mask[:] = mask_before_search
        ctx.canvas[:] = canvas_before_search

    # Every OTHER run's own actual ink extent, at its CURRENT (already-
    # final — QA only ever runs after every run in the page has been
    # placed) position — computed once, up front, since none of them
    # move while this run's own search runs. Deliberately the tight
    # _ink_footprint, not _run_footprint's own padded erase box — see
    # _ink_footprint's docstring for why the padded box on BOTH sides
    # rejects every candidate for a label with ordinary, safe clearance
    # from its neighbours.
    other_footprints = [
        _ink_footprint(other, ctx.rendered[j]) for j, other in enumerate(ctx.runs) if j != i
    ]

    def _footprint_clear_of_other_runs(box: tuple[int, int, int, int]) -> bool:
        # box is THIS run's padded erase box (what _redraw_run will
        # actually wipe back to plate) — checked against every other
        # run's tight ink box, not another padded one, so the check
        # reflects what would really get erased, not two safety margins
        # stacked against each other.
        bx0, by0, bx1, by1 = box
        for ox0, oy0, ox1, oy1 in other_footprints:
            if bx0 < ox1 and ox0 < bx1 and by0 < oy1 and oy0 < by1:
                return False
        return True

    best_hidden = violation.measured
    best_offset = (0, 0)
    for step in _CLEARANCE_NUDGE_PX:
        offsets = (
            [(0, -step), (0, step), (-step, 0), (step, 0)]
            if perpendicular else
            [(-step, 0), (step, 0), (0, -step), (0, step)]
        )
        for dx, dy in offsets:
            run.center_out = (cx + dx, cy + dy)
            if not _footprint_clear_of_other_runs(_run_footprint(run, ctx.targets[i])):
                continue
            _restore()
            rendered = _redraw_run(ctx, i, run.style, ctx.targets[i])
            if rendered.hidden < best_hidden:
                best_hidden, best_offset = rendered.hidden, (dx, dy)
            if rendered.hidden == 0:
                return True  # canvas/mask already reflect this winning trial

    _restore()
    run.center_out = (cx + best_offset[0], cy + best_offset[1])
    _redraw_run(ctx, i, run.style, ctx.targets[i])
    # Reports "did this help", not "is it fully fixed" — the latter is
    # what the next pass's check_clearance re-verifies. A genuine (if
    # partial) reduction is worth another pass: it starts the SAME
    # bounded nudge search again from this new, already-better
    # position, which can reach a clean zero across a couple of passes
    # that no single pass's fixed nudge set would have reached alone.
    # Reporting "resolved" here on ANY improvement, instead, would let
    # the outer loop accept the sheet with real overlap still on it.
    return best_hidden < violation.measured


# ---------------------------------------------------------------------------
# Rule 5 — OCR confidence
# ---------------------------------------------------------------------------

# S3/S4's own flag codes for "OCR was not sure this is what the source
# actually says" (raster/pipeline.py's LOW_CONFIDENCE / non-text-symbol
# checks, lexicon/snap.py's confusable-annotation fallback). Deliberately
# NOT including "non-text-symbol": that code means the opposite of
# uncertain content being shipped — it is OCR correctly recognising a mark
# as NOT text and leaving it untouched as geometry, the safe outcome this
# rule has nothing to add to. "lexicon-snap" is excluded for the same
# reason — it reports an ORDINARY, confident correction (a known
# abbreviation or diacritic fix), not a guess needing confirmation.
_UNRESOLVED_CONFIDENCE_CODES = frozenset({"low-confidence", "implausible", "annotation-confusable"})

# Which of those codes _correct_ocr_confidence is allowed to accept
# automatically. Two of the three DO have a real answer already sitting
# behind them — see the corrector's own docstring for why "low-confidence"
# and "annotation-confusable" are safe to auto-accept and "implausible" is
# not, and is therefore deliberately left out of this set.
_AUTO_ACCEPTABLE_CONFIDENCE_CODES = frozenset({"low-confidence", "annotation-confusable"})


def check_ocr_confidence(ctx: QAContext) -> list[Violation]:
    violations: list[Violation] = []
    for i, run in enumerate(ctx.runs):
        for flag in run.flags:
            if flag.code in _UNRESOLVED_CONFIDENCE_CODES:
                violations.append(Violation(
                    Rule.OCR_CONFIDENCE, i, flag.code, flag.message,
                    0.0, 1.0, correctable=flag.code in _AUTO_ACCEPTABLE_CONFIDENCE_CODES,
                ))
    return violations


def _correct_ocr_confidence(ctx: QAContext, violation: Violation) -> bool:
    """Accept the pipeline's own already-computed reading rather than
    keep blocking on it — for the two codes where that reading genuinely
    IS the best answer available, not a guess invented here.

    ``"low-confidence"`` means OCR wasn't fully sure, but nothing else
    changed: `run.text` is already the one string S3/S4 settled on, and
    there is no second candidate this stage could compare it against —
    accepting it is accepting the only answer that exists, not guessing
    among several. ``"annotation-confusable"`` is the same shape: by the
    time this flag exists, lexicon/snap.py has ALREADY substituted the
    confusable annotation into `run.text` (see its own
    ``_ANNOTATION_CONFUSABLES`` table) — the "correction" already
    happened upstream; this only decides whether to trust it.

    ``"implausible"`` is excluded from ``_AUTO_ACCEPTABLE_CONFIDENCE_CODES``
    on purpose and stays refused: it means NOTHING matched — no exact
    lexicon entry, no fold, no fuzzy hit, no confusable substitution —
    so unlike the other two there is no "the pipeline's own best answer"
    to accept here, only the raw, uninterpreted OCR string. Auto-
    accepting that would be a materially different and riskier act than
    the other two — trusting a read this pipeline has no lexicon
    grounds to believe at all, on what may be a real construction
    document — so this corrector still returns False for it and the
    gate keeps refusing until a human looks.

    Marks the flag accepted rather than deleting it — renamed to its own
    ``-accepted`` code so `check_ocr_confidence` won't re-raise it on the
    next pass, but still visible on `run.flags` (and therefore the
    Review panel) afterward: accepted is not the same as never having
    been uncertain, and a human scanning the saved file's own flags
    should still be able to see exactly which reads were close calls.
    """
    if violation.kind not in _AUTO_ACCEPTABLE_CONFIDENCE_CODES:
        return False
    run = ctx.runs[violation.run_index]
    for idx, flag in enumerate(run.flags):
        if flag.code == violation.kind and flag.message == violation.message:
            run.flags[idx] = replace(flag, code=f"{flag.code}-accepted")
            return True
    return False  # the flag this violation was built from is already gone


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

_CHECKERS: tuple[Callable[[QAContext], list[Violation]], ...] = (
    check_spatial_integrity,
    check_layer_ordering,
    check_text_fidelity,
    check_clearance,
    check_ocr_confidence,
)

# Rule.SPATIAL_INTEGRITY has no entry, deliberately — see the module
# docstring for why that rule can only ever be detected, not repaired,
# here. Rule.OCR_CONFIDENCE DOES have one, but check_ocr_confidence
# already marks an "implausible" violation correctable=False (see
# _AUTO_ACCEPTABLE_CONFIDENCE_CODES) — the loop below skips calling any
# corrector at all for those (see its own `if not v.correctable:
# continue`), so they still escalate exactly as if this rule had no
# entry here either, for that one code only. _correct_ocr_confidence's
# own guard against "implausible" is therefore never actually reached
# through the loop — kept anyway so the function is correct standalone,
# not just correct because of a filter one call site up.
_CORRECTORS: dict[Rule, Callable[[QAContext, Violation], bool]] = {
    Rule.OCR_CONFIDENCE: _correct_ocr_confidence,
    Rule.LAYER_ORDERING: _correct_layer_ordering,
    Rule.TEXT_FIDELITY: _correct_text_fidelity,
    Rule.CLEARANCE: _correct_clearance,
}


@dataclass
class SelfCorrectingQAGate:
    """Check, correct what's correctable, re-check — bounded, not
    "loop until it passes". Every correction here is a deterministic,
    mechanical retry over a small fixed set of alternatives (restore
    from the plate, widen a shrink target, swap to the next covering
    font); none of it is a stochastic search that more iterations would
    plausibly help. So either the whole page settles within a couple of
    passes, or what's left is structural and no amount of further
    looping fixes it — at which point looping again would only burn
    time re-confirming the same failure, not find a new one.
    """

    max_iterations: int = 3

    def run(self, ctx: QAContext) -> QAReport:
        history: list[list[Violation]] = []
        corrected: list[Violation] = []

        for i in range(1, self.max_iterations + 1):
            violations = [v for check in _CHECKERS for v in check(ctx)]
            history.append(violations)
            if not violations:
                return QAReport(True, i, [], corrected, history)

            made_progress = False
            for v in violations:
                if not v.correctable:
                    continue
                corrector = _CORRECTORS.get(v.rule)
                if corrector is not None and corrector(ctx, v):
                    made_progress = True
                    corrected.append(v)

            if not made_progress:
                # Nothing left we know how to fix locally — stop here
                # rather than re-running the same checks against the
                # same canvas for no new information.
                return QAReport(False, i, violations, corrected, history)

        final = [v for check in _CHECKERS for v in check(ctx)]
        return QAReport(not final, self.max_iterations, final, corrected, history)


def run_qa_gate(ctx: QAContext, *, max_iterations: int = 3) -> QAReport:
    """Convenience entry point — see SelfCorrectingQAGate."""
    return SelfCorrectingQAGate(max_iterations=max_iterations).run(ctx)
