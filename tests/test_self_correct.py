"""qa/self_correct.py's S8 gate — each rule's checker and corrector in
isolation, then the loop's own bounds.

No OCR, no full pipeline: a small synthetic canvas plus fake runs,
matching how tests/test_render_text.py exercises the renderer directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
import pytest

from spejl.models import Axis
from spejl.qa.self_correct import (
    _CORRECTORS,
    QAContext,
    Rule,
    SelfCorrectingQAGate,
    Violation,
    _correct_layer_ordering,
    _correct_text_fidelity,
    _font_covers,
    check_layer_ordering,
    check_spatial_integrity,
    check_text_fidelity,
)
from spejl.render.text import drawing_alpha_for, render_run
from spejl.style.metrics import TextStyle, resolve_font


@pytest.fixture(scope="module")
def font_path() -> str:
    return resolve_font()


def _style(font_path: str, px_size: int = 30) -> TextStyle:
    return TextStyle(
        font_path=font_path,
        px_size=px_size,
        tracking=0.0,
        ink=(0, 0, 0),
        paper=(255, 255, 255),
        cap_height_px=float(px_size) * 0.7,
    )


@dataclass
class _FakeRun:
    """Matches self_correct.RunLike's structural shape without pulling
    in raster/pipeline.py's MirroredRun (self_correct.py itself avoids
    that import to stay free of a circular dependency — the test
    mirrors that)."""

    text: str
    center_out: tuple[float, float]
    angle_out: float
    bbox_src: tuple[float, float, float, float]
    style: TextStyle


def _plate() -> np.ndarray:
    """A scrap of plan: poché wall, a dimension line — same fixture
    shape as tests/test_drawing_z_order.py, so a violation here means
    the same thing a violation there would."""
    plate = np.full((260, 520, 3), 255, np.uint8)
    cv2.rectangle(plate, (0, 0), (519, 18), (0, 0, 0), -1)
    cv2.line(plate, (0, 130), (519, 130), (0, 0, 0), 2)
    return plate


def _context_with_one_run(
    font_path: str, text: str = "1383", px_size: int = 30
) -> tuple[QAContext, _FakeRun, tuple[float, float]]:
    """A QAContext whose one run is genuinely clean: the target is the
    run's OWN naturally-rendered ink extent, not an arbitrary guess —
    the fidelity check compares rendered ink against `target`, so a
    made-up target that doesn't match what this text/size/font actually
    produces would trip the check by construction, independent of
    anything the module under test does.
    """
    plate = _plate()
    style = _style(font_path, px_size)
    center = (300.0, 200.0)
    run = _FakeRun(text=text, center_out=center, angle_out=0.0, bbox_src=(0, 0, 1, 1), style=style)

    natural = render_run(
        canvas=plate.copy(), text=text, center=center, angle_deg=0.0, style=style,
    )
    target = (natural.ink_width, natural.ink_height)

    canvas = plate.copy()
    drawing_alpha = drawing_alpha_for(plate)
    lines = np.zeros(plate.shape[:2], np.uint8)
    rendered = render_run(
        canvas=canvas, text=text, center=center, angle_deg=0.0, style=style,
        target_size=target, linework_mask=lines, drawing_alpha=drawing_alpha,
    )
    ctx = QAContext(
        canvas=canvas, plate=plate, source=cv2.flip(plate, 1), axis=Axis.VERTICAL,
        runs=[run], targets=[target], rendered=[rendered],
        drawing_alpha=drawing_alpha, linework_mask=lines,
    )
    return ctx, run, target


# ---------------------------------------------------------------------------
# Rule 1 — spatial integrity
# ---------------------------------------------------------------------------


def test_spatial_integrity_passes_on_an_untouched_mirror(font_path: str):
    ctx, _run, _target = _context_with_one_run(font_path)
    ctx.canvas = cv2.flip(ctx.plate, 1)  # a plain, correct mirror — no runs drawn on it
    ctx.source = ctx.plate
    assert check_spatial_integrity(ctx) == []


def test_spatial_integrity_catches_geometry_missing_outside_any_run(font_path: str):
    ctx, _run, _target = _context_with_one_run(font_path)
    ctx.canvas = cv2.flip(ctx.plate, 1)
    ctx.source = ctx.plate
    # Erase a chunk of the mirrored wall nowhere near the run's own
    # footprint (run sits around x=265-335,y=190-210 after mirroring).
    ctx.canvas[0:18, 400:460] = 255

    violations = check_spatial_integrity(ctx)
    assert len(violations) == 1
    assert violations[0].rule is Rule.SPATIAL_INTEGRITY
    assert not violations[0].correctable, "geometry loss must never be silently patched"


def test_spatial_integrity_is_never_in_the_corrector_table():
    """Structural guarantee, not just this test's own opinion: a rule
    with no safe local fix must have no entry a loop could call."""
    assert Rule.SPATIAL_INTEGRITY not in _CORRECTORS


# ---------------------------------------------------------------------------
# Rule 2 — layer ordering
# ---------------------------------------------------------------------------


def test_layer_ordering_passes_when_render_run_did_its_job(font_path: str):
    ctx, _run, _target = _context_with_one_run(font_path)
    assert check_layer_ordering(ctx) == []


def test_layer_ordering_catches_text_painted_directly_over_solid_geometry(font_path: str):
    ctx, _run, _target = _context_with_one_run(font_path)
    # Simulate the exact bug this rule exists to catch: some OTHER code
    # path draws directly onto the canvas, bypassing render_run's own
    # drawing_alpha compositing, and paints over the solid wall.
    cv2.putText(ctx.canvas, "X", (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

    violations = check_layer_ordering(ctx)
    assert len(violations) == 1
    assert violations[0].correctable


def test_layer_ordering_correction_restores_the_plate_exactly(font_path: str):
    ctx, _run, _target = _context_with_one_run(font_path)
    cv2.putText(ctx.canvas, "X", (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
    violation = check_layer_ordering(ctx)[0]

    fixed = _correct_layer_ordering(ctx, violation)
    assert fixed
    assert check_layer_ordering(ctx) == []
    solid = ctx.drawing_alpha >= 0.999
    assert np.array_equal(ctx.canvas[solid], ctx.plate[solid])


# ---------------------------------------------------------------------------
# Rule 3 — text fidelity: size lock
# ---------------------------------------------------------------------------


def test_text_fidelity_passes_for_a_run_rendered_at_its_measured_size(font_path: str):
    ctx, _run, _target = _context_with_one_run(font_path)
    assert check_text_fidelity(ctx) == []


def test_text_fidelity_catches_a_run_squished_well_under_its_target(font_path: str):
    ctx, _run, target = _context_with_one_run(font_path)
    # Force the SAME failure mode described in the module docstring: S7's
    # shrink loop successfully fitting a run by compressing it a lot.
    # RenderedRun is frozen (see render/text.py) — replace, not mutate.
    ctx.rendered[0] = replace(ctx.rendered[0], ink_height=target[1] * 0.5)

    violations = check_text_fidelity(ctx)
    assert any(v.rule is Rule.TEXT_FIDELITY and v.correctable for v in violations)


def test_text_fidelity_correction_widens_the_target_until_in_band(font_path: str):
    ctx, _run, target = _context_with_one_run(font_path)
    # Same forced-violation technique as the checker test above: assert
    # a squish happened, independent of coaxing the real shrink loop
    # into a specific outcome (that loop has its own test coverage in
    # test_render_text.py) — this test is about the CORRECTOR's own
    # widen-and-retry behaviour, not about reproducing squish naturally.
    ctx.rendered[0] = replace(ctx.rendered[0], ink_height=target[1] * 0.5)
    violations = check_text_fidelity(ctx)
    size_violation = next(v for v in violations if "cap height" in v.message)

    fixed = _correct_text_fidelity(ctx, size_violation)
    assert fixed
    ratio = ctx.rendered[0].ink_height / target[1]
    assert 0.85 <= ratio <= 1.10  # back in (or reasonably close to) band


# ---------------------------------------------------------------------------
# Rule 3 — text fidelity: font glyph coverage
# ---------------------------------------------------------------------------


def test_font_covers_reports_nothing_missing_for_danish_diacritics(font_path: str):
    assert _font_covers(font_path, "Køkken Værelse Åben") == ""


def test_font_covers_reports_a_character_the_font_cannot_render(font_path: str):
    missing = _font_covers(font_path, "Stue一")  # a CJK character appended
    assert missing == "一"


def test_text_fidelity_catches_a_missing_glyph(font_path: str):
    ctx, run, _target = _context_with_one_run(font_path, text="Stue")
    run.text = "Stue一"  # mutate after render — the check reads run.text directly

    violations = check_text_fidelity(ctx)
    assert any("no glyph for" in v.message for v in violations)


def test_text_fidelity_correction_swaps_to_a_covering_font(font_path: str):
    ctx, run, _target = _context_with_one_run(font_path, text="Køkken")
    run.text = "Køkken"
    # Fabricate a violation by asserting the CURRENT font is missing a
    # character it actually has — exercises the correction path (which
    # only checks whether the NEXT candidate covers the text) without
    # needing every real system font to genuinely lack a glyph.
    fake_violation = Violation(
        Rule.TEXT_FIDELITY, 0, "font-coverage",
        "'Køkken' has no glyph for 'Ø' in the resolved font", 0.0, 1.0, True,
    )
    fixed = _correct_text_fidelity(ctx, fake_violation)
    # Either it found and applied a covering candidate (font_path
    # unchanged is possible if the ONLY candidate is the original one
    # and it "covers" per _font_covers — Køkken's Ø is covered by every
    # real candidate on this system, so the search should succeed).
    assert fixed
    assert _font_covers(run.style.font_path, "Køkken") == ""


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_gate_passes_immediately_on_clean_output(font_path: str):
    ctx, _run, _target = _context_with_one_run(font_path)
    report = SelfCorrectingQAGate(max_iterations=3).run(ctx)
    assert report.passed
    assert report.iterations == 1
    assert report.violations == []


def test_gate_corrects_a_layer_ordering_violation_within_the_loop(font_path: str):
    ctx, _run, _target = _context_with_one_run(font_path)
    cv2.putText(ctx.canvas, "X", (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

    report = SelfCorrectingQAGate(max_iterations=3).run(ctx)
    assert report.passed
    assert report.corrected, "the gate reported passing without recording what it fixed"


def test_gate_stops_within_max_iterations_on_an_unfixable_violation(font_path: str):
    """Spatial-integrity loss has no corrector — the loop must recognise
    that immediately (no progress made) rather than burn every
    remaining iteration re-checking the same failure.

    Built to isolate that specific rule from the other one that LOOKS
    similar but isn't: `plate` is already in mirrored/output space (see
    QAContext's own docstring) and `source` is the pre-mirror original,
    so a hole punched into `plate` (and inherited by `canvas`, built
    from it) while `source` — a SEPARATE, still-intact copy taken
    before the corruption — keeps the real geometry, models a loss that
    happened upstream of this gate (inside erase/repair, before the
    plate snapshot it receives), which is exactly what makes it
    unfixable: plate itself no longer has the answer.

    Punching the SAME hole into `plate` alone (leaving `source` derived
    FROM the corrupted plate, as a naive test might) would instead
    produce a *layer-ordering* violation — canvas disagreeing with an
    still-correct source-of-truth — which genuinely is fixable by
    restoring from the plate. That's a different bug than this test
    means to exercise, which is why `source` is built from a clean copy
    taken first.
    """
    ctx, _run, _target = _context_with_one_run(font_path)
    good_plate = ctx.plate.copy()
    ctx.source = cv2.flip(good_plate, 1)  # pristine, pre-mirror — untouched by the corruption below

    bad_plate = good_plate.copy()
    bad_plate[0:18, 400:460] = 255  # the geometry loss, already baked into the plate
    ctx.plate = bad_plate
    ctx.canvas = bad_plate.copy()  # plate and canvas share ONE space — no additional flip
    ctx.drawing_alpha = drawing_alpha_for(bad_plate)

    # Sanity check the fixture actually isolates the rule as intended:
    # no disagreement between plate and canvas, so layer-ordering alone
    # would see nothing wrong here.
    assert check_layer_ordering(ctx) == []

    report = SelfCorrectingQAGate(max_iterations=5).run(ctx)
    assert not report.passed
    assert report.iterations == 1, "an uncorrectable violation should stop the loop immediately"
    assert any(v.rule is Rule.SPATIAL_INTEGRITY for v in report.violations)


def test_spatial_integrity_does_not_flag_geometry_erase_legitimately_removed(font_path: str):
    """Regression: confirmed on a REAL plan (demo/test_angle_fix.png,
    not a synthetic fixture), not a hypothetical. The exclusion zone
    check_spatial_integrity used was built from a run's MEASURED INK
    extent (_run_footprint) — correct for knowing where the RE-RENDERED
    glyph sits, and deliberately tighter than the raw OCR box, since
    fit_style/measure_ink_extent filter out padding and non-glyph
    content from that raw box on purpose (style/metrics.py). But S5's
    own erase (erase.clean.erase_text) wipes the WIDER raw OCR box
    (run.bbox_src, dilated by 3px) — not the tighter measured extent —
    so a real plan's own drawing content sitting in the GAP between the
    two (inside the raw box, outside the measured one) is legitimately
    erased by S5 and then misread as "missing" by a check that only
    knew about the tighter box. Confirmed on the real file: erasing
    '4504' correctly took a couple of real source pixels that sat
    exactly in that gap, at the edge of a nearby diagonal wall line.

    This constructs the same shape of case synthetically: bbox_src
    genuinely wider than the run's own measured ink target, with a
    short real drawing mark sitting in the gap between them, and
    confirms erase_text really does remove it (so this isn't testing
    an unreachable setup) before checking that check_spatial_integrity
    no longer flags the loss.
    """
    from spejl.erase.clean import erase_text

    plate = _plate()
    # A short real drawing mark sitting where a raw OCR box would
    # plausibly reach (a few px right of the measured ink) but a
    # tighter, ink-only extent would not.
    plate[195:205, 355:365] = 0
    style = _style(font_path, 30)
    center_src = (300.0, 200.0)
    bbox_src = (240.0, 185.0, 370.0, 215.0)  # deliberately wider than the run's own ink
    run = _FakeRun(text="1383", center_out=center_src, angle_out=0.0, bbox_src=bbox_src, style=style)

    erased = erase_text(plate, [bbox_src])
    # Sanity check the fixture is actually exercising the real gap this
    # test is about: the mark must be gone from the erased plate, or
    # this isn't testing what it claims to.
    assert erased.image[195:205, 355:365].min() > 200, "test setup itself doesn't erase the mark — fixture is wrong"

    canvas = erased.image.copy()
    drawing_alpha = drawing_alpha_for(erased.image)
    lines = np.zeros(plate.shape[:2], np.uint8)
    natural = render_run(canvas=canvas.copy(), text="1383", center=center_src, angle_deg=0.0, style=style)
    target = (natural.ink_width, natural.ink_height)
    rendered = render_run(
        canvas=canvas, text="1383", center=center_src, angle_deg=0.0, style=style,
        target_size=target, linework_mask=lines, drawing_alpha=drawing_alpha,
    )

    ctx = QAContext(
        canvas=canvas, plate=erased.image, source=plate, axis=Axis.VERTICAL,
        runs=[run], targets=[target], rendered=[rendered],
        drawing_alpha=drawing_alpha, linework_mask=lines,
    )
    violations = check_spatial_integrity(ctx)
    assert violations == [], violations
