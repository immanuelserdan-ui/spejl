"""detect/rotations.py's quad-derived angle resolution — the fix for
dimension text drawn at a genuine diagonal (following a sloped wall)
instead of a clean 0/90/-90. Tested directly against real quad geometry
and synthetic Detection objects; no OCR, no image.
"""

from __future__ import annotations

from spejl.detect.ocr import Detection
from spejl.detect.rotations import (
    _orientation_is_plausible,
    _quad_angle_deg,
    _resolve_ambiguous_tilts,
)


def _det(text: str, angle_deg: float, quad=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))) -> Detection:
    return Detection(text=text, quad=quad, conf=0.99, angle_deg=angle_deg)

# Real quads pulled directly from 722-0553-0006-1034-T26-S -- the file
# a user-reported screenshot showed several dimension numbers rendering
# broken on (oversized, running off their own room's walls). All follow
# the same diagonal partition wall. Pulled from detect_all_orientations'
# own output (the actual merge-winning candidate each run resolves to),
# not a single raw detection pass -- confirmed to matter: a quad read
# directly from one isolated pass measured a smaller, borderline tilt
# for the same two runs (~5.1-5.4°, under the snap floor) than the one
# the real multi-pass merge actually settles on and renders with
# (~6.9-7.3°) -- natural OCR/quad variance between passes reading the
# same physical text, not an error in either measurement on its own.
_QUAD_4504 = ((784.0, 488.0), (793.0, 558.0), (760.0, 562.0), (751.0, 492.0))
_QUAD_4381 = ((864.0, 1087.0), (872.0, 1153.0), (839.0, 1157.0), (831.0, 1091.0))
_QUAD_ENTRE = ((725.0, 1132.0), (843.0, 1132.0), (843.0, 1177.0), (725.0, 1177.0))
_QUAD_2303 = ((721.0, 1379.0), (788.0, 1379.0), (788.0, 1408.0), (721.0, 1408.0))
# 'H*', a fixed annotation glyph always drawn upright by convention --
# short enough (2 characters) that its own quad reads several degrees
# off true (confirmed: -9.3 deg) from ordinary detection noise, not a
# real physical tilt.
_QUAD_H_STAR = ((1044.0, 780.0), (1100.0, 771.0), (1108.0, 815.0), (1052.0, 824.0))


def test_a_genuinely_tilted_dimension_resolves_to_its_real_angle():
    """Confirmed on the real plan: this dimension's own detected quad
    already carries its true tilt (following the sloped wall it labels)
    — resolving it, instead of forcing it to the nearest of 0/90/-90,
    is the whole fix. ~95 degrees: bottom-to-top (ISO convention,
    matching every other vertical dimension on the same sheet), tilted
    ~5 degrees off pure vertical to match the wall.
    """
    angle = _quad_angle_deg(_QUAD_4504, fallback=90.0)
    assert 90.0 < angle < 100.0
    angle2 = _quad_angle_deg(_QUAD_4381, fallback=90.0)
    assert 90.0 < angle2 < 100.0


def test_axis_aligned_text_still_resolves_to_exactly_cardinal():
    """A cleanly horizontal run's own quad measures a fraction of a
    degree off true (ordinary detection noise) -- must snap to exactly
    0.0, not carry that noise into the render as a barely-perceptible
    but real-looking tilt."""
    assert _quad_angle_deg(_QUAD_ENTRE, fallback=0.0) == 0.0
    assert _quad_angle_deg(_QUAD_2303, fallback=0.0) == 0.0


def test_noisy_axis_aligned_quads_are_not_snapped_by_magnitude_alone():
    """'Gang' and 'Stue's own quads (real, from a DIFFERENT file than
    the diagonal-wall fix was built against, no diagonal wall anywhere
    near either room) measure 4.86 and 3.58 degrees off true -- ABOVE
    _ALWAYS_SNAP_DEG (3.0), so _quad_angle_deg alone, working from one
    detection's own geometry with no visibility into anything else on
    the sheet, correctly can't tell this apart from a genuine gentle
    tilt by magnitude alone. That's not a bug in this function -- see
    _resolve_ambiguous_tilts, applied afterwards with the SHEET-WIDE
    view this function deliberately doesn't have, for where the actual
    snap decision for a case like this gets made.
    """
    quad_stue = ((718.0, 234.0), (798.0, 239.0), (795.0, 274.0), (715.0, 269.0))
    quad_gang = ((1203.0, 527.0), (1297.0, 535.0), (1293.0, 571.0), (1200.0, 562.0))
    assert _quad_angle_deg(quad_stue, fallback=0.0) != 0.0
    assert _quad_angle_deg(quad_gang, fallback=0.0) != 0.0


def test_an_uncorroborated_tilt_is_snapped_to_cardinal():
    """Regression: found from a user screenshot circling 'Gang' as
    visibly, wrongly tilted in the mirrored output -- confirmed across
    THREE separate real files, no diagonal wall anywhere near any of
    them, so a property of how that word's own quad measures, not a
    fluke of one detection. Its own deviation (4.86°) sits ABOVE
    _ALWAYS_SNAP_DEG and would have been kept as a "real" tilt by
    magnitude alone; with nothing else on the sheet corroborating a
    similar deviation, _resolve_ambiguous_tilts must snap it back to
    exactly horizontal regardless.
    """
    gang = _det("Gang", 4.86)
    unrelated = _det("Stue", 0.0)  # cleanly axis-aligned, no ambiguity
    resolved = _resolve_ambiguous_tilts([gang, unrelated])
    assert resolved[0].angle_deg == 0.0


def test_corroborated_tilts_from_independent_runs_are_kept():
    """The other side of the same mechanism: real dimension numbers
    from the same real plan, all following the SAME diagonal wall,
    corroborate each other closely (three vertical-family runs within
    6.1-7.3° of 90°) and must all survive un-snapped -- the whole
    point of the diagonal-wall fix this corroboration check must not
    quietly undo.
    """
    run_a = _det("3921", 96.1)
    run_b = _det("4504", 97.3)
    run_c = _det("4381", 96.9)
    resolved = _resolve_ambiguous_tilts([run_a, run_b, run_c])
    assert [d.angle_deg for d in resolved] == [96.1, 97.3, 96.9]


def test_a_lone_tilt_with_no_corroborator_anywhere_still_snaps():
    """A single genuinely-tilted-looking detection with nothing else
    on the sheet anywhere close to its own deviation has no way to be
    told apart from noise -- the safe default (snap) applies, the same
    behaviour this whole angle-resolution feature had before it could
    detect diagonal text at all."""
    lone = _det("4504", 97.3)
    unrelated = _det("Stue", 0.0)
    resolved = _resolve_ambiguous_tilts([lone, unrelated])
    assert resolved[0].angle_deg == 90.0


def test_extreme_angles_never_need_corroboration():
    """Genuinely unambiguous tilts (well past _UNAMBIGUOUS_TILT_DEG --
    the diagonal-dimension-line case a separate fix in style/metrics.py
    handles) are trusted outright, alone, with no risk of being
    confused for ordinary detection noise in the first place."""
    lone_extreme = _det("1346", 130.6)
    resolved = _resolve_ambiguous_tilts([lone_extreme])
    assert resolved[0].angle_deg == 130.6


def test_a_short_near_square_glyph_falls_back_instead_of_trusting_noise():
    """Regression: 'H*' is a fixed annotation, always drawn upright by
    convention -- but with only 2 characters, its own quad has too
    little geometry to anchor a reliable angle estimate (measured -9.3
    degrees on the real plan, purely from measurement noise). Trusting
    that would rotate a label that was never tilted. Below the aspect
    floor, the pass's own canonical angle is used instead.
    """
    angle = _quad_angle_deg(_QUAD_H_STAR, fallback=0.0)
    assert angle == 0.0


def test_plausibility_is_checked_against_the_pass_angle_not_the_refined_one():
    """Regression: a rotated pass misread a room label's trailing digit
    as a separate fragment AND misread the label itself as truncated
    (missing that digit) -- at HIGHER confidence than the correct,
    full 0-degree read. The truncated read's own box is far wider than
    tall (118x55, real numbers from the same file): checked against
    the PASS's forced 90 degree claim, that mismatch is exactly what
    _orientation_is_plausible exists to catch, and correctly rejects
    it. Checked against the truncated read's OWN quad angle instead
    (which genuinely IS near-horizontal -- it's just missing a
    character, not tilted) it would incorrectly pass, letting the
    truncated read compete in the merge and win by confidence alone --
    exactly the regression a fix here must not reintroduce.
    """
    truncated = Detection(
        text="Vaer.",
        quad=((1019.0, 1177.0), (1137.0, 1177.0), (1137.0, 1232.0), (1019.0, 1232.0)),
        conf=0.997,
        angle_deg=90.0,  # the pass's own forced claim
    )
    assert not _orientation_is_plausible(truncated)

    # The genuine vertical dimension from the same sheet must still pass.
    genuine = Detection(text="4504", quad=_QUAD_4504, conf=1.0, angle_deg=90.0)
    assert _orientation_is_plausible(genuine)
