"""detect/rotations.py's quad-derived angle resolution — the fix for
dimension text drawn at a genuine diagonal (following a sloped wall)
instead of a clean 0/90/-90. Tested directly against real quad geometry
and synthetic Detection objects; no OCR, no image.
"""

from __future__ import annotations

from spejl.detect.ocr import Detection
from spejl.detect.rotations import _orientation_is_plausible, _quad_angle_deg

# Real quads pulled directly from 722-0553-0006-1034-T26-S -- the file
# a user-reported screenshot showed several dimension numbers rendering
# broken on (oversized, running off their own room's walls). All follow
# the same diagonal partition wall.
_QUAD_4504 = ((752.0, 489.0), (787.0, 486.0), (794.0, 560.0), (759.0, 563.0))
_QUAD_4381 = ((832.0, 1086.0), (867.0, 1083.0), (874.0, 1161.0), (839.0, 1164.0))
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
