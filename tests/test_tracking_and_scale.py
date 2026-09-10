"""solve_tracking / solve_horizontal_scale — the asymmetric clamp
design, and the regression it exists to prevent.

The core finding this file locks in: compressing tracking/scale risks
real illegibility (two glyphs squeezed together can read as a
different letter — confirmed on this project's own golden fixture,
where a first, symmetrically-tightened clamp turned 'Bad' into
something RapidOCR read back as 'Stue'), while expanding does not — it
just looks loose. The clamp bounds are asymmetric because that risk is
asymmetric, not by accident.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spejl.style.metrics import (
    font_measure_width,
    resolve_font,
    solve_horizontal_scale,
    solve_tracking,
)


@pytest.fixture(scope="module")
def font_path() -> str:
    return resolve_font()


def test_compression_side_stays_at_the_original_proven_bound(font_path: str):
    """A target much NARROWER than natural width must not compress
    tracking more aggressively than the pre-existing, already-proven
    -8% bound — see the module docstring for why that bound is a
    legibility line, not an arbitrary one."""
    natural = font_measure_width("Bad", font_path, 50, 0.0)
    tracking = solve_tracking("Bad", font_path, 50, target_width=natural * 0.5)
    assert tracking == pytest.approx(-0.08, abs=1e-6)


def test_expansion_side_is_tighter_than_the_old_fifty_percent_bound(font_path: str):
    """The regression this whole change addresses: a target much WIDER
    than natural width used to be absorbed entirely as letter-spacing,
    up to +50% of the em per gap — visibly 'gapped out' lettering on a
    real (non-synthetic) plan. Now capped at +15%."""
    natural = font_measure_width("Vaer.", font_path, 24, 0.0)
    tracking = solve_tracking("Vaer.", font_path, 24, target_width=natural * 1.6)
    assert tracking == pytest.approx(0.15, abs=1e-6)


def test_horizontal_scale_compression_bound_is_conservative():
    assert solve_horizontal_scale(natural_width=100, target_width=40) == pytest.approx(0.96)


def test_horizontal_scale_expansion_bound_has_real_range():
    assert solve_horizontal_scale(natural_width=57, target_width=91.2) == pytest.approx(1.25)


def test_horizontal_scale_is_the_identity_when_widths_already_match():
    assert solve_horizontal_scale(natural_width=80, target_width=80) == pytest.approx(1.0)


def test_a_short_word_needing_heavy_compression_stays_legible_end_to_end(tmp_path):
    """Direct regression test for the 'Bad' -> 'Stue' failure: render a
    short word into a box much narrower than its natural width (as
    Arial substituting for whatever house font the source used can
    require) and confirm the actual pixels, run back through OCR,
    still read as the original word — not just that the numeric clamp
    is right, but that the rendered result is genuinely legible.
    """
    from spejl.detect.ocr import RapidOcrBackend
    from spejl.detect.rotations import detect_all_orientations
    from spejl.render.text import render_run
    from spejl.style.metrics import TextStyle

    font_path = resolve_font()
    px_size = 50
    natural = font_measure_width("Bad", font_path, px_size, 0.0)
    target_along = natural * 0.66  # the exact ratio the golden-fixture regression hit

    tracking = solve_tracking("Bad", font_path, px_size, target_along)
    natural_tracked = font_measure_width("Bad", font_path, px_size, tracking * px_size)
    width_scale = solve_horizontal_scale(natural_tracked, target_along)
    style = TextStyle(
        font_path=font_path, px_size=px_size, tracking=tracking,
        ink=(0, 0, 0), paper=(255, 255, 255), cap_height_px=36.0,
        ink_along_px=target_along, width_scale=width_scale,
    )

    canvas = np.full((150, 200, 3), 255, np.uint8)
    render_run(canvas, "Bad", center=(100, 75), angle_deg=0.0, style=style)

    detections = detect_all_orientations(canvas, RapidOcrBackend())
    texts = [d.text for d in detections]
    assert "Bad" in texts, f"rendered 'Bad' became illegible — OCR read {texts!r}"
