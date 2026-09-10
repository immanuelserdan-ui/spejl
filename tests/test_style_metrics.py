"""style/metrics.py edge cases the round-trip fixture never exercises —
degenerate boxes and pathological text, not the clean synthetic case.
"""

from __future__ import annotations

import numpy as np
import pytest

from spejl.style.metrics import fit_font_size, measure_ink_extent, resolve_font


@pytest.fixture(scope="module")
def font_path() -> str:
    return resolve_font()


def test_out_of_bounds_bbox_falls_back_to_its_own_real_size():
    """Regression: the fallback used the ALREADY-CLAMPED box deltas,
    which collapse toward zero for a box that lies outside the image —
    handing fit_font_size a bogus ~1px target regardless of the box's
    actual (unclamped) size."""
    image = np.full((50, 50, 3), 255, np.uint8)
    # Entirely outside the image, but a real 80x30 box.
    along, across = measure_ink_extent(image, (100.0, 100.0, 180.0, 130.0), 0.0)
    assert along == pytest.approx(80.0)
    assert across == pytest.approx(30.0)


def test_bbox_straddling_the_image_edge_uses_unclamped_size_too():
    image = np.full((50, 50, 3), 255, np.uint8)
    # Starts inside, ends far outside — clamping alone collapses nothing
    # here (x1>x0 survives clamping), so this exercises the *other*
    # fallback branch (blank crop, `if not ink.any()`), which was
    # already correct; included as a boundary check alongside the fix.
    along, across = measure_ink_extent(image, (40.0, 10.0, 200.0, 40.0), 0.0)
    assert along > 1.0  # not the old degenerate ~1px result


def test_whitespace_only_text_does_not_freeze_the_binary_search(font_path: str):
    """Regression: getbbox(' ') returns a zero-height box at every
    candidate size, so every iteration's error was identical and the
    search froze at the first midpoint probed — unrelated to the
    requested target height."""
    small = fit_font_size(" ", target_cap_height=10.0, font_path=font_path)
    large = fit_font_size(" ", target_cap_height=200.0, font_path=font_path)
    # A real target-height dependence, not a frozen constant for both.
    assert small != large
    assert small < large


def test_empty_string_still_handled():
    fit_font_size("", target_cap_height=20.0, font_path=resolve_font())  # must not raise
