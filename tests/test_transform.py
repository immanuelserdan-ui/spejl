"""The mirror math, tested directly — it is what both routes turn on.

Fast (no OCR, no rendering), and doubles as executable documentation of
the angle convention described in ``transform/mirror.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from spejl.models import Axis
from spejl.transform import mirror as M

W, H = 1000.0, 800.0


# --------------------------------------------------------------------------
# Points and boxes
# --------------------------------------------------------------------------


def test_vertical_mirror_reflects_x_only():
    assert M.mirror_point(200, 300, W, H, Axis.VERTICAL) == (800, 300)


def test_horizontal_mirror_reflects_y_only():
    assert M.mirror_point(200, 300, W, H, Axis.HORIZONTAL) == (200, 500)


def test_both_axes_is_a_point_reflection():
    assert M.mirror_point(200, 300, W, H, Axis.BOTH) == (800, 500)


def test_bbox_corners_are_renormalised():
    """Reflection swaps which corner is minimal; mapping corner-for-corner
    would yield an inverted box that silently reads as empty."""
    box = (100.0, 200.0, 300.0, 400.0)
    out = M.mirror_bbox(box, W, H, Axis.VERTICAL)
    assert out == (700.0, 200.0, 900.0, 400.0)
    assert out[0] < out[2] and out[1] < out[3]


def test_mirroring_a_box_twice_is_the_identity():
    box = (100.0, 200.0, 300.0, 400.0)
    once = M.mirror_bbox(box, W, H, Axis.VERTICAL)
    assert M.mirror_bbox(once, W, H, Axis.VERTICAL) == box


# --------------------------------------------------------------------------
# The angle convention
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("angle", "direction"),
    [(0, (1, 0)), (90, (0, -1)), (180, (-1, 0)), (-90, (0, 1))],
)
def test_angle_to_direction_matches_the_documented_convention(angle, direction):
    dx, dy = M.direction_from_angle(angle)
    assert dx == pytest.approx(direction[0], abs=1e-9)
    assert dy == pytest.approx(direction[1], abs=1e-9)


def test_horizontal_text_stays_horizontal_under_a_vertical_mirror():
    """Its reading direction reverses under reflection, so it must be
    flipped back — only the anchor should move."""
    assert M.mirror_angle(0, Axis.VERTICAL) == 0


def test_vertical_text_is_invariant_under_a_vertical_mirror():
    """The worked example from build plan Figure 2: '5155' crosses to the
    other side of the sheet and still reads bottom-to-top, because a
    purely vertical direction has no x-component for the mirror to flip.
    """
    assert M.mirror_angle(90, Axis.VERTICAL) == 90


def test_vertical_text_reading_convention_is_axis_independent():
    """A top/bottom mirror must not turn vertical text upside down: the
    ISO convention is fixed, not a function of which axis was chosen."""
    assert M.mirror_angle(90, Axis.HORIZONTAL) == 90
    assert M.mirror_angle(90, Axis.BOTH) == 90


def test_text_is_never_emitted_reading_right_to_left():
    for axis in (Axis.VERTICAL, Axis.HORIZONTAL, Axis.BOTH):
        for angle in (0, 90, -90, 180, 30, -30, 150):
            out = M.mirror_angle(angle, axis)
            dx, dy = M.direction_from_angle(out)
            assert M.is_canonical_direction(dx, dy), (angle, axis, out)


def test_diagonal_keeps_its_mirrored_slope():
    """Negating a direction names the same infinite line, so restoring
    left-to-right reading cannot move the text off the reflected line.
    A 30° run mirrored left-right must end up at -30°, not back at 30°.
    """
    assert M.mirror_angle(30, Axis.VERTICAL) == pytest.approx(-30, abs=1e-6)
    assert M.mirror_angle(-30, Axis.VERTICAL) == pytest.approx(30, abs=1e-6)


def test_angle_round_trips_through_direction():
    for angle in (0, 30, 45, 90, -30, -90):
        dx, dy = M.direction_from_angle(angle)
        assert M.angle_from_direction(dx, dy) == pytest.approx(angle, abs=1e-9)


# --------------------------------------------------------------------------
# Raster
# --------------------------------------------------------------------------


def test_flip_image_matches_the_axis_semantics():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(M.flip_image(img, Axis.VERTICAL), img[:, ::-1])
    assert np.array_equal(M.flip_image(img, Axis.HORIZONTAL), img[::-1, :])
    assert np.array_equal(M.flip_image(img, Axis.BOTH), img[::-1, ::-1])
