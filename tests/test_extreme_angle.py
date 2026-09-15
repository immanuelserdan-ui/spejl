"""style/metrics.py's rotated-crop measurement path — for a run tilted
too far from any cardinal (0/90/-90/180) for the plain axis-aligned
crop to work. Found on two real dimension numbers running along a
diagonal dimension line at ~47deg and ~131deg: both measured wildly
oversized, and one measured its own ink colour as pure white (the
SAME as paper) because so little of its much-larger-than-necessary
axis-aligned crop was actual ink that the darkest-decile percentile
cut never left the white background.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from spejl.style.metrics import (
    _map_point_from_straightened,
    _needs_rotated_measurement,
    _straighten_crop,
    measure_ink_and_paper,
    measure_ink_center,
    measure_ink_extent,
    resolve_font,
)


@pytest.fixture(scope="module")
def font_path() -> str:
    return resolve_font()


def _diagonal_text_image(
    text: str, angle_deg: float, font_path: str, px_size: int = 30, canvas_size: int = 250
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """A synthetic image with ``text`` placed so that it reads correctly
    when straightened by :func:`_straighten_crop` at ``angle_deg`` —
    i.e. genuinely tilted source content, the same shape of input a
    real diagonal dimension number gives the real pipeline, not a
    horizontal string with a claimed angle nobody actually drew that
    way. Built by applying the INVERSE of _straighten_crop's own
    rotation (``angle_deg - 180``, not ``180 - angle_deg``) to place
    the text, then independently confirmed (see this module's own
    real-plan investigation) that straightening it back at the SAME
    ``angle_deg`` recovers upright, correctly-oriented text.
    """
    font = ImageFont.truetype(font_path, px_size)
    tile = Image.new("RGBA", (px_size * len(text), px_size * 2), (255, 255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.text((5, 5), text, font=font, fill=(0, 0, 0, 255))
    tile = tile.crop(tile.getbbox() or (0, 0, tile.width, tile.height))

    placed = tile.rotate(
        angle_deg - 180.0, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255, 255)
    )
    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    px0 = (canvas_size - placed.width) // 2
    py0 = (canvas_size - placed.height) // 2
    canvas.paste(placed.convert("RGB"), (px0, py0))

    bbox = (float(px0), float(py0), float(px0 + placed.width), float(py0 + placed.height))
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR), bbox


@pytest.mark.parametrize("angle_deg", [130.6, 46.9])
def test_needs_rotated_measurement_for_the_real_extreme_angles(angle_deg: float):
    """The exact two real-plan angles that motivated this whole path."""
    assert _needs_rotated_measurement(angle_deg)


@pytest.mark.parametrize("angle_deg", [0.0, 3.0, -2.5, 90.0, 95.0, -90.0, 84.6, 180.0, -178.0])
def test_does_not_need_rotated_measurement_near_a_cardinal(angle_deg: float):
    """Every gentle wall-following tilt confirmed on a real plan (5-9°)
    already works fine with the plain axis-aligned crop — this path
    must stay off for them, not just for exact cardinals."""
    assert not _needs_rotated_measurement(angle_deg)


@pytest.mark.parametrize("angle_deg", [130.6, 46.9, 20.0, -160.0])
def test_straighten_and_unstraighten_round_trips_a_point(angle_deg: float):
    """The straightened crop's own centre, mapped back through
    _map_point_from_straightened, must land back at the original
    bbox's centre — the rotation is performed about that exact point,
    so this is a direct, real-geometry check of the round trip, not
    just of visual appearance."""
    image = np.full((250, 250, 3), 255, np.uint8)
    bbox = (80.0, 90.0, 170.0, 160.0)
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    result = _straighten_crop(image, bbox, angle_deg)
    assert result is not None
    crop, origin_x, origin_y, matrix = result
    side = crop.shape[0]

    mapped_x, mapped_y = _map_point_from_straightened(side / 2.0, side / 2.0, matrix, origin_x, origin_y)
    assert mapped_x == pytest.approx(cx, abs=1.0)
    assert mapped_y == pytest.approx(cy, abs=1.0)


def test_a_diagonal_run_measures_black_ink_not_white_on_white(font_path: str):
    """Regression: on a real plan, a dimension number tilted ~47°
    measured its own ink colour as (255, 255, 255) -- identical to
    paper -- because under 10% of its (much larger than necessary)
    straightened crop was actual ink, and the darkest-decile percentile
    cut used to find ink colour never left the white background. The
    run was being sized and positioned correctly; it was invisible,
    rendered in white ink on a white page.
    """
    image, bbox = _diagonal_text_image("1346", 130.6, font_path)
    ink, paper = measure_ink_and_paper(image, bbox, angle_deg=130.6, expected_glyphs=4)
    assert ink != paper
    assert sum(ink) < 200  # genuinely dark, not a pale near-white reading


def test_a_diagonal_run_measures_a_reasonable_size(font_path: str):
    """Regression: the same two real dimension numbers measured 2-3x
    their siblings' size through the plain axis-aligned crop (px_size
    75-100 against a normal 26-45) -- an axis-aligned box around a
    ~45°-tilted run is dominated by empty corner padding, and whatever
    of that padding's own contents (here: nothing, deliberately, so
    this test isolates the padding-inflation problem on its own) get
    swept into the same measurement. Straightening first keeps the
    crop tight around the real ink again, the way it already is for a
    genuinely axis-aligned run.
    """
    image, bbox = _diagonal_text_image("1346", 130.6, font_path, px_size=30)
    along, across = measure_ink_extent(image, bbox, angle_deg=130.6, expected_glyphs=4)
    # The drawn tile's own true ink is roughly 30px cap height and a
    # comparable multiple of that in width for 4 digits -- generous
    # bounds around that, nowhere near what an inflated axis-aligned
    # crop of a ~130px-square bbox would report.
    assert 15 < across < 55
    assert 50 < along < 160


def test_a_diagonal_runs_measured_centre_is_close_to_its_true_centre(font_path: str):
    image, bbox = _diagonal_text_image("1346", 130.6, font_path, px_size=30)
    true_cx, true_cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    center = measure_ink_center(image, bbox, angle_deg=130.6, expected_glyphs=4)
    # A loose bound, deliberately: `bbox` here is the ROTATED TILE's own
    # bounding box (from PIL's expand=True), not a tight crop of just
    # the ink within it, so its centre is only an approximate stand-in
    # for the ink's true centre -- the point of this test is confirming
    # the measurement lands in the right neighbourhood at all (not off
    # by some large, wrong amount), not pixel-exact agreement with an
    # imperfect ground truth.
    assert center[0] == pytest.approx(true_cx, abs=25.0)
    assert center[1] == pytest.approx(true_cy, abs=25.0)


def test_a_sparse_component_spanning_the_crop_is_rejected_even_when_not_classically_elongated():
    """Regression: straightening a real dimension line (with arrowhead
    triangles at each end) turns it from a thin diagonal stroke into a
    component that spans most of the crop's width but is no longer
    thin enough to read as classically 'elongated' -- the arrowheads
    make it locally tall. Confirmed on a real plan (fill ratio 0.14):
    it escaped the pre-existing elongation check, and with too few
    OTHER components surviving to satisfy the expected_glyphs gate the
    three contamination filters share, was never caught by any of
    them either -- inflating a dimension number's measured size by
    nearly 3x. A plain fill-ratio floor, unconditional like the
    elongation check beside it, catches this without needing the gate.
    """
    from spejl.style.metrics import measure_ink_extent as _measure

    image = np.full((100, 200, 3), 255, np.uint8)
    # A thin line spanning most of the width, with two "arrowhead"
    # triangles at its ends thick enough to escape a pure thinness test.
    cv2.line(image, (5, 50), (195, 50), (0, 0, 0), 2)
    cv2.fillPoly(image, [np.array([[5, 40], [5, 60], [25, 50]])], (0, 0, 0))
    cv2.fillPoly(image, [np.array([[195, 40], [195, 60], [175, 50]])], (0, 0, 0))
    # A single compact "digit" well clear of the line -- enough real
    # content that the crop has SOME usable ink, but only one
    # component, so expected_glyphs > component count and the three
    # gated filters are blocked, same as the real 3-components-for-
    # 4-characters shape that let the line through undetected before.
    cv2.rectangle(image, (90, 65), (110, 90), (0, 0, 0), -1)

    bbox = (0.0, 0.0, 200.0, 100.0)
    along, across = _measure(image, bbox, angle_deg=0.0, expected_glyphs=4)
    # With the line correctly excluded, the measurement reflects only
    # the compact digit block (20x25) -- nowhere near the ~190px width
    # the line itself would report if it survived into the cluster.
    assert along < 40
    assert across < 40
