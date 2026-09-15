"""render/text.py's shrink-to-fit guard and its drawing-over-type
z-order — no OCR, no image pipeline, just the renderer against a
synthetic TextStyle."""

from __future__ import annotations

import numpy as np
import pytest

from spejl.render.text import drawing_alpha_for, render_run
from spejl.style.metrics import TextStyle, resolve_font


@pytest.fixture(scope="module")
def font_path() -> str:
    return resolve_font()


def _style(font_path: str, px_size: int = 24) -> TextStyle:
    return TextStyle(
        font_path=font_path,
        px_size=px_size,
        tracking=0.0,
        ink=(0, 0, 0),
        paper=(255, 255, 255),
        cap_height_px=float(px_size) * 0.7,
    )


def test_a_short_string_that_fits_is_not_flagged_shrunk(font_path: str):
    canvas = np.full((200, 200, 3), 255, np.uint8)
    rendered = render_run(
        canvas, "1", (100, 100), 0.0, _style(font_path), target_size=(30.0, 20.0)
    )
    assert not rendered.shrunk


def test_a_long_replacement_string_shrinks_but_may_not_fully_converge(font_path: str):
    """A lexicon snap can replace a short OCR read with a much longer
    canonical word (e.g. 'Kok' -> 'Køkken') — the shrink loop is bounded
    (6 attempts) and, for a target size far below what the text needs,
    can legitimately give up still oversized. The loop must still have
    made real progress (checked against the unshrunk render), which is
    what the pipeline-level escalated flag (raster/pipeline.py) then
    uses ink_width to distinguish from a clean, near-target shrink.
    """
    canvas = np.full((300, 300, 3), 255, np.uint8)
    target_along = 40.0  # deliberately much smaller than "Soveværelse" needs
    unshrunk = render_run(
        np.full((300, 300, 3), 255, np.uint8), "Soveværelse", (150, 150), 0.0,
        _style(font_path, px_size=40), target_size=None,
    )
    rendered = render_run(
        canvas, "Soveværelse", (150, 150), 0.0, _style(font_path, px_size=40),
        target_size=(target_along, 30.0),
    )
    assert rendered.shrunk
    assert rendered.ink_width < unshrunk.ink_width, "shrink loop made no progress at all"


def test_a_target_height_smaller_than_the_rendered_tile_triggers_a_shrink(font_path: str):
    """Regression: target_size[1] (the across/height component) was
    accepted but never checked — only width drove the shrink loop's
    break condition, contradicting this module's own header promise
    ("Let a run outgrow its box... anything more than 4% over the
    original is shrunk", no axis singled out). A generously large
    target width paired with a target height far smaller than the text
    actually needs must still trigger a shrink and make real height
    progress — not silently render at full size with `shrunk` staying
    False just because width alone already fit.
    """
    canvas = np.full((300, 300, 3), 255, np.uint8)
    unshrunk = render_run(
        np.full((300, 300, 3), 255, np.uint8), "Bad", (150, 150), 0.0,
        _style(font_path, px_size=60), target_size=None,
    )
    rendered = render_run(
        canvas, "Bad", (150, 150), 0.0, _style(font_path, px_size=60),
        target_size=(1000.0, 20.0),  # width generous, height far too small
    )
    assert rendered.shrunk
    assert rendered.ink_height < unshrunk.ink_height, "shrink loop made no height progress at all"


def test_ink_dimensions_are_measured_before_rotation_not_after(font_path: str):
    """Regression: ink_width/ink_height used to be measured from the
    tile AFTER PIL's `expand=True` rotation, whose axis-aligned bounding
    box is strictly larger than the un-rotated rectangle on BOTH axes
    for any non-cardinal angle — inflating the reported size purely
    from rotation geometry, worst on the narrow axis of a long thin
    run. Confirmed on real, gently-tilted dimension numbers (6-8 degrees
    off cardinal, following a sloped wall): once the height check above
    started comparing ink_height against a target measured in the SAME
    pre-rotation frame, every one of them reported 120-250% height
    "overflow" and a spurious fit-shrink-incomplete warning, despite
    never needing to shrink at all -- the shrink loop's OWN internal
    check (against the un-rotated tile) already agreed the glyph fit
    its target cleanly.

    A rotated render and an unrotated render of the IDENTICAL string at
    the IDENTICAL size must report the identical ink_width/ink_height —
    rotating where it's drawn must not change what "how big is this
    text" means.
    """
    style = _style(font_path, px_size=40)
    unrotated = render_run(
        np.full((300, 300, 3), 255, np.uint8), "4504", (150, 150), 0.0, style
    )
    rotated = render_run(
        np.full((300, 300, 3), 255, np.uint8), "4504", (150, 150), 83.9, style
    )
    assert rotated.ink_width == pytest.approx(unrotated.ink_width, abs=1.0)
    assert rotated.ink_height == pytest.approx(unrotated.ink_height, abs=1.0)


def test_render_run_reports_the_actual_ink_width_for_callers_to_check(font_path: str):
    """The caller-side fix (raster/pipeline.py) depends on ink_width
    being the real measured extent, not an estimate."""
    canvas = np.full((100, 100, 3), 255, np.uint8)
    rendered = render_run(canvas, "44", (50, 50), 0.0, _style(font_path))
    assert rendered.ink_width > 0


def test_a_run_that_overlaps_a_previously_rendered_run_is_flagged_collided(font_path: str):
    """Regression: on a real plan, a mirrored '4381' dimension landed on
    top of the 'Entre' room label. Neither run overlapped the original
    linework, so the old collision check — which only ever compared
    against a static linework mask — never fired for either one, and
    both rendered "clean" while visibly overlapping on the canvas.

    raster/pipeline.py passes the SAME mask object to every render_run()
    call for a page; each call must stamp its own ink into it so the
    next call sees it, turning the check into "against linework AND
    every run already drawn," not just linework.
    """
    canvas = np.full((200, 200, 3), 255, np.uint8)
    shared_mask = np.zeros((200, 200), np.uint8)

    first = render_run(
        canvas, "Entre", (100, 100), 0.0, _style(font_path), linework_mask=shared_mask
    )
    assert not first.collided  # nothing drawn yet — mask started empty

    second = render_run(
        canvas, "4381", (100, 100), 0.0, _style(font_path), linework_mask=shared_mask
    )
    assert second.collided  # lands squarely on top of the first run's ink


def test_two_runs_far_apart_do_not_collide(font_path: str):
    canvas = np.full((300, 300, 3), 255, np.uint8)
    shared_mask = np.zeros((300, 300), np.uint8)

    render_run(canvas, "Bad", (30, 30), 0.0, _style(font_path), linework_mask=shared_mask)
    second = render_run(
        canvas, "Stue", (250, 250), 0.0, _style(font_path), linework_mask=shared_mask
    )
    assert not second.collided


# ---------------------------------------------------------------------------
# Z-order: type goes behind the drawing
# ---------------------------------------------------------------------------


def _plate_with_linework() -> np.ndarray:
    """A scrap of plan: poché wall, a dimension line, a door-swing arc."""
    import cv2

    plate = np.full((260, 520, 3), 255, np.uint8)
    cv2.rectangle(plate, (0, 0), (519, 18), (0, 0, 0), -1)
    cv2.line(plate, (0, 130), (519, 130), (0, 0, 0), 2)
    cv2.ellipse(plate, (40, 250), (190, 190), 0, -90, 0, (0, 0, 0), 2)
    cv2.line(plate, (260, 40), (260, 240), (0, 0, 0), 1)
    return plate


def test_a_run_drawn_over_linework_leaves_that_linework_untouched(font_path: str):
    """The guarantee the drawing layer exists to make: geometry is the
    authoritative content and type is reconstructed, so a re-rendered
    run may never alter a wall, an arc or a dimension line — whatever
    its anchor, size or colour.

    Checked in *grey* ink rather than black on purpose. Black type over
    black linework is indistinguishable from the linework itself, so a
    naive over-composite passes that test by accident while still
    overwriting every pixel it lands on; the earlier failure this
    covers was only ever visible where type and geometry differed in
    tone — an antialiased glyph flank lightening a solid line, or a
    grey annotation drawn across one.
    """
    plate = _plate_with_linework()
    alpha = drawing_alpha_for(plate)
    style = TextStyle(
        font_path=font_path, px_size=34, tracking=0.0,
        ink=(128, 128, 128), paper=(255, 255, 255), cap_height_px=24.0,
    )

    canvas = plate.copy()
    render_run(canvas, "1383", (250, 130), 0.0, style, drawing_alpha=alpha)

    solid = alpha >= 0.999
    assert solid.sum() > 0, "fixture has no fully-covered drawing pixels to protect"
    assert np.array_equal(canvas[solid], plate[solid])


def test_without_a_drawing_layer_the_same_run_does_damage_that_linework(font_path: str):
    """The negative half of the test above — proof it is testing
    something. Same plate, same run, no layer: the type wins, which is
    precisely the behaviour the layer replaces."""
    plate = _plate_with_linework()
    alpha = drawing_alpha_for(plate)
    style = TextStyle(
        font_path=font_path, px_size=34, tracking=0.0,
        ink=(128, 128, 128), paper=(255, 255, 255), cap_height_px=24.0,
    )

    canvas = plate.copy()
    render_run(canvas, "1383", (250, 130), 0.0, style)

    solid = alpha >= 0.999
    assert not np.array_equal(canvas[solid], plate[solid])


def test_the_drawing_layer_is_never_mutated_by_the_runs_drawn_under_it(font_path: str):
    """Unlike the collision mask, which every run stamps itself into,
    the drawing layer describes the plan alone. If a run's own ink leaked
    into it, later runs would start hiding behind earlier ones and the
    result would depend on the order `runs` happens to be in."""
    plate = _plate_with_linework()
    alpha = drawing_alpha_for(plate)
    before = alpha.copy()

    canvas = plate.copy()
    render_run(canvas, "Entre", (250, 200), 0.0, _style(font_path, 34), drawing_alpha=alpha)
    render_run(canvas, "Bad", (250, 200), 0.0, _style(font_path, 34), drawing_alpha=alpha)

    assert np.array_equal(alpha, before)


def test_hidden_reports_how_much_of_a_run_the_drawing_covers(font_path: str):
    """`hidden` is what raster/pipeline.py escalates on: a run buried in
    poché is unreadable and needs a human, a run merely crossed by its
    own dimension line does not."""
    plate = _plate_with_linework()
    alpha = drawing_alpha_for(plate)
    style = _style(font_path, 30)

    in_the_clear = render_run(plate.copy(), "Stue", (400, 210), 0.0, style, drawing_alpha=alpha)
    buried = render_run(plate.copy(), "Stue", (260, 9), 0.0, style, drawing_alpha=alpha)

    assert in_the_clear.hidden < 0.05
    assert buried.hidden > 0.9


def test_drawing_alpha_is_zero_on_paper_and_one_on_ink(font_path: str):
    plate = _plate_with_linework()
    alpha = drawing_alpha_for(plate)

    assert alpha[210, 400] == pytest.approx(0.0)   # open floor
    assert alpha[9, 260] == pytest.approx(1.0)     # solid poché wall


def test_drawing_alpha_of_a_blank_sheet_hides_nothing():
    """Otsu has no split to find on an empty sheet. The layer must come
    back empty rather than saturating and hiding every label on the
    page."""
    blank = np.full((80, 80, 3), 255, np.uint8)
    assert not drawing_alpha_for(blank).any()
