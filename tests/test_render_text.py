"""render/text.py's shrink-to-fit guard — no OCR, no image pipeline,
just the renderer against a synthetic TextStyle."""

from __future__ import annotations

import numpy as np
import pytest

from spejl.render.text import render_run
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
