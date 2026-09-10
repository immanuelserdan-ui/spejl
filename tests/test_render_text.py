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
