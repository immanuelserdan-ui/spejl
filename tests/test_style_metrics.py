"""style/metrics.py edge cases the round-trip fixture never exercises —
degenerate boxes and pathological text, not the clean synthetic case.
"""

from __future__ import annotations

import cv2
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


def test_a_neighbouring_runs_ink_is_excluded_from_this_boxs_measurement():
    """Regression: on a real plan, a vertical '4381' dimension's tight
    detection box clipped a corner of the 'Entre' label sitting right
    next to it — that foreign ink became its own connected component
    inside '4381's crop and got swept into its glyph cluster, inflating
    the measured extent with a mark that belongs to a different string
    entirely. `other_boxes` is the same exclusion erase.clean's
    build_text_mask already applies, for the same real-plan reason.
    """
    image = np.full((40, 60, 3), 255, np.uint8)
    cv2.rectangle(image, (5, 5), (20, 25), (0, 0, 0), -1)   # this run's own glyph
    cv2.rectangle(image, (25, 0), (38, 30), (0, 0, 0), -1)  # a neighbour's glyph, clipping in

    bbox = (0.0, 0.0, 40.0, 30.0)
    without_masking = measure_ink_extent(image, bbox, angle_deg=0.0)
    with_masking = measure_ink_extent(
        image, bbox, angle_deg=0.0, other_boxes=[(20.0, -5.0, 45.0, 35.0)]
    )

    assert with_masking[0] < without_masking[0]
    assert with_masking[0] == pytest.approx(15.0, abs=2.0)  # just this run's own 15px-wide glyph


def test_a_wall_fused_with_the_runs_own_glyphs_is_left_unpruned():
    """Regression: pruning a component whose along-axis extent dwarfs
    its siblings' fixes a wall that sits SEPARATE from a run's own
    glyphs (the '4381' case above) — but on the same real plan, '3306'
    has a diagonal wall physically touching two of its own digits,
    fusing them into ONE connected component: 3 components total for a
    4-character string. That component legitimately contains real digit
    ink and can't be cleanly separated from the wall at the component
    level; pruning it anyway (an earlier version of this fix did) threw
    away real ink and nearly broke the render (a 'fit-shrink-incomplete'
    warning at 147% of target width, where there had never been a
    problem before).

    The guard: only prune when at least as many components survive as
    the string has characters — i.e. nothing suggests a real glyph is
    already fused with something else.
    """
    image = np.full((40, 80, 3), 255, np.uint8)
    cv2.rectangle(image, (0, 10), (80, 32), (0, 0, 0), -1)    # wall fused with 2 digits
    cv2.rectangle(image, (30, 0), (45, 8), (0, 0, 0), -1)     # a separate, real digit
    cv2.rectangle(image, (30, 34), (45, 40), (0, 0, 0), -1)   # another separate, real digit
    # 3 disconnected components; the first spans nearly the full width
    # (cwid=80) while the other two are compact (cwid=15) — exactly the
    # shape that WOULD get the wide one pruned if nothing stopped it.

    bbox = (0.0, 0.0, 80.0, 40.0)
    unguarded = measure_ink_extent(image, bbox, angle_deg=0.0)  # expected_glyphs=0: never prunes
    still_fused = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=4)
    assert still_fused == unguarded, "3 components for a 4-char string must block pruning"

    # Positive control: with a component count that DOES meet or exceed
    # the character count, the same wide component is recognised as an
    # outlier and pruned — proving the guard above is what's protecting
    # the 4-character case, not that pruning never fires at all.
    pruned = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=3)
    assert pruned != unguarded
    assert pruned[0] < unguarded[0]
