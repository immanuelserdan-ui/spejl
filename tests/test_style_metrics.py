"""style/metrics.py edge cases the round-trip fixture never exercises —
degenerate boxes and pathological text, not the clean synthetic case.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spejl.style.metrics import fit_font_size, measure_ink_center, measure_ink_extent, resolve_font


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
    cv2.rectangle(image, (0, 12), (80, 27), (0, 0, 0), -1)    # wall fused with 2 digits
    cv2.rectangle(image, (30, 0), (50, 10), (0, 0, 0), -1)    # a separate, real digit
    cv2.rectangle(image, (30, 29), (50, 40), (0, 0, 0), -1)   # another separate, real digit
    # 3 disconnected components; the first spans nearly the full width
    # (cwid=80) while the other two are compact (cwid=20) — exactly the
    # shape that WOULD get the wide one pruned if nothing stopped it.
    # Sized so the two real digits still clear _core_candidates' own
    # 8%-of-the-largest-component threshold (their area must be a
    # sizeable fraction of the wall's, not a sliver), so this test
    # continues to exercise the pruning path it's meant to.

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


def test_real_glyphs_outnumbered_by_small_marks_are_not_wrongly_pruned():
    """Regression: on the same real plan as 'Entre' (same project,
    same dashed reference line), a garbled read of 'Vær. 3' ('--Vr.3')
    crossed by that line had 5 tiny dash/dot fragments against only 4
    real letter-ish components — contaminants OUTNUMBERING the glyphs
    they're contaminating, the opposite ratio from the '4381'/'Entre'
    cases the two outlier filters were built for.

    Taking the median across all 9 components (as an earlier version of
    this fix did) dragged the along-axis median down to the dashes' own
    ~7px width, making the real (and simply average-width) 32px-wide
    'r' glyph look like the oversized outlier and get wrongly dropped —
    under-measuring the whole run and nearly breaking its render.

    _core_candidates fixes this by excluding tiny-by-pixel-count
    fragments from the median calculation itself (not just from the
    final answer), so the median reflects only plausibly-glyph-sized
    components regardless of how many small contaminants sit alongside
    them.
    """
    image = np.full((34, 140, 3), 255, np.uint8)
    # 4 real letters, deliberately uneven widths (24/32/11/23) --
    # matching 'V'/'r'/[mid]/'3's own measured widths on the real plan.
    cv2.rectangle(image, (10, 5), (34, 25), (0, 0, 0), -1)
    cv2.rectangle(image, (40, 5), (72, 25), (0, 0, 0), -1)
    cv2.rectangle(image, (78, 5), (89, 25), (0, 0, 0), -1)
    cv2.rectangle(image, (95, 5), (118, 25), (0, 0, 0), -1)
    # 5 tiny dash fragments (a crossing reference line), OUTNUMBERING
    # the 4 real letters -- sitting in their own row so they stay
    # disconnected components rather than touching any letter.
    for dx0 in (0, 36, 74, 91, 120):
        cv2.rectangle(image, (dx0, 28), (dx0 + 7, 29), (0, 0, 0), -1)

    bbox = (0.0, 0.0, 140.0, 34.0)
    along, _across = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=4)

    # The true letters span x=10..118 (108px). A wrongly-pruned run
    # collapses to whichever single narrow component survived (as low
    # as 11px) -- so demand the measurement reflect all 4 real letters,
    # not a fragment of them.
    assert along > 90.0


def test_a_wall_touching_the_crops_edge_is_dropped_even_when_neither_other_filter_catches_it():
    """Regression: on the same real plan, 'Depot's detection box has a
    wall stroke running along its left edge -- moderate width (not an
    outlier vs. its siblings' along-axis extent) and solidly filled
    (not sparse like the door-jamb near 'Entre'), so it tripped NEITHER
    _drop_foreign_strokes NOR _drop_sparse_linework. Being the single
    largest component by bounding-box area, it still became the glyph
    cluster's anchor and, since it spanned the crop's full height,
    single-handedly set the measured cap height to the full crop height
    -- fitting a font a third larger than every sibling room label.

    The fix relies on this codebase's own stated assumption (see
    measure_ink_extent's docstring): a detector's box is padded around
    its own text, so real glyph ink should never reach the crop edge.
    Something that DOES touch the cross-baseline edge is presumed to be
    linework continuing beyond the box, regardless of its width or
    fill ratio.
    """
    image = np.full((40, 100, 3), 255, np.uint8)
    cv2.rectangle(image, (0, 0), (20, 40), (0, 0, 0), -1)     # wall: touches top AND bottom
    cv2.rectangle(image, (30, 8), (45, 32), (0, 0, 0), -1)    # 3 real letters, comfortably
    cv2.rectangle(image, (50, 8), (65, 32), (0, 0, 0), -1)    # inset from every crop edge
    cv2.rectangle(image, (70, 8), (85, 32), (0, 0, 0), -1)

    bbox = (0.0, 0.0, 100.0, 40.0)
    unguarded = measure_ink_extent(image, bbox, angle_deg=0.0)  # expected_glyphs=0: never prunes
    guarded = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=3)

    assert unguarded[1] == pytest.approx(40.0)  # wall sets across to the full crop height
    assert guarded[1] == pytest.approx(24.0, abs=2.0)  # letters' own true height, wall excluded


def test_measure_ink_center_ignores_asymmetric_contamination():
    """Regression: a real plan's '---Entre' detection box was padded
    much further on its LEFT edge than its right (dash-noise from a
    crossing reference line, the same contamination
    _drop_sparse_linework's door-jamb case is), so the RAW box's own
    geometric centre sat measurably left of where the word 'Entre'
    itself actually centres. The pipeline used to anchor the re-render
    on that raw centre — mirroring turned a small leftward bias in the
    source into a visible RIGHTWARD one in the output, crowding the
    label against the wrong wall of its own room.

    measure_ink_center must report the ACTUAL glyph cluster's centre,
    not the padded box's — matching measure_ink_extent's own long-
    standing reasoning for why it measures ink instead of trusting the
    box for size, applied here to position instead.
    """
    image = np.full((40, 80, 3), 255, np.uint8)
    cv2.rectangle(image, (5, 0), (20, 38), (0, 0, 0), 1)      # sparse jamb, LEFT side only
    cv2.rectangle(image, (35, 10), (48, 30), (0, 0, 0), -1)   # real letter 1
    cv2.rectangle(image, (52, 10), (65, 30), (0, 0, 0), -1)   # real letter 2

    bbox = (0.0, 0.0, 80.0, 40.0)
    raw_center_x = 40.0  # the box's own midpoint, pulled left by the jamb padding
    true_center = measure_ink_center(image, bbox, angle_deg=0.0, expected_glyphs=2)

    # The real letters span x=35..65 (centre 50.0) -- the corrected
    # centre must land there, not at the raw box's midpoint.
    assert true_center[0] > raw_center_x
    assert true_center[0] == pytest.approx(50.0, abs=3.0)


def test_measure_ink_center_falls_back_to_the_raw_box_when_no_ink_is_found():
    image = np.full((20, 20, 3), 255, np.uint8)  # blank -- no ink anywhere
    bbox = (0.0, 0.0, 20.0, 20.0)
    center = measure_ink_center(image, bbox, angle_deg=0.0)
    assert center == pytest.approx((10.0, 10.0))
