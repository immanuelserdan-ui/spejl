"""style/metrics.py edge cases the round-trip fixture never exercises —
degenerate boxes and pathological text, not the clean synthetic case.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spejl.style.metrics import (
    _reads_vertically,
    fit_font_size,
    measure_ink_center,
    measure_ink_extent,
    resolve_font,
)


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


@pytest.mark.parametrize(
    ("angle_deg", "expected_vertical"),
    [
        (0.0, False),
        (10.0, False),
        (170.0, False),   # near +180 -- a HORIZONTAL direction, same as 0
        (-175.0, False),  # near -180 -- same physical direction as +180
        (90.0, True),
        (-90.0, True),
        (95.0, True),
        (-88.0, True),
    ],
)
def test_reads_vertically_handles_the_180_wraparound(angle_deg: float, expected_vertical: bool):
    """Regression: a plain abs(angle_deg) > 45 test mislabelled anything
    near 180/-180 (a HORIZONTAL direction, same as 0) as vertical, since
    raw magnitude has no notion that 180 and -180 are the same nearby
    direction. Upside-down horizontal text (e.g. 170 deg) got its width
    and height swapped everywhere this decision is used.
    """
    assert _reads_vertically(angle_deg) is expected_vertical


def test_near_180_degree_text_is_measured_as_horizontal_not_swapped():
    """End-to-end version of the wraparound fix, through
    measure_ink_extent's own fallback path: an 80x30 box read at 170
    deg (near +180, genuinely horizontal) must measure along=80,
    across=30 -- the SAME as the 0-degree case -- not swapped to
    along=30, across=80 as a vertical run would be.
    """
    image = np.full((50, 50, 3), 255, np.uint8)
    along, across = measure_ink_extent(image, (100.0, 100.0, 180.0, 130.0), 170.0)
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

    # Positive control: with an expected_glyphs value BOTH the outer
    # gate (>= before pruning) and the inner one (>= after pruning — see
    # _drop_foreign_strokes' own equivalent guard, added for the exact
    # same reason: the outlier itself might be the one holding real
    # glyph ink) can be satisfied by, the same wide component is
    # recognised as an outlier and pruned — proving the guards above are
    # what's protecting the 4-character case, not that pruning never
    # fires at all. 2, not 3: pruning the wide component always leaves
    # exactly the 2 real digits, so 3 would trip the inner guard too.
    pruned = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=2)
    assert pruned != unguarded
    assert pruned[0] < unguarded[0]


def test_a_genuinely_wide_glyph_is_not_pruned_if_that_would_undercount_the_text():
    """Regression: _drop_foreign_strokes only ever checked that ENOUGH
    components survived to reach this point (the call site's own outer
    gate) — not that pruning its own outlier wouldn't drop the count
    BELOW the character count. An outlier by along-span is not
    necessarily a foreign stroke; it could be a genuinely wide real
    glyph (mixing 'W'/'M' with narrower siblings, e.g. this project's
    own 'Walk-in' lexicon entry). If dropping it would leave fewer
    components than the text has characters, there is no way to tell
    the two apart, so it must be kept — the same reasoning the outer
    gate already applies, extended to the filter's own result.
    """
    image = np.full((40, 130, 3), 255, np.uint8)
    cv2.rectangle(image, (0, 10), (80, 30), (0, 0, 0), -1)    # a wide component
    cv2.rectangle(image, (90, 10), (100, 30), (0, 0, 0), -1)  # 3 normal-width siblings
    cv2.rectangle(image, (105, 10), (115, 30), (0, 0, 0), -1)
    cv2.rectangle(image, (120, 10), (130, 30), (0, 0, 0), -1)
    bbox = (0.0, 0.0, 130.0, 40.0)

    # 4 components total; dropping the wide one leaves 3 -- short of a
    # 4-character text, so it must stay.
    protected = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=4)
    assert protected[0] > 100.0  # the wide component's own ink is still included

    # With only 3 characters, dropping it leaves exactly enough (3) --
    # the safety net doesn't block it here, proving it's the count that
    # matters, not that this filter never fires at all.
    pruned = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=3)
    assert pruned[0] < protected[0]


def test_a_sparse_outlier_is_not_dropped_if_that_would_undercount_the_text():
    """Same safety net as above, for _drop_sparse_linework: a low-fill
    outlier is not dropped when doing so would leave fewer surviving
    components than the text has characters.
    """
    image = np.full((40, 130, 3), 255, np.uint8)
    cv2.rectangle(image, (0, 5), (60, 35), (0, 0, 0), 4)       # hollow (sparse) component, fill ~0.37
    cv2.rectangle(image, (70, 10), (80, 30), (0, 0, 0), -1)    # 3 solid siblings, fill 1.0
    cv2.rectangle(image, (90, 10), (100, 30), (0, 0, 0), -1)
    cv2.rectangle(image, (110, 10), (120, 30), (0, 0, 0), -1)
    bbox = (0.0, 0.0, 130.0, 40.0)

    protected = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=4)
    assert protected[0] > 100.0  # the sparse component's own ink is still included

    pruned = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=3)
    assert pruned[0] < protected[0]


def test_a_small_edge_toucher_is_clipped_not_dropped_if_dropping_would_undercount():
    """Same safety net for _drop_edge_touching_intrusions: a small
    edge-touching component that would normally be dropped outright is
    instead CLIPPED (its own real ink kept, cross-axis excess trimmed)
    when dropping it entirely would leave too few components — the same
    "clip, don't discard" treatment this function already gives a LARGE
    edge intrusion, extended to a small one the text can't spare.
    """
    image = np.full((40, 130, 3), 255, np.uint8)
    cv2.rectangle(image, (0, 0), (5, 40), (0, 0, 0), -1)       # small, touches top AND bottom edge
    cv2.rectangle(image, (25, 10), (35, 30), (0, 0, 0), -1)    # 3 ordinary interior siblings
    cv2.rectangle(image, (45, 10), (55, 30), (0, 0, 0), -1)
    cv2.rectangle(image, (65, 10), (75, 30), (0, 0, 0), -1)
    bbox = (0.0, 0.0, 130.0, 40.0)

    protected = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=4)
    # The edge-toucher's own left edge (x=0) is still represented.
    assert protected[0] > 70.0

    dropped = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=3)
    assert dropped[0] < protected[0]


def test_a_small_real_glyph_fragment_joins_the_cluster_if_dropping_it_would_undercount():
    """Same safety net for _main_glyph_cluster's own 8%-of-anchor-area
    floor: a real but small letter fragment (an 'i' or 'j' dot, in
    spirit — disconnected from the rest of the glyph, small relative to
    a much larger anchor in the same run) that falls under the floor is
    still admitted as a growth candidate when the text doesn't have
    enough other components to spare it. It still has to earn its way
    into the cluster by proximity (the region-growing pass below) —
    this only stops it being excluded from consideration outright.
    """
    image = np.full((40, 110, 3), 255, np.uint8)
    cv2.rectangle(image, (10, 10), (90, 30), (0, 0, 0), -1)   # a large anchor glyph
    cv2.rectangle(image, (95, 10), (100, 15), (0, 0, 0), -1)  # a tiny fragment, close by

    bbox = (0.0, 0.0, 110.0, 40.0)
    unprotected = measure_ink_extent(image, bbox, angle_deg=0.0)  # expected_glyphs=0: floor always applies
    protected = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=2)

    assert unprotected[0] < 85.0  # only the anchor's own extent
    assert protected[0] > 85.0    # the small fragment is now included too


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


def test_a_mid_span_sparse_component_is_excluded_even_below_the_span_box_cutoff():
    """Regression: the initial per-component filter only excluded a
    sparse (low fill-ratio) component when it ALSO spanned at least 80%
    of the crop on some axis (the same span_box test the elongation
    check uses) — a rule/dimension line spanning a more modest majority
    (short of that 80% cutoff) escaped entirely, regardless of how
    sparse it was. Sparse alone, with no span requirement, is already
    established elsewhere in this file as a safe, unconditional signal
    (a genuine glyph never measures below ~0.3 fill even fused);
    span_box only still gates the classically-thin `elongated` case,
    where a narrow REAL glyph ('1', 'l') could otherwise be caught.
    """
    image = np.full((40, 100, 3), 255, np.uint8)
    # A hollow rectangle -- low fill ratio -- spanning 60% of the crop's
    # width and 75% of its height, both short of the 80% spans_box cutoff.
    cv2.rectangle(image, (5, 5), (65, 35), (0, 0, 0), 1)
    # Two real, solidly-filled glyphs, comfortably inset from every edge.
    cv2.rectangle(image, (75, 12), (85, 28), (0, 0, 0), -1)
    cv2.rectangle(image, (90, 12), (98, 28), (0, 0, 0), -1)

    bbox = (0.0, 0.0, 100.0, 40.0)
    along, _across = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=2)

    # The true glyphs span x=75..98 (23px). If the sparse rectangle had
    # survived, it would dominate the measured extent (spanning x=5..98,
    # 93px), since it is by far the largest component.
    assert along < 30.0


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


def test_a_glyph_fused_with_an_edge_touching_wall_is_clipped_not_dropped():
    """Regression: on the same real plan, 'Depot's own 'D' is physically
    TOUCHING the wall beside it -- not merely adjacent like the door-jamb
    near 'Entre', fused into one connected component no component-level
    shape test (nor, confirmed separately, a small erosion) can cleanly
    split. Dropping that fused component outright (what an earlier
    version of _drop_edge_touching_intrusions did, correctly fixing the
    HEIGHT-inflation bug that motivated it) also discarded 'D' along
    with the wall, undermeasuring the run's WIDTH (69px instead of the
    true ~96px) and forcing an unrelated render-time shrink -- 'Depot'
    rendered visibly smaller than 'Gang'/'Kælderrum' right next to it,
    with no flag pointing at why.

    The fix: when an edge-touching component is too big to be pure
    linework (compared against the run's own unambiguous letters), CLIP
    its cross-axis extent to what those letters actually occupy instead
    of discarding it outright -- the wall's excess height is cut away,
    but whatever of its along-baseline extent is real, fused-in glyph
    ink is kept. A genuinely small edge-toucher (noise, not a fusion)
    still gets dropped outright, unaffected by this change.
    """
    image = np.full((44, 100, 3), 255, np.uint8)
    # A wall fused with a letter -- touches both the top and bottom.
    cv2.rectangle(image, (0, 0), (24, 44), (0, 0, 0), -1)
    # 3 more real, unambiguous letters, well clear of every crop edge.
    cv2.rectangle(image, (40, 10), (55, 34), (0, 0, 0), -1)
    cv2.rectangle(image, (60, 10), (75, 34), (0, 0, 0), -1)
    cv2.rectangle(image, (80, 10), (95, 34), (0, 0, 0), -1)

    bbox = (0.0, 0.0, 100.0, 44.0)
    along, across = measure_ink_extent(image, bbox, angle_deg=0.0, expected_glyphs=4)

    # Height stays correct -- clipped to the real letters' own span, not
    # the full 44px crop the wall alone would have set.
    assert across == pytest.approx(24.0, abs=2.0)
    # Width now reflects the fused component's contribution too, not
    # just the 3 unambiguous letters (which alone would measure ~55px).
    assert along > 80.0
