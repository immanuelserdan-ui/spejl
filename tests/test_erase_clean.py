"""erase/clean.py edge cases and the pattern-detection regression."""

from __future__ import annotations

import numpy as np

import cv2

from spejl.erase.clean import (
    _looks_patterned,
    _ring_modal_colour,
    build_text_mask,
    erase_text,
    line_pixel_mask,
)


def test_ring_colour_on_pure_white_is_pure_white_not_the_quantisation_floor():
    """The critical regression: quantising ring pixels into 8-wide
    buckets to find the modal colour (robust against a ring that clips
    a black wall) previously RETURNED the bucket's own floor value —
    `255 // 8 * 8 == 248` — instead of the true colour of the pixels in
    that bucket. Every single erase fill was drawn in (248,248,248), a
    uniform ~3% darkening of every run on every plan, invisible at a
    glance but visible at zoom as a faint ghost with the exact
    silhouette of whatever text was erased. Confirmed on the golden
    fixture's 'Bad'/'1400' cluster before this fix."""
    img = np.full((60, 60, 3), 255, np.uint8)
    box = (20, 20, 40, 40)
    img[box[1]:box[3], box[0]:box[2]] = 0
    colour = _ring_modal_colour(img, box)
    assert tuple(int(c) for c in colour) == (255, 255, 255)


def test_erased_text_region_returns_to_true_paper_not_a_darker_bucket():
    """End-to-end version of the regression above, through the real
    erase_text() path rather than calling the helper directly.

    Checks the region's *mean*, not its min: a lone antialiasing-edge
    pixel right at the mask's own paper-tolerance boundary is expected
    and harmless. What the old bug produced was every fill pixel across
    the whole region landing at 248 instead of 255 — a small but
    uniform, image-wide shift the mean catches cleanly.
    """
    img = np.full((80, 80, 3), 255, np.uint8)
    cv2.putText(img, "Bad", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    box = (5, 20, 75, 60)
    result = erase_text(img, [box])
    region = result.image[box[1]:box[3], box[0]:box[2]]
    assert region.mean() >= 253.0, f"erase left the region averaging {region.mean():.1f}, not ~255"


def test_pattern_detection_is_not_diluted_by_an_earlier_boxs_fresh_fill():
    """Regression: erase_text's own box loop calls _looks_patterned
    against `work` — the SAME array it progressively flat-fills one box
    at a time — so by the time a LATER box is checked, its 6px ring can
    dip into an EARLIER box that was just filled a moment ago. Real
    plans crowd runs only a few pixels apart (a room name beside its
    own area figure), close enough for exactly this. Sampling the
    neighbour's fresh fill instead of its true original texture dilutes
    the measured variance below the pattern threshold, silently
    suppressing this warning for the crowded-label case it exists to
    catch. Checked against the ORIGINAL image now, not `work`.
    """
    img = np.full((100, 150, 3), 255, np.uint8)
    box_a = (10.0, 10.0, 65.0, 60.0)  # erased FIRST
    box_b = (68.0, 20.0, 78.0, 50.0)  # erased second; its own left ring
    #                                   reaches back into box A's territory
    # A patterned strip inside box A, near its right edge — within box
    # B's own ring reach, but nowhere near box A's own ring (so erasing
    # box A's own local-paper estimate isn't thrown off by it).
    shades = [130, 255, 140, 250, 135, 245, 150, 255]
    for i, y in enumerate(range(10, 60, 3)):
        img[y : y + 2, 58:65] = shades[i % len(shades)]

    result = erase_text(img, [box_a, box_b])
    codes = [f.code for f in result.flags]
    assert codes.count("erase-over-pattern") >= 1


def test_pattern_detection_ignores_the_just_filled_interior():
    """Regression: _looks_patterned sampled box+ring together with no
    exclusion of the box interior. Called right after that interior was
    flat-filled (erase_text's own sequence), the flat fill dwarfed the
    thin ring for any normal-width run, dragging variance to ~0 and
    silently suppressing the 'erase-over-pattern' warning it exists to
    raise — regardless of how patterned the real surround was.
    """
    img = np.full((120, 120, 3), 255, np.uint8)
    # A WIDE flat-filled box (like a real multi-character label) —
    # wide enough that the old box+ring sampling would be dominated by
    # its flat interior.
    box = (10, 40, 110, 70)
    img[box[1]:box[3], box[0]:box[2]] = 0  # simulates the post-fill state
    # Patterned surround (hatch lines) just outside the box.
    img[35, :] = 180
    img[38, :] = 180
    img[72, :] = 180
    img[75, :] = 180

    assert _looks_patterned(img, box) is True


def test_pattern_detection_false_on_a_genuinely_plain_surround():
    img = np.full((120, 120, 3), 255, np.uint8)
    box = (10, 40, 110, 70)
    img[box[1]:box[3], box[0]:box[2]] = 0
    assert _looks_patterned(img, box) is False


def test_ring_modal_colour_still_correct_after_the_shared_helper_refactor():
    img = np.full((60, 60, 3), 240, np.uint8)  # near-white paper
    box = (20, 20, 40, 40)
    img[box[1]:box[3], box[0]:box[2]] = 0
    colour = _ring_modal_colour(img, box)
    assert tuple(int(c) for c in colour) in {(240, 240, 240), (232, 232, 232)}


def test_a_tall_narrow_stroke_fully_inside_its_own_box_is_not_restored():
    """Regression: a room label's own letter strokes (a straight
    full-height vertical, as in 'K'/'k'/'l'/'d'/'b'/'h') can independently
    satisfy line_pixel_mask's own >=40px straight-run detector once the
    sheet's cap height clears it — confirmed on a real plan: 'Køkken'
    (cap height 54px) produced three separate components this way, one
    per straight vertical stroke in 'K', 'k', 'k', each ENTIRELY inside
    its own text box (never within 9px of the box's own edge). Before
    this fix, _restore_line_pixels painted every one of them straight
    back onto the erased canvas regardless — a ghost stroke rendered
    right through the freshly re-drawn glyph ('Køkken' came out as a
    smeared 'Kølkken!', 'Bad' as 'Badl'). A component that never extends
    past its own box is exactly this failure mode, not a real line, and
    must stay erased.
    """
    img = np.full((160, 200, 3), 255, np.uint8)
    # A 3px-wide, 50px-tall vertical stroke — long enough (>40px) for
    # line_pixel_mask's own detector — entirely inside the box below.
    img[60:110, 50:53] = 0
    box = (40.0, 55.0, 70.0, 115.0)  # comfortably contains the stroke

    # Sanity check the test actually exercises the fix: line_pixel_mask
    # must genuinely flag this stroke as "linework", the same false
    # positive the real 'K'/'k'/'k' strokes produced.
    assert np.count_nonzero(line_pixel_mask(img)[60:110, 50:53]) > 0

    result = erase_text(img, [box])
    assert result.repaired_px == 0
    region = result.image[55:115, 40:70]
    assert region.min() >= 250


def test_a_line_that_extends_well_past_its_box_is_still_restored():
    """The other side of the same mechanism: a genuine wall or dimension
    line that happens to run under a label continues into the
    surrounding sheet well past that one label — confirmed on a real
    plan extending 55 to 1428px beyond the text box whose erasure
    exposed it. This must still be repaired after the erase — the whole
    point of line_pixel_mask/_restore_line_pixels in the first place,
    and the case the fix above must not quietly break.
    """
    img = np.full((160, 200, 3), 255, np.uint8)
    img[79:82, :] = 0  # a line spanning nearly the full width
    box = (80.0, 70.0, 120.0, 90.0)  # sits in the middle of the line

    result = erase_text(img, [box])
    assert result.repaired_px > 0
    # The line must reappear under where the box was.
    assert result.image[80, 100].max() < 50


def test_a_repaired_line_keeps_its_antialiased_flank_not_just_its_core():
    """Regression: the erase and the repair disagreed on where a line
    ends, and the gap between them was a visible nick.

    ``build_text_mask`` erases everything below *local* paper — a
    deliberately low bar, so antialiased glyph fringes go too (see
    PAPER_TOLERANCE) — while ``line_pixel_mask`` labels linework by
    Otsu, which keeps only each line's solid core. So the erase
    consistently took one pixel more of every line than the repair knew
    to give back: repaired lines came out a pixel thin down one side,
    for the exact width of the label that crossed them. On the golden
    fixture that pixel accounted for 190 of the 319 pixels of real
    linework the pipeline was losing.

    The grey flank here is what an antialiased CAD export actually
    produces, and is dark enough for the erase to take but too light for
    Otsu to call linework — the precise gap between the two thresholds.
    """
    img = np.full((160, 200, 3), 255, np.uint8)
    img[79:82, :] = 0        # solid core, Otsu sees this
    img[78, :] = 150         # antialiased flank, only the erase sees it
    img[82, :] = 150
    box = (80.0, 70.0, 120.0, 90.0)

    result = erase_text(img, [box])

    under_the_box = result.image[:, 90:110]
    assert under_the_box[80].max() < 50, "core not repaired"
    assert under_the_box[78].max() < 200, "flank left erased — the nick this test is about"
    assert under_the_box[82].max() < 200


def test_a_repaired_line_has_the_same_weight_as_its_own_continuation():
    """The flank is restored at the line's own grey, not filled solid.
    Filling it with ink would thicken the line — the one thing
    _restore_line_pixels promises it never does — so the repaired
    stretch has to match the profile of the same line a few pixels
    outside the box, top edge, core and bottom edge alike.
    """
    img = np.full((160, 200, 3), 255, np.uint8)
    img[79:82, :] = 0
    img[78, :] = 150
    img[82, :] = 150
    box = (80.0, 70.0, 120.0, 90.0)

    result = erase_text(img, [box])

    # Inside the erased box, the line must look exactly like the stretch
    # of itself just outside it.
    repaired = result.image[76:85, 90:110]
    untouched = img[76:85, 10:30]
    assert np.array_equal(repaired.mean(axis=1).round(), untouched.mean(axis=1).round())


def test_a_repair_still_cannot_paint_outside_the_erased_footprint():
    """The flank band is dilated, so it is worth pinning down that the
    dilation cannot reach pixels the erase never touched: the band is
    intersected with the erase mask last, and everything beyond the box
    must come through the pipeline byte-identical."""
    img = np.full((160, 200, 3), 255, np.uint8)
    img[79:82, :] = 0
    img[78, :] = 150
    img[82, :] = 150
    box = (80.0, 70.0, 120.0, 90.0)

    result = erase_text(img, [box])

    assert np.array_equal(result.image[:, :60], img[:, :60])
    assert np.array_equal(result.image[:, 140:], img[:, 140:])
    assert np.array_equal(result.image[:60, :], img[:60, :])


def test_a_repair_does_not_paint_the_old_glyph_back_onto_the_line_it_crossed():
    """The trap in repairing a line from its own original pixels: where
    a glyph sat ON the line, those pixels are part glyph, so restoring
    them faithfully stamps the old, pre-mirror string back into the
    line — which then survives into the mirrored sheet as a ghost of
    text that is supposed to have moved.

    Invisible in the ordinary black-type-on-black-linework case, and
    plainly legible the moment the two differ in tone, which is why the
    mark here is *lighter* than the wall it crosses (a dark grey
    annotation over black poché). The line's core is therefore repainted
    flat, in ink — a flat fill cannot reproduce a glyph shape at all —
    and only the flank, where no flat value would be right, comes from
    the original.

    The mark stays dark enough to fall on the ink side of Otsu's split,
    so the wall remains one solid component and this test isolates the
    repair's choice of *value*. A mark light enough to punch a hole in
    the binary would instead break the wall apart for
    :func:`line_pixel_mask`, which is a different matter entirely.
    """
    img = np.full((160, 200, 3), 255, np.uint8)
    img[70:90, :] = 0                 # a thick black wall, full width
    img[76:84, 95:105] = 60           # a dark grey mark on it, label-shaped
    box = (85.0, 65.0, 115.0, 95.0)   # the label box around that mark

    result = erase_text(img, [box])

    wall = cv2.cvtColor(result.image[74:86, 92:108], cv2.COLOR_BGR2GRAY)
    assert wall.max() == 0, f"the grey mark survived the erase at {wall.max()}, back on the wall"


# ---------------------------------------------------------------------------
# Reverse-out text: ink lighter than its own background
# ---------------------------------------------------------------------------


def test_reverse_out_text_is_actually_erased_not_left_standing():
    """Regression: build_text_mask used to read 'local paper' from the
    BOX'S OWN light end — a percentile of its own interior. That works
    when the glyph is the box's darkest content, which is ordinary type,
    but inverts completely for reverse-out text (white lettering knocked
    out of a filled panel, e.g. a room name on a shaded background): the
    interior's light end IS the glyph, so 'darker than the light end'
    never once fired. The panel got erased as 'not paper' and the white
    lettering was judged paper and left untouched — confirmed surviving
    whole, unmirrored, sitting exactly where it always was, on a
    synthetic reverse-out label.

    The fix reads local paper from the RING outside the box, which has
    no such ambiguity, and masks by absolute deviation from it in either
    direction. Checked here at the level that actually matters: after
    erase_text, no run of white pixels big enough to still read as a
    letter may survive inside the panel.
    """
    img = np.full((160, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (40, 30), (260, 130), (0, 0, 0), -1)  # a dark panel
    cv2.putText(img, "BAD", (70, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
               (255, 255, 255), 2, cv2.LINE_AA)               # reverse-out label
    box = (65.0, 60.0, 200.0, 105.0)

    result = erase_text(img, [box])

    panel = cv2.cvtColor(result.image[35:125, 45:255], cv2.COLOR_BGR2GRAY)
    survivors = int(np.count_nonzero(panel > 200))
    # Before the fix, this was 813 white pixels — the whole word, fully
    # legible. A modest antialiased fringe (a known, documented, and far
    # less severe residual — see _restore_line_pixels) is not what this
    # test is guarding against; a run anywhere near "still a readable
    # word" is.
    assert survivors < 250, f"{survivors} white pixels survived — the label reads as text, not a fringe"


def test_reverse_out_masks_exactly_the_glyph_not_the_panel_around_it():
    """The other side of the same fix: the panel itself — real, correct
    material — must not be swept up as "not paper" just because it is
    dark and the glyph inside it is light. build_text_mask's job is
    still to flag the glyph ALONE.
    """
    img = np.full((160, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (40, 30), (260, 130), (0, 0, 0), -1)
    cv2.putText(img, "BAD", (70, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
               (255, 255, 255), 2, cv2.LINE_AA)
    box = (65.0, 60.0, 200.0, 105.0)

    mask = build_text_mask(img, [box])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Every pixel the mask flags for erase, within the box, must be part
    # of the glyph or its own antialiased fringe — never TRUE panel ink
    # (near-pure black, well below the fringe's own gradient) misread as
    # ink. <10 rather than a looser cut: the glyph's own antialiasing
    # legitimately spans a wide grey ramp up from true black (confirmed
    # values 16-49 flagged here, all fringe, none of it a bug) — this
    # test is about the panel's own solid interior, not that ramp.
    x0, y0, x1, y1 = (int(v) for v in box)
    flagged = mask[y0:y1, x0:x1] > 0
    panel_pixels_flagged = int(np.count_nonzero(flagged & (gray[y0:y1, x0:x1] < 10)))
    assert panel_pixels_flagged == 0, "the panel's own solid ink was flagged for erase, not just the glyph"


def test_ordinary_dark_on_light_text_is_unaffected_by_the_ring_based_paper():
    """The common case this change touches the machinery of but must not
    change the outcome of: plain dark type on a plain light sheet, no
    panel in sight, must erase exactly as before."""
    img = np.full((100, 200, 3), 255, np.uint8)
    cv2.putText(img, "Bad", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    box = (25.0, 30.0, 110.0, 70.0)

    result = erase_text(img, [box])

    x0, y0, x1, y1 = (int(v) for v in box)
    region = cv2.cvtColor(result.image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    assert region.min() >= 240, "ordinary dark-on-light text was not fully erased"
