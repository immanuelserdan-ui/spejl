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
