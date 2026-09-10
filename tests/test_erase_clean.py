"""erase/clean.py edge cases and the pattern-detection regression."""

from __future__ import annotations

import numpy as np

from spejl.erase.clean import _looks_patterned, _ring_modal_colour, build_text_mask, erase_text


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
    import cv2

    cv2.putText(img, "Bad", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    box = (5, 20, 75, 60)
    result = erase_text(img, [box])
    region = result.image[box[1]:box[3], box[0]:box[2]]
    assert region.mean() >= 253.0, f"erase left the region averaging {region.mean():.1f}, not ~255"


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
