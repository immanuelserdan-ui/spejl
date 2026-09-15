"""Stage S7 — draw each run upright at its mirrored anchor.

Three things this module refuses to do, each of which would show:

* **Anchor on a corner.** A re-rendered string is never exactly the
  original's pixel width, so corner-anchoring drifts. The centre is
  anchored instead, which keeps a label on the centreline the drafter
  put it on.
* **Let a run outgrow its box.** After rendering, actual ink bounds are
  measured; anything more than 4% over the original is shrunk (tracking
  first, then size). A label can therefore never grow into a wall.
* **Draw aliased type.** Rendering happens at 4x and downsamples, so the
  edges match the CAD export's own antialiasing rather than looking
  stamped on.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spejl.style.metrics import TextStyle, font_measure_width

SUPERSAMPLE = 4
OVERFLOW_TOLERANCE = 1.04


@dataclass(frozen=True)
class RenderedRun:
    """One drawn run and what had to be done to make it fit."""

    shrunk: bool
    final_px_size: int
    final_tracking: float
    ink_width: float
    ink_height: float
    collided: bool


def _draw_string(
    text: str, style: TextStyle, px_size: int, tracking_em: float, scale: int
) -> Image.Image:
    """Render to a tight transparent RGBA tile, tracking applied per glyph.

    ``tracking_em`` is a fraction of ``px_size`` (see
    ``style.metrics.solve_tracking``), converted to pixels here at
    whatever size is actually being drawn — so a shrink-to-fit pass that
    reduces ``px_size`` shrinks the letter-spacing right along with it,
    rather than reapplying an unchanged pixel gap to smaller and smaller
    glyphs until they overlap.

    ``style.width_scale`` (see ``style.metrics.solve_horizontal_scale``)
    is applied last, as a horizontal resize of the finished tile — baked
    in here rather than by the caller, so every consumer of this tile's
    width (the fit-to-box shrink loop included) measures the scaled
    width automatically, with nothing further to keep in sync.
    """
    font = ImageFont.truetype(style.font_path, max(1, px_size * scale))
    track = tracking_em * px_size * scale

    widths = [font.getlength(ch) for ch in text]
    total_w = max(1, int(np.ceil(sum(widths) + track * max(0, len(text) - 1))))
    ascent, descent = font.getmetrics()
    total_h = max(1, ascent + descent)

    pad = max(2, px_size * scale // 4)
    tile = Image.new("RGBA", (total_w + pad * 2, total_h + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)

    x = float(pad)
    for ch, advance in zip(text, widths):
        draw.text((x, pad), ch, font=font, fill=(*style.ink, 255))
        x += advance + track

    tile = tile.crop(tile.getbbox() or (0, 0, tile.width, tile.height))

    if abs(style.width_scale - 1.0) > 1e-3:
        scaled_w = max(1, round(tile.width * style.width_scale))
        tile = tile.resize((scaled_w, tile.height), Image.LANCZOS)

    return tile


def render_run(
    canvas: np.ndarray,
    text: str,
    center: tuple[float, float],
    angle_deg: float,
    style: TextStyle,
    target_size: tuple[float, float] | None = None,
    linework_mask: np.ndarray | None = None,
) -> RenderedRun:
    """Composite ``text`` onto ``canvas`` (BGR, modified in place).

    ``target_size`` is the original run's (along, across) ink extent; when
    given, it drives the fit-to-box guard.

    ``linework_mask``, if given, is mutated in place: this run's own ink
    is stamped into it after the collision check, so a caller that passes
    the *same* mask object to every run in a page (see raster/pipeline.py)
    gets collision detection against every run rendered so far, not just
    the static geometry the mask started with.
    """
    px_size, tracking = style.px_size, style.tracking  # tracking: em-relative
    shrunk = False

    tile = _draw_string(text, style, px_size, tracking, SUPERSAMPLE)

    if target_size is not None:
        target_along = max(1.0, target_size[0])
        target_across = max(1.0, target_size[1])
        for _attempt in range(6):
            along = tile.width / SUPERSAMPLE
            across = tile.height / SUPERSAMPLE
            # Both dimensions must fit, not just width: the module's own
            # header promises "a run can therefore never grow into a
            # wall" without singling out an axis, but only `along` was
            # ever checked here — a font substitution with taller
            # ascent+descent than the original detected box (a
            # different font, or glyphs with descenders the original
            # crop didn't have) could overflow vertically with `shrunk`
            # staying False and no shrink ever attempted, silently
            # breaking that promise on the axis nobody was checking.
            if (
                along <= target_along * OVERFLOW_TOLERANCE
                and across <= target_across * OVERFLOW_TOLERANCE
            ):
                break
            shrunk = True
            # Tracking is the cheaper knob — reducing it preserves the
            # glyph size the rest of the sheet is drawn at. Compared as
            # an em fraction throughout, so this stays meaningful as
            # px_size drops on later iterations.
            if tracking > 0.02:
                tracking = max(0.0, tracking - max(0.02, tracking * 0.4))
            else:
                px_size = max(1, px_size - 1)
            tile = _draw_string(text, style, px_size, tracking, SUPERSAMPLE)

    # Downsample to final size, then rotate. Rotating after downsampling
    # keeps the supersample cost linear and PIL's bicubic rotation is
    # already smooth at 1x.
    final_w = max(1, int(round(tile.width / SUPERSAMPLE)))
    final_h = max(1, int(round(tile.height / SUPERSAMPLE)))
    tile = tile.resize((final_w, final_h), Image.LANCZOS)

    # Captured HERE, before rotation — these are the (along, across)
    # dimensions in the glyph's own frame, the same frame `target_size`
    # is measured in. A rotated rectangle's axis-aligned bounding box
    # (PIL's `expand=True`) is strictly larger than the rectangle itself
    # on both axes for any non-cardinal angle, so measuring ink extent
    # AFTER rotation and comparing it against a pre-rotation target
    # inflates the apparent size purely from rotation geometry — most
    # visible on the narrow "across" axis of a long, thin, gently-tilted
    # dimension number: a few degrees of tilt add only a modest fraction
    # to the long "along" axis but a LARGE fraction to the short one
    # (confirmed on real tilted dimension numbers: 120-250% "overflow"
    # reported on height alone, for runs whose actual glyph tile fit
    # its target cleanly and never even attempted a shrink).
    ink_width, ink_height = float(final_w), float(final_h)

    if abs(angle_deg) > 1e-6:
        # PIL rotates counter-clockwise in a y-down image, which matches
        # Spejl's angle convention (90° = reads bottom-to-top).
        tile = tile.rotate(angle_deg, expand=True, resample=Image.BICUBIC)

    collided = _composite(canvas, tile, center, linework_mask)
    return RenderedRun(
        shrunk=shrunk,
        final_px_size=px_size,
        final_tracking=tracking,
        ink_width=ink_width,
        ink_height=ink_height,
        collided=collided,
    )


def _composite(
    canvas: np.ndarray,
    tile: Image.Image,
    center: tuple[float, float],
    linework_mask: np.ndarray | None,
) -> bool:
    """Alpha-blend ``tile`` centred on ``center``; report any collision."""
    h, w = canvas.shape[:2]
    tw, th = tile.width, tile.height
    x0 = int(round(center[0] - tw / 2))
    y0 = int(round(center[1] - th / 2))

    # Clip to canvas.
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(w, x0 + tw), min(h, y0 + th)
    if dx1 <= dx0 or dy1 <= dy0:
        return False

    patch = np.array(tile)[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    if patch.size == 0:
        return False

    alpha = (patch[:, :, 3:4].astype(np.float32)) / 255.0
    rgb = patch[:, :, :3].astype(np.float32)
    bgr = rgb[:, :, ::-1]  # PIL is RGB, OpenCV canvas is BGR

    region = canvas[dy0:dy1, dx0:dx1].astype(np.float32)
    canvas[dy0:dy1, dx0:dx1] = (bgr * alpha + region * (1 - alpha)).astype(np.uint8)

    if linework_mask is None:
        return False
    ink_here = (patch[:, :, 3] > 128)
    region_mask = linework_mask[dy0:dy1, dx0:dx1]  # a view, not a copy
    lines_here = region_mask > 0
    collided = bool(np.logical_and(ink_here, lines_here).any())
    # Stamp this run's own ink into the shared mask before returning, so
    # the *next* call (the next run rendered onto the same canvas) is
    # checked against it too. Without this, two runs whose boxes overlap
    # only each other — never the original linework — never collide with
    # anything as far as either call can tell, and both render clean
    # while silently overlapping on the canvas (confirmed on a real
    # plan: a mirrored '4381' dimension landing on top of the 'Entre'
    # label, with the pipeline's own collision flag staying silent
    # because it only ever compared against linework).
    region_mask[ink_here] = 255
    return collided


def linework_mask_for(image: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    """Dark pixels that are *not* text — what a re-rendered run must avoid.

    Built from the cleaned plate, so the mask reflects the geometry as it
    will actually appear under the new text.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _t, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.bitwise_and(dark, cv2.bitwise_not(text_mask))
