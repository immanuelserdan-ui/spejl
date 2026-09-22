"""Stage S7 — draw each run upright at its mirrored anchor.

Type goes **behind** the drawing. The sheet is composited in two layers
— paper, then the re-rendered runs, then the geometry over the top — so
no label can paint over a wall, an arc or a dimension line. The two
layers are not equally recoverable: a run is reconstructed from a string
and a measured style, and can be redrawn at any time, whereas linework a
glyph painted over is simply gone from the output. Keeping the drawing
in front means a mis-anchored label is a legible mistake sitting under
intact geometry rather than a silent hole in the plan.

Three further things this module refuses to do, each of which would show:

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

from dataclasses import dataclass, replace
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spejl.style.metrics import TextStyle, font_measure_width

SUPERSAMPLE = 4
OVERFLOW_TOLERANCE = 1.04

# A proximity-based collision check (flagging text that comes within a
# fixed margin of linework, not just literal pixel touching) was tried
# here and reverted: a dimension number sitting close to — but not
# touching — a nearby line is not actually unusual. It's the ORDINARY
# case for a dimension number sitting close to its OWN dimension line,
# which is exactly where drafting convention puts it. A fixed-distance
# threshold has no way to tell "close to the line it labels" apart from
# "close to something unrelated it shouldn't be near" — tested against
# the real project corpus, it flagged the clear majority of ordinary,
# correctly-placed dimension numbers on every file, and broke this
# project's own golden-fixture regression test
# (test_no_text_is_drawn_over_linework), which exists specifically to
# confirm a clean render produces NO collision flags at all. Reverted;
# exact pixel-touching stays the collision signal.


@dataclass(frozen=True)
class RenderedRun:
    """One drawn run and what had to be done to make it fit."""

    shrunk: bool
    final_px_size: int
    final_tracking: float
    ink_width: float
    ink_height: float
    collided: bool
    # Share of this run's own ink the drawing layer covers, 0.0 to 1.0.
    # A few percent is ordinary — a dimension number sitting on its own
    # dimension line — but a run that is mostly or entirely behind
    # geometry is unreadable on the output sheet, which is the one thing
    # drawing-over-type costs and therefore the one thing worth
    # reporting. Zero when no drawing layer was supplied.
    hidden: float = 0.0


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
    if style.font_variation is not None:
        font.set_variation_by_name(style.font_variation)
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
    drawing_alpha: np.ndarray | None = None,
) -> RenderedRun:
    """Composite ``text`` onto ``canvas`` (BGR, modified in place).

    ``target_size`` is the original run's (along, across) ink extent; when
    given, it drives the fit-to-box guard.

    ``linework_mask``, if given, is mutated in place: this run's own ink
    is stamped into it after the collision check, so a caller that passes
    the *same* mask object to every run in a page (see raster/pipeline.py)
    gets collision detection against every run rendered so far, not just
    the static geometry the mask started with.

    ``drawing_alpha`` (see :func:`drawing_alpha_for`) is the geometry
    layer this run is drawn UNDERNEATH, and is never modified — unlike
    ``linework_mask`` it describes the sheet's drawing alone, so runs
    hide behind the plan but not behind each other, and the layer stays
    identical no matter what order the page's runs happen to be drawn
    in.
    """
    px_size, tracking = style.px_size, style.tracking  # tracking: em-relative
    width_scale = style.width_scale
    numeric_run = bool(re.fullmatch(r"[0-9][0-9 .,:/\\-]*", text.strip()))
    shrunk = False

    tile = _draw_string(text, replace(style, width_scale=width_scale), px_size, tracking, SUPERSAMPLE)

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
            if numeric_run and width_scale > 0.76:
                # Dimension strings are the strongest style cue on a plan.
                # Preserve their measured cap height and spacing; if a
                # substitute font is wider, compress its advance width
                # instead of shrinking px_size (which made labels such as
                # 1961 and 1383 visibly smaller than their host text).
                width_scale = max(0.76, width_scale * 0.94)
            elif tracking > 0.02:
                tracking = max(0.0, tracking - max(0.02, tracking * 0.4))
            else:
                px_size = max(1, px_size - 1)
            tile = _draw_string(
                text, replace(style, width_scale=width_scale), px_size, tracking, SUPERSAMPLE
            )

        # The source measurement is the text's ink box, while the font
        # renderer's tile is measured from its antialiased alpha bounds.
        # Those two bounds can differ by several pixels after a font
        # substitution (especially for short labels), leaving the output
        # visibly narrower or wider even though the fit loop accepted it.
        # Apply the remaining measured correction before downsampling.  Keep
        # it deliberately bounded: this closes the small raster/font-metric
        # gap without turning a genuinely different font into a visibly
        # stretched typeface.
        current_along = max(1.0, tile.width / SUPERSAMPLE)
        current_across = max(1.0, tile.height / SUPERSAMPLE)
        sx = target_along / current_along
        sy = target_across / current_across
        # PDF/CAD fonts often have materially different cap-height metrics
        # from the fallback font used by the raster route.  This is most
        # visible on short numeric labels along slanted walls: the measured
        # source cap can be 80–90% of the fallback tile even though the
        # baseline and angle are correct.  Correct the tile in its own
        # (unrotated) frame before rotation, within a bounded range, so the
        # rendered cap height matches the source without changing tracking or
        # the label's wall-aligned orientation.
        if 0.80 <= sx <= 1.20 and 0.80 <= sy <= 1.20:
            corrected_w = max(1, int(round(tile.width * sx)))
            corrected_h = max(1, int(round(tile.height * sy)))
            if corrected_w != tile.width or corrected_h != tile.height:
                tile = tile.resize((corrected_w, corrected_h), Image.LANCZOS)

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

    collided, hidden = _composite(canvas, tile, center, linework_mask, drawing_alpha)
    return RenderedRun(
        shrunk=shrunk,
        final_px_size=px_size,
        final_tracking=tracking,
        ink_width=ink_width,
        ink_height=ink_height,
        collided=collided,
        hidden=hidden,
    )


def _composite(
    canvas: np.ndarray,
    tile: Image.Image,
    center: tuple[float, float],
    linework_mask: np.ndarray | None,
    drawing_alpha: np.ndarray | None = None,
) -> tuple[bool, float]:
    """Alpha-blend ``tile`` centred on ``center``, *underneath* the
    drawing layer; report any collision and how much of the run the
    drawing ended up covering."""
    h, w = canvas.shape[:2]
    tw, th = tile.width, tile.height
    x0 = int(round(center[0] - tw / 2))
    y0 = int(round(center[1] - th / 2))

    # Clip to canvas.
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(w, x0 + tw), min(h, y0 + th)
    if dx1 <= dx0 or dy1 <= dy0:
        return False, 0.0

    patch = np.array(tile)[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    if patch.size == 0:
        return False, 0.0

    alpha = (patch[:, :, 3:4].astype(np.float32)) / 255.0
    rgb = patch[:, :, :3].astype(np.float32)
    bgr = rgb[:, :, ::-1]  # PIL is RGB, OpenCV canvas is BGR

    region = canvas[dy0:dy1, dx0:dx1].astype(np.float32)
    blended = bgr * alpha + region * (1 - alpha)

    hidden = 0.0
    if drawing_alpha is not None:
        # The z-order this module exists to guarantee. Interpolating
        # back toward the untouched `region` by the drawing's own
        # coverage is what "type goes behind the drawing" means in a
        # flattened raster: where the geometry covers a pixel fully the
        # canvas keeps precisely the byte it already held, so a run
        # cannot alter linework at all; where it covers partially — the
        # antialiased flank of every line on the sheet — the two mix in
        # proportion, so the type fades under the line's own soft edge
        # instead of stopping dead against a hard cut-out of it.
        over = drawing_alpha[dy0:dy1, dx0:dx1][:, :, None]
        blended = region * over + blended * (1 - over)
        own_ink = alpha[:, :, 0] > 0.5
        if own_ink.any():
            hidden = float(over[:, :, 0][own_ink].mean())

    canvas[dy0:dy1, dx0:dx1] = blended.astype(np.uint8)

    if linework_mask is None:
        return False, hidden
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
    return collided, hidden


def linework_mask_for(image: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    """Dark pixels that are *not* text — what a re-rendered run must avoid.

    Built from the cleaned plate, so the mask reflects the geometry as it
    will actually appear under the new text.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _t, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.bitwise_and(dark, cv2.bitwise_not(text_mask))


def drawing_alpha_for(image: np.ndarray) -> np.ndarray:
    """How much of each pixel the drawing layer covers, 0.0 to 1.0.

    Built from the type-free plate, so every dark pixel left on it *is*
    geometry and the layer needs no text mask to stay out of: this is
    the whole sheet's linework — walls, arcs, dimension and witness
    lines, fixtures — as the thing every re-rendered run is composited
    underneath.

    Coverage, not a binary mask. Every line on a CAD export is
    antialiased, and a hard 0/1 occluder would stop the type dead
    against the line's thresholded core while the line's own grey flank
    kept blending over it — a bright fringe tracing every wall that a
    label passes behind. Grading the occlusion by the same coverage the
    export itself drew means the type simply disappears under the line,
    edge included.

    Ink and paper levels come from Otsu's own split rather than a fixed
    percentile: what share of a sheet is ink swings wildly between a
    dense plan and a sparse detail, and a decile cut tuned for one
    measures "ink" as near-paper on the other — which would collapse
    the span below, drive this alpha to 1.0 across the entire sheet, and
    hide every label on it completely.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    level, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    flat = gray.reshape(-1)
    is_dark = flat <= level
    if not is_dark.any() or is_dark.all():
        return np.zeros(gray.shape, dtype=np.float32)  # blank or solid: nothing to hide behind

    ink = float(np.median(flat[is_dark]))
    paper = float(np.median(flat[~is_dark]))
    alpha = np.clip(
        (paper - gray.astype(np.float32)) / max(1.0, paper - ink), 0.0, 1.0
    )

    # Confined to the geometry's own pixels plus the one-pixel fringe
    # around them, dilated from the thresholded core so the antialiased
    # flank the paragraph above is about stays inside. A scanned or
    # JPEG-compressed sheet's paper is nowhere near uniform, and without
    # this every run on such a sheet would be faintly greyed by the
    # page's own noise — an occluder covering the whole sheet at a few
    # percent — rather than only where real linework crosses it.
    alpha[cv2.dilate(dark, np.ones((3, 3), np.uint8)) == 0] = 0.0
    return alpha
