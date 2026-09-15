"""Route A — lossless mirroring for vector PDFs.

The build plan's whole argument in one paragraph: a page's geometry
(walls, doors, dimension lines) is correct under a straight reflection.
Its text is not — reflecting a glyph run reverses the glyphs and, for
text on a diagonal, points the baseline the wrong way. So this module
never reflects a glyph. It reads every text span off the original page,
computes where and which way each one *should* sit once the drawing is
mirrored, discards the original spans, redraws the vector geometry
mirrored, and re-inserts each string upright at its mirrored anchor.

Geometry is reconstructed path-by-path from ``Page.get_drawings()``
rather than pushed through ``Page.show_pdf_page()``, because the
``show_pdf_page`` in the pinned PyMuPDF build takes no arbitrary
transform matrix (only 90°-step ``rotate``) — reconstruction is the one
route that supports a true left-right or top-bottom mirror.

Known P1 limitations, both flagged in the sidecar rather than silently
swallowed:
    * embedded raster images are DROPPED, not mirrored — geometry and
      text are reconstructed on a fresh page, and nothing in this module
      ever copies image XObjects onto it. Correct handling means judging
      per image whether its pixels should flip too (a hatch fill,
      probably; a scanned stamp or logo, probably not — the same
      protected-region judgement call Route B makes explicitly) and this
      codebase has no PDF fixture with an embedded image to develop or
      verify that against yet. Tracked as real follow-up work, not
      silently patched over: the sidecar flag says "removed", not
      "unmirrored", so nobody mistakes a dropped image for a mirrored
      one on a plan that happens to have one;
    * diagonal text is rounded to the nearest 90° on re-insertion,
      because ``Page.insert_text`` only rotates in quarter turns.
      Genuinely diagonal dimension text turned out NOT to be a
      hypothetical: Route B's own fix history this session confirmed
      real, recurring dimension numbers following a sloped partition
      wall at a real, deliberate angle (not a cardinal) on this
      project's own real plans. A plan with a diagonal run that reaches
      this route the same way needs the ``morph`` rotation path before
      it could claim losslessness for those runs — not yet implemented,
      so every span whose own true direction is rounded by more than a
      trivial floating-point sliver is flagged (``diagonal-text-
      rounded``) rather than silently mis-rotated with nothing in the
      sidecar to show for it.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from spejl.models import Axis, Document, Flag, PageResult, Route
from spejl.transform import mirror as M

# Point/direction mirroring and the readability convention live in
# transform/mirror.py, shared with Route B — two copies of that rule
# would be free to drift, and it is the rule the whole tool turns on.
_mirror_point = M.mirror_point

# Font-family fallback: CAD-exported PDFs overwhelmingly set a
# Helvetica/Arial-alike, so that is the default; a document that
# clearly asks for a serif or monospace face gets one. Exact family
# matching (the build plan's style/family.py, NCC template matching)
# is Route B's job — Route A only ever loses a glyph's *exact* outline
# for a font PyMuPDF doesn't ship, never its position or orientation.
_HELV = {False: {False: "helv", True: "heit"}, True: {False: "hebo", True: "hebi"}}
_TIMES = {False: {False: "tiro", True: "tiit"}, True: {False: "tibo", True: "tibi"}}
_COURIER = {False: {False: "cour", True: "coit"}, True: {False: "cobo", True: "cobi"}}


def mirror_pdf(input_path: Path, output_path: Path, axis: Axis = Axis.VERTICAL) -> Document:
    """Mirror every page of ``input_path`` and write ``output_path``.

    Returns the :class:`Document` record — the same object that
    ``to_sidecar()`` turns into the ``*.spejl.json`` written beside the
    output.
    """
    result = Document(source=input_path, output=output_path, axis=axis, route=Route.VECTOR)

    src = pymupdf.open(str(input_path))
    try:
        out = pymupdf.open()
    except Exception:
        src.close()  # opening `out` failing must not leak the already-open `src`
        raise

    try:
        for page_index in range(src.page_count):
            page = src[page_index]
            W, H = page.rect.width, page.rect.height
            spans = _extract_text_spans(page)
            drawings = page.get_drawings()

            new_page = out.new_page(width=W, height=H)
            shape = new_page.new_shape()
            for dwg in drawings:
                _draw_mirrored_path(shape, dwg, W, H, axis)
            shape.commit()

            flags: list[Flag] = []
            diagonal_text_rounded = False
            for span in spans:
                if _reinsert_mirrored_span(new_page, span, W, H, axis):
                    diagonal_text_rounded = True

            if page.get_images():
                flags.append(
                    Flag(
                        code="image-removed",
                        message="Page contains embedded raster image(s); these "
                        "are REMOVED, not mirrored — see module docstring.",
                        severity="warn",
                    )
                )
            if diagonal_text_rounded:
                flags.append(
                    Flag(
                        code="diagonal-text-rounded",
                        message="Page contains text at a genuine diagonal angle; "
                        "Page.insert_text only rotates in quarter turns, so it "
                        "was rounded to the nearest 90° — see module docstring.",
                        severity="warn",
                    )
                )

            result.pages.append(
                PageResult(
                    index=page_index,
                    width=W,
                    height=H,
                    text_runs_mirrored=len(spans),
                    flags=flags,
                )
            )

        out.save(str(output_path))
    finally:
        out.close()
        src.close()

    return result


# --------------------------------------------------------------------------
# Text: extract → mirror the anchor and angle → redraw upright
# --------------------------------------------------------------------------


def _extract_text_spans(page: pymupdf.Page) -> list[dict]:
    """One dict per glyph run, carrying exactly what re-rendering needs."""
    spans: list[dict] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # 0 = text, 1 = image
            continue
        for line in block["lines"]:
            direction = line["dir"]  # (dx, dy), screen coords, unit vector
            for span in line["spans"]:
                text = span["text"]
                if not text.strip():
                    continue
                spans.append(
                    {
                        "text": text,
                        "bbox": span["bbox"],  # (x0, y0, x1, y1)
                        "size": span["size"],
                        "color": _int_to_rgb(span["color"]),
                        "font": span["font"],
                        "bold": bool(span["flags"] & (1 << 4)),
                        "italic": bool(span["flags"] & (1 << 1)),
                        "dir": direction,
                    }
                )
    return spans


def _int_to_rgb(color_int: int) -> tuple[float, float, float]:
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return (r / 255, g / 255, b / 255)


# Vector PDF text direction is exact glyph-run geometry lifted straight
# from the page, not noisy OCR measurement — unlike the raster route's
# ordinary-detection-noise snap floor (_ALWAYS_SNAP_DEG = 3.0 in
# detect/rotations.py), a deviation here of even a couple of degrees is
# a genuine, deliberately-drawn diagonal, not measurement noise. This
# tiny tolerance exists only to absorb floating-point rounding in the
# PDF's own stored direction vector, not to forgive a real tilt.
_DIAGONAL_ROUNDING_TOLERANCE_DEG = 0.5


def _rotate_param(dx: float, dy: float) -> tuple[int, float]:
    """Map a canonicalised direction to PyMuPDF's ``insert_text(rotate=)``
    quarter-turn steps.

    ``rotate`` is the negative of the screen-space ``atan2(dy, dx)``
    angle — i.e. exactly Spejl's own angle convention, confirmed
    empirically during the build: ``rotate=90`` produces direction
    (0, -1). Quarter turns only, hence the rounding; the module docstring
    covers what that costs for diagonal text.

    Also returns the signed deviation (degrees) between the run's own
    true angle and the cardinal it was rounded to, so the caller can
    judge whether that rounding actually cost anything worth flagging —
    see ``_DIAGONAL_ROUNDING_TOLERANCE_DEG``.
    """
    angle = M.angle_from_direction(dx, dy)
    nearest_quarter_turn = round(angle / 90)
    return int(nearest_quarter_turn * 90) % 360, angle - nearest_quarter_turn * 90


def _font_alias(font_name: str, bold: bool, italic: bool) -> str:
    name = font_name.lower()
    if "courier" in name or "mono" in name:
        table = _COURIER
    elif "times" in name or "serif" in name or "georgia" in name or "garamond" in name:
        table = _TIMES
    else:
        table = _HELV
    return table[bold][italic]


def _reinsert_mirrored_span(
    page: pymupdf.Page, span: dict, W: float, H: float, axis: Axis
) -> bool:
    """Re-insert ``span`` upright at its mirrored anchor.

    Returns True if this span's own true reading direction was a
    meaningfully diagonal one (see ``_DIAGONAL_ROUNDING_TOLERANCE_DEG``),
    rounded to the nearest quarter turn on insertion — the caller flags
    this so it's visible in the sidecar, not silently swallowed.
    """
    x0, y0, x1, y1 = span["bbox"]
    mx0, my0 = _mirror_point(x0, y0, W, H, axis)
    mx1, my1 = _mirror_point(x1, y1, W, H, axis)
    cx, cy = (mx0 + mx1) / 2, (my0 + my1) / 2  # mirrored box centre — the anchor

    dxf, dyf = M.mirror_direction(span["dir"][0], span["dir"][1], axis)
    rotate, deviation_deg = _rotate_param(dxf, dyf)

    fontname = _font_alias(span["font"], span["bold"], span["italic"])
    fontsize = span["size"]
    width = pymupdf.get_text_length(span["text"], fontname=fontname, fontsize=fontsize)

    # Baseline insertion point for a centred result. insert_text() places
    # `point` at the baseline start, so we offset by half the measured
    # width along the reading direction and by a fixed baseline/cap-height
    # fraction across it. Exact tracking-aware centring (the build plan's
    # style/fit_box.py) is Route B's job — this is the P1 approximation,
    # accurate enough that no re-rendered label visibly drifts off its
    # original centreline on the golden fixture.
    baseline_frac = 0.32 * fontsize
    if rotate == 0:
        point = (cx - width / 2, cy + baseline_frac)
    elif rotate == 90:
        point = (cx - baseline_frac, cy + width / 2)
    elif rotate == 180:
        point = (cx + width / 2, cy - baseline_frac)
    else:  # 270
        point = (cx + baseline_frac, cy - width / 2)

    page.insert_text(
        point,
        span["text"],
        fontsize=fontsize,
        fontname=fontname,
        color=span["color"],
        rotate=rotate,
    )
    return abs(deviation_deg) >= _DIAGONAL_ROUNDING_TOLERANCE_DEG


# --------------------------------------------------------------------------
# Geometry: reconstruct every drawn path, mirrored, from get_drawings()
# --------------------------------------------------------------------------


def _draw_mirrored_path(shape, dwg: dict, W: float, H: float, axis: Axis) -> None:
    for item in dwg["items"]:
        op = item[0]
        if op == "re":
            rect: pymupdf.Rect = item[1]
            p0 = _mirror_point(rect.x0, rect.y0, W, H, axis)
            p1 = _mirror_point(rect.x1, rect.y1, W, H, axis)
            shape.draw_rect(
                pymupdf.Rect(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))
            )
        elif op == "l":
            p0 = _mirror_point(item[1].x, item[1].y, W, H, axis)
            p1 = _mirror_point(item[2].x, item[2].y, W, H, axis)
            shape.draw_line(pymupdf.Point(*p0), pymupdf.Point(*p1))
        elif op == "c":
            pts = [pymupdf.Point(*_mirror_point(p.x, p.y, W, H, axis)) for p in item[1:5]]
            shape.draw_bezier(*pts)
        elif op == "qu":
            quad: pymupdf.Quad = item[1]
            pts = [
                pymupdf.Point(*_mirror_point(p.x, p.y, W, H, axis))
                for p in (quad.ul, quad.ur, quad.ll, quad.lr)
            ]
            shape.draw_quad(pymupdf.Quad(*pts))
        # Unrecognised ops are rare (this covers line/rect/curve/quad, which
        # is everything CAD/vector-PDF exporters emit for plan geometry)
        # and are skipped rather than raising, so one odd path never aborts
        # a whole-document mirror.

    line_cap = dwg.get("lineCap")
    # PDF/CAD convention: an explicit stroke width of 0 means "hairline —
    # thinnest the device can render". `dwg.get("width") or 1.0` treats
    # that 0 as falsy and silently thickens every hairline wall/gridline
    # (what DWG->PDF exporters commonly use) to a full 1pt stroke.
    #
    # The fix is not simply "pass 0 through", though: PyMuPDF's own
    # Shape.finish() gives `width=0` a THIRD, different meaning again —
    # its source sets `color = None` whenever `width == 0` ("border
    # color makes no sense then"), which suppresses the stroke operator
    # entirely, so the line would vanish rather than render thin. A
    # small positive width is the only value that survives PyMuPDF's own
    # writer as a visibly hairline-thin *stroke*, which is what "0 w" in
    # the source actually meant. Only a genuinely missing key (None)
    # falls back to the ordinary default.
    _HAIRLINE_PT = 0.1
    dwg_width = dwg.get("width")
    if dwg_width is None:
        resolved_width = 1.0
    elif dwg_width == 0:
        resolved_width = _HAIRLINE_PT
    else:
        resolved_width = dwg_width
    shape.finish(
        width=resolved_width,
        color=dwg.get("color"),
        fill=dwg.get("fill"),
        lineCap=line_cap[0] if isinstance(line_cap, tuple) else (line_cap or 0),
        lineJoin=int(dwg.get("lineJoin") or 0),
        dashes=dwg.get("dashes"),
        closePath=bool(dwg.get("closePath", False)),
        even_odd=bool(dwg.get("even_odd", False)),
        fill_opacity=dwg.get("fill_opacity") if dwg.get("fill_opacity") is not None else 1,
        stroke_opacity=dwg.get("stroke_opacity") if dwg.get("stroke_opacity") is not None else 1,
    )
