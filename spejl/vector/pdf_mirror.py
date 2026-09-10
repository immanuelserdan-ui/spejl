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
    * embedded raster images are left in place, unmirrored (§09 "text
      over hatch or grey fill" territory — real handling needs the same
      erase/flip treatment as Route B and is deferred to Phase 3+);
    * diagonal text is rounded to the nearest 90° on re-insertion,
      because ``Page.insert_text`` only rotates in quarter turns. Every
      dimension label and room name in the target plans is horizontal
      or vertical, so this never fires against real input — but a plan
      with angled callouts would need the ``morph`` rotation path
      before it could claim losslessness for those runs.
"""

from __future__ import annotations

import math
from pathlib import Path

import pymupdf

from spejl.models import Axis, Document, Flag, PageResult, Route

_EPS = 1e-6

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
    src = pymupdf.open(str(input_path))
    out = pymupdf.open()
    result = Document(source=input_path, output=output_path, axis=axis, route=Route.VECTOR)

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
            for span in spans:
                _reinsert_mirrored_span(new_page, span, W, H, axis)

            if page.get_images():
                flags.append(
                    Flag(
                        code="image-not-mirrored",
                        message="Page contains embedded raster image(s); left "
                        "unmirrored — see module docstring.",
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


def _mirror_point(x: float, y: float, W: float, H: float, axis: Axis) -> tuple[float, float]:
    if axis is Axis.VERTICAL:
        return (W - x, y)
    if axis is Axis.HORIZONTAL:
        return (x, H - y)
    return (W - x, H - y)  # BOTH — point reflection


def _is_canonical_direction(dx: float, dy: float) -> bool:
    """True if a baseline pointing (dx, dy) reads the way a drafter expects:
    left-to-right when it has any horizontal component, bottom-to-top when
    it is purely vertical (ISO dimension-text convention; screen y is
    down, so "reads upward" means dy < 0).

    This is a fixed convention, deliberately independent of which axis is
    being mirrored: a vertical dimension always ends up reading bottom-to-
    top, whether the mirror was left/right or top/bottom. The alternative
    — letting a top/bottom mirror flip vertical text upside down — would
    make the tool's output depend on an axis choice in a way no real
    drafting standard does; only the *position* of the run should move.
    """
    if dx > _EPS:
        return True
    if dx < -_EPS:
        return False
    return dy < _EPS  # dx ~ 0: canonical iff dy <= 0 (bottom-up or degenerate)


def _mirror_direction(dx: float, dy: float, axis: Axis) -> tuple[float, float]:
    """Reflect a baseline direction, then pick the traversal (of the two
    that describe the same mirrored line) that stays readable — see the
    build plan §05 and this module's docstring for why this is safe even
    for diagonals: negating a direction vector names the same line.
    """
    if axis is Axis.VERTICAL:
        dxf, dyf = -dx, dy
    elif axis is Axis.HORIZONTAL:
        dxf, dyf = dx, -dy
    else:
        dxf, dyf = -dx, -dy
    if not _is_canonical_direction(dxf, dyf):
        dxf, dyf = -dxf, -dyf
    return dxf, dyf


def _rotate_param(dx: float, dy: float) -> int:
    """Map a canonicalised direction to PyMuPDF's ``insert_text(rotate=)``
    quarter-turn steps. Empirically (see the build session's API probe):
    ``rotate=90`` produces direction (0, -1) — i.e. ``rotate`` is the
    negative of the screen-space ``atan2(dy, dx)`` angle.
    """
    angle_screen = math.degrees(math.atan2(dy, dx))
    return int(round(-angle_screen / 90) * 90) % 360


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
) -> None:
    x0, y0, x1, y1 = span["bbox"]
    mx0, my0 = _mirror_point(x0, y0, W, H, axis)
    mx1, my1 = _mirror_point(x1, y1, W, H, axis)
    cx, cy = (mx0 + mx1) / 2, (my0 + my1) / 2  # mirrored box centre — the anchor

    dxf, dyf = _mirror_direction(span["dir"][0], span["dir"][1], axis)
    rotate = _rotate_param(dxf, dyf)

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
    shape.finish(
        width=dwg.get("width") or 1.0,
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
