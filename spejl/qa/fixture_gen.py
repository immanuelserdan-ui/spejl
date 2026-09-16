"""Generate the golden raster fixture — with ground truth, for free.

Route B needs a raster plan whose every string, box and angle is known
exactly, so OCR can be *scored* rather than eyeballed. Hand-annotating a
real PNG is slow and error-prone; instead this module draws the Danish
two-room unit plan as vector, asks PyMuPDF what text it just placed
(exact boxes, exact angles), and rasterises the same page to PNG. The
vector source *is* the annotation, so fixture and ground truth can never
drift apart.

Geometry follows the real sheet: 2900 / 3200 / 4045 / 5155 / 4060 /
2105 / 1680 / 1400 / 870 mm, and the numbers close — left column
3200 + 150 + 4060 = 7410 mm equals right column 5155 + 150 + 2105.
Dimension text sits *inside* the rooms, as it does on the real plan,
which is what makes the "text touching linework" case (§09) show up
here rather than only in production.

Usage:
    python -m spejl.qa.fixture_gen --dpi 150 --out spejl/qa/golden
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# ---------------------------------------------------------------------------
# Sheet setup. Plan coordinates are millimetres, y down, origin at the
# inside face of the top-left exterior wall.
# ---------------------------------------------------------------------------

MM = 0.07          # PDF points per millimetre
MARGIN = 42.0      # points of white around the drawing
EXT_WALL = 200.0   # mm
INT_WALL = 150.0   # mm

IW, IH = 7095.0, 7410.0  # interior clear width / height in mm

# Rooms: name -> (x0, y0, x1, y1) interior clear dimensions, mm
ROOMS: dict[str, tuple[float, float, float, float]] = {
    "Koekken": (0, 0, 2900, 3200),
    "Stue": (3050, 0, 7095, 5155),
    "Vaer1": (0, 3350, 2900, 7410),
    "Entre": (3050, 5305, 4245, 7410),
    "Toilet": (4395, 5305, 6075, 7410),
    "Bad": (6225, 6010, 7095, 7410),
    "Closet": (6225, 5305, 7095, 5860),
}


@dataclass(frozen=True)
class TextPlacement:
    """A string to draw, and what it means — the ground-truth record."""

    text: str
    x_mm: float          # centre, plan mm
    y_mm: float
    size_pt: float
    rotate: int          # 0 or 90 (bottom-to-top), matching the real sheet
    kind: str            # "room" | "dimension" | "annotation"


# Room labels and the one annotation glyph, placed as on the real sheet.
LABELS: list[TextPlacement] = [
    TextPlacement("Køkken", 1600, 1500, 20, 0, "room"),
    TextPlacement("Stue", 5000, 2400, 20, 0, "room"),
    TextPlacement("Vær. 1", 1450, 5300, 20, 0, "room"),
    # Not this room's own centre (3648, 6358) either, for the same
    # reason 'Bad' wasn't: the "Entre -> Toilet" door's swing arc is an
    # 800mm-radius quarter-circle centred at (4395, 7100), sweeping
    # into Entre's own south-east corner (down to roughly x=3595,
    # y=6300 at its tip) — the original (3640, 6650) sat only ~80mm
    # outside that curve, close enough that the label's own rendered
    # ink actually touched it (see qa/self_correct.py's
    # Rule.CLEARANCE; confirmed as a real, reachable case, not a
    # theoretical one — an 8px corrective nudge got the check to pass
    # at 0% overlap while leaving the arc close enough that OCR still
    # fused it with the label's own 'e' on the golden fixture's
    # round-trip, misreading 'Entre' as 'Entrel'). Moved north-west,
    # comfortably outside the arc's radius from its own centre instead
    # of just past the letter of the check.
    TextPlacement("Entre", 3550, 5750, 15, 0, "room"),
    TextPlacement("Toilet", 5230, 6150, 16, 0, "room"),
    # Moved from the original (6650, 7050): that sat only 130mm from
    # the '870' dimension line (y=7180), close enough that the label's
    # own rendered ink actually overlapped it (see qa/self_correct.py's
    # Rule.CLEARANCE) — unlike every OTHER room label here, which sits
    # close to its own room's centre without incident. The room's own
    # geometric centre (6660, 6710) is a worse spot, not a fix: it
    # lands almost exactly on the '1400' dimension's OWN text anchor
    # ((y0+y1)/2 = 6710, text_x = 6770 — see V_DIMS/
    # _all_text_placements below), stacking two labels together instead
    # of one label on a line. This corner of the room has real
    # obstacles on three sides — the '1400' line at x=6900 (east), the
    # '870' line at y=7180 (south), the "bad" door's swing arc (an
    # 800mm-radius quarter-circle centred at (6075, 7100), reaching
    # into the room's south-west) — so the position below sits west of
    # the 1400 line/text, north of the 870 line, and outside the door
    # arc's own radius.
    #
    # A single mirror is clean at any of a wide range of positions in
    # this pocket. A ROUND trip (mirror the mirrored output again) is
    # not: moving 'Bad' at all — to any of 60+ positions tried, this
    # one included — perturbs the OCR-measured cap height of nearby
    # dimension runs (fit_style's own ``other_boxes`` exclusion sees a
    # different 'Bad' box at a different spot) just enough to
    # occasionally flip which side of raster/pipeline.py's own
    # _snap_consistent_sizes clustering tolerance a run like '870'
    # lands on. That function's own docstring already documents this
    # exact class of noise as real and confirmed on this fixture
    # (a 25-30px spread even withOUT moving anything) — this is that
    # same pre-existing fragility, not something introduced by moving
    # this label, and not fixable by finding a smarter (x, y). Chosen
    # empirically as the cleanest of many candidates tried: a clean
    # single mirror, a clean two-pass round trip (no QA refusal on
    # either pass), and the smallest residual round-trip misread
    # ('870' -> 'A70', nothing else) of any position tried other than
    # the original one this rule now correctly rejects.
    TextPlacement("Bad", 6600, 6250, 16, 0, "room"),
    TextPlacement("H*", 300, 380, 13, 0, "annotation"),
]

# Dimensions: (text, size, orientation, line span, line offset axis pos,
# text offset side). Encoded concretely below rather than abstractly —
# a dimension's placement on a real sheet is a judgement call, not a formula.
H_DIMS: list[tuple[str, float, float, float, float, float]] = [
    # text,  x0,     x1,     line_y,  text_y,  size
    ("2900", 0, 2900, 430, 300, 12),
    ("4045", 3050, 7095, 430, 300, 12),
    ("2900", 0, 2900, 6980, 7180, 12),
    ("1300", 3050, 4245, 5420, 5610, 11),
    ("1680", 4395, 6075, 7180, 7350, 11),
    ("870", 6225, 7095, 7180, 7350, 11),
]
V_DIMS: list[tuple[str, float, float, float, float, float]] = [
    # text,  y0,     y1,     line_x,  text_x,  size
    ("3200", 0, 3200, 2820, 2690, 12),
    ("5155", 0, 5155, 7000, 6870, 12),
    ("4060", 3350, 7410, 180, 320, 12),
    ("2105", 5305, 7410, 4520, 4660, 11),
    ("1400", 6010, 7410, 6900, 6770, 11),
]

# Openings in walls: (x0, y0, x1, y1) of the gap, plus a swing arc hint.
DOORS: list[tuple[float, float, float, float, str]] = [
    (2900, 2400, 3050, 3200, "stue"),        # Køkken -> Stue
    (2900, 6200, 3050, 7000, "entre"),       # Vær. 1 -> Entre
    (4245, 6300, 4395, 7100, "entre"),       # Entre -> Toilet
    (6075, 6300, 6225, 7100, "bad"),         # Toilet -> Bad
    (7095, 1200, 7295, 2000, "ext"),         # Stue exterior door
    (3400, 7410, 4200, 7610, "ext"),         # entry door
]

WINDOWS: list[tuple[float, float, float, float]] = [
    (400, -200, 1600, 0),        # Køkken
    (3400, -200, 4600, 0),       # Stue left
    (5400, -200, 6600, 0),       # Stue right
    (600, 7410, 1800, 7610),     # Vær. 1
]


def _page_size() -> tuple[float, float]:
    w = (IW + 2 * EXT_WALL) * MM + 2 * MARGIN
    h = (IH + 2 * EXT_WALL) * MM + 2 * MARGIN
    return w, h


def _pt(x_mm: float, y_mm: float) -> tuple[float, float]:
    """Plan millimetres -> PDF points."""
    return (MARGIN + (x_mm + EXT_WALL) * MM, MARGIN + (y_mm + EXT_WALL) * MM)


def _rect(x0: float, y0: float, x1: float, y1: float) -> pymupdf.Rect:
    p0 = _pt(x0, y0)
    p1 = _pt(x1, y1)
    return pymupdf.Rect(*p0, *p1)


def _draw_walls(shape) -> None:
    """Solid black poché, exactly as the source sheet renders it."""
    walls = [
        (-EXT_WALL, -EXT_WALL, IW + EXT_WALL, 0),        # top
        (-EXT_WALL, IH, IW + EXT_WALL, IH + EXT_WALL),   # bottom
        (-EXT_WALL, -EXT_WALL, 0, IH + EXT_WALL),        # left
        (IW, -EXT_WALL, IW + EXT_WALL, IH + EXT_WALL),   # right
        (2900, 0, 3050, IH),                             # left col | right col
        (0, 3200, 2900, 3350),                           # Køkken | Vær. 1
        (3050, 5155, IW, 5305),                          # Stue | bottom strip
        (4245, 5305, 4395, IH),                          # Entre | Toilet
        (6075, 5305, 6225, IH),                          # Toilet | Bad
        (6225, 5860, IW, 6010),                          # closet floor over Bad
    ]
    for w in walls:
        shape.draw_rect(_rect(*w))
    shape.finish(color=(0, 0, 0), fill=(0, 0, 0), width=0.4)


def _punch_openings(shape) -> None:
    """White out door and window gaps so the poché reads as a real plan."""
    for x0, y0, x1, y1, _ in DOORS:
        shape.draw_rect(_rect(x0, y0, x1, y1))
    for w in WINDOWS:
        shape.draw_rect(_rect(*w))
    shape.finish(color=(1, 1, 1), fill=(1, 1, 1), width=0.3)


def _draw_window_glazing(shape) -> None:
    for x0, y0, x1, y1 in WINDOWS:
        horizontal = (x1 - x0) > (y1 - y0)
        if horizontal:
            for frac in (0.32, 0.68):
                y = y0 + (y1 - y0) * frac
                shape.draw_line(pymupdf.Point(*_pt(x0, y)), pymupdf.Point(*_pt(x1, y)))
        else:
            for frac in (0.32, 0.68):
                x = x0 + (x1 - x0) * frac
                shape.draw_line(pymupdf.Point(*_pt(x, y0)), pymupdf.Point(*_pt(x, y1)))
    shape.finish(color=(0, 0, 0), width=0.5)


def _draw_door_swings(shape) -> None:
    """A leaf line plus a quarter-circle arc — the mark that makes a
    mirrored plan visibly mirrored."""
    for x0, y0, x1, y1, side in DOORS:
        w, h = x1 - x0, y1 - y0
        if h >= w:  # door in a vertical wall, leaf swings horizontally
            hinge = (x0 if side in ("stue", "bad", "ext") else x1, y1)
            tip = (hinge[0] + (h if side in ("stue", "bad", "ext") else -h), y1)
            centre = hinge
        else:       # door in a horizontal wall
            hinge = (x0, y0 if side == "ext" else y1)
            tip = (x0, hinge[1] - w)
            centre = hinge
        shape.draw_line(pymupdf.Point(*_pt(*centre)), pymupdf.Point(*_pt(*tip)))
        try:
            shape.draw_sector(
                pymupdf.Point(*_pt(*centre)), pymupdf.Point(*_pt(*tip)), 90, fullSector=False
            )
        except Exception:
            pass  # arc is decoration; never let it break fixture generation
    shape.finish(color=(0, 0, 0), width=0.5)


def _draw_dim_lines(shape) -> None:
    tick = 55.0  # mm
    for _text, x0, x1, line_y, _text_y, _size in H_DIMS:
        shape.draw_line(pymupdf.Point(*_pt(x0, line_y)), pymupdf.Point(*_pt(x1, line_y)))
        for x in (x0, x1):
            shape.draw_line(
                pymupdf.Point(*_pt(x, line_y - tick)), pymupdf.Point(*_pt(x, line_y + tick))
            )
    for _text, y0, y1, line_x, _text_x, _size in V_DIMS:
        shape.draw_line(pymupdf.Point(*_pt(line_x, y0)), pymupdf.Point(*_pt(line_x, y1)))
        for y in (y0, y1):
            shape.draw_line(
                pymupdf.Point(*_pt(line_x - tick, y)), pymupdf.Point(*_pt(line_x + tick, y))
            )
    shape.finish(color=(0, 0, 0), width=0.45)


def _all_text_placements() -> list[TextPlacement]:
    placements = list(LABELS)
    for text, x0, x1, _line_y, text_y, size in H_DIMS:
        placements.append(TextPlacement(text, (x0 + x1) / 2, text_y, size, 0, "dimension"))
    for text, y0, y1, _line_x, text_x, size in V_DIMS:
        placements.append(TextPlacement(text, text_x, (y0 + y1) / 2, size, 90, "dimension"))
    return placements


def _insert_text(page: pymupdf.Page, tp: TextPlacement) -> None:
    """Place a string centred on its plan coordinate.

    Mirrors the centring arithmetic in vector/pdf_mirror.py so the fixture
    and the tool agree on what "centred" means.
    """
    width = pymupdf.get_text_length(tp.text, fontname="helv", fontsize=tp.size_pt)
    cx, cy = _pt(tp.x_mm, tp.y_mm)
    baseline = 0.32 * tp.size_pt
    if tp.rotate == 0:
        point = (cx - width / 2, cy + baseline)
    else:  # 90 — reads bottom-to-top
        point = (cx - baseline, cy + width / 2)
    page.insert_text(
        point, tp.text, fontsize=tp.size_pt, fontname="helv", color=(0, 0, 0), rotate=tp.rotate
    )


def build_pdf(pdf_path: Path) -> pymupdf.Document:
    w, h = _page_size()
    doc = pymupdf.open()
    page = doc.new_page(width=w, height=h)

    shape = page.new_shape()
    _draw_walls(shape)
    _punch_openings(shape)
    _draw_window_glazing(shape)
    _draw_door_swings(shape)
    _draw_dim_lines(shape)
    shape.commit()

    for tp in _all_text_placements():
        _insert_text(page, tp)

    doc.save(str(pdf_path))
    return doc


def extract_ground_truth(doc: pymupdf.Document, zoom: float) -> list[dict]:
    """Ask the PDF what text it contains — that answer is the annotation.

    Boxes come back in PDF points and are scaled by ``zoom`` into the pixel
    space of the rendered PNG, so ground truth and image share coordinates.
    """
    kind_by_text: dict[str, str] = {}
    for tp in _all_text_placements():
        kind_by_text.setdefault(tp.text, tp.kind)

    truth: list[dict] = []
    page = doc[0]
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            dx, dy = line["dir"]
            angle = 0 if abs(dx) > abs(dy) else 90
            for span in line["spans"]:
                text = span["text"].strip()
                if not text:
                    continue
                x0, y0, x1, y1 = span["bbox"]
                truth.append(
                    {
                        "text": text,
                        "bbox_px": [
                            round(x0 * zoom, 2),
                            round(y0 * zoom, 2),
                            round(x1 * zoom, 2),
                            round(y1 * zoom, 2),
                        ],
                        "angle_deg": angle,
                        "kind": kind_by_text.get(text, "unknown"),
                        "cap_height_px": round((y1 - y0) * zoom * 0.72, 1),
                    }
                )
    return truth


def generate(out_dir: Path, dpi: int = 150) -> dict:
    """Write plan.pdf, plan_<dpi>.png and ground_truth_<dpi>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "plan.pdf"
    doc = build_pdf(pdf_path)

    zoom = dpi / 72.0
    png_path = out_dir / f"plan_{dpi}.png"
    pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    pix.save(str(png_path))

    truth = extract_ground_truth(doc, zoom)
    truth_path = out_dir / f"ground_truth_{dpi}.json"
    payload = {
        "image": png_path.name,
        "dpi": dpi,
        "zoom": round(zoom, 4),
        "width_px": pix.width,
        "height_px": pix.height,
        "runs": truth,
    }
    truth_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    doc.close()

    return {
        "pdf": pdf_path,
        "png": png_path,
        "truth": truth_path,
        "runs": len(truth),
        "size_px": (pix.width, pix.height),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Generate the golden raster fixture.")
    ap.add_argument("--dpi", type=int, nargs="+", default=[150])
    ap.add_argument("--out", type=Path, default=Path("spejl/qa/golden"))
    args = ap.parse_args()

    for dpi in args.dpi:
        info = generate(args.out, dpi=dpi)
        print(
            f"dpi={dpi:>4}  {info['size_px'][0]}x{info['size_px'][1]}px  "
            f"{info['runs']} text runs  ->  {info['png'].name}"
        )


if __name__ == "__main__":
    main()
