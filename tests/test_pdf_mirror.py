"""Route A acceptance tests, scaled down from the build plan's golden
fixture: a room rectangle, a horizontal room label, and a vertical
dimension reading bottom-to-top — exactly the three things that have to
survive a mirror simultaneously (geometry, horizontal text, vertical text).
"""

from __future__ import annotations

import math
from pathlib import Path

import pymupdf
import pytest

from spejl.models import Axis
from spejl.vector.pdf_mirror import (
    _rotate_param,
    mirror_pdf,
    text_drawing_overlap_findings,
)

PAGE_W, PAGE_H = 400.0, 300.0
ROOM = pymupdf.Rect(40, 40, 240, 260)  # room occupies the LEFT side of the sheet
DIM_X = 260.0  # the "5155" dimension line sits just right of the room


def _make_fixture_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)

    shape = page.new_shape()
    shape.draw_rect(ROOM)
    shape.finish(color=(0, 0, 0), width=2)
    shape.draw_line(pymupdf.Point(DIM_X, 60), pymupdf.Point(DIM_X, 240))
    shape.finish(color=(0, 0, 0), width=1)
    shape.commit()

    # Room label — reads left-to-right, centred in the room.
    page.insert_text((90, 150), "Stue", fontsize=16, fontname="helv", color=(0, 0, 0))

    # Vertical dimension — reads bottom-to-top (rotate=90 -> dir (0,-1)),
    # per the probe in the build session: this is the ISO convention the
    # golden two-room plan itself uses for its 5155 mm wall run.
    page.insert_text((DIM_X + 12, 200), "5155", fontsize=12, fontname="helv", rotate=90, color=(0, 0, 0))

    doc.save(str(path))
    doc.close()


def _spans(path: Path) -> list[dict]:
    doc = pymupdf.open(str(path))
    out = []
    for block in doc[0].get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].strip():
                    out.append({
                        "text": span["text"],
                        "bbox": span["bbox"],
                        "dir": line["dir"],
                        "origin": span["origin"],
                    })
    doc.close()
    return out


def _rects(path: Path) -> list[pymupdf.Rect]:
    doc = pymupdf.open(str(path))
    rects = [
        item[1]
        for dwg in doc[0].get_drawings()
        for item in dwg["items"]
        if item[0] == "re"
    ]
    doc.close()
    return rects


@pytest.fixture
def source_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "plan.pdf"
    _make_fixture_pdf(p)
    return p


def test_page_size_is_preserved(source_pdf: Path, tmp_path: Path):
    out = tmp_path / "plan_mirrored.pdf"
    mirror_pdf(source_pdf, out, axis=Axis.VERTICAL)
    doc = pymupdf.open(str(out))
    assert (doc[0].rect.width, doc[0].rect.height) == (PAGE_W, PAGE_H)
    doc.close()


def test_room_geometry_moves_to_the_opposite_side(source_pdf: Path, tmp_path: Path):
    out = tmp_path / "plan_mirrored.pdf"
    mirror_pdf(source_pdf, out, axis=Axis.VERTICAL)
    rects = _rects(out)
    assert len(rects) == 1
    mirrored = rects[0]
    # x' = W - x, corners swap
    assert mirrored.x0 == pytest.approx(PAGE_W - ROOM.x1, abs=0.5)
    assert mirrored.x1 == pytest.approx(PAGE_W - ROOM.x0, abs=0.5)
    assert mirrored.y0 == pytest.approx(ROOM.y0, abs=0.5)
    assert mirrored.y1 == pytest.approx(ROOM.y1, abs=0.5)
    # The room's centre was left of the sheet's midline; it must now be right of it.
    src_cx = (ROOM.x0 + ROOM.x1) / 2
    mirrored_cx = (mirrored.x0 + mirrored.x1) / 2
    assert src_cx < PAGE_W / 2
    assert mirrored_cx > PAGE_W / 2


def test_room_label_is_readable_and_repositioned(source_pdf: Path, tmp_path: Path):
    out = tmp_path / "plan_mirrored.pdf"
    mirror_pdf(source_pdf, out, axis=Axis.VERTICAL)
    spans = _spans(out)
    stue = next(s for s in spans if s["text"] == "Stue")

    # Not reversed — this is the entire point of the tool.
    assert stue["text"] == "Stue"

    # Reads left-to-right (dx > 0), same as the source.
    assert stue["dir"][0] > 0.9

    # Its box centre crossed to the mirrored side of the sheet.
    cx = (stue["bbox"][0] + stue["bbox"][2]) / 2
    assert cx > PAGE_W / 2


def test_vertical_dimension_still_reads_bottom_to_top(source_pdf: Path, tmp_path: Path):
    out = tmp_path / "plan_mirrored.pdf"
    mirror_pdf(source_pdf, out, axis=Axis.VERTICAL)
    spans = _spans(out)
    dim = next(s for s in spans if s["text"] == "5155")

    assert dim["text"] == "5155"  # never "5515" or a reversed glyph run

    # A purely vertical mirror axis leaves vertical reading direction
    # untouched: still (0, -1) — bottom-to-top, ISO convention. This is
    # the build plan's central worked example (§05, Figure 2).
    assert dim["dir"][0] == pytest.approx(0, abs=1e-3)
    assert dim["dir"][1] < -0.9

    # The dimension sat right of the room; it must now sit left of it.
    cx = (dim["bbox"][0] + dim["bbox"][2]) / 2
    assert cx < PAGE_W - DIM_X + 5


def test_rotated_text_uses_the_mirrored_baseline_origin(source_pdf: Path, tmp_path: Path):
    """Vertical text must keep its exact wall offset after reflection.

    An axis-aligned bbox centre includes font ascender/descender padding and
    shifts the baseline sideways; the PDF origin does not.
    """
    out = tmp_path / "plan_mirrored_origin.pdf"
    mirror_pdf(source_pdf, out, axis=Axis.VERTICAL)
    source_dim = next(s for s in _spans(source_pdf) if s["text"] == "5155")
    mirrored_dim = next(s for s in _spans(out) if s["text"] == "5155")
    assert mirrored_dim["origin"][0] == pytest.approx(PAGE_W - source_dim["origin"][0], abs=0.1)
    assert mirrored_dim["origin"][1] == pytest.approx(source_dim["origin"][1], abs=0.1)


def test_horizontal_axis_mirror_keeps_the_same_reading_convention(source_pdf, tmp_path):
    """A top/bottom mirror moves the *position* of a vertical run just
    like a left/right mirror does, but the readability rule (build plan
    §05: horizontal always left-to-right, vertical always bottom-to-top)
    is a fixed drafting convention, not something that should flip back
    and forth depending on which axis happened to be mirrored — so both
    runs keep the same reading direction they had before, and only the
    room-relative position test (below) differs between axes.
    """
    out = tmp_path / "plan_mirrored_h.pdf"
    mirror_pdf(source_pdf, out, axis=Axis.HORIZONTAL)
    spans = _spans(out)

    stue = next(s for s in spans if s["text"] == "Stue")
    assert stue["dir"][0] > 0.9  # still left-to-right

    dim = next(s for s in spans if s["text"] == "5155")
    assert dim["dir"][0] == pytest.approx(0, abs=1e-3)
    assert dim["dir"][1] < -0.9  # still bottom-to-top

    # What DID change: the room (and everything else) flipped top/bottom.
    src_cy = (ROOM.y0 + ROOM.y1) / 2
    room = _rects(out)[0]
    mirrored_cy = (room.y0 + room.y1) / 2
    assert mirrored_cy == pytest.approx(PAGE_H - src_cy, abs=0.5)


def test_sidecar_reports_one_page_and_three_text_runs(source_pdf: Path, tmp_path: Path):
    out = tmp_path / "plan_mirrored.pdf"
    doc = mirror_pdf(source_pdf, out, axis=Axis.VERTICAL)
    assert len(doc.pages) == 1
    assert doc.pages[0].text_runs_mirrored == 2  # "Stue" + "5155"
    assert doc.pages[0].flags == []  # no embedded images to flag


def test_genuinely_diagonal_text_keeps_its_mirrored_slope(tmp_path: Path):
    """A slanted dimension stays slanted and readable after reflection."""
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    # A genuinely diagonal run (30 degrees off horizontal).
    pivot = pymupdf.Point(150, 150)
    mat = pymupdf.Matrix(1, 0, 0, 1, 0, 0).prerotate(30)
    page.insert_text((150, 150), "4381", fontsize=14, fontname="helv", morph=(pivot, mat), color=(0, 0, 0))
    src = tmp_path / "diagonal.pdf"
    doc.save(str(src))
    doc.close()

    out = tmp_path / "diagonal_mirrored.pdf"
    result = mirror_pdf(src, out, axis=Axis.VERTICAL)

    assert result.pages[0].flags == []
    span = next(s for s in _spans(out) if s["text"] == "4381")
    # Vertical reflection reverses the slope but preserves readable
    # left-to-right traversal: +30° becomes -30° in Spejl's convention.
    out_angle = -math.degrees(math.atan2(span["dir"][1], span["dir"][0]))
    assert out_angle == pytest.approx(-30, abs=1.0)


def test_hairline_stroke_survives_as_thin_not_thickened_to_1pt(tmp_path: Path):
    """Regression: a genuine PDF '0 w' hairline (DWG->PDF exporters use
    this for wall/gridlines) was read correctly as width 0.0 by
    get_drawings(), then silently thickened to a full 1pt stroke by
    `dwg.get("width") or 1.0` treating 0 as falsy.

    The naive fix (pass 0 straight through to Shape.finish) is also
    wrong: PyMuPDF's own writer gives width=0 a third meaning — it
    discards the stroke color entirely ("border color makes no sense
    then"), which would make the line disappear rather than render
    thin. A small positive width is what actually survives as a visible
    hairline through PyMuPDF's own Shape.finish().

    Built via a raw content-stream operator rather than
    Shape.finish(color=..., width=0), because Shape.finish() itself
    already intercepts width=0 on the way in — the only way to produce
    a get_drawings() width of exactly 0.0 to mirror is the same way a
    real CAD-exported PDF would: a literal '0 w' operator in the stream.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    # A no-op draw first, so the page has a contents stream to overwrite.
    shape = page.new_shape()
    shape.draw_line(pymupdf.Point(0, 0), pymupdf.Point(1, 1))
    shape.finish(color=(1, 1, 1), width=1)
    shape.commit()
    xref = page.get_contents()[0]
    doc.update_stream(xref, b"0 w 0 0 0 RG 50 50 m 250 50 l S")

    src = tmp_path / "hairline.pdf"
    doc.save(str(src))
    doc.close()

    assert next(dwg["width"] for dwg in pymupdf.open(str(src))[0].get_drawings()) == 0.0

    out = tmp_path / "hairline_mirrored.pdf"
    mirror_pdf(src, out, axis=Axis.VERTICAL)

    mirrored = pymupdf.open(str(out))
    width = next(dwg["width"] for dwg in mirrored[0].get_drawings())
    mirrored.close()
    # The content-stream route preserves the original PDF ``0 w`` operator
    # exactly.  Unlike the former Shape reconstruction it does not need to
    # approximate hairline semantics with a non-zero width.
    assert width == 0.0, f"hairline was altered to {width}pt"


def test_content_stream_retains_spacing_and_text_state(tmp_path: Path):
    """The vector route must not replace CAD ``TJ`` spacing with a font fit."""
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    # Create the /helv resource, then replace the content with a compact
    # CAD-style text object that uses character spacing, word spacing,
    # horizontal scaling, text rise and a TJ kerning adjustment.
    page.insert_text((1, 1), "x", fontsize=1, fontname="helv")
    xref = page.get_contents()[0]
    page.parent.update_stream(
        xref,
        b"BT /helv 12 Tf 0.5 Tc 1 Tw 90 Tz 2 Ts 1 0 0 1 50 250 Tm [(A) 120 ( B)] TJ ET",
    )
    src = tmp_path / "spacing.pdf"
    doc.save(str(src))
    doc.close()

    out = tmp_path / "spacing_mirrored.pdf"
    mirror_pdf(src, out, axis=Axis.VERTICAL)

    import pikepdf
    pdf = pikepdf.Pdf.open(out)
    operators = [(str(op), operands) for operands, op in pikepdf.parse_content_stream(pdf.pages[0])]
    pdf.close()
    names = [name for name, _ in operators]
    for operator in ("Tf", "Tc", "Tw", "Tz", "Ts", "TJ"):
        assert operator in names
    tj = next(operands[0] for name, operands in operators if name == "TJ")
    assert str(tj[0]) == "A"
    assert tj[1] == 120
    assert str(tj[2]) == " B"


def test_text_drawing_overlap_findings_name_only_colliding_text(tmp_path: Path):
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    for x in (100, 150, 200):
        page.draw_line((x, 40), (x, 260), color=(0, 0, 0), width=2)
    for x, y, label in (
        (94, 90, "First overlap"),
        (144, 150, "Second overlap"),
        (194, 210, "Third overlap"),
        (280, 150, "Clear label"),
    ):
        page.insert_text((x, y), label, fontsize=12)
    pdf = tmp_path / "overlap.pdf"
    doc.save(pdf)
    doc.close()

    assert text_drawing_overlap_findings(pdf) == [
        (1, "First overlap"),
        (1, "Second overlap"),
        (1, "Third overlap"),
    ]


def test_double_mirror_is_close_to_idempotent(source_pdf: Path, tmp_path: Path):
    """Build plan §10: mirroring twice should return close to the source —
    the QA harness's idempotency gate, exercised here on anchor position."""
    once = tmp_path / "once.pdf"
    twice = tmp_path / "twice.pdf"
    mirror_pdf(source_pdf, once, axis=Axis.VERTICAL)
    mirror_pdf(once, twice, axis=Axis.VERTICAL)

    src_stue = next(s for s in _spans(source_pdf) if s["text"] == "Stue")
    rt_stue = next(s for s in _spans(twice) if s["text"] == "Stue")
    src_cx = (src_stue["bbox"][0] + src_stue["bbox"][2]) / 2
    rt_cx = (rt_stue["bbox"][0] + rt_stue["bbox"][2]) / 2
    assert rt_cx == pytest.approx(src_cx, abs=2.0)


@pytest.mark.parametrize(
    ("angle_deg", "expected_rotate", "expected_deviation"),
    [
        (0.0, 0, 0.0),
        (90.0, 90, 0.0),
        (180.0, 180, 0.0),
        (-90.0, 270, 0.0),
        (30.0, 0, 30.0),     # a real diagonal -- rounds to 0, real cost
        (96.9, 90, 6.9),     # this project's own confirmed real tilt
    ],
)
def test_rotate_param_reports_the_true_rounding_cost(angle_deg, expected_rotate, expected_deviation):
    dx, dy = math.cos(math.radians(angle_deg)), -math.sin(math.radians(angle_deg))
    rotate, deviation = _rotate_param(dx, dy)
    assert rotate == expected_rotate
    assert deviation == pytest.approx(expected_deviation, abs=0.1)
