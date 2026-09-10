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
from spejl.vector.pdf_mirror import mirror_pdf

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
                    out.append({"text": span["text"], "bbox": span["bbox"], "dir": line["dir"]})
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
