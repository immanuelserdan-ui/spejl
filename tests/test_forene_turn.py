import pymupdf as pdf
import pytest
from PIL import Image, ImageDraw

from spejl.forene import compose, import_floors, match_landings, placement


def test_return_stair_alignment_and_slot_order(tmp_path):
    floors = []
    turns = [215, 142]
    for slot, turn in enumerate(turns):
        img = Image.new("L", (350, 600), 255)
        draw = ImageDraw.Draw(img)
        draw.rectangle((5, 5, 345, 595), outline=0, width=3)
        for x in range(90, 148, 15):
            draw.line((x, turn-55, x, turn+55), fill=0)
        for offset in (-2, 2):
            draw.line((68, turn+offset, 147, turn+offset), fill=0)
        for dy in (-30, 30):
            draw.line((25, turn+dy, 68, turn), fill=0)
        # Long balcony hatch is a competing straight tread bank.
        for x in range(170, 320, 8):
            draw.line((x, 30, x, 95), fill=0)
        path = tmp_path / f"slot{slot}.png"
        img.save(path, dpi=(96, 96))
        floors.extend(import_floors(path))
    floors = match_landings(floors)
    rects, _ = placement(floors)
    ys = []
    for floor, rect, turn in zip(floors, rects, turns):
        bounds = pdf.Rect(floor.bounds)
        assert floor.detected is not None
        assert bounds.y0 + bounds.height*floor.detected == pytest.approx(turn*.75, abs=2)
        assert rect.width == pytest.approx(bounds.width)
        assert rect.height == pytest.approx(bounds.height)
        ys.append(rect.y0 + turn*.75 - bounds.y0)
    assert rects[0].x1 < rects[1].x0
    assert ys[0] == pytest.approx(ys[1], abs=2)
    with pdf.open(stream=compose(floors, 1.5, False), filetype="pdf") as doc:
        assert len(doc) == 1
