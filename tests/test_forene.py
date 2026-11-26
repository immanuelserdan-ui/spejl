"""Geometric and desktop workflow regressions for Forene."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import pytest
import pymupdf as pdf
from PIL import Image, ImageDraw

from spejl.forene import compose, export_sheet, import_floors, match_landings, placement


def make_pdf(path, pages=1):
    with pdf.open() as doc:
        for i in range(pages):
            page = doc.new_page(width=180, height=250)
            page.draw_rect((15, 12, 165, 232))
            page.insert_text((25, 35), f"Level {i}")
            for j in range(6):
                y = 80 + i*5 + j*6
                page.draw_line((32, y), (60, y))
        doc.save(path)
    return path


@pytest.mark.parametrize("count", [2, 3, 8])
def test_storeys(count, tmp_path):
    floors = import_floors(make_pdf(tmp_path / "levels.pdf", count))
    assert all(f.detected is not None for f in floors)
    rects, datum = placement(floors)
    for i, (f, r) in enumerate(zip(floors, rects)):
        assert f.name.endswith(f"p{i+1}")
        assert r.y0 + f.datum*r.height == pytest.approx(datum)
        assert r.y0 == pytest.approx((count-1-i)*5, abs=1)
        if i:
            assert r.x0-rects[i-1].x1 == pytest.approx(r.width*.015)
    with pdf.open(stream=compose(floors, 1.5, False), filetype="pdf") as out:
        assert len(out) == 1
        for i, r in enumerate(rects):
            assert f"Level {i}" in out[0].get_text(clip=r)
        assert out[0].get_drawings()


def test_dpi_fallback_manual_and_export(tmp_path):
    img = Image.new("RGBA", (240, 120))
    ImageDraw.Draw(img).rectangle((10, 10, 230, 110), outline="black", width=3)
    path = tmp_path / "upper.png"
    img.save(path, dpi=(144, 144))
    floor = import_floors(path)[0]
    assert pdf.Rect(floor.bounds).width == pytest.approx(111, abs=1)
    assert floor.detected is None
    other = replace(floor, manual=.15, scale=2)
    rects, datum = placement([floor, other], 0)
    assert rects[0].x1 == rects[1].x0
    assert rects[1].y0 + .15*rects[1].height == pytest.approx(datum)
    plain = compose([floor, other], 0, False)
    marked = compose([floor, other], 0, True)
    with pdf.open(stream=plain, filetype="pdf") as a, pdf.open(stream=marked, filetype="pdf") as b:
        assert b[0].rect.height > a[0].rect.height
    with ThreadPoolExecutor(2) as pool:
        outputs = list(pool.map(lambda _: export_sheet(plain, tmp_path, "pdf"), range(2)))
    assert {p.name for p in outputs} == {"Forene.pdf", "Forene_2.pdf"}
    assert all(p.read_bytes() == plain for p in outputs)
    jpeg = export_sheet(marked, tmp_path, "jpg")
    with Image.open(jpeg) as image:
        assert image.info["dpi"] == (300, 300)


def test_invalid_file(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-broken")
    with pytest.raises(ValueError, match="bad.pdf"):
        import_floors(p)


def test_missing_stair_never_snaps_to_wall(tmp_path):
    floors = []
    for index, landing in enumerate([130, 100, 118]):
        image = Image.new("RGB", (190, 240), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 4, 185, 235), outline="black", width=3)
        draw.rectangle((4, landing-8, 185, landing), fill="black")
        if index == 2:
            for j in range(6):
                y = landing + 8 + j*6
                draw.line((15, y, 45, y), fill="black", width=2)
        path = tmp_path / f"{index}.png"
        image.save(path)
        floors.extend(import_floors(path))
    floors = match_landings(floors)
    assert floors[2].detected is not None
    assert all(f.detected is None and f.stair_edge is None for f in floors[:2])
    assert all(f.datum == .56 for f in floors[:2])


def test_rotated_pdf(tmp_path):
    path = make_pdf(tmp_path / "source.pdf")
    with pdf.open(path) as doc:
        doc[0].set_rotation(90)
        rotated = tmp_path / "rotated.pdf"
        doc.save(rotated)
    floors = import_floors(rotated)
    with pdf.open(stream=compose(floors, 0, False), filetype="pdf") as doc:
        assert doc[0].rect.width > doc[0].rect.height
        assert "Level 0" in doc[0].get_text()


def test_stair_edge_wins_over_different_wall_thicknesses_and_deck_lines(tmp_path):
    floors = []
    tops = [63, 102]
    for i, top in enumerate(tops):
        path = tmp_path / f"floor-{i}.pdf"
        with pdf.open() as doc:
            page = doc.new_page(width=230, height=320)
            page.draw_rect((15, 20, 210, 310))
            for x in range(20, 205, 9):
                page.draw_line((x, 20), (x, top))
            for rect in [(15, top, 24, top+85), (15, top, 80, top+8+5*i)]:
                page.draw_rect(rect, color=None, fill=(0, 0, 0))
            page.insert_text((55, top+(34 if i == 0 else 76)), ("Op", "Ned")[i])
            for j in range(6):
                page.draw_line((30+j*6, top+8+5*i), (30+j*6, top+45+5*i))
                page.draw_line((30+j*6, top+53+5*i), (30+j*6, top+90+5*i))
            doc.save(path)
        floors.extend(import_floors(path))
    matched = match_landings(floors)
    assert all(f.shared_edge for f in matched)
    rects, _ = placement(matched)
    sheet_y = [rect.y0 + top + 8 + 5*i - floor.bounds[1]
               for i, (rect, top, floor) in enumerate(zip(rects, tops, matched))]
    assert sheet_y[0] == pytest.approx(sheet_y[1], abs=.001)
    assert replace(matched[0], manual=.2).datum == .2
    for missing in (None, (.1, 1.5, True)):
        assert not any(f.shared_edge for f in match_landings(
            [matched[0], replace(matched[1], stair_edge=missing)]))


def test_panel_workflow(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QPushButton
    from spejl.gui.forene import ImportDialog
    from spejl.gui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    try:
        switch = next(b for b in window.findChildren(QPushButton) if b.text()=="Switch to Forene")
        switch.click()
        panel = window._views.currentWidget()
        path = make_pdf(tmp_path/"levels.pdf", 3)
        monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *args: ([str(path)], ""))
        def accept_import(dialog):
            dialog.browse(0)
            dialog.move(1)
            return QDialog.DialogCode.Accepted
        monkeypatch.setattr(ImportDialog, "exec", accept_import)
        panel.insert()
        assert [f.name[-2:] for f in panel.floors] == ["p2", "p1", "p3"]
        assert not panel.save_button.isEnabled()
        assert "1, 2, 3" in panel.status.text()
        with monkeypatch.context() as guard:
            guard.setattr(QDialog, "exec", lambda self: pytest.fail("Export dialog opened"))
            panel.save()
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from spejl.gui.forene import StairEdgeDialog
        original = list(panel.floors)
        sheet = panel.sheet
        app.processEvents()
        r = placement(panel.floors)[0][1]
        point = panel.view.mapFromScene(r.x0*panel.zoom+1, (r.y0+r.height*.31)*panel.zoom)
        QTest.mouseClick(panel.view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        QTest.mouseDClick(panel.view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        assert panel.floors == original and panel.sheet is sheet
        assert panel.floor.currentIndex() == 0 and not panel.save_button.isEnabled()

        def choose_edge(dialog):
            dialog.show()
            app.processEvents()
            frame = dialog.view.sceneRect()
            point = dialog.view.mapFromScene(frame.width()*.5, frame.height()*.31)
            QTest.mouseClick(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=point)
            assert dialog.datum == pytest.approx(.31, abs=.01)
            return outcome

        monkeypatch.setattr(StairEdgeDialog, "exec", choose_edge)
        outcome = QDialog.DialogCode.Rejected
        panel.adjust.click()
        assert panel.floors == original and panel.sheet is sheet
        outcome = QDialog.DialogCode.Accepted
        for i in range(3):
            panel.floor.setCurrentIndex(i)
            panel.adjust.click()
            assert panel.floors[i].manual == pytest.approx(.31, abs=.01)
        assert panel.save_button.isEnabled()
        panel.reset_datum()
        assert not panel.save_button.isEnabled()
        panel.adjust.click()
        first = list(panel.floors)
        def cancel(dialog):
            dialog.clear()
            return QDialog.DialogCode.Rejected
        monkeypatch.setattr(ImportDialog, "exec", cancel)
        panel.insert()
        assert panel.floors == first
        monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
        panel.save()
        assert (tmp_path/"Forene.jpg").is_file()
        panel.back.emit()
        assert window._views.currentIndex() == 0
        switch.click()
        assert panel.floors == first
        app.processEvents()
    finally:
        window.close()
