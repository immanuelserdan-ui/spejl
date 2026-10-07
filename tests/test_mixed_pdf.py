"""Assisted mirroring of mixed PDFs: vector text over a raster drawing."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("cv2")
pytest.importorskip("scipy")

from spejl.detect.ocr import Detection
from spejl.models import Axis
from spejl.vector import mixed_pdf
from spejl.vector.mixed_pdf import PictureLabel, find_picture_labels, mirror_mixed_pdf
from spejl.vector.pdf_mirror import text_drawing_overlap_findings

PAGE_W, PAGE_H = 200.0, 300.0
PICTURE_DPI = 150
LABEL_ORIGIN = (120.0, 150.0)  # where "H*" is lettered inside the picture (page points)


def _make_mixed_pdf(path: Path) -> tuple[Path, pymupdf.Rect]:
    """A Revit-style raster export: linework + one label as a picture, vector text on top."""
    drawing = pymupdf.open()
    sheet = drawing.new_page(width=PAGE_W, height=PAGE_H)
    shape = sheet.new_shape()
    shape.draw_rect(pymupdf.Rect(30, 40, 170, 260))  # the walls
    shape.draw_line(pymupdf.Point(30, 120), pymupdf.Point(100, 120))  # an inner wall, left side only
    shape.finish(color=(0, 0, 0), width=3)
    shape.commit()
    sheet.insert_text(LABEL_ORIGIN, "H*", fontsize=9, fontname="helv", rotate=90)  # lettering in the model
    label_ink = _ink_box(sheet, pymupdf.Rect(105, 125, 135, 160))
    pix = sheet.get_pixmap(dpi=PICTURE_DPI, alpha=False)

    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, pixmap=pix)
    page.insert_text((60, 200), "Stue", fontsize=12, fontname="helv")  # real vector text
    doc.save(path)
    return path, label_ink


def _ink_box(page: pymupdf.Page, clip: pymupdf.Rect) -> pymupdf.Rect:
    pix = page.get_pixmap(dpi=600, clip=clip, colorspace=pymupdf.csGRAY, alpha=False)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
    ys, xs = np.where(a < 128)
    s = 72 / 600
    return pymupdf.Rect(clip.x0 + xs.min() * s, clip.y0 + ys.min() * s,
                        clip.x0 + (xs.max() + 1) * s, clip.y0 + (ys.max() + 1) * s)


def _fake_detections(label_ink: pymupdf.Rect):
    """OCR stand-in: finds the label (confidently) and a symbol (not)."""

    def detect(image, backend, **_kwargs):
        scale = image.shape[1] / (PAGE_W * PICTURE_DPI / 72)  # the upscale find_picture_labels chose
        k = PICTURE_DPI / 72 * scale
        x0, y0, x1, y1 = (v * k for v in (label_ink.x0 - 1, label_ink.y0 - 1, label_ink.x1 + 1, label_ink.y1 + 1))
        quad = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        sx0, sy0 = 150 * k, 60 * k
        symbol = ((sx0, sy0), (sx0 + 8 * k, sy0), (sx0 + 8 * k, sy0 + 8 * k), (sx0, sy0 + 8 * k))
        return [Detection("H*", quad, 0.95, 90.0), Detection("←", symbol, 0.6, 0.0)]

    return detect


@pytest.fixture
def mixed(tmp_path, monkeypatch):
    path, label_ink = _make_mixed_pdf(tmp_path / "mixed.pdf")
    import spejl.detect.rotations as rotations

    monkeypatch.setattr(rotations, "detect_all_orientations", _fake_detections(label_ink))
    return path, label_ink


def _spans(path: Path) -> dict[str, tuple[pymupdf.Rect, tuple[float, float]]]:
    with pymupdf.open(str(path)) as doc:
        return {s["text"].strip(): (pymupdf.Rect(s["bbox"]), line["dir"])
                for b in doc[0].get_text("dict")["blocks"] for line in b.get("lines", [])
                for s in line["spans"] if s["text"].strip()}


def test_finds_picture_lettering_and_preticks_only_confident_text(mixed):
    path, label_ink = mixed
    labels = find_picture_labels(path, backend=object())
    assert [(label.ocr_text, label.include) for label in labels] == [("←", False), ("H*", True)]
    found = next(label for label in labels if label.text == "H*")
    assert found.angle_deg == 90.0
    # The OCR box is tightened to the lettering's own ink (within a pixel at 150 dpi).
    assert found.page_box == pytest.approx(tuple(label_ink), abs=0.6)


def test_confirmed_lettering_is_redrawn_as_real_text_and_mirrored(mixed, tmp_path):
    path, label_ink = mixed
    labels = find_picture_labels(path, backend=object())
    out = tmp_path / "mirrored.pdf"
    document = mirror_mixed_pdf(path, out, Axis.VERTICAL, labels)

    spans = _spans(out)
    assert set(spans) == {"H*", "Stue"}
    h_box, h_dir = spans["H*"]
    assert h_dir == pytest.approx((0.0, -1.0), abs=1e-3)  # still reads bottom to top
    want = pymupdf.Rect(PAGE_W - label_ink.x1, label_ink.y0, PAGE_W - label_ink.x0, label_ink.y1)
    assert h_box.intersects(want) and abs(h_box.x0 + h_box.x1 - want.x0 - want.x1) / 2 < 1.0
    assert spans["Stue"][1] == pytest.approx((1.0, 0.0), abs=1e-3)

    # The lettering is gone from the picture; the walls are not.
    with pymupdf.open(str(out)) as doc:
        page = doc[0]
        assert len(page.get_images(full=True)) == 1
        picture = mixed_pdf.picture_rgb(doc, page.get_images(full=True)[0][0]).mean(axis=2)
    k = picture.shape[1] / PAGE_W
    lettering = picture[int(label_ink.y0 * k):int(label_ink.y1 * k) + 1, int(label_ink.x0 * k):int(label_ink.x1 * k) + 1]
    assert (lettering < 128).sum() == 0
    wall = picture[int(40 * k) - 2:int(40 * k) + 3, int(60 * k):int(140 * k)]
    assert (wall < 128).mean() > 0.3

    codes = [flag.code for flag in document.pages[0].flags]
    assert "picture-linework" in codes and "picture-text" in codes
    assert any("'←'" in flag.message for flag in document.pages[0].flags if flag.code == "picture-text-skipped")


def test_linework_is_an_exact_left_right_flip(mixed, tmp_path):
    path, _ = mixed
    out = tmp_path / "mirrored.pdf"
    mirror_mixed_pdf(path, out, Axis.VERTICAL, find_picture_labels(path, backend=object()))

    def gray(p):
        with pymupdf.open(str(p)) as doc:
            pix = doc[0].get_pixmap(dpi=72, colorspace=pymupdf.csGRAY, alpha=False)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).astype(int)

    source, mirrored = gray(path)[:, ::-1], gray(out)
    walls = np.zeros(source.shape, bool)
    walls[38:43, 32:168] = True  # top wall band, away from any text
    assert np.abs(source - mirrored)[walls].max() < 60
    # the inner wall started on the left; it now runs on the right
    assert (mirrored[118:123, 110:165] < 128).any() and not (mirrored[118:123, 35:90] < 128).any()


def test_unticked_and_unsupported_lettering(mixed, tmp_path):
    path, _ = mixed
    labels = find_picture_labels(path, backend=object())
    for label in labels:
        label.include = False
    out = tmp_path / "mirrored.pdf"
    mirror_mixed_pdf(path, out, Axis.VERTICAL, labels)
    assert set(_spans(out)) == {"Stue"}  # nothing redrawn; the picture keeps its lettering

    slanted = PictureLabel(0, labels[0].image_xref, "X", 33.0, (1, 1, 5, 5), (1, 1, 2, 2), 0.9, True)
    with pytest.raises(ValueError, match="not supported"):
        mirror_mixed_pdf(path, tmp_path / "bad.pdf", Axis.VERTICAL, [slanted])


def test_overlap_check_reads_picture_ink_not_its_rectangle(mixed, tmp_path):
    path, _ = mixed
    # "Stue" sits on white paper inside a page-wide picture: not an overlap.
    assert [text for _, text in text_drawing_overlap_findings(path)] == []
    with pymupdf.open(str(path)) as doc:
        doc[0].insert_text((40, 42), "Wall", fontsize=8, fontname="helv")  # on the top wall
        doc.save(tmp_path / "on-wall.pdf")
    assert [text for _, text in text_drawing_overlap_findings(tmp_path / "on-wall.pdf")] == ["Wall"]
    # The mirrored page draws the picture flipped; the check must follow it.
    out = tmp_path / "mirrored.pdf"
    mirror_mixed_pdf(tmp_path / "on-wall.pdf", out, Axis.VERTICAL, find_picture_labels(path, backend=object()))
    assert sorted(text for _, text in text_drawing_overlap_findings(out)) == ["Wall"]


@pytest.fixture
def qapp(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from spejl.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_check_for_updates", lambda self: None)
    return QApplication.instance() or QApplication([])


def test_review_unlocks_a_mixed_plan_and_feeds_the_mirror_job(qapp, mixed, monkeypatch):
    import time

    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QDialog

    from spejl.gui import main_window as mw
    from spejl.gui.picture_review import PictureTextDialog

    path, _ = mixed
    real_find = mixed_pdf.find_picture_labels
    monkeypatch.setattr(mixed_pdf, "find_picture_labels", lambda p: real_find(p, backend=object()))
    window = mw.MainWindow()
    try:
        window._on_files_chosen([path])
        entry = window._batch_entries[path.resolve()]
        assert entry.blocking_problem is not None and entry.blocking_problem.kind == "mixed-image"
        assert window._review_picture_button.isVisibleTo(window)
        assert "Review picture text" in window._status_label.text()

        seen = {}

        def confirm(dialog):
            seen["rows"] = [(label.text, label.include) for label in dialog._labels]
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(PictureTextDialog, "exec", confirm)
        window._on_review_picture_clicked()
        end = time.time() + 30
        while time.time() < end and entry.picture_labels is None:
            QCoreApplication.processEvents()
            time.sleep(0.01)
        assert seen["rows"] == [("←", False), ("H*", True)]
        assert entry.blocking_problem is None
        assert window._route_badge.text() == "✓ Picture text confirmed"
        assert window._review_picture_button.text() == "Edit picture text…"
        assert not window._uploaded_list.item(0).text().startswith("⚠")

        started = {}

        def fake_start(*args):
            started["args"] = args
            raise _Stopped

        monkeypatch.setattr(window._job_controller, "start", fake_start)
        window._batch_queue = [path.resolve()]
        with pytest.raises(_Stopped):
            window._start_next_batch_job()
        assert started["args"][3] is entry.picture_labels
    finally:
        window.close()


class _Stopped(Exception):
    pass


def test_dialog_blocks_confirming_a_ticked_find_without_text(qapp, mixed):
    from PySide6.QtWidgets import QDialogButtonBox

    from spejl.gui.picture_review import PictureTextDialog

    path, _ = mixed
    dialog = PictureTextDialog(path, find_picture_labels(path, backend=object()))
    try:
        ok = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
        assert ok.isEnabled()
        include, text, _direction = dialog._rows[1]
        text.setText("")
        assert not ok.isEnabled()
        include.setChecked(False)
        assert ok.isEnabled()
        assert [label.include for label in dialog.labels()] == [False, False]
    finally:
        dialog.close()
