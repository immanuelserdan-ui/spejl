"""Mixed image/vector PDFs are explained on upload, not only after Mirror."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pymupdf = pytest.importorskip("pymupdf")

from spejl.router import native_vector_pdf_problem


def _pdf(path: Path, *, image: pymupdf.Rect | None = None) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=300)
    shape = page.new_shape()
    shape.draw_rect(pymupdf.Rect(20, 20, 180, 280))
    shape.finish(color=(0, 0, 0), width=2)
    shape.commit()
    page.insert_text((60, 150), "Stue", fontsize=12)
    if image is not None:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4, 4), 0)
        pixmap.clear_with(200)
        page.insert_image(image, pixmap=pixmap)
    doc.save(path)
    doc.close()
    return path


def test_vector_pdf_has_no_problem(tmp_path):
    assert native_vector_pdf_problem(_pdf(tmp_path / "vector.pdf")) is None


def test_full_page_pictures_explain_the_revit_export(tmp_path):
    # Revit's raster processing: the drawing is two stacked page-wide JPEGs.
    path = tmp_path / "mixed.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=300)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4, 4), 0)
    pixmap.clear_with(255)
    page.insert_image(pymupdf.Rect(0, 0, 200, 150), pixmap=pixmap)
    page.insert_image(pymupdf.Rect(0, 150, 200, 300), pixmap=pixmap)
    page.insert_text((60, 150), "Stue", fontsize=12)
    doc.save(path)
    doc.close()

    problem = native_vector_pdf_problem(path)
    assert problem is not None
    assert problem.badge.startswith("Mixed image PDF")
    assert problem.where == "mixed.pdf, page 1"
    assert "2 pictures covering 100% of the page" in problem.reason
    assert "the walls and linework are pictures" in problem.reason
    assert "Vector processing" in problem.reason
    assert problem.detail.startswith("mixed.pdf, page 1: contains embedded image content")


def test_small_logo_is_reported_as_something_to_hide(tmp_path):
    problem = native_vector_pdf_problem(
        _pdf(tmp_path / "logo.pdf", image=pymupdf.Rect(150, 260, 190, 290))
    )
    assert problem is not None
    assert "1 picture covering 2% of the page" in problem.reason
    assert "probably a logo or underlay; hide it" in problem.reason


def test_unreadable_pdf_is_a_problem_not_a_crash(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7\nnot really a pdf")
    problem = native_vector_pdf_problem(path)
    assert problem is not None
    assert problem.badge == "Unreadable PDF"


@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_upload_flags_a_mixed_pdf_before_mirroring(qapp, tmp_path):
    from spejl.gui.main_window import MainWindow

    good = _pdf(tmp_path / "634-T08-A00-R-V00-R00.pdf")
    mixed = _pdf(tmp_path / "634-T01-A00-S-V00-R00.pdf", image=pymupdf.Rect(0, 0, 200, 300))
    window = MainWindow()
    try:
        window._on_files_chosen([good, mixed])
        assert "1 cannot be mirrored" not in window._status_label.text()
        assert "634-T01-A00-S-V00-R00.pdf, page 1: contains embedded image content" in (
            window._status_label.text()
        )
        assert window._uploaded_list.item(0).text() == good.name
        assert window._uploaded_list.item(1).text().startswith("⚠")
        assert window._mirrored_list.item(1).text().startswith("⚠")
        assert "Vector processing" in window._uploaded_list.item(1).toolTip()

        window._select_batch_entry(mixed.resolve())
        assert window._route_badge.property("route") == "blocked"
        assert "Mixed image PDF" in window._route_badge.text()
        assert window._status_label.text().startswith("⚠ 634-T01-A00-S-V00-R00.pdf, page 1")

        window._select_batch_entry(good.resolve())
        assert window._route_badge.property("route") == "vector"
    finally:
        window.close()


def test_mirror_with_only_problem_plans_explains_instead_of_failing(qapp, tmp_path, monkeypatch):
    from spejl.gui.batch_dialogs import MirrorSelectionDialog
    from spejl.gui.main_window import MainWindow

    # A problem no review can fix (picture-drawn plans go through the
    # picture text review instead; see test_mixed_pdf.py).
    from spejl.router import InputProblem

    paths = [_pdf(tmp_path / f"634-T0{i}-A00-S-V00-R00.pdf") for i in (1, 2)]
    window = MainWindow()
    try:
        window._on_files_chosen(paths)
        for path in paths:
            window._batch_entries[path.resolve()].input_problem = InputProblem(
                "Unreadable PDF", path.name, "could not read as a valid PDF (test).")
        monkeypatch.setattr(
            MirrorSelectionDialog, "exec",
            lambda self: pytest.fail("no dialog when nothing can be mirrored"),
        )
        window._on_mirror_clicked()
        assert window._worker is None and window._picture_scan_worker is None
        text = window._status_label.text()
        assert text.startswith("⚠ None of the 2 uploaded plans can be mirrored: could not read as a valid PDF")
    finally:
        window.close()


def test_cancelled_review_of_the_only_plan_says_so(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from spejl.gui import main_window as mw
    from spejl.gui.batch_dialogs import MirrorSelectionDialog
    from spejl.gui.picture_review import PictureTextDialog
    from spejl.vector import mixed_pdf

    path = _pdf(tmp_path / "634-T01-A00-S-V00-R00.pdf", image=pymupdf.Rect(0, 0, 200, 300))
    monkeypatch.setattr(mixed_pdf, "find_picture_labels", lambda p: [])
    monkeypatch.setattr(PictureTextDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    monkeypatch.setattr(MirrorSelectionDialog, "exec", lambda self: pytest.fail("nothing to mirror"))
    window = mw.MainWindow()
    try:
        window._on_files_chosen([path])
        window._on_mirror_clicked()
        import time

        from PySide6.QtCore import QCoreApplication

        end = time.time() + 10
        while time.time() < end and "review was cancelled" not in window._status_label.text():
            QCoreApplication.processEvents()
            time.sleep(0.01)
        assert window._status_label.text().startswith("Nothing to mirror: the picture text review was cancelled")
    finally:
        window.close()


def test_selection_dialog_leaves_problem_plans_unchecked(qapp, tmp_path):
    from PySide6.QtCore import Qt

    from spejl.gui.batch_dialogs import MirrorSelectionDialog
    from spejl.gui.batch_model import BatchEntry

    good = _pdf(tmp_path / "good.pdf")
    mixed = _pdf(tmp_path / "mixed.pdf", image=pymupdf.Rect(0, 0, 200, 300))
    entries = [BatchEntry(good, good.name), BatchEntry(mixed, mixed.name)]
    entries[1].input_problem = native_vector_pdf_problem(mixed)
    dialog = MirrorSelectionDialog(entries)
    try:
        assert dialog.list.item(0).checkState() == Qt.CheckState.Checked
        assert dialog.list.item(1).checkState() == Qt.CheckState.Unchecked
        assert dialog.list.item(1).text().startswith("⚠  mixed.pdf — Mixed image PDF")
        assert dialog.selected_paths() == [good]
    finally:
        dialog.close()


def test_batch_summary_names_the_shared_failure_reason(tmp_path):
    pytest.importorskip("PySide6")
    from spejl.gui.batch_model import BatchEntry
    from spejl.gui.main_window import _failure_summary

    entries = []
    for name in ("a.pdf", "b.pdf"):
        path = _pdf(tmp_path / name, image=pymupdf.Rect(0, 0, 200, 300))
        entry = BatchEntry(path, name, status="failed")
        entry.input_problem = native_vector_pdf_problem(path)
        entry.error = entry.input_problem.detail
        entries.append(entry)
    assert _failure_summary(entries).startswith("Both: contains embedded image content")
    assert _failure_summary(entries[:1]) == entries[0].error

    other = BatchEntry(tmp_path / "c.pdf", "c.pdf", status="failed", error="Text transform QA failed")
    assert _failure_summary([*entries, other]) == "Select a ✖ plan in Mirrored to see why it failed."


def test_corrupt_pdf_upload_is_explained_not_a_crash(qapp, tmp_path, monkeypatch):
    """A damaged .pdf next to a good plan: no exception, the reason is shown,
    and the good plan still mirrors."""
    import time

    from PySide6.QtCore import QCoreApplication

    from spejl.gui.batch_dialogs import MirrorSelectionDialog, PdfPreviewDialog
    from spejl.gui.main_window import MainWindow

    broken = tmp_path / "634-T02-A00-S-V00-R00.pdf"
    broken.write_bytes(b"%PDF-1.7\nnot really a pdf")
    good = _pdf(tmp_path / "634-T01-A00-S-V00-R00.pdf")
    selection = {}

    def choose(dialog):
        selection["ticked"] = dialog.selected_paths()
        return 1

    monkeypatch.setattr(MirrorSelectionDialog, "exec", choose)
    window = MainWindow()
    try:
        window._on_files_chosen([good, broken])  # used to raise FileDataError
        window._select_batch_entry(broken.resolve())
        assert window._route_badge.text() == "⚠ Unreadable PDF"
        assert window._status_label.text().startswith("⚠ 634-T02-A00-S-V00-R00.pdf: could not read as a valid PDF")
        preview = PdfPreviewDialog(broken.resolve())
        assert "could not be read" in preview.image.text()
        preview.close()

        window._on_mirror_clicked()
        assert selection["ticked"] == [good.resolve()]
        end = time.time() + 30
        entry = window._batch_entries[good.resolve()]
        while time.time() < end and entry.status != "completed":
            QCoreApplication.processEvents()
            time.sleep(0.01)
        assert entry.status == "completed"
        assert window._batch_entries[broken.resolve()].status == "queued"
    finally:
        window.close()
