"""Regression checks for isolated modern GUI layers.

These tests intentionally avoid invoking the mirror engine. They prove that
the new controller and page navigation do not need to modify protected
rendering code to provide their features.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_document_session_only_changes_preview_page(tmp_path):
    from spejl.gui.document_session import DocumentSession

    pdf = tmp_path / "two_pages.pdf"
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.save(pdf)
    doc.close()

    session = DocumentSession()
    session.open(pdf)
    assert session.page_count == 2
    assert session.page_index == 0
    assert session.move(1)
    assert session.page_index == 1
    assert session.path == pdf


def test_window_exposes_isolated_modern_controls(qapp):
    from spejl.gui.main_window import MainWindow

    window = MainWindow()
    try:
        window.show()
        qapp.processEvents()
        assert window._cancel_button.isHidden()
        assert window._page_label.text() == "Single page"
        assert not window._previous_page_button.isEnabled()
        assert not window._next_page_button.isEnabled()
        assert "Vector PDF" in window._drop_zone._subtitle.text()
    finally:
        window.close()


def test_text_nudging_only_snaps_when_alt_is_held(qapp):
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtTest import QTest
    from spejl.gui.widgets import ScaledImageLabel

    view = ScaledImageLabel()
    view.set_text_regions([(0, QRectF(10, 10, 20, 12))], selected={0})
    moves = []
    view.text_nudged.connect(lambda *args: moves.append(args))
    view.show()
    view.setFocus()
    qapp.processEvents()
    try:
        QTest.keyClick(view, Qt.Key.Key_Right)
        QTest.keyClick(view, Qt.Key.Key_Right, Qt.KeyboardModifier.AltModifier)
        QTest.keyClick(view, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        assert moves == [
            ([0], 1.0, 0.0, False),
            ([0], 1.0, 0.0, True),
            ([0], 5.0, 0.0, False),
        ]
    finally:
        view.close()
