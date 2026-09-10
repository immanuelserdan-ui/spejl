"""Lightweight GUI regression coverage.

A full click-through (load a real plan, run the threaded mirror,
verify the preview and flags, save the result) is scripted manually in
the build session rather than kept as a slow, environment-sensitive
pytest — offscreen Qt platforms often lack real font/rendering support,
which makes a screenshot-diffing CI test brittle for the wrong reasons.
This file keeps the fast, deterministic part: the window constructs,
wires its signals, and its pure state-transition logic behaves — so a
refactor that breaks the wiring fails immediately, without needing a
real OCR run or a real display.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from spejl.models import Axis  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    from spejl.gui.main_window import MainWindow

    win = MainWindow()
    # Qt's isVisible() reflects actual on-screen visibility, which
    # requires the whole parent chain to have been shown — not just
    # `.show()` called on a child widget. show() the window itself so
    # child-widget visibility assertions behave the way they do for a
    # real running app, matching how MainWindow is actually used.
    win.show()
    yield win
    win.close()


def test_mirror_button_actually_renders_its_accent_colour(window, tmp_path):
    """Regression for a real bug found in manual testing: the sidebar
    container had its own setStyleSheet() call for its background/
    border, which — silently, with no error and no visible sign beyond
    the widget just not being there — blocks Qt's style-sheet cascade
    from carrying the MAIN WINDOW's ID-selector rules (#mirrorButton,
    #saveButton, #routeBadge) down to that container's children at all.
    The button reported correct geometry, text, and enabled state the
    whole time; only the actual painted pixels were wrong.

    Property checks (isVisible, isEnabled, geometry) do NOT catch this
    class of bug — only sampling real rendered pixels does, which is
    why this test grabs the widget and checks its own paint output
    rather than asking Qt about its logical state.
    """
    from spejl.qa import fixture_gen

    info = fixture_gen.generate(tmp_path, dpi=100)
    window._on_file_chosen(info["png"])
    window._mirror_button.setEnabled(True)  # exercise the enabled (accent-blue) paint path
    window.resize(1172, 750)
    window.show()
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()

    pixmap = window._mirror_button.grab()
    image = pixmap.toImage()
    assert image.width() > 0 and image.height() > 0

    # Sample the centre pixel: accent blue (#0A6E8A) if rendering
    # correctly, or the sidebar's plain white/base background if the
    # cascade is broken again and the button silently isn't painting.
    centre = image.pixelColor(image.width() // 2, image.height() // 2)
    is_near_white = centre.red() > 230 and centre.green() > 230 and centre.blue() > 230
    assert not is_near_white, (
        f"mirror button centre pixel is {centre.getRgb()} — looks unpainted, "
        "not the accent-coloured button"
    )
    # And specifically: it should be recognisably the dark teal accent,
    # not just "not white" for some unrelated reason.
    assert centre.blue() > centre.red(), f"expected a blue-leaning accent, got {centre.getRgb()}"


def test_window_constructs_with_expected_widgets(window):
    assert window._mirror_button.isEnabled() is False  # no file chosen yet
    assert window._save_button.isEnabled() is False
    assert window._axis_buttons[Axis.VERTICAL].isChecked()  # sensible default
    assert not window._progress.isVisible()


def test_choosing_a_file_enables_mirror_and_shows_route(window, tmp_path):
    from spejl.qa import fixture_gen

    info = fixture_gen.generate(tmp_path, dpi=100)
    window._on_file_chosen(info["png"])

    assert window._input_path == info["png"]
    assert window._mirror_button.isEnabled()
    assert window._route_badge.isVisible()
    assert "Raster" in window._route_badge.text()


def test_choosing_a_pdf_shows_the_vector_route(window, tmp_path):
    from spejl.qa import fixture_gen

    info = fixture_gen.generate(tmp_path, dpi=100)
    window._on_file_chosen(info["pdf"])

    assert window._route_badge.isVisible()
    assert "Vector" in window._route_badge.text()


def test_choosing_a_second_file_resets_prior_results(window, tmp_path):
    from spejl.qa import fixture_gen

    info = fixture_gen.generate(tmp_path, dpi=100)
    window._save_button.setEnabled(True)  # simulate a completed prior mirror
    window._flags_list.addItem("stale flag from a previous run")

    window._on_file_chosen(info["png"])

    assert window._save_button.isEnabled() is False
    assert window._flags_list.count() == 0
    assert window._mirrored_view._source is None


def test_mirror_failure_is_shown_not_raised(window, tmp_path, qapp):
    """An unreadable file must surface via the failed signal and a
    dialog, never as an unhandled exception on the Qt event loop."""
    from PySide6.QtCore import QEventLoop, QTimer

    bad = tmp_path / "not_a_real_image.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)  # valid magic, garbage body
    window._on_file_chosen(bad)

    loop = QEventLoop()
    outcome = {}

    def on_fail(message: str) -> None:
        outcome["failed"] = message
        loop.quit()

    def on_ok(document, route) -> None:
        outcome["ok"] = True
        loop.quit()

    # Patch QMessageBox so the failure dialog doesn't block the test.
    import spejl.gui.main_window as mw

    original = mw.QMessageBox.critical
    mw.QMessageBox.critical = staticmethod(lambda *a, **k: None)
    try:
        window._on_mirror_clicked()
        window._worker.failed.connect(on_fail)
        window._worker.succeeded.connect(on_ok)
        QTimer.singleShot(15000, loop.quit)
        loop.exec()
    finally:
        mw.QMessageBox.critical = original

    assert "failed" in outcome, outcome
    assert window._mirror_button.isEnabled()  # re-enabled after failure
