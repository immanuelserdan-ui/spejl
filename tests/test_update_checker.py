from spejl.gui.update_checker import find_update


def test_new_release_returns_installer_asset():
    info = find_update(
        {
            "tag_name": "v0.1.2",
            "html_url": "https://github.com/immanuelserdan-ui/spejl/releases/tag/v0.1.2",
            "assets": [
                {
                    "name": "Spejl-0.1.2-Windows-x64-Setup.exe",
                    "browser_download_url": "https://example/setup.exe",
                }
            ],
        },
        current_version="0.1.1",
    )
    assert info is not None
    assert info.version == "0.1.2"
    assert info.download_url.endswith("setup.exe")


def test_same_release_is_ignored():
    assert find_update({"tag_name": "v0.1.1", "assets": []}, current_version="0.1.1") is None


def test_window_skips_the_update_check_when_opted_out(monkeypatch):
    """The suite sets SPEJL_NO_UPDATE_CHECK (tests/conftest.py): a test
    window must never ask GitHub, or a modal update box hangs the run."""
    import os
    import time

    import pytest

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    from spejl.gui import update_checker
    from spejl.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    calls = []
    monkeypatch.setattr(update_checker, "fetch_update", lambda: calls.append(1))
    window = MainWindow()
    try:
        assert os.environ.get("SPEJL_NO_UPDATE_CHECK") == "1"
        window._check_for_updates()
        assert window._update_thread is None

        # Without the opt-out the check still runs (against the stand-in).
        monkeypatch.delenv("SPEJL_NO_UPDATE_CHECK")
        window._check_for_updates()
        assert window._update_thread is not None
        end = time.time() + 10
        while time.time() < end and window._update_thread is not None:
            app.processEvents()
            QCoreApplication.processEvents()
            time.sleep(0.01)
        # Windows left open by earlier tests still hold a pending 1.5 s check
        # and may also fire while the opt-out is lifted; all reach the stand-in.
        assert calls
    finally:
        window.close()


import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox
from spejl.gui.main_window import MainWindow
from spejl.gui.update_checker import UpdateInfo, UpdateCheckWorker


def test_notifications_start_only_after_window_is_ready(qtbot, monkeypatch):
    monkeypatch.delenv("SPEJL_NO_UPDATE_CHECK", raising=False)
    window = MainWindow()
    qtbot.addWidget(window)
    calls = []
    monkeypatch.setattr(window, "_check_for_updates", lambda: calls.append(1))
    qtbot.wait(1600)
    assert calls == []  # Old code checked while the splash covered the window.
    window.start_update_notifications()
    assert calls == []
    window.show()
    window.start_update_notifications()
    window.start_update_notifications()
    assert calls == [1]


def test_centered_notice_reappears_on_next_launch(qtbot, monkeypatch):
    from spejl.gui import main_window
    downloads = []
    monkeypatch.setattr(main_window.QDesktopServices, "openUrl", lambda url: downloads.append(url.toString()))
    info = UpdateInfo("0.1.99", "https://github.com/immanuelserdan-ui/spejl/releases", "https://github.com/immanuelserdan-ui/spejl/setup.exe")
    for launch in range(2):
        window = MainWindow()
        qtbot.addWidget(window)
        window.setGeometry(80, 100, 1000, 650)
        window.show()
        qtbot.wait(20)
        window._on_update_result(info)
        dialog = window._update_dialog
        assert dialog is not None and dialog.isVisible()
        assert dialog.windowModality() == Qt.WindowModality.WindowModal
        assert (dialog.frameGeometry().center() - window.frameGeometry().center()).manhattanLength() <= 2
        button = next(b for b in dialog.buttons() if b.text() == ("Later" if launch == 0 else "Download update"))
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
        assert window._update_dialog is None
        window.close()
    assert downloads == [info.download_url]


def test_update_waits_for_visible_parent(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._on_update_result(UpdateInfo("0.1.99", "https://github.com/immanuelserdan-ui/spejl/releases", None))
    assert window._update_dialog is None
    assert window._pending_update is not None
    window.show()
    window._show_update_notice()
    assert window._update_dialog.isVisible()
    window._update_dialog.reject()
    window.close()


def test_transient_network_failure_is_retried(monkeypatch):
    from spejl.gui import update_checker
    calls = []
    info = UpdateInfo("0.1.99", "https://github.com/immanuelserdan-ui/spejl/releases", None)
    def fetch():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("Temporary network failure")
        return info
    monkeypatch.setattr(update_checker, "fetch_update", fetch)
    monkeypatch.setattr(update_checker, "sleep", lambda seconds: None)
    worker = UpdateCheckWorker()
    results = []
    worker.finished.connect(results.append)
    worker.run()
    assert len(calls) == 3
    assert results == [info]


def test_equal_or_newer_installed_version_is_not_prompted():
    assert find_update({"tag_name": "v0.1.24"}, "0.1.24") is None
    assert find_update({"tag_name": "v0.1.24"}, "0.1.25") is None
