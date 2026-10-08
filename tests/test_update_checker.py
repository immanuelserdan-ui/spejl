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
