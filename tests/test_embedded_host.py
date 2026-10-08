"""Embedded Spejl must not outlive OmniBIM's host window."""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


class _FakeUser32:
    def __init__(self) -> None:
        self.host_alive = True

        def is_window(hwnd):
            assert hwnd == 0x1234
            return 1 if self.host_alive else 0

        self.IsWindow = is_window


def test_embedded_editor_quits_once_its_host_window_is_gone(monkeypatch):
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    from spejl.gui import __main__ as gui_main

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui_main, "_HOST_CHECK_MS", 20)
    quits = []
    monkeypatch.setattr(app, "quit", lambda: quits.append(1))
    user32 = _FakeUser32()
    timer = gui_main._quit_when_host_closes(app, user32, 0x1234)
    try:
        end = time.time() + 0.3
        while time.time() < end:
            QCoreApplication.processEvents()
            time.sleep(0.01)
        assert quits == []  # host still open: keep running

        user32.host_alive = False  # OmniBIM closed
        end = time.time() + 2
        while time.time() < end and not quits:
            QCoreApplication.processEvents()
            time.sleep(0.01)
        assert quits
    finally:
        timer.stop()


def test_installer_closes_a_leftover_spejl_before_copying():
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "installer" / "Spejl.iss").read_text(encoding="utf-8")
    code = script[script.index("[Code]"):]
    for needed in ("function PrepareToInstall", "function InitializeUninstall",
                   'IMAGENAME eq Spejl.exe', "taskkill.exe", "USERNAME eq"):
        assert needed in code, needed
