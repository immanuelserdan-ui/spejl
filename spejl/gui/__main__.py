"""``python -m spejl.gui`` — launch the desktop app."""

from __future__ import annotations

import sys
import os
import json
import ctypes
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QTimer, Qt
from PySide6.QtWidgets import QApplication, QGraphicsOpacityEffect

from spejl.gui.main_window import MainWindow
from spejl.gui.splash import StartupSplash


def run_embedded(parent_hwnd: int) -> int:
    """Host the full editor inside OmniBIM's native child window on Windows."""
    if sys.platform != "win32" or parent_hwnd <= 0:
        raise ValueError("Embedding requires a valid Windows parent window")

    app = QApplication(sys.argv)
    app.setApplicationName("Spejl")
    window = MainWindow(embedded=True)
    window.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
    child_hwnd = int(window.winId())

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.SetParent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    user32.SetParent.restype = ctypes.c_void_p
    user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    user32.SetWindowPos.restype = ctypes.c_int

    gwl_style = -16
    ws_child, ws_popup = 0x40000000, 0x80000000
    ws_chrome = 0x00CF0000
    style = user32.GetWindowLongPtrW(child_hwnd, gwl_style)
    user32.SetWindowLongPtrW(child_hwnd, gwl_style, (style | ws_child) & ~(ws_popup | ws_chrome))
    if not user32.SetParent(child_hwnd, parent_hwnd):
        raise ctypes.WinError(ctypes.get_last_error())
    swp_framechanged, swp_nozorder = 0x0020, 0x0004
    user32.SetWindowPos(child_hwnd, None, 0, 0, 1, 1, swp_framechanged | swp_nozorder)
    window.show()
    # Qt can restore native frame bits while showing the window. Apply the
    # child style again so the hosted editor fills the entire WPF panel.
    style = user32.GetWindowLongPtrW(child_hwnd, gwl_style)
    user32.SetWindowLongPtrW(child_hwnd, gwl_style, (style | ws_child) & ~(ws_popup | ws_chrome))
    user32.SetWindowPos(child_hwnd, None, 0, 0, 1, 1, swp_framechanged | swp_nozorder)
    _quit_when_host_closes(app, user32, parent_hwnd)
    return app.exec()


_HOST_CHECK_MS = 1000


def _quit_when_host_closes(app: QApplication, user32, parent_hwnd: int) -> QTimer:
    """Exit once OmniBIM's host window is gone.

    Windows destroys the reparented editor window together with its host, but
    Qt gets no close event for that, so Spejl.exe used to keep running with no
    window -- holding Spejl.exe and its DLLs open, which made the next
    installer fail with "DeleteFile failed; code 5".
    """
    user32.IsWindow.argtypes = [ctypes.c_void_p]
    user32.IsWindow.restype = ctypes.c_int
    timer = QTimer(app)
    timer.setInterval(_HOST_CHECK_MS)
    timer.timeout.connect(lambda: None if user32.IsWindow(parent_hwnd) else app.quit())
    timer.start()
    return timer


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1:3] == ["--embedded", "--parent-hwnd"]:
        return run_embedded(int(sys.argv[3], 16))
    app = QApplication(sys.argv)
    app.setApplicationName("Spejl")
    app.setQuitOnLastWindowClosed(False)
    splash = StartupSplash()
    splash.showFullScreen()
    window_holder = []

    def prepare_main_window() -> None:
        if not window_holder:
            window_holder.append(MainWindow())

    def reveal_main_window() -> None:
        # Keep painting the live preview while its opacity falls, with the
        # fully rendered main window already underneath it.
        prepare_main_window()
        window = window_holder[0]
        splash.setParent(window, Qt.WindowType.Widget)
        splash.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        opacity = QGraphicsOpacityEffect(splash)
        splash.setGraphicsEffect(opacity)
        window.showFullScreen()
        splash.setGeometry(window.rect())
        splash.raise_()
        splash.show()

        fade = QPropertyAnimation(opacity, b"opacity", window)
        fade.setDuration(StartupSplash.TRANSITION_MS)
        fade.setStartValue(1.0)
        fade.setEndValue(0.0)
        fade.setEasingCurve(QEasingCurve.Type.InOutCubic)
        def finish_transition() -> None:
            splash.close()
            splash.deleteLater()
            app.setQuitOnLastWindowClosed(True)
            # Confirm the actual main window, not a splash or error dialog.
            report_path = os.environ.get("SPEJL_SMOKE_TEST_REPORT")
            if report_path:
                def report_ready() -> None:
                    if not window.isVisible():
                        app.exit(1)
                        return
                    Path(report_path).write_text(
                        json.dumps({"status": "ready", "pid": os.getpid()}),
                        encoding="utf-8",
                    )
                    app.exit(0)

                QTimer.singleShot(250, report_ready)

        fade.finished.connect(finish_transition)
        fade.start()
        window_holder.append(fade)

    QTimer.singleShot(100, prepare_main_window)
    presentation_allowance_ms = 600
    QTimer.singleShot(
        StartupSplash.DURATION_MS - StartupSplash.TRANSITION_MS - presentation_allowance_ms,
        reveal_main_window,
    )
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
