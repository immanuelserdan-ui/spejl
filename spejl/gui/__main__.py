"""``python -m spejl.gui`` — launch the desktop app."""

from __future__ import annotations

import sys
import os
import json
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QTimer, Qt
from PySide6.QtWidgets import QApplication, QGraphicsOpacityEffect

from spejl.gui.main_window import MainWindow
from spejl.gui.splash import StartupSplash


def main() -> int:
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
