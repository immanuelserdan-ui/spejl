"""``python -m spejl.gui`` — launch the desktop app."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from spejl.gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Spejl")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
