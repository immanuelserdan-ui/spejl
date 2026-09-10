"""Small reusable widgets for the main window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import QFileDialog, QFrame, QLabel, QVBoxLayout, QWidget

SUPPORTED_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class DropZone(QFrame):
    """Drag a plan in, or click to browse — the whole zone is a button."""

    file_chosen = Signal(Path)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(120)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("dropZone")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon = QLabel("\U0001F5CE")  # 🗎 — a plain document glyph, no colour dependency
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon.setStyleSheet("font-size: 28px;")
        self._title = QLabel("Drop a plan here, or click to browse")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setWordWrap(True)
        self._subtitle = QLabel("PDF · PNG · JPEG · TIFF · BMP")
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Styled inline, not via an ancestor's ID-selector rule: DropZone
        # sets its own stylesheet on itself below (to redraw the dashed
        # border on drag-active), and Qt's cascade doesn't carry an
        # ancestor's selector-based rules past a widget that has done
        # that — see main_window.py's _STYLESHEET comment for the fuller
        # story (the same pattern made the Mirror Plan button invisible).
        self._subtitle.setStyleSheet("color: palette(mid); font-size: 11px;")
        layout.addWidget(self._icon)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)

        self._update_style(active=False)

    def set_file(self, path: Path) -> None:
        self._title.setText(path.name)
        self._subtitle.setText(str(path.parent))

    def _update_style(self, active: bool) -> None:
        border = "#4FC3E0" if active else "palette(mid)"
        self.setStyleSheet(
            f"#dropZone {{ border: 2px dashed {border}; border-radius: 8px; }}"
        )

    # -- drag and drop -----------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 (Qt override)
        if event.mimeData().hasUrls() and self._first_supported_path(event):
            self._update_style(active=True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._update_style(active=False)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._update_style(active=False)
        path = self._first_supported_path(event)
        if path is not None:
            self.file_chosen.emit(path)
            event.acceptProposedAction()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._browse()

    def _browse(self) -> None:
        exts = " ".join(f"*{s}" for s in SUPPORTED_SUFFIXES)
        path_str, _filter = QFileDialog.getOpenFileName(
            self, "Choose a floor plan", "", f"Floor plans ({exts});;All files (*)"
        )
        if path_str:
            self.file_chosen.emit(Path(path_str))

    @staticmethod
    def _first_supported_path(event) -> Path | None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
                return path
        return None


class ScaledImageLabel(QLabel):
    """A QLabel that keeps its pixmap fit to whatever size it's given."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)  # allow shrinking below the pixmap's natural size

    def set_pixmap_source(self, pixmap: QPixmap | None) -> None:
        self._source = pixmap
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source is None or self._source.isNull():
            self.setPixmap(QPixmap())
            return
        scaled = self._source.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
