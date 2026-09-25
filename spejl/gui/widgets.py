"""Small reusable widgets for the main window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QFileDialog, QFrame, QLabel, QVBoxLayout, QWidget

SUPPORTED_SUFFIXES = (".pdf",)


class DropZone(QFrame):
    """Drag a plan in, or click to browse — the whole zone is a button."""

    file_chosen = Signal(Path)
    files_chosen = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(152)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("dropZone")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon = QLabel("\U0001F5CE")  # 🗎 — a plain document glyph, no colour dependency
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon.setStyleSheet("font-size: 28px;")
        self._title = QLabel("Drop plans here, or click to browse")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setWordWrap(True)
        self._subtitle = QLabel("Vector PDF only — image-free pages keep text editable")
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle.setWordWrap(True)
        # Styled inline, not via an ancestor's ID-selector rule: DropZone
        # sets its own stylesheet on itself below (to redraw the dashed
        # border on drag-active), and Qt's cascade doesn't carry an
        # ancestor's selector-based rules past a widget that has done
        # that — see main_window.py's _STYLESHEET comment for the fuller
        # story (the same pattern made the Mirror Plan button invisible).
        self._subtitle.setStyleSheet("color: #8CB3C0; font-size: 11px;")
        layout.addWidget(self._icon)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)
        self._formats = QLabel("Scans and mixed image PDFs are rejected; outlined lettering is not editable text")
        self._formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._formats.setWordWrap(True)
        self._formats.setStyleSheet("color: #8CB3C0; font-size: 10px;")
        layout.addWidget(self._formats)

        self._update_style(active=False)

    def set_file(self, path: Path) -> None:
        self._title.setText(path.name)
        self._subtitle.setText(str(path.parent))

    def set_files(self, paths: list[Path]) -> None:
        if len(paths) == 1:
            self.set_file(paths[0])
        elif paths:
            self._title.setText(f"{len(paths)} plans uploaded")
            self._subtitle.setText("Select a filename below to preview it")

    def clear(self) -> None:
        """Restore the empty-file prompt after clearing the current session."""
        self._title.setText("Drop plans here, or click to browse")
        self._subtitle.setText("Vector PDF only — image-free pages keep text editable")

    def _update_style(self, active: bool) -> None:
        border = "#4FC3E0" if active else "#2B6479"
        self.setStyleSheet(
            f"#dropZone {{ background: #07111F; border: 2px dashed {border}; border-radius: 8px; }}"
        )

    # -- drag and drop -----------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 (Qt override)
        if event.mimeData().hasUrls() and self._supported_paths(event):
            self._update_style(active=True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._update_style(active=False)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._update_style(active=False)
        paths = self._supported_paths(event)
        if paths:
            self.files_chosen.emit(paths)
            event.acceptProposedAction()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._browse()

    def _browse(self) -> None:
        self.browse()

    def browse(self) -> None:
        """Open the normal file picker; shared by click and shortcuts."""
        exts = " ".join(f"*{s}" for s in SUPPORTED_SUFFIXES)
        path_strings, _filter = QFileDialog.getOpenFileNames(
            self, "Choose vector PDF floor plans", "", f"Vector PDF files ({exts})"
        )
        if path_strings:
            self.files_chosen.emit([Path(path) for path in path_strings])

    @staticmethod
    def _first_supported_path(event) -> Path | None:
        paths = DropZone._supported_paths(event)
        return paths[0] if paths else None

    @staticmethod
    def _supported_paths(event) -> list[Path]:
        paths: list[Path] = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
                paths.append(path)
        return paths


class ScaledImageLabel(QLabel):
    """A QLabel that draws its pixmap at an EXTERNALLY chosen scale.

    Deliberately not "fit myself to my own size" — that independent
    behaviour is what let the Source and Mirrored panes show the exact
    same drawing at two different on-screen sizes: a QSplitter does not
    guarantee its two sides end up equal width, and two labels each
    independently maximising their own pixmap to fill whatever space
    *they* got will happily draw the same content at two different
    zoom levels with no error and no visual cue beyond "one of these
    looks bigger than the other." A pair of these is meant to be driven
    by one shared-scale coordinator (see MainWindow._sync_preview_scale)
    so a before/after comparison is actually comparable.
    """

    resized = Signal()
    text_selection_changed = Signal(list)
    text_nudged = Signal(list, float, float, bool)
    zoom_requested = Signal(int, float, float)
    pan_requested = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source: QPixmap | None = None
        self._text_regions: list[tuple[int, QRectF]] = []
        self._selected_text: set[int] = set()
        self._content_sized = False
        self._pan_start: QPointF | None = None
        self._pan_dragging = False
        self._alignment_guides: list[tuple[str, float]] = []
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)  # allow shrinking below the pixmap's natural size
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_pixmap_source(self, pixmap: QPixmap | None) -> None:
        self._source = pixmap
        if pixmap is None or pixmap.isNull():
            if self._content_sized:
                self.resize(1, 1)
            self.setPixmap(QPixmap())

    def set_content_sized(self, enabled: bool) -> None:
        """When embedded in a scroll area, make the label match the image.

        A normal preview label fills its pane. A zoomable preview instead
        becomes as large as its rendered image, allowing the scroll area to
        pan to a selected detail instead of clipping it.
        """
        self._content_sized = enabled

    def set_text_regions(
        self, regions: list[tuple[int, QRectF]], selected: set[int] | None = None
    ) -> None:
        """Set selectable text boxes in source-pixmap coordinates."""
        self._text_regions = regions
        available = {index for index, _rect in regions}
        self._selected_text = set(selected or ()) & available
        self._alignment_guides = []
        self.update()

    def snap_text_nudge(
        self, right: float, down: float, *, pixels_per_point: float
    ) -> tuple[float, float, list[tuple[str, float]]]:
        """Snap a selected group's edges or centre to nearby text.

        The comparison is entirely between text boxes in the rendered PDF;
        it never alters a font matrix, glyph spacing, or rotation. ``right``
        and ``down`` remain PDF points so the caller can pass them directly
        to the vector content-stream editor.
        """
        if not self._selected_text or pixels_per_point <= 0:
            return right, down, []
        selected_rects = [rect for index, rect in self._text_regions if index in self._selected_text]
        other_rects = [rect for index, rect in self._text_regions if index not in self._selected_text]
        if not selected_rects or not other_rects:
            return right, down, []

        group = QRectF(selected_rects[0])
        for rect in selected_rects[1:]:
            group = group.united(rect)
        tolerance = 4.0 * pixels_per_point
        guides: list[tuple[str, float]] = []

        def features(rect: QRectF, axis: str) -> tuple[float, float, float]:
            if axis == "x":
                return rect.left(), rect.center().x(), rect.right()
            return rect.top(), rect.center().y(), rect.bottom()

        if right:
            proposed = right * pixels_per_point
            guide: tuple[float, float] | None = None
            snap: tuple[float, float] | None = None
            for candidate in other_rects:
                for source_feature in features(group, "x"):
                    for target_feature in features(candidate, "x"):
                        correction = target_feature - (source_feature + proposed)
                        if abs(correction) <= tolerance and (
                            guide is None or abs(correction) < abs(guide[0])
                        ):
                            guide = (correction, target_feature)
                        distance = target_feature - source_feature
                        # Snap only when this press reaches or crosses a
                        # guide in its requested direction. This preserves
                        # smooth one-point movement next to aligned labels.
                        if (
                            distance * proposed > 0
                            and abs(distance) <= abs(proposed)
                            and (snap is None or abs(correction) < abs(snap[0]))
                        ):
                            snap = (correction, target_feature)
            if snap is not None:
                right += snap[0] / pixels_per_point
            if guide is not None:
                guides.append(("x", guide[1]))

        if down:
            proposed = down * pixels_per_point
            guide = None
            snap = None
            for candidate in other_rects:
                for source_feature in features(group, "y"):
                    for target_feature in features(candidate, "y"):
                        correction = target_feature - (source_feature + proposed)
                        if abs(correction) <= tolerance and (
                            guide is None or abs(correction) < abs(guide[0])
                        ):
                            guide = (correction, target_feature)
                        distance = target_feature - source_feature
                        if (
                            distance * proposed > 0
                            and abs(distance) <= abs(proposed)
                            and (snap is None or abs(correction) < abs(snap[0]))
                        ):
                            snap = (correction, target_feature)
            if snap is not None:
                down += snap[0] / pixels_per_point
            if guide is not None:
                guides.append(("y", guide[1]))

        self._alignment_guides = guides
        self.update()
        return right, down, guides

    def show_alignment_guides(self, guides: list[tuple[str, float]]) -> None:
        """Display guides calculated before a PDF refresh, if any."""
        self._alignment_guides = guides
        self.update()

    def source_size(self) -> QSize:
        return self._source.size() if self._source and not self._source.isNull() else QSize(0, 0)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self.resized.emit()  # a coordinator recomputes the SHARED scale, not this label alone

    def render_at_scale(self, scale: float) -> None:
        """Draw the source pixmap scaled by exactly ``scale`` — the
        same physical drawing at the same zoom the sibling pane is
        using, not independently fit to this label's own box."""
        if self._source is None or self._source.isNull() or scale <= 0:
            self.setPixmap(QPixmap())
            return
        target = QSize(
            max(1, round(self._source.width() * scale)),
            max(1, round(self._source.height() * scale)),
        )
        scaled = self._source.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self._content_sized:
            # QScrollArea preserves this child size as its scrollable
            # canvas. Do not make it fixed: programmatic layout can still
            # supply a tighter comparison viewport when required.
            self.resize(scaled.size())
        self.setPixmap(scaled)

    def _display_transform(self) -> tuple[float, float, float] | None:
        shown = self.pixmap()
        if self._source is None or shown is None or shown.isNull() or self._source.isNull():
            return None
        scale = shown.width() / self._source.width()
        return scale, (self.width() - shown.width()) / 2.0, (self.height() - shown.height()) / 2.0

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if not self._selected_text:
            return
        transform = self._display_transform()
        if transform is None:
            return
        scale, ox, oy = transform
        painter = QPainter(self)
        painter.setPen(QPen(QColor("#E53935"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for index, rect in self._text_regions:
            if index not in self._selected_text:
                continue
            painter.drawRect(QRectF(ox + rect.x() * scale, oy + rect.y() * scale, rect.width() * scale, rect.height() * scale))
        guide_pen = QPen(QColor("#1976D2"), 1, Qt.PenStyle.DashLine)
        painter.setPen(guide_pen)
        for axis, coordinate in self._alignment_guides:
            if axis == "x":
                painter.drawLine(round(ox + coordinate * scale), round(oy), round(ox + coordinate * scale), round(oy + self._source.height() * scale))
            else:
                painter.drawLine(round(ox), round(oy + coordinate * scale), round(ox + self._source.width() * scale), round(oy + coordinate * scale))
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            transform = self._display_transform()
            if transform is not None:
                scale, ox, oy = transform
                x, y = (event.position().x() - ox) / scale, (event.position().y() - oy) / scale
                hit: int | None = None
                for index, rect in reversed(self._text_regions):
                    if rect.contains(x, y):
                        hit = index
                        break
                additive = event.modifiers() & (
                    Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
                )
                if hit is not None:
                    if additive:
                        if hit in self._selected_text:
                            self._selected_text.remove(hit)
                        else:
                            self._selected_text.add(hit)
                    else:
                        self._selected_text = {hit}
                    self.setFocus()
                    self.text_selection_changed.emit(sorted(self._selected_text))
                    self.update()
                    event.accept()
                    return
                if self._selected_text:
                    self._selected_text.clear()
                    self.text_selection_changed.emit([])
                    self.update()
                event.accept()
                return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = event.position()
            self._pan_dragging = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._pan_start is not None and event.buttons() & Qt.MouseButton.MiddleButton:
            delta = event.position() - self._pan_start
            if delta.manhattanLength() >= 1:
                self._pan_dragging = True
                self.pan_requested.emit(delta.x(), delta.y())
                self._pan_start = event.position()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_start is not None:
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._pan_dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape and self._selected_text:
            self._selected_text.clear()
            self.text_selection_changed.emit([])
            self.update()
            event.accept()
            return
        if self._selected_text:
            step = 5.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            moves = {
                Qt.Key.Key_Left: (-step, 0.0), Qt.Key.Key_Right: (step, 0.0),
                Qt.Key.Key_Up: (0.0, -step), Qt.Key.Key_Down: (0.0, step),
            }
            delta = moves.get(event.key())
            if delta is not None:
                # Ordinary arrow movement is exact. Snap only on the explicit
                # Alt+arrow gesture shown in the selection guidance; otherwise
                # a nearby alignment guide can silently shorten a user's nudge.
                snap_to_text = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                self.text_nudged.emit(sorted(self._selected_text), *delta, snap_to_text)
                event.accept()
                return
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """Use the pointer wheel for direct, detail-oriented PDF editing."""
        if self._source is not None and not self._source.isNull() and event.angleDelta().y():
            self.zoom_requested.emit(
                1 if event.angleDelta().y() > 0 else -1,
                event.position().x(),
                event.position().y(),
            )
            event.accept()
            return
        super().wheelEvent(event)
