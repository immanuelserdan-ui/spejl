"""Selection and quick-preview dialogs for uploaded PDF plans."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from spejl.gui.batch_model import BatchEntry
from spejl.gui.imaging import load_preview, pdf_page_count


class MirrorSelectionDialog(QDialog):
    def __init__(self, entries: list[BatchEntry], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose plans to mirror")
        self.resize(540, 440)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select the uploaded PDFs to mirror:"))
        self.list = QListWidget()
        for entry in entries:
            item = QListWidgetItem(entry.display_name or entry.source.name)
            item.setData(Qt.ItemDataRole.UserRole, str(entry.source))
            item.setToolTip(str(entry.source))
            item.setCheckState(Qt.CheckState.Checked)
            self.list.addItem(item)
        layout.addWidget(self.list)
        controls = QHBoxLayout()
        for label, state in (("Select all", Qt.CheckState.Checked), ("Select none", Qt.CheckState.Unchecked)):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, state=state: self._set_all(state))
            controls.addWidget(button)
        controls.addStretch()
        layout.addLayout(controls)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Mirror selected")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all(self, state: Qt.CheckState) -> None:
        for index in range(self.list.count()):
            self.list.item(index).setCheckState(state)

    def selected_paths(self) -> list[Path]:
        return [Path(self.list.item(index).data(Qt.ItemDataRole.UserRole))
                for index in range(self.list.count())
                if self.list.item(index).checkState() == Qt.CheckState.Checked]


class JpegExportDialog(QDialog):
    def __init__(self, entries: list[BatchEntry], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose PDFs to export as JPEG")
        self.resize(780, 570)
        self.setStyleSheet("""
            QDialog { background: #07111F; color: #EAF7FB; }
            QLabel { color: #D7EEF5; }
            QLabel#columnHeading { color: #37D5FF; font-weight: 700; padding: 4px; }
            QLabel#planName {
                color: #EAF7FB; background: #0D2638; border: 1px solid #1F4D61;
                border-radius: 5px; padding: 8px; font-weight: 600;
            }
            QLabel#displayName { color: #8CB3C0; padding: 4px 8px; }
            QLabel#mirroredFileName { color: #EAF7FB; font-weight: 600; padding: 2px 4px; }
            QCheckBox { color: #D7EEF5; spacing: 6px; }
            QScrollArea, QWidget#exportBody { background: #07111F; }
            QWidget#exportCell { background: #0A1928; border: 1px solid #1F4D61; border-radius: 6px; }
            QPushButton { color: #D7EEF5; background: #0D2638; border: 1px solid #2B6479; border-radius: 5px; padding: 6px 10px; }
            QPushButton:hover { background: #14364C; }
        """)
        self._choices: list[tuple[Path, str, QCheckBox]] = []
        self._filename_labels: list[QLabel] = []
        layout = QVBoxLayout(self)
        intro = QLabel("Select source and mirrored PDFs to export. Every page becomes a JPEG.")
        layout.addWidget(intro)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body.setObjectName("exportBody")
        grid = QGridLayout(body)
        for column, title in enumerate(("File name", "Source", "Mirrored")):
            heading = QLabel(title)
            heading.setObjectName("columnHeading")
            grid.addWidget(heading, 0, column)
        for row, entry in enumerate(entries, start=1):
            name_cell = QWidget()
            name_layout = QVBoxLayout(name_cell)
            name_layout.setContentsMargins(2, 2, 2, 2)
            filename = QLabel(entry.source.name)
            filename.setObjectName("planName")
            filename.setWordWrap(True)
            filename.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            filename.setToolTip(str(entry.source))
            self._filename_labels.append(filename)
            name_layout.addWidget(filename)
            if entry.display_name and entry.display_name != entry.source.name:
                display_name = QLabel(f"Display name: {entry.display_name}")
                display_name.setObjectName("displayName")
                display_name.setWordWrap(True)
                name_layout.addWidget(display_name)
            grid.addWidget(name_cell, row, 0)
            for col, kind, path in ((1, "source", entry.source), (2, "mirrored", entry.output)):
                cell = QWidget()
                cell.setObjectName("exportCell")
                cell_layout = QVBoxLayout(cell)
                cell_layout.setContentsMargins(4, 4, 4, 4)
                if path is not None and path.is_file():
                    if kind == "mirrored":
                        mirrored_name = QLabel(path.name)
                        mirrored_name.setObjectName("mirroredFileName")
                        mirrored_name.setWordWrap(True)
                        mirrored_name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                        mirrored_name.setToolTip(str(path))
                        cell_layout.addWidget(mirrored_name)
                    try:
                        preview = load_preview(path, dpi=54)
                    except Exception:
                        preview = None
                    thumb = QLabel()
                    thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    thumb.setMinimumSize(145, 120)
                    thumb.setText("Preview unavailable")
                    if preview is not None:
                        thumb.setPixmap(preview.scaled(145, 120, Qt.AspectRatioMode.KeepAspectRatio,
                                                       Qt.TransformationMode.SmoothTransformation))
                    cell_layout.addWidget(thumb)
                    checkbox = QCheckBox("Export " + kind)
                    checkbox.setChecked(True)
                    cell_layout.addWidget(checkbox)
                    self._choices.append((entry.source, kind, checkbox))
                else:
                    cell_layout.addWidget(QLabel("Not mirrored yet"))
                grid.addWidget(cell, row, col)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        scroll.setWidget(body)
        layout.addWidget(scroll)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Choose export folder")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_items(self) -> list[tuple[Path, str]]:
        return [(path, kind) for path, kind, checkbox in self._choices if checkbox.isChecked()]


class PdfPreviewDialog(QDialog):
    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = path
        self.index = 0
        self.count = pdf_page_count(path)
        self.setWindowTitle(f"Preview — {path.name}")
        self.resize(850, 750)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(self.image)
        layout.addWidget(scroll)
        controls = QHBoxLayout()
        self.previous = QPushButton("‹ Previous")
        self.next = QPushButton("Next ›")
        self.counter = QLabel()
        self.previous.clicked.connect(lambda: self._show_page(self.index - 1))
        self.next.clicked.connect(lambda: self._show_page(self.index + 1))
        controls.addWidget(self.previous)
        controls.addWidget(self.counter)
        controls.addWidget(self.next)
        controls.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        controls.addWidget(close)
        layout.addLayout(controls)
        self._show_page(0)

    def _show_page(self, index: int) -> None:
        if not 0 <= index < self.count:
            return
        self.index = index
        preview = load_preview(self.path, dpi=130, page_index=index)
        if preview is None:
            self.image.setText("Preview unavailable")
        else:
            self.image.setPixmap(preview)
        self.counter.setText(f"Page {index + 1} of {self.count}")
        self.previous.setEnabled(index > 0)
        self.next.setEnabled(index < self.count - 1)
