"""Screens 1 and 2 of the build plan's §06 UI flow, in one window:
import on the left, before/after preview on the right. Screen 3
(batch) is a CLI job (``spejl mirror plans/ --batch``), not a GUI one —
see the build plan's rationale for why the CLI, not the app, is what a
40-unit project actually runs.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from spejl.gui.imaging import load_preview
from spejl.gui.widgets import DropZone, ScaledImageLabel
from spejl.gui.worker import MirrorWorker
from spejl.models import Axis, Document, Route

_ACCENT = "#0A6E8A"

_DISABLED_TEXT = "#9AA5AB"

# Every rule lives in ONE stylesheet applied once, at the QMainWindow —
# deliberately never on an intermediate container in between. Qt's style
# sheet cascade stops propagating an ancestor's ID-selector rules (like
# #mirrorButton here) past any widget that has its own setStyleSheet()
# call, even an unrelated one — a container that styles itself directly
# silently blocks *every* selector-based rule meant for its descendants,
# with no error and no visual sign beyond "that widget just isn't there".
# Found exactly that way: the sidebar container's own background/border
# setStyleSheet() call was silently making the Mirror Plan button (its
# child) invisible, despite the button reporting correct geometry, text,
# and an enabled state — see #sidebar below for the fix, and
# widgets.py's DropZone for the same pattern applied to its own subtitle
# label rather than to an ancestor's stylesheet, for the same reason.
_STYLESHEET = f"""
QMainWindow {{ background: palette(window); }}
#sidebar {{ background: palette(base); border-right: 1px solid palette(mid); }}
QPushButton#mirrorButton {{
    background: {_ACCENT}; color: white; border: none;
    border-radius: 6px; padding: 10px 16px; font-weight: 600;
}}
QPushButton#mirrorButton:disabled {{ background: palette(mid); color: {_DISABLED_TEXT}; }}
QPushButton#mirrorButton:hover:!disabled {{ background: #085a72; }}
QPushButton#saveButton {{
    border: 1px solid {_ACCENT}; color: {_ACCENT}; border-radius: 6px;
    padding: 8px 14px; background: transparent; font-weight: 600;
}}
QPushButton#saveButton:disabled {{ border-color: palette(mid); color: {_DISABLED_TEXT}; }}
QPushButton#saveButton:hover:!disabled {{ background: rgba(10, 110, 138, 0.08); }}
#routeBadge {{ font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px; }}
#routeBadge[route="vector"] {{ background: rgba(10, 110, 138, 0.15); color: {_ACCENT}; }}
#routeBadge[route="raster"] {{ background: rgba(162, 76, 7, 0.15); color: #A24C07; }}
#previewCaption {{ font-weight: 600; padding: 4px 0; color: palette(mid); }}
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Spejl — mirror floor plans without mirroring the text")
        self.resize(1180, 720)
        self.setStyleSheet(_STYLESHEET)

        self._input_path: Path | None = None
        self._output_path: Path | None = None
        self._worker: MirrorWorker | None = None
        self._temp_dir = tempfile.TemporaryDirectory(prefix="spejl_gui_")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())
        root.addWidget(self._build_preview_area(), stretch=1)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")  # styled from _STYLESHEET, not a call here — see its comment
        sidebar.setFixedWidth(300)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("Spejl")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        subtitle = QLabel("Mirror floor plans without mirroring the text")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: palette(mid);")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self._drop_zone = DropZone()
        self._drop_zone.file_chosen.connect(self._on_file_chosen)
        layout.addWidget(self._drop_zone)

        self._route_badge = QLabel()
        self._route_badge.setObjectName("routeBadge")
        self._route_badge.hide()
        layout.addWidget(self._route_badge)

        layout.addWidget(self._build_axis_group())

        self._mirror_button = QPushButton("Mirror Plan")
        self._mirror_button.setObjectName("mirrorButton")
        self._mirror_button.setEnabled(False)
        self._mirror_button.clicked.connect(self._on_mirror_clicked)
        layout.addWidget(self._mirror_button)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate — OCR has no known duration up front
        self._progress.hide()
        layout.addWidget(self._progress)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: palette(mid); font-size: 12px;")
        layout.addWidget(self._status_label)

        flags_box = QGroupBox("Review")
        flags_layout = QVBoxLayout(flags_box)
        self._flags_list = QListWidget()
        self._flags_list.setAlternatingRowColors(True)
        flags_layout.addWidget(self._flags_list)
        layout.addWidget(flags_box, stretch=1)

        self._save_button = QPushButton("Save As…")
        self._save_button.setObjectName("saveButton")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._on_save_clicked)
        layout.addWidget(self._save_button)

        return sidebar

    def _build_axis_group(self) -> QGroupBox:
        box = QGroupBox("Mirror axis")
        layout = QVBoxLayout(box)
        self._axis_group = QButtonGroup(self)

        self._axis_buttons: dict[Axis, QRadioButton] = {}
        options = (
            (Axis.VERTICAL, "Vertical — left / right (most plans)"),
            (Axis.HORIZONTAL, "Horizontal — top / bottom"),
            (Axis.BOTH, "Both — 180°"),
        )
        for axis, label in options:
            btn = QRadioButton(label)
            self._axis_group.addButton(btn)
            layout.addWidget(btn)
            self._axis_buttons[axis] = btn
        self._axis_buttons[Axis.VERTICAL].setChecked(True)

        return box

    def _build_preview_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._source_view, source_pane = self._make_preview_pane("Source")
        self._mirrored_view, mirrored_pane = self._make_preview_pane("Mirrored")
        splitter.addWidget(source_pane)
        splitter.addWidget(mirrored_pane)
        splitter.setSizes([1, 1])
        layout.addWidget(splitter)
        return container

    @staticmethod
    def _make_preview_pane(caption: str) -> tuple[ScaledImageLabel, QWidget]:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(16, 12, 16, 16)
        label = QLabel(caption)
        label.setObjectName("previewCaption")
        layout.addWidget(label)
        image_view = ScaledImageLabel()
        image_view.setStyleSheet(
            "background: palette(alternate-base); border: 1px solid palette(mid); border-radius: 6px;"
        )
        layout.addWidget(image_view, stretch=1)
        return image_view, pane

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------

    def _on_file_chosen(self, path: Path) -> None:
        self._input_path = path
        self._output_path = None
        self._drop_zone.set_file(path)
        self._mirror_button.setEnabled(True)
        self._save_button.setEnabled(False)
        self._flags_list.clear()
        self._status_label.setText("")
        self._mirrored_view.set_pixmap_source(None)

        from spejl.router import sniff_route

        try:
            route = sniff_route(path)
        except ValueError:
            route = None
        self._show_route_badge(route)

        pixmap = load_preview(path)
        self._source_view.set_pixmap_source(pixmap)
        if pixmap is None:
            self._status_label.setText("Could not preview this file — mirroring may still work.")

    def _show_route_badge(self, route: Route | None) -> None:
        if route is Route.VECTOR:
            self._route_badge.setText("Vector PDF — lossless")
            self._route_badge.setProperty("route", "vector")
        elif route is Route.RASTER:
            self._route_badge.setText("Raster image — reconstructed")
            self._route_badge.setProperty("route", "raster")
        else:
            self._route_badge.hide()
            return
        self._route_badge.style().unpolish(self._route_badge)
        self._route_badge.style().polish(self._route_badge)
        self._route_badge.show()

    def _on_mirror_clicked(self) -> None:
        if self._input_path is None:
            return
        axis = next(a for a, btn in self._axis_buttons.items() if btn.isChecked())
        out_name = f"{self._input_path.stem}_mirrored{self._input_path.suffix}"
        output_path = Path(self._temp_dir.name) / out_name

        self._mirror_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._progress.show()
        self._flags_list.clear()
        self._status_label.setText("Mirroring… (OCR can take a few seconds)")

        self._worker = MirrorWorker(self._input_path, output_path, axis)
        self._worker.succeeded.connect(self._on_mirror_succeeded)
        self._worker.failed.connect(self._on_mirror_failed)
        self._worker.start()

    def _on_mirror_succeeded(self, document: Document, route: Route) -> None:
        self._progress.hide()
        self._mirror_button.setEnabled(True)
        self._output_path = document.output
        self._save_button.setEnabled(True)

        pixmap = load_preview(document.output)
        self._mirrored_view.set_pixmap_source(pixmap)

        total_runs = sum(p.text_runs_mirrored for p in document.pages)
        total_flags = sum(len(p.flags) for p in document.pages)
        self._status_label.setText(
            f"{total_runs} text run(s) mirrored"
            + (f" · {total_flags} flag(s) below" if total_flags else " · no flags")
        )
        self._populate_flags(document)

    def _on_mirror_failed(self, message: str) -> None:
        self._progress.hide()
        self._mirror_button.setEnabled(True)
        self._status_label.setText("")
        QMessageBox.critical(self, "Couldn't mirror this plan", message)

    def _populate_flags(self, document: Document) -> None:
        self._flags_list.clear()
        severity_marks = {"warn": "⚠", "info": "ℹ", "error": "✖"}
        for page in document.pages:
            for flag in page.flags:
                mark = severity_marks.get(flag.severity, "•")
                item = QListWidgetItem(f"{mark}  {flag.message}")
                if flag.severity == "warn":
                    item.setForeground(Qt.GlobalColor.darkYellow)
                elif flag.severity == "error":
                    item.setForeground(Qt.GlobalColor.red)
                self._flags_list.addItem(item)
        if self._flags_list.count() == 0:
            self._flags_list.addItem("No issues to review.")

    def _on_save_clicked(self) -> None:
        if self._output_path is None or self._input_path is None:
            return
        suggested = str(self._input_path.with_name(self._output_path.name))
        path_str, _filter = QFileDialog.getSaveFileName(
            self, "Save mirrored plan", suggested,
            f"{self._output_path.suffix.upper().lstrip('.')} files (*{self._output_path.suffix})",
        )
        if not path_str:
            return
        dest = Path(path_str)
        shutil.copyfile(self._output_path, dest)

        sidecar_src = self._output_path.with_suffix(self._output_path.suffix + ".spejl.json")
        if sidecar_src.exists():
            shutil.copyfile(sidecar_src, dest.with_suffix(dest.suffix + ".spejl.json"))

        self._status_label.setText(f"Saved to {dest.name}")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        self._temp_dir.cleanup()
        super().closeEvent(event)
