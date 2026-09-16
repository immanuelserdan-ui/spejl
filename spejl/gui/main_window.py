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
        # setSizes([1, 1]) only proposes a starting split — Qt does not
        # guarantee the two sides end up equal width from it (nor after
        # any later resize or a user drag of the handle), which is
        # exactly what let the Source and Mirrored panes end up
        # different widths and, since each independently fit its own
        # pixmap to its own box, show the identical drawing at two
        # visibly different zoom levels. Equal stretch factors are the
        # part of this that actually holds under resize; the shared-
        # scale coordinator below (_sync_preview_scale) is what holds
        # even if the user drags the handle to something unequal anyway.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        self._source_view.resized.connect(self._sync_preview_scale)
        self._mirrored_view.resized.connect(self._sync_preview_scale)

        return container

    def _sync_preview_scale(self) -> None:
        """Render Source and Mirrored at ONE shared scale, so the same
        physical drawing is always the same size in both panes —
        whichever pane/image pairing is the tighter fit sets the scale
        for both, rather than each independently maximising itself into
        whatever space it happens to have. See ScaledImageLabel's
        docstring for why "each fits its own box" broke this."""
        src_size = self._source_view.source_size()
        mir_size = self._mirrored_view.source_size()
        if src_size.isEmpty() and mir_size.isEmpty():
            return

        def fit_scale(img_size, box_size) -> float | None:
            if img_size.isEmpty() or box_size.width() <= 0 or box_size.height() <= 0:
                return None
            return min(box_size.width() / img_size.width(), box_size.height() / img_size.height())

        candidates = [
            s
            for s in (
                fit_scale(src_size, self._source_view.size()),
                fit_scale(mir_size, self._mirrored_view.size()),
            )
            if s is not None
        ]
        if not candidates:
            return
        scale = min(candidates)  # the tighter of the two pane/image pairings

        self._source_view.render_at_scale(scale)
        self._mirrored_view.render_at_scale(scale)

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

    def _job_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _on_file_chosen(self, path: Path) -> None:
        self._input_path = path
        self._output_path = None
        self._drop_zone.set_file(path)
        # Not re-enabled while a job is still in flight: the worker
        # already running holds its OWN captured input/output paths
        # (MirrorWorker.__init__ copies them), so swapping the file here
        # is harmless to it — but enabling this button would let a click
        # start a SECOND MirrorWorker and overwrite self._worker with it
        # while the first is still running. Nothing then holds a Python
        # reference to that first worker any more, even though its
        # background thread keeps running — a silent resource leak at
        # best, a use-after-free crash at worst if PySide6 garbage-
        # collects the orphaned QThread wrapper out from under its own
        # still-executing C++ thread. _on_mirror_succeeded re-enables it
        # once the in-flight job actually finishes.
        self._mirror_button.setEnabled(not self._job_running())
        self._save_button.setEnabled(False)
        self._flags_list.clear()
        if self._job_running():
            self._status_label.setText("Mirroring the previous file — this one will be ready to mirror once it finishes.")
        else:
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
        self._sync_preview_scale()
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
        if self._job_running():
            # Defence in depth alongside the button-disable above: the
            # button being disabled during a run is what's SUPPOSED to
            # make this unreachable, but nothing here costs anything to
            # also refuse outright rather than trust that one piece of
            # UI state never gets out of sync with reality — see
            # _on_file_chosen's own comment for exactly the scenario
            # (a second worker silently orphaning the first, still-
            # running one) this and that guard together close off.
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
        # The worker's OWN input path, bound at connect time — not
        # read from self._input_path when the signal fires, since by
        # then the user may have loaded a different file (see
        # _on_file_chosen). Lets the handler tell "my job finished"
        # apart from "A job finished, possibly someone else's".
        job_input = self._input_path
        self._worker.succeeded.connect(
            lambda document, route, job_input=job_input: self._on_mirror_succeeded(document, route, job_input)
        )
        self._worker.failed.connect(
            lambda message, job_input=job_input: self._on_mirror_failed(message, job_input)
        )
        self._worker.start()

    def _on_mirror_succeeded(self, document: Document, route: Route, job_input: Path) -> None:
        self._progress.hide()
        if job_input != self._input_path:
            # This job's own result is for a file the user has since
            # navigated away from (see _on_file_chosen) — applying it
            # now would silently replace whatever the CURRENT file's
            # own state is with a stale result the user never asked to
            # see. _on_file_chosen already re-enabled the mirror button
            # for the current file once this (the job it was waiting
            # on) finishes; nothing else here is still relevant.
            self._mirror_button.setEnabled(not self._job_running())
            return
        self._mirror_button.setEnabled(True)
        self._output_path = document.output
        self._save_button.setEnabled(True)

        pixmap = load_preview(document.output)
        self._mirrored_view.set_pixmap_source(pixmap)
        self._sync_preview_scale()

        total_runs = sum(p.text_runs_mirrored for p in document.pages)
        total_flags = sum(len(p.flags) for p in document.pages)
        self._status_label.setText(
            f"{total_runs} text run(s) mirrored"
            + (f" · {total_flags} flag(s) below" if total_flags else " · no flags")
        )
        self._populate_flags(document)

    def _on_mirror_failed(self, message: str, job_input: Path) -> None:
        if job_input != self._input_path:
            self._progress.hide()
            self._mirror_button.setEnabled(not self._job_running())
            return
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
