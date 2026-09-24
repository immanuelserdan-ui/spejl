"""Multi-plan import, queued mirroring, and paired before/after review."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QRectF, QThread, QTimer, QUrl, Qt
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
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
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from spejl.gui.imaging import load_preview
from spejl.gui.batch_model import BatchEntry, mirrored_filename
from spejl.gui.document_session import DocumentSession
from spejl.gui.job_controller import MirrorJobController
from spejl.gui.review_model import summarize_document, summarize_verification
from spejl.gui.update_checker import UpdateCheckWorker, UpdateInfo
from spejl.gui.widgets import DropZone, ScaledImageLabel
from spejl.gui.worker import MirrorWorker, VerifyWorker
from spejl.models import Axis, Document, Route
from spejl.qa.verify import VerifyReport

_ACCENT = "#37D5FF"

_DISABLED_TEXT = "#6F8792"

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
QMainWindow {{ background: #07111F; color: #EAF7FB; }}
#windowHeader {{ background: #081725; border-bottom: 1px solid #1F4D61; }}
#windowTitle {{ color: #D7EEF5; font-size: 12px; font-weight: 600; }}
QPushButton#windowControl, QPushButton#closeControl {{
    min-width: 38px; max-width: 38px; min-height: 28px; max-height: 28px;
    border: none; border-radius: 4px; background: transparent; color: #D7EEF5;
    font-size: 16px; font-weight: 500;
}}
QPushButton#windowControl:hover {{ background: #14364C; }}
QPushButton#closeControl:hover {{ background: #C74343; color: white; }}
#sidebar {{ background: #0A1928; border-right: 1px solid #1F4D61; }}
QLabel {{ color: #EAF7FB; }}
QGroupBox {{ border: 1px solid #1F4D61; border-radius: 8px; margin-top: 10px; padding: 10px; color: #BFEFFF; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QRadioButton {{ color: #D7EEF5; spacing: 7px; }}
QRadioButton::indicator {{ width: 14px; height: 14px; border: 1px solid #47869B; border-radius: 7px; background: #07111F; }}
QRadioButton::indicator:checked {{ border: 4px solid {_ACCENT}; }}
QListWidget {{ background: #07111F; border: 1px solid #1F4D61; border-radius: 6px; color: #D7EEF5; }}
QListWidget::item:alternate {{ background: #0D2638; }}
QProgressBar {{ border: 1px solid #1F4D61; border-radius: 5px; background: #07111F; text-align: center; color: #EAF7FB; }}
QProgressBar::chunk {{ background: {_ACCENT}; border-radius: 4px; }}
QPushButton#cancelButton {{ border: 1px solid #A66037; color: #FFC39D; border-radius: 6px; padding: 7px 12px; background: transparent; }}
QPushButton#cancelButton:hover:!disabled {{ background: rgba(166, 96, 55, 0.18); }}
QPushButton#mirrorButton {{
    background: {_ACCENT}; color: #06101D; border: none;
    border-radius: 6px; padding: 10px 16px; font-weight: 600;
}}
QPushButton#mirrorButton:disabled {{ background: palette(mid); color: {_DISABLED_TEXT}; }}
QPushButton#mirrorButton:hover:!disabled {{ background: #83E7FF; }}
QPushButton#saveButton {{
    border: 1px solid {_ACCENT}; color: {_ACCENT}; border-radius: 6px;
    padding: 8px 14px; background: transparent; font-weight: 600;
}}
QPushButton#saveButton:disabled {{ border-color: palette(mid); color: {_DISABLED_TEXT}; }}
QPushButton#saveButton:hover:!disabled {{ background: rgba(10, 110, 138, 0.08); }}
QPushButton#verifyButton {{
    border: 1px solid #2B6479; color: #D7EEF5; border-radius: 6px;
    padding: 8px 14px; background: #0D2638;
}}
QPushButton#verifyButton:disabled {{ border-color: palette(mid); color: {_DISABLED_TEXT}; }}
QPushButton#verifyButton:hover:!disabled {{ background: #14364C; }}
#routeBadge {{ font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px; }}
#routeBadge[route="vector"] {{ background: rgba(10, 110, 138, 0.15); color: {_ACCENT}; }}
#routeBadge[route="raster"] {{ background: rgba(162, 76, 7, 0.15); color: #A24C07; }}
#previewCaption {{ font-weight: 600; padding: 4px 0; color: #9EE5F7; }}
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Spejl — mirror floor plans without mirroring the text")
        self.resize(1180, 720)
        self.setStyleSheet(_STYLESHEET)

        self._input_path: Path | None = None
        self._output_path: Path | None = None
        self._output_axis: Axis | None = None
        self._output_route: Route | None = None
        self._worker: MirrorWorker | None = None
        self._job_controller = MirrorJobController(self)
        self._job_controller.stage_changed.connect(self._on_job_stage_changed)
        self._job_controller.cancel_available_changed.connect(self._on_cancel_available_changed)
        self._job_controller.cancelled.connect(self._on_mirror_cancelled)
        self._verify_worker: VerifyWorker | None = None
        self._diff_overlay_path: Path | None = None
        self._selected_text_indexes: set[int] = set()
        self._mirrored_zoom = 1.0
        self._preview_dpi = 150
        self._syncing_viewport = False
        self._temp_dir = tempfile.TemporaryDirectory(prefix="spejl_gui_")
        self._document_session = DocumentSession()
        self._update_thread: QThread | None = None
        self._update_worker: UpdateCheckWorker | None = None
        self._batch_entries: dict[Path, BatchEntry] = {}
        self._batch_order: list[Path] = []
        self._batch_queue: list[Path] = []
        self._batch_axis: Axis | None = None
        self._active_batch_path: Path | None = None
        self._selecting_batch_item = False
        self._clear_pending = False

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_window_header())

        content = QWidget()
        root = QHBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())
        root.addWidget(self._build_preview_area(), stretch=1)
        outer.addWidget(content, stretch=1)
        self._install_shortcuts()
        QTimer.singleShot(1500, self._check_for_updates)

    def _install_shortcuts(self) -> None:
        """Discoverable desktop conventions; each delegates to existing UI actions."""
        QShortcut(QKeySequence.StandardKey.Open, self, activated=self._drop_zone.browse)
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self._on_save_clicked)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._on_mirror_clicked)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self._reset_mirrored_view)
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)

    def _check_for_updates(self) -> None:
        if self._update_thread is not None:
            return
        self._update_thread = QThread(self)
        self._update_worker = UpdateCheckWorker()
        self._update_worker.moveToThread(self._update_thread)
        self._update_thread.started.connect(self._update_worker.run)
        self._update_worker.finished.connect(self._on_update_result)
        self._update_worker.finished.connect(self._update_thread.quit)
        self._update_thread.finished.connect(self._finish_update_check)
        self._update_thread.start()

    def _on_update_result(self, info: UpdateInfo | None) -> None:
        if info is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Spejl update available")
        box.setText(f"Spejl {info.version} is ready to install.")
        box.setInformativeText(
            "Download the installer to update Spejl. Your floor-plan files are not affected."
        )
        open_button = box.addButton("Open download page", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is open_button:
            target = info.download_url or info.release_url
            QDesktopServices.openUrl(QUrl(target))

    def _finish_update_check(self) -> None:
        if self._update_thread is not None:
            self._update_thread.deleteLater()
        if self._update_worker is not None:
            self._update_worker.deleteLater()
        self._update_thread = None
        self._update_worker = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_window_header(self) -> QWidget:
        """Provide window controls when native chrome is hidden full-screen."""
        header = QWidget()
        header.setObjectName("windowHeader")
        header.setFixedHeight(38)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(14, 4, 6, 4)
        layout.setSpacing(4)

        title = QLabel("Spejl — mirror floor plans without mirroring the text")
        title.setObjectName("windowTitle")
        layout.addWidget(title)
        layout.addStretch(1)

        minimize = QPushButton("−")
        minimize.setObjectName("windowControl")
        minimize.setToolTip("Minimize")
        minimize.clicked.connect(self.showMinimized)
        layout.addWidget(minimize)

        self._maximize_button = QPushButton("□")
        self._maximize_button.setObjectName("windowControl")
        self._maximize_button.setToolTip("Restore from full screen")
        self._maximize_button.clicked.connect(self._toggle_fullscreen)
        layout.addWidget(self._maximize_button)

        close = QPushButton("×")
        close.setObjectName("closeControl")
        close.setToolTip("Close")
        close.clicked.connect(self.close)
        layout.addWidget(close)
        return header

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
            self._maximize_button.setText("⛶")
            self._maximize_button.setToolTip("Enter full screen")
        else:
            self.showFullScreen()
            self._maximize_button.setText("□")
            self._maximize_button.setToolTip("Restore from full screen")

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
        subtitle.setStyleSheet("color: #8CB3C0;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self._drop_zone = DropZone()
        self._drop_zone.files_chosen.connect(self._on_files_chosen)
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

        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setObjectName("cancelButton")
        self._cancel_button.setToolTip("Finish the current protected operation, then discard its temporary result")
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        self._cancel_button.hide()
        layout.addWidget(self._cancel_button)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #8CB3C0; font-size: 12px;")
        layout.addWidget(self._status_label)

        uploaded_box = QGroupBox("Uploaded")
        uploaded_layout = QVBoxLayout(uploaded_box)
        self._uploaded_list = QListWidget()
        self._uploaded_list.setAlternatingRowColors(True)
        self._uploaded_list.currentItemChanged.connect(self._on_uploaded_item_selected)
        uploaded_layout.addWidget(self._uploaded_list)
        layout.addWidget(uploaded_box, stretch=1)

        mirrored_box = QGroupBox("Mirrored")
        mirrored_layout = QVBoxLayout(mirrored_box)
        self._mirrored_list = QListWidget()
        self._mirrored_list.setAlternatingRowColors(True)
        self._mirrored_list.currentItemChanged.connect(self._on_mirrored_item_selected)
        mirrored_layout.addWidget(self._mirrored_list)
        layout.addWidget(mirrored_box, stretch=1)

        flags_box = QGroupBox("Checks")
        flags_layout = QVBoxLayout(flags_box)
        self._review_summary = QLabel("Choose a plan to begin")
        self._review_summary.setStyleSheet("color: #9EE5F7; font-size: 12px; font-weight: 600;")
        flags_layout.addWidget(self._review_summary)
        self._flags_list = QListWidget()
        self._flags_list.setAlternatingRowColors(True)
        self._flags_list.itemClicked.connect(self._on_review_item_clicked)
        flags_layout.addWidget(self._flags_list)
        self._flags_list.setMaximumHeight(85)
        layout.addWidget(flags_box)
        # The review/verification panel was removed from the primary workflow
        # so the batch lists and Save As action remain easy to reach. Keep the
        # widgets alive because their state is still used by verification code.
        flags_box.hide()

        self._save_button = QPushButton("Save As…")
        self._save_button.setObjectName("saveButton")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._on_save_clicked)
        layout.addWidget(self._save_button)

        verify_row = QHBoxLayout()
        self._verify_button = QPushButton("Verify")
        self._verify_button.setObjectName("verifyButton")
        self._verify_button.setEnabled(False)
        self._verify_button.setToolTip(
            "Compare geometry outside text exclusions and cross-check "
            "output wording using OCR. Uncertain readings require review."
        )
        self._verify_button.clicked.connect(self._on_verify_clicked)
        verify_row.addWidget(self._verify_button)

        self._view_diff_button = QPushButton("View Diff")
        self._view_diff_button.setObjectName("verifyButton")
        self._view_diff_button.setEnabled(False)
        self._view_diff_button.setToolTip(
            "Open the verification image: source geometry in black, every "
            "pixel that differs from the output highlighted in red."
        )
        self._view_diff_button.clicked.connect(self._on_view_diff_clicked)
        verify_row.addWidget(self._view_diff_button)
        layout.addLayout(verify_row)
        self._verify_button.hide()
        self._view_diff_button.hide()

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

        page_bar = QWidget()
        # Keep the navigation row compact.  Without an explicit fixed
        # vertical policy Qt may distribute the preview area's spare height
        # to this otherwise un-stretched widget, pushing the comparison panes
        # far below the toolbar on tall windows.
        page_bar.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        page_layout = QHBoxLayout(page_bar)
        page_layout.setContentsMargins(16, 8, 16, 0)
        page_layout.addStretch(1)
        self._clear_button = QPushButton("Clear")
        self._clear_button.setObjectName("verifyButton")
        self._clear_button.setToolTip("Clear loaded plans and temporary work, returning Spejl to its fresh-open state")
        self._clear_button.clicked.connect(self._on_clear_clicked)
        page_layout.addWidget(self._clear_button)
        self._resolve_button = QPushButton("Resolve")
        self._resolve_button.setObjectName("verifyButton")
        self._resolve_button.setToolTip(
            "Move every currently flagged text run 3 points to the right, then refresh the overlap findings"
        )
        self._resolve_button.setEnabled(False)
        self._resolve_button.clicked.connect(self._on_resolve_clicked)
        page_layout.addWidget(self._resolve_button)
        self._previous_page_button = QPushButton("‹ Previous")
        self._previous_page_button.setObjectName("verifyButton")
        self._previous_page_button.clicked.connect(lambda: self._change_preview_page(-1))
        self._next_page_button = QPushButton("Next ›")
        self._next_page_button.setObjectName("verifyButton")
        self._next_page_button.clicked.connect(lambda: self._change_preview_page(1))
        self._page_label = QLabel("Single page")
        self._page_label.setStyleSheet("color: #8CB3C0; font-size: 12px;")
        page_layout.addWidget(self._previous_page_button)
        page_layout.addWidget(self._page_label)
        page_layout.addWidget(self._next_page_button)
        layout.addWidget(page_bar)
        self._update_page_navigation()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._preview_splitter = splitter
        self._source_view, source_pane, self._source_scroll, self._source_filename_label = self._make_preview_pane(
            "Source", scrollable=True
        )
        self._mirrored_view, mirrored_pane, self._mirrored_scroll, self._mirrored_filename_label = self._make_preview_pane(
            "Mirrored", zoomable=True, scrollable=True
        )
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
        splitter.setChildrenCollapsible(False)
        source_pane.setMinimumWidth(0)
        mirrored_pane.setMinimumWidth(0)
        # The panes are the content area and must receive all remaining
        # vertical space after the compact navigation row.
        layout.addWidget(splitter, stretch=1)

        self._mirrored_view.text_selection_changed.connect(self._on_preview_text_selection_changed)
        self._mirrored_view.text_nudged.connect(self._on_preview_text_nudged)
        self._mirrored_view.zoom_requested.connect(self._change_mirrored_zoom)
        self._mirrored_view.pan_requested.connect(self._pan_mirrored_view)
        self._mirrored_scroll.horizontalScrollBar().valueChanged.connect(self._sync_source_viewport)
        self._mirrored_scroll.verticalScrollBar().valueChanged.connect(self._sync_source_viewport)
        # QScrollArea reports its preferred content width while Qt is
        # laying out the first frame. Resetting the splitter after that
        # layout prevents the editable pane from claiming almost all of
        # the window simply because it contains a zoomable image.
        QTimer.singleShot(0, self._equalize_preview_panes)

        return container

    def _equalize_preview_panes(self) -> None:
        """Keep Source and Mirrored as equally sized comparison panes."""
        available = self._preview_splitter.width()
        if available > 1:
            half = available // 2
            self._preview_splitter.setSizes([half, available - half])

    def _refresh_preview_layout(self) -> None:
        self._equalize_preview_panes()
        self._sync_preview_scale()
        self._sync_source_viewport()

    def _update_page_navigation(self) -> None:
        session = self._document_session
        plan_index = (
            self._batch_order.index(self._input_path)
            if self._input_path in self._batch_order else -1
        )
        plan_count = len(self._batch_order)
        if plan_count > 1 and plan_index >= 0:
            plan_label = f"Plan {plan_index + 1} of {plan_count}"
            self._page_label.setText(
                f"{plan_label} · {session.label.lower()}" if session.is_paginated else plan_label
            )
        else:
            self._page_label.setText(session.label)
        self._previous_page_button.setEnabled(session.page_index > 0 or plan_index > 0)
        self._next_page_button.setEnabled(
            session.page_index < session.page_count - 1
            or (plan_index >= 0 and plan_index < plan_count - 1)
        )
        self._update_resolve_button_state()

    def _update_resolve_button_state(self) -> None:
        entry = self._batch_entries.get(self._input_path) if self._input_path else None
        self._resolve_button.setEnabled(
            not self._clear_pending
            and entry is not None
            and entry.output is not None
            and entry.route is Route.VECTOR
            and entry.overlaps_checked
            and bool(entry.overlap_findings)
        )

    def _on_clear_clicked(self) -> None:
        """Clear the current work session without interrupting protected writes."""
        if self._clear_pending:
            return
        if self._job_running() or (
            self._verify_worker is not None and self._verify_worker.isRunning()
        ):
            self._clear_pending = True
            self._clear_button.setEnabled(False)
            self._resolve_button.setEnabled(False)
            if self._job_running():
                self._job_controller.request_cancel()
            self._status_label.setText("Clearing after the current operation finishes safely…")
            return
        self._clear_session()

    def _on_deferred_clear_ready(self) -> None:
        if not self._clear_pending:
            return
        if self._job_running() or (
            self._verify_worker is not None and self._verify_worker.isRunning()
        ):
            return
        self._clear_pending = False
        self._clear_session()

    def _clear_session(self) -> None:
        """Return all loaded and generated plan state to its initial UI state."""
        self._batch_entries.clear()
        self._batch_order.clear()
        self._batch_queue.clear()
        self._batch_axis = None
        self._active_batch_path = None
        self._input_path = None
        self._output_path = None
        self._output_axis = None
        self._output_route = None
        self._diff_overlay_path = None
        self._selected_text_indexes.clear()
        self._document_session = DocumentSession()
        self._mirrored_zoom = 1.0
        self._preview_dpi = 150

        self._uploaded_list.clear()
        self._mirrored_list.clear()
        self._flags_list.clear()
        self._review_summary.setText("Choose a plan to begin")
        self._drop_zone.clear()
        self._route_badge.hide()
        self._source_filename_label.setText("No file selected")
        self._source_filename_label.setToolTip("")
        self._mirrored_filename_label.setText("No file selected")
        self._mirrored_filename_label.setToolTip("")
        self._source_view.set_pixmap_source(None)
        self._mirrored_view.set_pixmap_source(None)
        self._mirrored_view.set_text_regions([])
        self._reset_view_button.hide()
        self._previous_page_button.setEnabled(False)
        self._next_page_button.setEnabled(False)
        self._page_label.setText("Single page")
        self._mirror_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._verify_button.setEnabled(False)
        self._view_diff_button.setEnabled(False)
        self._progress.hide()
        self._cancel_button.hide()
        self._cancel_button.setEnabled(False)
        self._clear_button.setEnabled(True)
        self._status_label.setStyleSheet("color: #8CB3C0; font-size: 12px;")
        self._status_label.setToolTip("")
        self._status_label.setText("")
        self._axis_buttons[Axis.VERTICAL].setChecked(True)
        self._sync_preview_scale()
        for scrollbar in (
            self._source_scroll.horizontalScrollBar(),
            self._source_scroll.verticalScrollBar(),
            self._mirrored_scroll.horizontalScrollBar(),
            self._mirrored_scroll.verticalScrollBar(),
        ):
            scrollbar.setValue(0)

        # A Clear action discards generated output and verification images in
        # the app's private temp directory; original source files are untouched.
        self._temp_dir.cleanup()
        self._temp_dir = tempfile.TemporaryDirectory(prefix="spejl_gui_")
        self._worker = None
        self._verify_worker = None
        self._job_controller.worker = None
        self._job_controller._cancel_requested = False
        self._update_resolve_button_state()

    def _change_preview_page(self, offset: int) -> None:
        if self._input_path is None:
            return
        if not self._document_session.move(offset):
            if self._input_path not in self._batch_order:
                return
            target_row = self._batch_order.index(self._input_path) + offset
            if 0 <= target_row < len(self._batch_order):
                self._select_batch_entry(self._batch_order[target_row])
            return
        entry = self._batch_entries.get(self._input_path)
        if entry is not None:
            entry.page_index = self._document_session.page_index
        self._source_view.set_pixmap_source(
            load_preview(self._input_path, dpi=self._preview_dpi,
                         page_index=self._document_session.page_index)
        )
        if self._output_path is not None and self._output_path.suffix.lower() == ".pdf":
            import pymupdf
            output_doc = pymupdf.open(str(self._output_path))
            try:
                output_page = min(self._document_session.page_index, output_doc.page_count - 1)
            finally:
                output_doc.close()
            self._mirrored_view.set_pixmap_source(
                load_preview(self._output_path, dpi=self._preview_dpi,
                             page_index=max(0, output_page))
            )
        self._refresh_mirrored_text_regions()
        self._update_page_navigation()
        self._sync_preview_scale()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        # Run after Qt has calculated the splitter's new available width.
        QTimer.singleShot(0, self._refresh_preview_layout)

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

        candidate_sizes = [
            (src_size, self._source_scroll.viewport().size()),
            (mir_size, self._mirrored_scroll.viewport().size()),
        ]
        # Honour a programmatically constrained image label (used by UI
        # tests and embedding hosts) without feeding an ordinary current
        # scroll-canvas size back into a live zoom calculation.
        for image_size, view in ((src_size, self._source_view), (mir_size, self._mirrored_view)):
            shown = view.pixmap()
            if shown is not None and not shown.isNull() and view.size() != shown.size():
                candidate_sizes.append((image_size, view.size()))
        candidates = [
            s
            for s in (fit_scale(image_size, box_size) for image_size, box_size in candidate_sizes)
            if s is not None
        ]
        if not candidates:
            return
        scale = min(candidates)  # the tighter of the two pane/image pairings

        shared_scale = scale * self._mirrored_zoom
        self._source_view.render_at_scale(shared_scale)
        self._mirrored_view.render_at_scale(shared_scale)

    def _preview_dpi_for_zoom(self) -> int:
        """Render vector PDFs with enough pixels for the current zoom level."""
        import math
        dpi = min(300, round(150 * math.sqrt(max(1.0, self._mirrored_zoom))))
        longest_page_points = 0.0
        import pymupdf
        for path in (self._input_path, self._output_path):
            if path is None or path.suffix.lower() != ".pdf":
                continue
            try:
                doc = pymupdf.open(str(path))
                try:
                    if doc.page_count:
                        longest_page_points = max(
                            longest_page_points,
                            max(doc[min(self._document_session.page_index, doc.page_count - 1)].rect.width,
                                doc[min(self._document_session.page_index, doc.page_count - 1)].rect.height),
                        )
                finally:
                    doc.close()
            except Exception:  # Preview resolution must never block opening a plan.
                continue
        if longest_page_points:
            # Avoid multiplying an already large sheet's base preview size
            # as the user zooms, which could consume excessive memory.
            dpi = min(dpi, max(150, int(6000 * 72 / longest_page_points)))
        return max(150, dpi)

    def _reload_previews_for_zoom(self) -> None:
        self._preview_dpi = self._preview_dpi_for_zoom()
        page_index = self._document_session.page_index
        if self._input_path is not None:
            self._source_view.set_pixmap_source(load_preview(
                self._input_path, dpi=self._preview_dpi, page_index=page_index
            ))
        if self._output_path is not None:
            output_page = page_index
            if self._output_path.suffix.lower() == ".pdf":
                import pymupdf
                doc = pymupdf.open(str(self._output_path))
                try:
                    output_page = min(page_index, max(doc.page_count - 1, 0))
                finally:
                    doc.close()
            self._mirrored_view.set_pixmap_source(load_preview(
                self._output_path, dpi=self._preview_dpi, page_index=output_page
            ))
        self._refresh_mirrored_text_regions(selected=self._selected_text_indexes)
        self._sync_preview_scale()

    def _make_preview_pane(
        self, caption: str, *, zoomable: bool = False, scrollable: bool = False
    ) -> tuple[ScaledImageLabel, QWidget, QScrollArea | None, QLabel]:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(16, 12, 16, 16)
        heading = QHBoxLayout()
        label = QLabel(caption)
        label.setObjectName("previewCaption")
        heading.addWidget(label)
        filename_label = QLabel("No file selected")
        filename_label.setStyleSheet("color: #8CB3C0; font-size: 11px;")
        filename_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        heading.addWidget(filename_label, stretch=1)
        if zoomable:
            self._reset_view_button = QPushButton("Reset View")
            self._reset_view_button.setToolTip("Return the mirrored plan to its default fit view")
            self._reset_view_button.clicked.connect(self._reset_mirrored_view)
            self._reset_view_button.hide()
            heading.addWidget(self._reset_view_button)
        layout.addLayout(heading)
        image_view = ScaledImageLabel()
        image_view.setStyleSheet(
            "background: #07111F; border: 1px solid #1F4D61; border-radius: 6px;"
        )
        if not scrollable:
            layout.addWidget(image_view, stretch=1)
            return image_view, pane, None, filename_label
        image_view.set_content_sized(True)
        scroll = QScrollArea()
        scroll.setWidget(image_view)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if not zoomable:
            # The source follows the editable view; visible source scrollbars
            # only add visual noise and invite an unsynchronised manual pan.
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border: 0px;")
        layout.addWidget(scroll, stretch=1)
        return image_view, pane, scroll, filename_label

    def _change_mirrored_zoom(
        self, direction: int, cursor_x: float | None = None, cursor_y: float | None = None
    ) -> None:
        """Change the shared comparison scale around the pointer location."""
        # Start at 150% so a plan that initially fits the panel gains a
        # real scrollable canvas on its first wheel step; 125% can still
        # fit entirely and cannot preserve an off-centre anchor.
        levels = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
        current = min(range(len(levels)), key=lambda i: abs(levels[i] - self._mirrored_zoom))
        target = max(0, min(len(levels) - 1, current + direction))
        if target == current:
            return
        anchor = None
        if cursor_x is not None and cursor_y is not None:
            content_size = self._mirrored_view.size()
            if content_size.width() and content_size.height():
                from PySide6.QtCore import QPoint

                viewport_point = self._mirrored_view.mapTo(
                    self._mirrored_scroll.viewport(), QPoint(round(cursor_x), round(cursor_y))
                )
                anchor = (
                    cursor_x / content_size.width(),
                    cursor_y / content_size.height(),
                    viewport_point.x(),
                    viewport_point.y(),
                )
        self._mirrored_zoom = levels[target]
        self._reset_view_button.show()
        self._reload_previews_for_zoom()
        if anchor is None:
            QTimer.singleShot(0, self._sync_source_viewport)
        else:
            # QScrollArea publishes new scroll ranges after the child resize
            # reaches the event loop. Re-applying after layout keeps the
            # exact floor-plan coordinate beneath the pointer.
            QTimer.singleShot(0, lambda anchor=anchor: self._restore_zoom_anchor(*anchor))
            QTimer.singleShot(35, lambda anchor=anchor: self._restore_zoom_anchor(*anchor))

    def _restore_zoom_anchor(
        self, fraction_x: float, fraction_y: float, viewport_x: int, viewport_y: int
    ) -> None:
        """Keep the plan coordinate under the mouse stationary after zoom."""
        horizontal = self._mirrored_scroll.horizontalScrollBar()
        vertical = self._mirrored_scroll.verticalScrollBar()
        horizontal.setValue(round(fraction_x * self._mirrored_view.width() - viewport_x))
        vertical.setValue(round(fraction_y * self._mirrored_view.height() - viewport_y))
        self._sync_source_viewport()

    def _reset_mirrored_view(self) -> None:
        """Restore the default fit scale and remove the temporary reset control."""
        self._mirrored_zoom = 1.0
        self._preview_dpi = 150
        self._reset_view_button.hide()
        self._reload_previews_for_zoom()
        QTimer.singleShot(0, lambda: self._mirrored_scroll.verticalScrollBar().setValue(0))
        QTimer.singleShot(0, lambda: self._mirrored_scroll.horizontalScrollBar().setValue(0))
        QTimer.singleShot(0, self._sync_source_viewport)

    def _pan_mirrored_view(self, dx: float, dy: float) -> None:
        """Pan the editable canvas and mirror the resulting viewport."""
        horizontal = self._mirrored_scroll.horizontalScrollBar()
        vertical = self._mirrored_scroll.verticalScrollBar()
        horizontal.setValue(horizontal.value() - round(dx))
        vertical.setValue(vertical.value() - round(dy))

    def _sync_source_viewport(self, *_unused) -> None:
        """Show the source area corresponding to the mirrored viewport.

        The viewports share a scale. Horizontal and vertical offsets are
        inverted when that axis is reflected, so the same physical room or
        dimension remains visible in both panels while editing.
        """
        if self._syncing_viewport or self._source_scroll is None or self._mirrored_scroll is None:
            return
        self._syncing_viewport = True
        try:
            axis = self._output_axis or Axis.VERTICAL
            for source_bar, mirrored_bar, reflected in (
                (
                    self._source_scroll.horizontalScrollBar(),
                    self._mirrored_scroll.horizontalScrollBar(),
                    axis in {Axis.VERTICAL, Axis.BOTH},
                ),
                (
                    self._source_scroll.verticalScrollBar(),
                    self._mirrored_scroll.verticalScrollBar(),
                    axis in {Axis.HORIZONTAL, Axis.BOTH},
                ),
            ):
                mirrored_maximum = mirrored_bar.maximum()
                fraction = mirrored_bar.value() / mirrored_maximum if mirrored_maximum else 0.0
                if reflected:
                    fraction = 1.0 - fraction
                source_bar.setValue(round(source_bar.maximum() * fraction))
        finally:
            self._syncing_viewport = False

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------

    def _job_running(self) -> bool:
        # Keep the direct worker check as a compatibility guard for existing
        # callers/tests that hold the legacy worker reference. In production
        # the controller owns that same worker; no engine behavior changes.
        return self._job_controller.is_running or (
            self._worker is not None and self._worker.isRunning()
        )

    def _on_files_chosen(self, paths: list[Path]) -> None:
        added: list[Path] = []
        for raw_path in paths:
            path = raw_path.resolve()
            if path in self._batch_entries:
                continue
            name, warning = mirrored_filename(path)
            self._batch_entries[path] = BatchEntry(path, name, warning)
            self._batch_order.append(path)
            added.append(path)
        if not added:
            self._mirror_button.setEnabled(bool(self._batch_order) and not self._job_running())
            self._status_label.setText("Those plans are already in the Uploaded list.")
            return
        self._drop_zone.set_files(self._batch_order)
        self._refresh_batch_lists()
        self._select_batch_entry(added[0])
        self._mirror_button.setEnabled(not self._job_running())
        warnings = sum(self._batch_entries[path].naming_warning for path in added)
        self._status_label.setText(
            f"Added {len(added)} plan(s)."
            + (f" {warnings} filename(s) have no R/S orientation field." if warnings else "")
        )

    def _on_file_chosen(self, path: Path) -> None:
        """Backward-compatible single-file entry point used by integrations/tests."""
        self._on_files_chosen([path])

    def _refresh_batch_lists(self) -> None:
        selected = self._input_path
        self._selecting_batch_item = True
        try:
            self._uploaded_list.clear()
            self._mirrored_list.clear()
            marks = {"queued": "○", "processing": "◌", "completed": "✓", "failed": "✖"}
            for path in self._batch_order:
                entry = self._batch_entries[path]
                uploaded = QListWidgetItem(path.name)
                uploaded.setData(Qt.ItemDataRole.UserRole, str(path))
                uploaded.setToolTip(str(path))
                self._uploaded_list.addItem(uploaded)
                mirrored = QListWidgetItem(f"{marks.get(entry.status, '○')}  {entry.mirrored_name}")
                mirrored.setData(Qt.ItemDataRole.UserRole, str(path))
                mirrored.setToolTip(entry.error or entry.mirrored_name)
                if entry.status == "completed":
                    mirrored.setForeground(Qt.GlobalColor.darkGreen)
                elif entry.status == "failed":
                    mirrored.setForeground(Qt.GlobalColor.red)
                elif entry.naming_warning:
                    mirrored.setForeground(Qt.GlobalColor.darkYellow)
                self._mirrored_list.addItem(mirrored)
                if path == selected:
                    self._uploaded_list.setCurrentRow(self._uploaded_list.count() - 1)
                    self._mirrored_list.setCurrentRow(self._mirrored_list.count() - 1)
        finally:
            self._selecting_batch_item = False

    def _path_from_list_item(self, item: QListWidgetItem | None) -> Path | None:
        return Path(item.data(Qt.ItemDataRole.UserRole)) if item is not None else None

    def _on_uploaded_item_selected(self, current: QListWidgetItem | None, _previous=None) -> None:
        if not self._selecting_batch_item and (path := self._path_from_list_item(current)):
            self._select_batch_entry(path)

    def _on_mirrored_item_selected(self, current: QListWidgetItem | None, _previous=None) -> None:
        if not self._selecting_batch_item and (path := self._path_from_list_item(current)):
            self._select_batch_entry(path)

    def _select_batch_entry(self, path: Path) -> None:
        if path not in self._batch_entries:
            return
        if self._input_path in self._batch_entries:
            self._batch_entries[self._input_path].page_index = self._document_session.page_index
        entry = self._batch_entries[path]
        self._mirrored_zoom = 1.0
        self._preview_dpi = 150
        self._selected_text_indexes.clear()
        # Keep both lists on the same plan whichever list initiated selection.
        # Block the selection callbacks while moving the counterpart row.
        self._selecting_batch_item = True
        try:
            row = self._batch_order.index(path)
            if self._uploaded_list.currentRow() != row:
                self._uploaded_list.setCurrentRow(row)
            if self._mirrored_list.currentRow() != row:
                self._mirrored_list.setCurrentRow(row)
        finally:
            self._selecting_batch_item = False
        self._input_path = path
        self._output_path = entry.output
        self._output_axis = entry.document.axis if entry.document is not None else None
        self._output_route = entry.route
        self._diff_overlay_path = entry.diff_overlay
        self._document_session.open(path)
        self._document_session.page_index = min(entry.page_index, self._document_session.page_count - 1)
        self._update_page_navigation()
        self._drop_zone.set_file(path)
        self._source_filename_label.setText(path.name)
        self._source_filename_label.setToolTip(str(path))
        self._mirrored_filename_label.setText(entry.mirrored_name if entry.output else "Waiting to be mirrored")
        self._mirrored_filename_label.setToolTip(entry.mirrored_name)
        self._mirror_button.setEnabled(bool(self._batch_order) and not self._job_running())
        completed = entry.status == "completed" and entry.output is not None
        self._save_button.setEnabled(completed)
        self._verify_button.setEnabled(completed and entry.route is Route.RASTER)
        self._view_diff_button.setEnabled(entry.diff_overlay is not None)
        self._mirrored_view.set_pixmap_source(
            load_preview(entry.output, dpi=self._preview_dpi,
                         page_index=self._document_session.page_index)
            if entry.output else None
        )
        self._mirrored_view.set_text_regions([])
        self._reset_mirrored_view()

        from spejl.router import sniff_route

        route = entry.route
        if route is None:
            try:
                route = sniff_route(path)
            except ValueError:
                route = None
        self._show_route_badge(route)

        pixmap = load_preview(path, dpi=self._preview_dpi,
                              page_index=self._document_session.page_index)
        self._source_view.set_pixmap_source(pixmap)
        if entry.output is not None:
            self._refresh_mirrored_text_regions()
        if entry.document is not None:
            self._populate_flags(entry.document)
        else:
            self._flags_list.clear()
            self._review_summary.setText(
                "Processing…" if entry.status == "processing" else
                "Processing failed" if entry.status == "failed" else "Ready to mirror"
            )
        if entry.error:
            self._status_label.setText(entry.error)
        elif entry.output is not None:
            self._set_status_with_overlap("Plan ready for review.")
        self._sync_preview_scale()
        if pixmap is None:
            self._status_label.setText("Could not preview this file — mirroring may still work.")

    def _show_route_badge(self, route: Route | None) -> None:
        if route is Route.VECTOR:
            self._route_badge.setText("Vector PDF — best accuracy; geometry and text stay editable")
            self._route_badge.setProperty("route", "vector")
        elif route is Route.RASTER:
            self._route_badge.setText("Raster input — review text and geometry before saving")
            self._route_badge.setProperty("route", "raster")
        else:
            self._route_badge.hide()
            return
        self._route_badge.style().unpolish(self._route_badge)
        self._route_badge.style().polish(self._route_badge)
        self._route_badge.show()

    def _on_mirror_clicked(self) -> None:
        if not self._batch_order:
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
        self._batch_axis = next(a for a, btn in self._axis_buttons.items() if btn.isChecked())
        self._batch_queue = list(self._batch_order)
        self._mirror_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._progress.show()
        self._cancel_button.show()
        self._cancel_button.setEnabled(False)
        for path in self._batch_queue:
            entry = self._batch_entries[path]
            entry.status, entry.error = "queued", None
        self._refresh_batch_lists()
        self._start_next_batch_job()

    def _start_next_batch_job(self) -> None:
        if self._clear_pending or not self._batch_order:
            return
        if not self._batch_queue:
            self._active_batch_path = None
            self._progress.hide()
            self._cancel_button.hide()
            self._mirror_button.setEnabled(bool(self._batch_order))
            completed = sum(entry.status == "completed" for entry in self._batch_entries.values())
            failed = sum(entry.status == "failed" for entry in self._batch_entries.values())
            self._status_label.setText(f"Batch complete: {completed} mirrored" + (f", {failed} failed" if failed else ""))
            return
        job_input = self._batch_queue.pop(0)
        entry = self._batch_entries[job_input]
        entry.status = "processing"
        self._active_batch_path = job_input
        index = self._batch_order.index(job_input)
        output_path = Path(self._temp_dir.name) / f"{index:04d}_{entry.mirrored_name}"
        self._status_label.setText(f"Mirroring {index + 1} of {len(self._batch_order)}: {job_input.name}")
        self._refresh_batch_lists()
        self._worker = self._job_controller.start(job_input, output_path, self._batch_axis or Axis.VERTICAL)
        self._worker.finished.connect(self._on_deferred_clear_ready)
        self._worker.succeeded.connect(
            lambda document, route, job_input=job_input: self._on_mirror_succeeded(document, route, job_input)
        )
        self._worker.failed.connect(
            lambda message, job_input=job_input: self._on_mirror_failed(message, job_input)
        )
        # Connect result handlers before starting; vector PDFs can finish
        # quickly enough that starting inside the controller loses the signal.
        self._worker.start()

    def _on_mirror_succeeded(self, document: Document, route: Route, job_input: Path) -> None:
        if self._job_controller.cancel_requested:
            return
        entry = self._batch_entries[job_input]
        entry.status = "completed"
        entry.output = document.output
        entry.document = document
        entry.route = route
        self._refresh_batch_lists()
        if job_input == self._input_path:
            self._output_path = document.output
            self._output_axis = document.axis
            self._output_route = route
            self._save_button.setEnabled(True)
            self._mirrored_filename_label.setText(entry.mirrored_name)
            self._reset_mirrored_view()
        # Verify re-detects text on the SOURCE to know what to exclude
        # (see qa/verify.py) — meaningful only for the raster route,
        # where the drawing is reconstructed from pixels and geometry
        # loss is a real possibility. PDF output uses a different page coordinate system;
        # this image-file verifier is only connected for raster input.
        # Image-bearing PDF pages run the raster checks during conversion.
            self._verify_button.setEnabled(route is Route.RASTER)
            self._view_diff_button.setEnabled(False)
            self._diff_overlay_path = None
            pixmap = load_preview(document.output, dpi=self._preview_dpi,
                                  page_index=self._document_session.page_index)
            self._mirrored_view.set_pixmap_source(pixmap)
            self._refresh_mirrored_text_regions()
            self._sync_preview_scale()

            total_runs = sum(p.text_runs_mirrored for p in document.pages)
            total_flags = sum(len(p.flags) for p in document.pages)
            self._status_label.setText(
                f"{total_runs} text run(s) mirrored"
                + (f" · {total_flags} flag(s) below" if total_flags else " · no flags")
            )
            self._set_status_with_overlap(self._status_label.text())
            self._populate_flags(document)
        if not self._batch_queue:
            self._mirror_button.setEnabled(True)
        QTimer.singleShot(0, self._start_next_batch_job)

    def _on_mirror_failed(self, message: str, job_input: Path) -> None:
        if self._clear_pending:
            self._on_deferred_clear_ready()
            return
        entry = self._batch_entries[job_input]
        entry.status, entry.error = "failed", message
        self._refresh_batch_lists()
        if job_input == self._input_path:
            self._review_summary.setText("Processing failed")
            self._status_label.setText(f"{job_input.name}: {message}")
        if not self._batch_queue:
            self._mirror_button.setEnabled(True)
        QTimer.singleShot(0, self._start_next_batch_job)

    def _on_job_stage_changed(self, message: str) -> None:
        self._status_label.setText(message)

    def _on_cancel_available_changed(self, available: bool) -> None:
        self._cancel_button.setEnabled(available)

    def _on_cancel_clicked(self) -> None:
        self._job_controller.request_cancel()

    def _on_mirror_cancelled(self) -> None:
        self._batch_queue.clear()
        if self._active_batch_path in self._batch_entries:
            self._batch_entries[self._active_batch_path].status = "queued"
        self._active_batch_path = None
        self._progress.hide()
        self._cancel_button.hide()
        self._mirror_button.setEnabled(bool(self._batch_order))
        self._review_summary.setText("Mirror cancelled — source remains unchanged")
        self._flags_list.clear()
        self._on_deferred_clear_ready()

    def _on_verify_clicked(self) -> None:
        if self._input_path is None or self._output_path is None or self._output_axis is None:
            return
        if self._verify_worker is not None and self._verify_worker.isRunning():
            return  # same defence-in-depth as _on_mirror_clicked — see its own comment
        overlay_path = Path(self._temp_dir.name) / f"{self._output_path.stem}_verify_diff.png"

        self._verify_button.setEnabled(False)
        self._view_diff_button.setEnabled(False)
        self._progress.show()
        self._status_label.setText("Verifying against the source… (re-reading the source's own text)")

        self._verify_worker = VerifyWorker(self._input_path, self._output_path, self._output_axis, overlay_path)
        self._verify_worker.finished.connect(self._on_deferred_clear_ready)
        # Bound at connect time, not read from self._output_path when the
        # signal fires — same reasoning as MirrorWorker's own job_input
        # capture (see _on_mirror_clicked): the user may have loaded and
        # mirrored a different file before this verify job finishes.
        job_output = self._output_path
        self._verify_worker.succeeded.connect(
            lambda report, overlay, job_output=job_output: self._on_verify_succeeded(report, overlay, job_output)
        )
        self._verify_worker.failed.connect(
            lambda message, job_output=job_output: self._on_verify_failed(message, job_output)
        )
        self._verify_worker.start()

    def _on_verify_succeeded(self, report: VerifyReport, overlay_path: Path, job_output: Path) -> None:
        self._progress.hide()
        if job_output != self._output_path:
            self._verify_button.setEnabled(self._output_path is not None and self._output_route is Route.RASTER)
            return
        self._verify_button.setEnabled(True)
        self._diff_overlay_path = overlay_path
        if self._input_path in self._batch_entries:
            self._batch_entries[self._input_path].diff_overlay = overlay_path
        self._view_diff_button.setEnabled(True)
        self._review_summary.setText(summarize_verification(report).label)

        if report.passed:
            self._status_label.setText(
                f"No changes above tolerance outside text exclusions; "
                f"{report.text_runs_checked} output readings agree with source OCR. "
                "Excluded geometry and source OCR still require review."
            )
            item = QListWidgetItem("✓  Geometry outside exclusions checked; output OCR agrees. Not a ground-truth guarantee.")
            item.setForeground(Qt.GlobalColor.darkGreen)
        else:
            self._status_label.setText(
                f"Verify found {report.outside_text_px}px across "
                f"{len(report.outside_text_regions)} region(s) outside text exclusions; "
                f"{len(report.text_issues)} wording concern(s)."
            )
            item = QListWidgetItem(
                f"✖  Verify: {report.outside_text_px}px across {len(report.outside_text_regions)} "
                f"region(s) outside exclusions; {len(report.text_issues)} wording concern(s)"
            )
            item.setForeground(Qt.GlobalColor.red)
        self._flags_list.addItem(item)
        for issue in report.text_issues:
            self._flags_list.addItem(f"⚠  Text verification: {issue}")
        if self._flags_list.item(0) is not None and self._flags_list.item(0).text() == "No issues to review.":
            self._flags_list.takeItem(0)

        if not report.passed:
            self._on_view_diff_clicked()  # a real finding is worth surfacing immediately, not just noting

    def _on_verify_failed(self, message: str, job_output: Path) -> None:
        self._progress.hide()
        if job_output != self._output_path:
            self._verify_button.setEnabled(self._output_path is not None and self._output_route is Route.RASTER)
            return
        self._verify_button.setEnabled(True)
        self._status_label.setText("")
        QMessageBox.critical(self, "Couldn't verify this plan", message)

    def _on_resolve_clicked(self) -> None:
        """Nudge every flagged text run three points right, across all pages."""
        if self._output_path is None or self._output_route is not Route.VECTOR:
            return
        entry = self._batch_entries.get(self._input_path) if self._input_path else None
        if entry is None:
            return

        from spejl.vector.pdf_mirror import text_drawing_overlap_runs, translate_pdf_text_runs

        try:
            overlaps = text_drawing_overlap_runs(self._output_path)
            entry.overlap_findings = [(page + 1, text) for page, _run, text in overlaps]
            entry.overlaps_checked = True
            if not overlaps:
                self._set_status_with_overlap("No flagged text needs resolving.")
                return

            translations = {
                (page_index, run_index): (3.0, 0.0)
                for page_index, run_index, _text in overlaps
            }
            edited = Path(self._temp_dir.name) / f"{self._output_path.stem}_resolved.pdf"
            translate_pdf_text_runs(
                self._output_path,
                edited,
                translations,
                axis=self._output_axis or Axis.VERTICAL,
            )
            edited.replace(self._output_path)

            page_index = self._document_session.page_index
            self._mirrored_view.set_pixmap_source(load_preview(
                self._output_path, dpi=self._preview_dpi, page_index=page_index
            ))
            selected = {
                run_index
                for page, run_index, _text in overlaps
                if page == page_index
            }
            self._selected_text_indexes = selected
            self._refresh_mirrored_text_regions(selected=selected)
            self._sync_preview_scale()
            count = len(overlaps)
            self._set_status_with_overlap(
                f"Moved {count} flagged text item(s) 3 pt to the right; checking remaining overlaps.",
                refresh=True,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the editor
            QMessageBox.critical(self, "Couldn't resolve flagged text", str(exc))

    def _on_view_diff_clicked(self) -> None:
        if self._diff_overlay_path is None or not self._diff_overlay_path.exists():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._diff_overlay_path)))

    def _populate_flags(self, document: Document) -> None:
        self._flags_list.clear()
        self._review_summary.setText(summarize_document(document).label)
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

    def _on_review_item_clicked(self, item: QListWidgetItem) -> None:
        """Surface the selected review item without modifying the output."""
        self._status_label.setText(f"Review item selected: {item.text()}")

    def _on_save_clicked(self) -> None:
        if self._output_path is None or self._input_path is None:
            return
        entry = self._batch_entries.get(self._input_path)
        mirrored_name = entry.mirrored_name if entry else self._output_path.name
        suggested = str(self._input_path.with_name(Path(mirrored_name).stem + ".jpg"))
        path_str, _selected_filter = QFileDialog.getSaveFileName(
            self, "Save mirrored page as JPEG", suggested,
            "JPEG image (*.jpg *.jpeg)",
        )
        if not path_str:
            return
        dest = Path(path_str)
        if not dest.suffix:
            dest = dest.with_suffix(".jpg")
        if dest.suffix.lower() not in {".jpg", ".jpeg"}:
            QMessageBox.warning(self, "JPEG required", "Save As exports a JPEG file. Choose .jpg or .jpeg.")
            return
        self._export_mirrored_image(dest)
        self._status_label.setText(f"Saved to {dest.name}")

    def _export_mirrored_image(self, destination: Path) -> None:
        """Render a PDF export at print quality or convert an image output."""
        if self._output_path is None:
            return
        suffix = destination.suffix.lower().lstrip(".")
        image_format = {"jpg": "jpeg", "jpeg": "jpeg", "tif": "tiff", "tiff": "tiff"}.get(suffix, suffix)
        if self._output_path.suffix.lower() == ".pdf":
            import cv2
            import numpy as np
            import pymupdf

            document = pymupdf.open(str(self._output_path))
            try:
                page_index = min(self._document_session.page_index, document.page_count - 1)
                pixmap = document[page_index].get_pixmap(dpi=300, alpha=False)
                image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                )
                # PyMuPDF supplies RGB samples; OpenCV writers expect BGR.
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(destination), image):
                    raise ValueError(f"Could not write {image_format.upper()} image export.")
            finally:
                document.close()
            return

        import cv2

        image = cv2.imread(str(self._output_path), cv2.IMREAD_UNCHANGED)
        if image is None or not cv2.imwrite(str(destination), image):
            raise ValueError(f"Could not write {image_format.upper()} image export.")

    def _on_remove_text_clicked(self) -> None:
        if self._output_path is None or self._output_route is not Route.VECTOR:
            return
        import pymupdf
        from spejl.vector.pdf_mirror import remove_pdf_text_runs

        doc = pymupdf.open(str(self._output_path))
        dialog = QDialog(self)
        dialog.setWindowTitle("Remove text from mirrored PDF")
        dialog.resize(420, 520)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select text objects to remove:"))
        runs = QListWidget()
        for page_index, page in enumerate(doc):
            run_index = 0
            for trace in page.get_texttrace():
                if trace.get("type") != 0 or not trace.get("chars"):
                    continue
                text = "".join(chr(char[0]) for char in trace["chars"])
                item = QListWidgetItem(f"Page {page_index + 1}: {text}")
                item.setData(Qt.ItemDataRole.UserRole, (page_index, run_index))
                item.setCheckState(Qt.CheckState.Unchecked)
                runs.addItem(item)
                run_index += 1
        doc.close()
        layout.addWidget(runs)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        removals: dict[int, set[int]] = {}
        for i in range(runs.count()):
            item = runs.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                page_index, run_index = item.data(Qt.ItemDataRole.UserRole)
                removals.setdefault(page_index, set()).add(run_index)
        if not removals:
            return
        edited = Path(self._temp_dir.name) / f"{self._output_path.stem}_edited.pdf"
        try:
            remove_pdf_text_runs(self._output_path, edited, removals)
            edited.replace(self._output_path)
            pixmap = load_preview(self._output_path, dpi=self._preview_dpi,
                                  page_index=self._document_session.page_index)
            self._mirrored_view.set_pixmap_source(pixmap)
            self._refresh_mirrored_text_regions()
            self._sync_preview_scale()
            self._set_status_with_overlap("Selected text removed from the mirrored PDF.", refresh=True)
        except Exception as exc:  # noqa: BLE001 - surfaced in the editor
            QMessageBox.critical(self, "Couldn't remove text", str(exc))

    def _on_move_text_clicked(self) -> None:
        if self._output_path is None or self._output_route is not Route.VECTOR:
            return
        import pymupdf
        from spejl.vector.pdf_mirror import translate_pdf_text_runs

        doc = pymupdf.open(str(self._output_path))
        dialog = QDialog(self)
        dialog.setWindowTitle("Move text in mirrored PDF")
        dialog.resize(420, 560)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select text, then set its movement in PDF points:"))
        runs = QListWidget()
        for page_index, page in enumerate(doc):
            run_index = 0
            for trace in page.get_texttrace():
                if trace.get("type") != 0 or not trace.get("chars"):
                    continue
                text = "".join(chr(char[0]) for char in trace["chars"])
                item = QListWidgetItem(f"Page {page_index + 1}: {text}")
                item.setData(Qt.ItemDataRole.UserRole, (page_index, run_index))
                item.setCheckState(Qt.CheckState.Unchecked)
                runs.addItem(item)
                run_index += 1
        doc.close()
        layout.addWidget(runs)
        offsets = QHBoxLayout()
        offsets.addWidget(QLabel("Right / left:"))
        right = QDoubleSpinBox()
        right.setRange(-100.0, 100.0)
        right.setDecimals(1)
        right.setSuffix(" pt")
        offsets.addWidget(right)
        offsets.addWidget(QLabel("Down / up:"))
        down = QDoubleSpinBox()
        down.setRange(-100.0, 100.0)
        down.setDecimals(1)
        down.setSuffix(" pt")
        offsets.addWidget(down)
        layout.addLayout(offsets)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        translations = {}
        for i in range(runs.count()):
            item = runs.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                translations[item.data(Qt.ItemDataRole.UserRole)] = (right.value(), down.value())
        if not translations:
            return
        edited = Path(self._temp_dir.name) / f"{self._output_path.stem}_moved.pdf"
        try:
            translate_pdf_text_runs(
                self._output_path, edited, translations,
                axis=self._output_axis or Axis.VERTICAL,
            )
            edited.replace(self._output_path)
            self._mirrored_view.set_pixmap_source(load_preview(
                self._output_path, dpi=self._preview_dpi,
                page_index=self._document_session.page_index
            ))
            self._refresh_mirrored_text_regions()
            self._sync_preview_scale()
            self._set_status_with_overlap(
                "Selected text moved without changing font, spacing, or orientation.",
                refresh=True,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the editor
            QMessageBox.critical(self, "Couldn't move text", str(exc))

    def _refresh_mirrored_text_regions(self, selected: set[int] | None = None) -> None:
        """Expose current-page vector text as click targets over the preview."""
        if self._output_path is None or self._output_route is not Route.VECTOR:
            self._mirrored_view.set_text_regions([])
            return
        import pymupdf
        doc = pymupdf.open(str(self._output_path))
        try:
            page_index = self._document_session.page_index
            if doc.page_count == 0 or page_index >= doc.page_count:
                self._mirrored_view.set_text_regions([])
                return
            zoom = self._preview_dpi / 72.0  # must match gui.imaging.load_preview
            regions = []
            index = 0
            for trace in doc[page_index].get_texttrace():
                if trace.get("type") != 0 or not trace.get("chars"):
                    continue
                x0, y0, x1, y1 = trace["bbox"]
                regions.append((index, QRectF(x0 * zoom, y0 * zoom, (x1 - x0) * zoom, (y1 - y0) * zoom)))
                index += 1
            self._mirrored_view.set_text_regions(regions, selected)
        finally:
            doc.close()

    def _set_status_with_overlap(self, message: str, *, refresh: bool = False) -> None:
        """Show current plan's text/drawing collision findings in the status area."""
        entry = self._batch_entries.get(self._input_path) if self._input_path else None
        if entry is None or entry.output is None:
            self._status_label.setText(message)
            self._update_resolve_button_state()
            return
        if refresh or not entry.overlaps_checked:
            try:
                if entry.output.suffix.lower() == ".pdf":
                    from spejl.vector.pdf_mirror import text_drawing_overlap_findings
                    entry.overlap_findings = text_drawing_overlap_findings(entry.output)
                else:
                    entry.overlap_findings = []
                entry.overlaps_checked = True
            except Exception as exc:  # Diagnostics should never interrupt editing.
                entry.overlap_findings = []
                entry.overlaps_checked = False
                self._status_label.setText(f"{message}\nOverlap check unavailable: {exc}")
                self._update_resolve_button_state()
                return
        findings = entry.overlap_findings
        self._update_resolve_button_state()
        if not findings:
            self._status_label.setStyleSheet("color: #8CB3C0; font-size: 12px;")
            self._status_label.setToolTip("No text and drawing overlaps were detected.")
            self._status_label.setText(f"{message}\nNo text/drawing overlaps detected.")
            return
        labels = [f"p.{page}: {text!r}" for page, text in findings]
        visible = ", ".join(labels[:3])
        if len(labels) > 3:
            visible += f", and {len(labels) - 3} more"
        self._status_label.setStyleSheet("color: #FFC27A; font-size: 12px; font-weight: 600;")
        self._status_label.setToolTip(
            "Text runs whose glyph bounds intersect drawing geometry:\n" + "\n".join(labels)
        )
        self._status_label.setText(
            f"{message}\n⚠ {len(findings)} text/drawing overlap(s): {visible}"
        )

    def _on_preview_text_selection_changed(self, indexes: list[int]) -> None:
        self._selected_text_indexes = set(indexes)
        if not indexes:
            self._set_status_with_overlap("Selection cleared.")
            return
        self._set_status_with_overlap(
            f"{len(indexes)} text item(s) selected. Arrow keys move all; Alt+arrow snaps to text; "
            "Shift uses 5 pt; Esc clears."
        )

    def _on_preview_text_nudged(
        self, indexes: list[int], right: float, down: float, snap_to_text: bool
    ) -> None:
        if self._output_path is None or self._output_route is not Route.VECTOR:
            return
        if not indexes:
            return
        guides = []
        if snap_to_text:
            right, down, guides = self._mirrored_view.snap_text_nudge(
                right, down, pixels_per_point=self._preview_dpi / 72.0
            )
        from spejl.vector.pdf_mirror import translate_pdf_text_runs
        edited = Path(self._temp_dir.name) / f"{self._output_path.stem}_nudge.pdf"
        try:
            translate_pdf_text_runs(
                self._output_path, edited,
                {(self._document_session.page_index, index): (right, down) for index in indexes},
                axis=self._output_axis or Axis.VERTICAL,
            )
            edited.replace(self._output_path)
            self._mirrored_view.set_pixmap_source(load_preview(
                self._output_path, dpi=self._preview_dpi,
                page_index=self._document_session.page_index
            ))
            self._refresh_mirrored_text_regions(selected=set(indexes))
            self._mirrored_view.show_alignment_guides(guides)
            self._sync_preview_scale()
            snapped = " Snapped to nearby text alignment." if guides else ""
            self._set_status_with_overlap(
                f"{len(indexes)} text item(s) moved.{snapped} "
                "Arrow keys are exact; Alt+arrow snaps to nearby text; Shift uses 5 pt; Esc clears.",
                refresh=True,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the preview
            QMessageBox.critical(self, "Couldn't move text", str(exc))

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        if self._verify_worker is not None and self._verify_worker.isRunning():
            self._verify_worker.wait(2000)
        self._temp_dir.cleanup()
        super().closeEvent(event)
