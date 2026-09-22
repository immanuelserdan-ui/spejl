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
        self._syncing_viewport = False
        self._temp_dir = tempfile.TemporaryDirectory(prefix="spejl_gui_")
        self._document_session = DocumentSession()
        self._update_thread: QThread | None = None
        self._update_worker: UpdateCheckWorker | None = None

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

        flags_box = QGroupBox("Review")
        flags_layout = QVBoxLayout(flags_box)
        self._review_summary = QLabel("Choose a plan to begin")
        self._review_summary.setStyleSheet("color: #9EE5F7; font-size: 12px; font-weight: 600;")
        flags_layout.addWidget(self._review_summary)
        self._flags_list = QListWidget()
        self._flags_list.setAlternatingRowColors(True)
        self._flags_list.itemClicked.connect(self._on_review_item_clicked)
        flags_layout.addWidget(self._flags_list)
        layout.addWidget(flags_box, stretch=1)

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
        self._source_view, source_pane, self._source_scroll = self._make_preview_pane(
            "Source", scrollable=True
        )
        self._mirrored_view, mirrored_pane, self._mirrored_scroll = self._make_preview_pane(
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
        self._page_label.setText(session.label)
        self._previous_page_button.setEnabled(session.page_index > 0)
        self._next_page_button.setEnabled(session.page_index < session.page_count - 1)

    def _change_preview_page(self, offset: int) -> None:
        if not self._document_session.move(offset) or self._input_path is None:
            return
        self._source_view.set_pixmap_source(
            load_preview(self._input_path, page_index=self._document_session.page_index)
        )
        if self._output_path is not None and self._output_path.suffix.lower() == ".pdf":
            self._mirrored_view.set_pixmap_source(
                load_preview(self._output_path, page_index=self._document_session.page_index)
            )
        if self._document_session.page_index:
            self._mirrored_view.set_text_regions([])
        else:
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

    def _make_preview_pane(
        self, caption: str, *, zoomable: bool = False, scrollable: bool = False
    ) -> tuple[ScaledImageLabel, QWidget, QScrollArea | None]:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(16, 12, 16, 16)
        heading = QHBoxLayout()
        label = QLabel(caption)
        label.setObjectName("previewCaption")
        heading.addWidget(label)
        if zoomable:
            self._reset_view_button = QPushButton("Reset View")
            self._reset_view_button.setToolTip("Return the mirrored plan to its default fit view")
            self._reset_view_button.clicked.connect(self._reset_mirrored_view)
            self._reset_view_button.hide()
            heading.addWidget(self._reset_view_button)
        heading.addStretch(1)
        layout.addLayout(heading)
        image_view = ScaledImageLabel()
        image_view.setStyleSheet(
            "background: #07111F; border: 1px solid #1F4D61; border-radius: 6px;"
        )
        if not scrollable:
            layout.addWidget(image_view, stretch=1)
            return image_view, pane, None
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
        return image_view, pane, scroll

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
        self._sync_preview_scale()
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
        self._reset_view_button.hide()
        self._sync_preview_scale()
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

    def _on_file_chosen(self, path: Path) -> None:
        self._input_path = path
        self._output_path = None
        self._document_session.open(path)
        self._update_page_navigation()
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
        self._verify_button.setEnabled(False)
        self._view_diff_button.setEnabled(False)
        self._diff_overlay_path = None
        self._flags_list.clear()
        self._review_summary.setText("Ready to mirror")
        if self._job_running():
            self._status_label.setText("Mirroring the previous file — this one will be ready to mirror once it finishes.")
        else:
            self._status_label.setText("")
        self._mirrored_view.set_pixmap_source(None)
        self._mirrored_view.set_text_regions([])
        self._reset_mirrored_view()

        from spejl.router import sniff_route

        try:
            route = sniff_route(path)
        except ValueError:
            route = None
        self._show_route_badge(route)

        pixmap = load_preview(path, page_index=self._document_session.page_index)
        self._source_view.set_pixmap_source(pixmap)
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
        self._cancel_button.show()
        self._cancel_button.setEnabled(False)
        self._flags_list.clear()
        self._review_summary.setText("Processing in progress")
        self._status_label.setText("Preparing plan…")

        self._worker = self._job_controller.start(self._input_path, output_path, axis)
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
        self._cancel_button.hide()
        if self._job_controller.cancel_requested:
            return
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
        self._output_axis = document.axis
        self._output_route = route
        self._save_button.setEnabled(True)
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

        pixmap = load_preview(document.output, page_index=self._document_session.page_index)
        self._mirrored_view.set_pixmap_source(pixmap)
        self._refresh_mirrored_text_regions()
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
            self._cancel_button.hide()
            self._mirror_button.setEnabled(not self._job_running())
            return
        self._progress.hide()
        self._cancel_button.hide()
        self._mirror_button.setEnabled(True)
        self._status_label.setText("")
        QMessageBox.critical(self, "Couldn't mirror this plan", message)

    def _on_job_stage_changed(self, message: str) -> None:
        self._status_label.setText(message)

    def _on_cancel_available_changed(self, available: bool) -> None:
        self._cancel_button.setEnabled(available)

    def _on_cancel_clicked(self) -> None:
        self._job_controller.request_cancel()

    def _on_mirror_cancelled(self) -> None:
        self._progress.hide()
        self._cancel_button.hide()
        self._mirror_button.setEnabled(self._input_path is not None)
        self._review_summary.setText("Mirror cancelled — source remains unchanged")
        self._flags_list.clear()

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
        suggested = str(self._input_path.with_name(self._output_path.name))
        save_filter = (
            "PDF files (*.pdf);;PNG image (*.png);;JPEG image (*.jpg *.jpeg);;"
            "TIFF image (*.tif *.tiff);;Bitmap image (*.bmp)"
        )
        path_str, selected_filter = QFileDialog.getSaveFileName(
            self, "Save mirrored plan", suggested,
            save_filter,
        )
        if not path_str:
            return
        dest = Path(path_str)
        suffixes = {
            "PDF files": ".pdf", "PNG image": ".png", "JPEG image": ".jpg",
            "TIFF image": ".tif", "Bitmap image": ".bmp",
        }
        if not dest.suffix:
            dest = dest.with_suffix(next((suffix for label, suffix in suffixes.items() if selected_filter.startswith(label)), ".pdf"))

        if dest.suffix.lower() == ".pdf":
            shutil.copyfile(self._output_path, dest)
        elif dest.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            self._export_mirrored_image(dest)
        else:
            QMessageBox.warning(self, "Unsupported export", "Choose PDF, PNG, JPEG, TIFF, or BMP.")
            return

        sidecar_src = self._output_path.with_suffix(self._output_path.suffix + ".spejl.json")
        if dest.suffix.lower() == ".pdf" and sidecar_src.exists():
            shutil.copyfile(sidecar_src, dest.with_suffix(dest.suffix + ".spejl.json"))

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
                if document.page_count != 1:
                    raise ValueError("Image export currently supports the first mirrored PDF page only.")
                pixmap = document[0].get_pixmap(dpi=300, alpha=False)
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
            pixmap = load_preview(self._output_path)
            self._mirrored_view.set_pixmap_source(pixmap)
            self._refresh_mirrored_text_regions()
            self._sync_preview_scale()
            self._status_label.setText("Selected text removed from the mirrored PDF.")
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
            self._mirrored_view.set_pixmap_source(load_preview(self._output_path))
            self._refresh_mirrored_text_regions()
            self._sync_preview_scale()
            self._status_label.setText("Selected text moved without changing font, spacing, or orientation.")
        except Exception as exc:  # noqa: BLE001 - surfaced in the editor
            QMessageBox.critical(self, "Couldn't move text", str(exc))

    def _refresh_mirrored_text_regions(self, selected: set[int] | None = None) -> None:
        """Expose first-page vector text as click targets over the preview."""
        if self._output_path is None or self._output_route is not Route.VECTOR:
            self._mirrored_view.set_text_regions([])
            return
        import pymupdf
        doc = pymupdf.open(str(self._output_path))
        try:
            if doc.page_count == 0:
                self._mirrored_view.set_text_regions([])
                return
            zoom = 150 / 72.0  # must match gui.imaging.load_preview
            regions = []
            index = 0
            for trace in doc[0].get_texttrace():
                if trace.get("type") != 0 or not trace.get("chars"):
                    continue
                x0, y0, x1, y1 = trace["bbox"]
                regions.append((index, QRectF(x0 * zoom, y0 * zoom, (x1 - x0) * zoom, (y1 - y0) * zoom)))
                index += 1
            self._mirrored_view.set_text_regions(regions, selected)
        finally:
            doc.close()

    def _on_preview_text_selection_changed(self, indexes: list[int]) -> None:
        self._selected_text_indexes = set(indexes)
        if not indexes:
            self._status_label.setText("Selection cleared.")
            return
        self._status_label.setText(
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
                right, down, pixels_per_point=150 / 72.0
            )
        from spejl.vector.pdf_mirror import translate_pdf_text_runs
        edited = Path(self._temp_dir.name) / f"{self._output_path.stem}_nudge.pdf"
        try:
            translate_pdf_text_runs(
                self._output_path, edited, {(0, index): (right, down) for index in indexes},
                axis=self._output_axis or Axis.VERTICAL,
            )
            edited.replace(self._output_path)
            self._mirrored_view.set_pixmap_source(load_preview(self._output_path))
            self._refresh_mirrored_text_regions(selected=set(indexes))
            self._mirrored_view.show_alignment_guides(guides)
            self._sync_preview_scale()
            snapped = " Snapped to nearby text alignment." if guides else ""
            self._status_label.setText(
                f"{len(indexes)} text item(s) moved.{snapped} "
                "Arrow keys are exact; Alt+arrow snaps to nearby text; Shift uses 5 pt; Esc clears."
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
