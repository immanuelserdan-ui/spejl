"""Review dialog for assisted mirroring of mixed image/vector PDFs.

OCR suggests the lettering it found inside the plan's pictures; the user
decides what is lettering, corrects its text and reading direction, and only
then may the plan be mirrored. Lettering the user leaves out stays part of the
drawing and is mirrored with it — the dialog says so plainly, because that is
the one way assisted mirroring can still produce backwards text.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QScrollArea, QVBoxLayout, QWidget,
)

from spejl.gui.imaging import load_preview
from spejl.vector.mixed_pdf import SUPPORTED_ANGLES, PictureLabel

_DIRECTIONS = (
    (0.0, "→  left to right"),
    (90.0, "↑  bottom to top"),
    (-90.0, "↓  top to bottom"),
)
_PREVIEW_DPI = 110
_BOX_COLOR = QColor("#E8590C")


def _crop_pixmap(picture: np.ndarray, label: PictureLabel, scale: int = 4) -> QPixmap:
    """The lettering as found, turned to read left to right, enlarged."""
    x0, y0, x1, y1 = label.pixel_box
    pad = 3
    h, w = picture.shape[:2]
    crop = picture[max(y0 - pad, 0):min(y1 + pad, h), max(x0 - pad, 0):min(x1 + pad, w)]
    if label.angle_deg == 90.0:
        crop = np.rot90(crop, k=-1)
    elif label.angle_deg == -90.0:
        crop = np.rot90(crop, k=1)
    crop = np.ascontiguousarray(np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1))
    image = QImage(crop.data, crop.shape[1], crop.shape[0], crop.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class PictureTextDialog(QDialog):
    """Confirm the lettering found inside a mixed PDF's pictures."""

    def __init__(self, pdf_path: Path, labels: list[PictureLabel], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Review picture text — {pdf_path.name}")
        self.resize(1100, 760)
        # Same palette as the app's other dialogs (see JpegExportDialog).
        self.setStyleSheet("""
            QDialog { background: #07111F; color: #EAF7FB; }
            QLabel { color: #D7EEF5; }
            QLabel#intro { background: #0D2638; border: 1px solid #1F4D61; border-radius: 6px; padding: 10px; }
            QLabel#summary { color: #FFB74D; font-weight: 600; }
            QScrollArea, QWidget#rows { background: #07111F; border: none; }
            QCheckBox { color: #D7EEF5; }
            QLineEdit, QComboBox { color: #EAF7FB; background: #0D2638; border: 1px solid #2B6479;
                                   border-radius: 4px; padding: 4px 6px; }
            QLineEdit:disabled, QComboBox:disabled { color: #5C7A88; border-color: #1F4D61; }
            QPushButton { color: #D7EEF5; background: #0D2638; border: 1px solid #2B6479;
                          border-radius: 5px; padding: 6px 12px; }
            QPushButton:hover { background: #14364C; }
            QPushButton:disabled { color: #5C7A88; }
        """)
        self._labels = [copy.copy(label) for label in labels]
        self._rows: list[tuple[QCheckBox, QLineEdit, QComboBox]] = []

        layout = QVBoxLayout(self)
        intro = QLabel(
            "<b>The walls and lines of this plan are a picture.</b> Spejl mirrors the picture as a "
            "whole, so any lettering inside it would come out backwards. OCR found the "
            "lettering below; <b>tick what is lettering and correct its text</b>. Ticked items "
            "are removed from the picture and redrawn as real text.<br>"
            "<b>Check the plan on the left:</b> lettering without a numbered box was not found "
            "and will be mirrored backwards. Untick finds that are symbols (stove burners, arrows)."
        )
        intro.setObjectName("intro")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        body = QHBoxLayout()
        preview = QScrollArea()
        preview.setWidgetResizable(True)
        preview.setMinimumWidth(420)
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setWidget(self._preview)
        body.addWidget(preview, 1)

        rows_widget = QWidget()
        rows_widget.setObjectName("rows")
        grid = QGridLayout(rows_widget)
        for column, heading in enumerate(("#", "Lettering?", "As found", "Text", "Reading direction", "OCR")):
            title = QLabel(f"<b>{heading}</b>")
            grid.addWidget(title, 0, column)
        pictures: dict[int, np.ndarray] = {}
        import pymupdf

        from spejl.vector.mixed_pdf import picture_rgb

        with pymupdf.open(str(pdf_path)) as doc:
            for label in self._labels:
                if label.image_xref not in pictures:
                    pictures[label.image_xref] = picture_rgb(doc, label.image_xref)
        for index, label in enumerate(self._labels, start=1):
            number = QLabel(str(index))
            include = QCheckBox()
            include.setChecked(label.include)
            crop = QLabel()
            crop.setPixmap(_crop_pixmap(pictures[label.image_xref], label))
            text = QLineEdit(label.text)
            text.setMinimumWidth(110)
            direction = QComboBox()
            for angle, caption in _DIRECTIONS:
                direction.addItem(caption, angle)
            if label.angle_deg in SUPPORTED_ANGLES:
                direction.setCurrentIndex([a for a, _ in _DIRECTIONS].index(label.angle_deg))
            else:
                direction.addItem(f"slanted {label.angle_deg:.0f}° (not supported)", label.angle_deg)
                direction.setCurrentIndex(direction.count() - 1)
            ocr = QLabel(f"'{label.ocr_text}' · {label.confidence:.0%}")
            ocr.setToolTip("What OCR read, and how sure it was")
            for column, widget in enumerate((number, include, crop, text, direction, ocr)):
                grid.addWidget(widget, index, column)
            include.toggled.connect(self._sync)
            text.textChanged.connect(self._sync)
            direction.currentIndexChanged.connect(self._sync)
            self._rows.append((include, text, direction))
        grid.setRowStretch(len(self._labels) + 1, 1)
        rows_scroll = QScrollArea()
        rows_scroll.setWidgetResizable(True)
        rows_scroll.setWidget(rows_widget)
        body.addWidget(rows_scroll, 1)
        layout.addLayout(body, 1)

        self._summary = QLabel()
        self._summary.setObjectName("summary")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Confirm and allow mirroring")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._base = load_preview(pdf_path, dpi=_PREVIEW_DPI, page_index=0)
        self._sync()

    def labels(self) -> list[PictureLabel]:
        """The labels as the user left them."""
        return [copy.copy(label) for label in self._labels]

    def _sync(self, *_unused) -> None:
        problems: list[str] = []
        for label, (include, text, direction) in zip(self._labels, self._rows):
            label.include = include.isChecked()
            label.text = text.text()
            label.angle_deg = float(direction.currentData())
            text.setEnabled(label.include)
            direction.setEnabled(label.include)
            if label.include and not label.text.strip():
                problems.append("a ticked item has no text")
            if label.include and label.angle_deg not in SUPPORTED_ANGLES:
                problems.append("slanted lettering can't be redrawn; untick it or pick a direction")
        ticked = sum(label.include for label in self._labels)
        message = (f"{ticked} of {len(self._labels)} find(s) will be redrawn as real text."
                   if self._labels else
                   "OCR found no lettering inside the picture. If the plan on the left shows "
                   "lettering in the picture, it will come out backwards.")
        if problems:
            message += " ⚠ Fix before confirming: " + "; ".join(dict.fromkeys(problems)) + "."
        self._summary.setText(message)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not problems)
        self._draw_preview()

    def _draw_preview(self) -> None:
        if self._base is None:
            self._preview.setText("No preview available.")
            return
        canvas = QPixmap(self._base)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale = _PREVIEW_DPI / 72.0
        font = QFont()
        font.setBold(True)
        font.setPointSizeF(8)
        painter.setFont(font)
        for index, label in enumerate(self._labels, start=1):
            if label.page_index != 0:
                continue
            x0, y0, x1, y1 = (v * scale for v in label.page_box)
            pen = QPen(_BOX_COLOR if label.include else QColor("#868E96"), 1.6,
                       Qt.PenStyle.SolidLine if label.include else Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRectF(x0 - 2, y0 - 2, x1 - x0 + 4, y1 - y0 + 4))
            painter.drawText(QPointF(x1 + 4, y0 + 8), str(index))
        painter.end()
        self._preview.setPixmap(canvas)
