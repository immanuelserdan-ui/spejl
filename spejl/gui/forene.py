"""Forene's ordered import dialog and live single-sheet workspace."""
from dataclasses import replace
import logging
from pathlib import Path

import pymupdf
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QListWidget,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from spejl.forene import REFERENCE_GAP, compose, export_sheet, import_floors, match_landings, placement, unconfirmed_floors

LOG = logging.getLogger(__name__)
FILTER = "Drawings (*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"
DIALOG_STYLE = """
QDialog { background: #0A1928; }
QDialog QLabel, QDialog QCheckBox { color: #D7EEF5; }
"""


def button(row, text, slot):
    widget = QPushButton(text)
    widget.setObjectName("verifyButton")
    widget.clicked.connect(slot)
    row.addWidget(widget)
    return widget


class ImportDialog(QDialog):
    def __init__(self, floors, parent):
        super().__init__(parent)
        self.setWindowTitle("Insert floor plans")
        self.setStyleSheet(DIALOG_STYLE)
        self.resize(610, 390)
        self.slots = list(floors)
        self.slots.extend([None] * max(0, 4-len(floors)))
        box = QVBoxLayout(self)
        box.addWidget(QLabel("Click a slot to insert a drawing. Place the lowest floor first."))
        self.rows = QListWidget()
        box.addWidget(self.rows)
        self.rows.itemClicked.connect(self.choose_empty)
        self.rows.itemActivated.connect(lambda item: self.browse(self.rows.row(item)))
        row = QHBoxLayout()
        box.addLayout(row)
        button(row, "Add more…", self.append_slot)
        button(row, "Choose file…", lambda: self.browse(self.rows.currentRow()))
        button(row, "Move earlier", lambda: self.move(-1))
        button(row, "Move later", lambda: self.move(1))
        button(row, "Remove", self.remove)
        button(row, "Reset / Clear", self.clear)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        box.addWidget(buttons)
        self.refresh(0)

    def choose_empty(self, item):
        index = self.rows.row(item)
        if self.slots[index] is None:
            self.browse(index)

    def refresh(self, selected):
        self.rows.clear()
        for i, floor in enumerate(self.slots, 1):
            self.rows.addItem(f"{i}.   {floor.name if floor else 'Choose drawing…'}")
        self.rows.setCurrentRow(selected)

    def browse(self, index):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose floor plans", "", FILTER)
        if not files:
            return
        try:
            added = [floor for file in files for floor in import_floors(Path(file))]
        except ValueError as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        end = index+1
        while end < len(self.slots) and self.slots[end] is None and end-index < len(added):
            end += 1
        self.slots[index:end] = added
        self.refresh(index)

    def append_slot(self):
        self.slots.append(None)
        self.refresh(len(self.slots)-1)
        self.browse(len(self.slots)-1)

    def move(self, direction):
        i = self.rows.currentRow()
        j = i + direction
        if 0 <= i < len(self.slots) and 0 <= j < len(self.slots):
            self.slots[i], self.slots[j] = self.slots[j], self.slots[i]
            self.refresh(j)

    def remove(self):
        i = self.rows.currentRow()
        if i >= 0:
            self.slots.pop(i)
            self.slots.append(None)
            self.refresh(i)

    def clear(self):
        self.slots = [None]*4
        self.refresh(0)


class SheetView(QGraphicsView):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def fit(self):
        if self.scene() and not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

class StairEdgePicker(SheetView):
    picked = Signal(float)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(point):
                self.picked.emit(point.y() / self.sceneRect().height())
        super().mousePressEvent(event)


class StairEdgeDialog(QDialog):
    def __init__(self, floor, parent):
        super().__init__(parent)
        self.setWindowTitle("Set stair edge")
        self.resize(680, 640)
        self.datum = None
        box = QVBoxLayout(self)
        box.addWidget(QLabel(f"{floor.name} — click the upper edge of the stairs."))
        scene = QGraphicsScene(self)
        self.view = StairEdgePicker(scene)
        box.addWidget(self.view)
        sheet = compose([floor], 0, False)
        rect = placement([floor], 0)[0][0]
        with pymupdf.open(stream=sheet, filetype="pdf") as doc:
            zoom = min(3., 1600 / max(rect.width, rect.height))
            pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=rect)
            image = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
            scene.addPixmap(QPixmap.fromImage(image))
        scene.setSceneRect(0, 0, pix.width, pix.height)
        guide = scene.addLine(0, 0, pix.width, 0, QPen(QColor("#0588a6"), 0))
        guide.hide()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(False)

        def select(fraction):
            self.datum = fraction
            guide.setY(fraction * scene.sceneRect().height())
            guide.show()
            ok.setEnabled(True)

        self.view.picked.connect(select)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        box.addWidget(buttons)


class ForenePanel(QWidget):
    back = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.floors = []
        self.sheet = None
        self.footer = True
        self.zoom = 1.
        box = QVBoxLayout(self)
        bar = QHBoxLayout()
        box.addLayout(bar)
        title = QLabel("Forene")
        title.setObjectName("previewCaption")
        bar.addWidget(title)
        button(bar, "Insert File", self.insert)
        bar.addStretch()
        self.save_button = button(bar, "Generate / Save", self.save)
        button(bar, "Back to Spejl", self.back.emit)
        self.scene = QGraphicsScene(self)
        self.view = SheetView(self.scene)
        self.view.setBackgroundBrush(QColor("#18212a"))
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setInteractive(False)
        box.addWidget(self.view, 1)
        controls = QHBoxLayout()
        box.addLayout(controls)
        self.floor = QComboBox()
        self.floor.setMaximumWidth(220)
        controls.addWidget(self.floor)
        self.scale = self.spin(controls, "Scale", 1, 1000, 100)
        self.gap = self.spin(controls, "Gap", 0, 100, REFERENCE_GAP)
        self.adjust = button(controls, "Set stair edge…", self.adjust_datum)
        self.reset = button(controls, "Reset datum", self.reset_datum)
        self.status = QLabel()
        box.addWidget(self.status)
        self.floor.currentIndexChanged.connect(self.selection)
        self.scale.valueChanged.connect(self.rescale)
        self.gap.valueChanged.connect(self.render)
        self.selection()

    @staticmethod
    def spin(row, label, low, high, value):
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        spin.setSuffix(" %")
        spin.setKeyboardTracking(False)
        row.addWidget(spin)
        return spin

    def insert(self):
        dialog = ImportDialog(self.floors, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.floors = match_landings([floor for floor in dialog.slots if floor is not None])
            self.floor.blockSignals(True)
            self.floor.clear()
            self.floor.addItems([f"{i}. {f.name}" for i, f in enumerate(self.floors, 1)])
            self.floor.blockSignals(False)
            self.selection()

    def selection(self):
        i = self.floor.currentIndex()
        self.scale.setEnabled(i >= 0)
        self.adjust.setEnabled(i >= 0)
        self.reset.setEnabled(i >= 0)
        if i >= 0:
            self.scale.blockSignals(True)
            self.scale.setValue(self.floors[i].scale * 100)
            self.scale.blockSignals(False)
        self.render()

    def rescale(self, percent):
        i = self.floor.currentIndex()
        if i >= 0:
            self.floors[i] = replace(self.floors[i], scale=percent / 100)
            self.render()

    def reset_datum(self):
        i = self.floor.currentIndex()
        self.floors[i] = replace(self.floors[i], manual=None)
        self.render()

    def adjust_datum(self):
        i = self.floor.currentIndex()
        if i < 0 or not self.sheet:
            return
        dialog = StairEdgeDialog(self.floors[i], self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.datum is not None:
            self.floors[i] = replace(self.floors[i], manual=dialog.datum)
            self.render()

    def render(self):
        self.sheet = None
        self.save_button.setEnabled(False)
        self.scene.clear()
        if not self.floors:
            self.status.setText("Insert floor plans to preview the sheet.")
            return
        try:
            sheet = compose(self.floors, self.gap.value(), self.footer)
            with pymupdf.open(stream=sheet, filetype="pdf") as doc:
                page = doc[0]
                self.zoom = min(3., 2400/page.rect.width, 1600/page.rect.height)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(self.zoom, self.zoom))
                image = QImage(pix.samples, pix.width, pix.height, pix.stride,
                               QImage.Format.Format_RGB888).copy()
            self.scene.addPixmap(QPixmap.fromImage(image))
            self.scene.setSceneRect(0, 0, pix.width, pix.height)
            rects, datum = placement(self.floors, self.gap.value())
            pen = QPen(QColor("#0588a6"), 0, Qt.PenStyle.DashLine)
            self.scene.addLine(0, datum*self.zoom, rects[-1].x1*self.zoom, datum*self.zoom, pen)
            self.view.fit()
            self.sheet = sheet
            pending = unconfirmed_floors(self.floors)
            self.save_button.setEnabled(not pending)
            floor = self.floors[self.floor.currentIndex()]
            mode = ("Manual" if floor.manual is not None else
                    "Shared stair edge" if floor.shared_edge else
                    "Stair edge detected" if floor.stair_edge is not None else
                    "Stair detected" if floor.detected is not None else "Reference fallback")
            if pending:
                slots = ", ".join(map(str, pending))
                self.status.setText(f"Before saving, use Set stair edge on floor(s) {slots}.")
            else:
                self.status.setText(f"{floor.name} · {mode}.")
        except Exception as exc:
            LOG.exception("Forene preview failed")
            self.status.setText(f"Preview failed: {exc}")

    def save(self):
        if not self.sheet or unconfirmed_floors(self.floors):
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Generate / Save")
        dialog.setStyleSheet(DIALOG_STYLE)
        form = QFormLayout(dialog)
        fmt = QComboBox()
        fmt.addItems(["JPEG · 300 dpi", "PDF · preserve vectors"])
        form.addRow("Format", fmt)
        footer = QCheckBox("Include footer text")
        footer.setChecked(self.footer)
        form.addRow(footer)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Choose export folder")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not folder:
            return
        self.footer = footer.isChecked()
        self.render()
        if self.sheet:
            try:
                path = export_sheet(self.sheet, folder, ("jpg", "pdf")[fmt.currentIndex()])
                self.status.setText(f"Saved {path}")
            except Exception as exc:
                LOG.exception("Forene export failed")
                QMessageBox.warning(self, "Export failed", str(exc))
