"""Branded startup screen shown before the main Spejl window."""

from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QElapsedTimer, QPointF, QRectF, QTimer, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import QWidget


class StartupSplash(QWidget):
    """A five-second branded startup animation for the desktop app."""

    DURATION_MS = 5_000
    TRANSITION_MS = 900

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Dynamisk Konsept Studio")
        asset = Path(__file__).with_name("assets") / "dynamic-konsept-spinning-mark.png"
        self._mark = QPixmap(str(asset))
        self._elapsed = QElapsedTimer()
        self._exit_progress = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.update)

    def begin_exit(self) -> None:
        """Finish the preview with an app-coloured cinematic fade."""
        animation = QVariantAnimation(self)
        animation.setDuration(self.TRANSITION_MS)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        animation.valueChanged.connect(self._set_exit_progress)
        animation.start()
        self._exit_animation = animation

    def _set_exit_progress(self, value: object) -> None:
        self._exit_progress = float(value)
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if not self._elapsed.isValid():
            self._elapsed.start()
            self._timer.start()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        elapsed = self._elapsed.elapsed() % 6_000
        phase = elapsed / 6_000
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Paint the animation in a 980x640 design space, scaled to fill any
        # display while preserving the original proportions.
        design_width, design_height = 980.0, 640.0
        # Keep the logo lockup comfortably inside the display rather than
        # enlarging it until it competes with the loading copy.
        scale = min(self.width() / design_width, self.height() / design_height)
        offset_x = (self.width() - design_width * scale) / 2
        offset_y = (self.height() - design_height * scale) / 2

        backdrop = QLinearGradient(0, 0, self.width(), self.height())
        backdrop.setColorAt(0, QColor("#07111F"))
        backdrop.setColorAt(0.50, QColor("#0B1F2E"))
        backdrop.setColorAt(1, QColor("#040912"))
        painter.fillRect(self.rect(), backdrop)

        painter.translate(offset_x, offset_y)
        painter.scale(scale, scale)

        centre_x, centre_y = design_width / 2, design_height * 0.40
        glow = QRadialGradient(centre_x, centre_y, 350)
        glow.setColorAt(0, QColor(35, 124, 149, 90))
        glow.setColorAt(0.52, QColor(255, 172, 45, 28))
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), glow)

        self._paint_orbits(painter, centre_x, centre_y, phase)
        self._paint_fire(painter, centre_x, centre_y, phase)
        self._paint_lightning(painter, centre_x, centre_y, phase)
        self._paint_mark(painter, centre_x, centre_y, elapsed)
        self._paint_copy(painter, centre_y)
        self._paint_progress(painter)
        self._paint_exit_overlay(painter)

    def _paint_exit_overlay(self, painter: QPainter) -> None:
        """Draw a radial cyan-to-navy dissolve before the UI reveal."""
        if not self._exit_progress:
            return
        painter.save()
        painter.resetTransform()
        centre_x, centre_y = self.width() / 2, self.height() * 0.40
        radius = max(self.width(), self.height()) * (0.22 + 0.95 * self._exit_progress)
        bloom = QRadialGradient(centre_x, centre_y, radius)
        bloom.setColorAt(0, QColor(55, 213, 255, int(78 * (1 - self._exit_progress))))
        bloom.setColorAt(0.42, QColor(11, 31, 46, int(120 * self._exit_progress)))
        bloom.setColorAt(1, QColor(4, 9, 18, int(245 * self._exit_progress)))
        painter.fillRect(self.rect(), bloom)
        painter.fillRect(self.rect(), QColor(4, 9, 18, int(160 * self._exit_progress)))
        painter.restore()

    def _paint_orbits(self, painter: QPainter, x: float, y: float, phase: float) -> None:
        painter.save()
        painter.translate(x, y)
        painter.rotate(phase * 35)
        for radius, colour in ((188, QColor(85, 220, 255, 95)), (162, QColor(255, 198, 85, 90))):
            painter.setPen(QPen(colour, 1.2))
            painter.drawEllipse(QRectF(-radius, -radius, radius * 2, radius * 2))
        painter.restore()

    def _paint_fire(self, painter: QPainter, x: float, y: float, phase: float) -> None:
        pulse = 0.34 + 0.66 * abs(sin(phase * pi * 3))
        painter.save()
        painter.translate(x, y)
        painter.rotate(-phase * 210)
        for angle in range(0, 360, 24):
            radians = angle * pi / 180
            inner, outer = 156, 184 + 14 * pulse
            colour = QColor(255, 96 if angle % 48 else 188, 26, int(115 * pulse))
            painter.setPen(QPen(colour, 8 if angle % 48 else 11, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(
                QPointF(cos(radians) * inner, sin(radians) * inner),
                QPointF(cos(radians) * outer, sin(radians) * outer),
            )
        painter.restore()

    def _paint_lightning(self, painter: QPainter, x: float, y: float, phase: float) -> None:
        flash = max(0.0, 1.0 - min(abs(phase - point) for point in (0.18, 0.51, 0.79)) / 0.018)
        if not flash:
            return
        painter.save()
        painter.translate(x, y)
        painter.scale(0.63, 0.63)
        painter.setPen(QPen(QColor(186, 248, 255, int(255 * flash)), 3.0))
        for rotation in (-26, 154):
            painter.save()
            painter.rotate(rotation)
            bolt = QPainterPath()
            bolt.moveTo(-295, -82)
            bolt.lineTo(-160, -54)
            bolt.lineTo(-185, -5)
            bolt.lineTo(-38, -48)
            bolt.lineTo(-112, 34)
            bolt.lineTo(10, 8)
            painter.drawPath(bolt)
            painter.restore()
        painter.restore()

    def _paint_mark(self, painter: QPainter, x: float, y: float, elapsed: int) -> None:
        if self._mark.isNull():
            return
        scaled = self._mark.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        painter.save()
        painter.translate(x, y)
        painter.rotate(-(elapsed / 5_000) * 360)
        painter.drawPixmap(-scaled.width() // 2, -scaled.height() // 2, scaled)
        painter.restore()

    def _paint_copy(self, painter: QPainter, y: float) -> None:
        painter.setPen(QColor("#F3F8FB"))
        painter.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        # Keep the brand lockup below the orbit and burst effects so it stays
        # clean and readable on every screen size.
        painter.drawText(0, int(y + 216), 980, 32, Qt.AlignmentFlag.AlignHCenter, "Dynamisk Konsept")
        painter.setPen(QColor("#B7D4DD"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.drawText(0, int(y + 250), 980, 18, Qt.AlignmentFlag.AlignHCenter, "S T U D I O   I N C O R P O R A T E D")

    def _paint_progress(self, painter: QPainter) -> None:
        progress = min(1.0, self._elapsed.elapsed() / self.DURATION_MS)
        label = "P R E P A R I N G   Y O U R   W O R K S P A C E"
        painter.setPen(QColor("#6F8792"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.drawText(0, 580, 980, 18, Qt.AlignmentFlag.AlignHCenter, label)

        # A small pulse preserves a sense of loading without a bar crossing
        # the brand lockup.
        painter.setPen(Qt.PenStyle.NoPen)
        alpha = int(45 + 170 * abs(sin(progress * pi * 4)))
        painter.setBrush(QColor(55, 213, 255, alpha))
        painter.drawEllipse(QRectF(745, 584, 7, 7))
