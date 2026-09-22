"""First-use guidance shown above an otherwise empty preview workspace."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class OnboardingOverlay(QFrame):
    open_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("onboardingOverlay")
        self.setStyleSheet(
            "#onboardingOverlay { background: rgba(7, 17, 31, 242); border: 1px solid #1F4D61; "
            "border-radius: 14px; } #onboardingTitle { font-size: 24px; font-weight: 700; color: #EAF7FB; } "
            "#onboardingText { color: #9CC3CF; font-size: 13px; } "
            "#onboardingOpen { background: #37D5FF; color: #06101D; border: 0; border-radius: 7px; "
            "padding: 10px 18px; font-weight: 700; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 36, 40, 36)
        layout.setSpacing(10)
        layout.addStretch(1)
        title = QLabel("Start with a floor plan")
        title.setObjectName("onboardingTitle")
        title.setWordWrap(True)
        layout.addWidget(title)
        text = QLabel(
            "Drop a PDF or image into the left panel. Spejl mirrors the drawing while "
            "preserving readable text, then shows source and result side by side."
        )
        text.setObjectName("onboardingText")
        text.setWordWrap(True)
        layout.addWidget(text)
        hint = QLabel("Tip: Vertical is the usual left-to-right floor-plan mirror.")
        hint.setObjectName("onboardingText")
        layout.addWidget(hint)
        row = QHBoxLayout()
        open_button = QPushButton("Choose a plan")
        open_button.setObjectName("onboardingOpen")
        open_button.clicked.connect(self.open_requested)
        row.addWidget(open_button)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)
