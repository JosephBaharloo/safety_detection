from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class AlarmOverlay(QFrame):
    """Visual overlay displayed on top of a stream when anomaly exists."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background-color: rgba(210, 25, 25, 150); border-radius: 6px;")

        self._label: QLabel = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "color: white; font-size: 16px; font-weight: 700; padding: 8px;"
        )

        layout: QVBoxLayout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._label)

        self.hide()

    @pyqtSlot(str)
    def show_message(self, message: str) -> None:
        self._label.setText(message)
        self.show()

    @pyqtSlot()
    def clear_message(self) -> None:
        self.hide()
