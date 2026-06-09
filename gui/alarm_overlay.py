from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QLabel, QWidget

class AlarmOverlay(QLabel):
    """Visual overlay displayed on top of a stream when anomaly exists."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        # Click-through transparency
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        
        # Center the text
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Apply ALL styling directly to the label. No cascading possible.
        self.setStyleSheet(
            "QLabel {"
            "  background-color: rgba(210, 25, 25, 220);"
            "  border-radius: 6px;"
            "  color: white;"
            "  font-size: 14px;"
            "  font-weight: 700;"
            "  padding: 6px 10px;" 
            "}"
        )

        self.hide()

    @pyqtSlot(str)
    def show_message(self, message: str) -> None:
        self.setText(message)
        self.adjustSize()
        self.show()

    @pyqtSlot()
    def clear_message(self) -> None:
        self.hide()