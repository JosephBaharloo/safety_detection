from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget


class AlarmPanel(QWidget):
    """Displays global alarm state and event history."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._status_label: QLabel = QLabel("System status: Idle")
        self._status_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #1f7a3e;")

        self._detail_label: QLabel = QLabel("No active alerts")
        self._detail_label.setWordWrap(True)

        self._clear_button: QPushButton = QPushButton("Clear History")
        self._clear_button.clicked.connect(self._clear_history)

        self._history_list: QListWidget = QListWidget()
        self._history_list.setAlternatingRowColors(True)

        layout: QVBoxLayout = QVBoxLayout(self)
        layout.addWidget(self._status_label)
        layout.addWidget(self._detail_label)
        layout.addWidget(self._clear_button)
        layout.addWidget(self._history_list, stretch=1)

    @pyqtSlot(bool, str)
    def set_alarm_state(self, active: bool, message: str) -> None:
        if active:
            self._status_label.setText("System status: ALERT")
            self._status_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #b71c1c;")
        else:
            self._status_label.setText("System status: Normal")
            self._status_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #1f7a3e;")

        self._detail_label.setText(message)
        self.append_event(message)

    @pyqtSlot(str)
    def append_event(self, message: str) -> None:
        timestamp: str = datetime.now().strftime("%H:%M:%S")
        self._history_list.insertItem(0, f"[{timestamp}] {message}")
        if self._history_list.count() > 200:
            self._history_list.takeItem(self._history_list.count() - 1)
