from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget


class AlarmPanel(QWidget):
    """Displays global alarm state and event history, with audio alerts via pygame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        from pygame import mixer
        mixer.init()

        self._status_label: QLabel = QLabel("System status: Idle")
        self._status_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #1f7a3e;")

        self._detail_label: QLabel = QLabel("No active alerts")
        self._detail_label.setWordWrap(True)

        self._clear_button: QPushButton = QPushButton("Clear History")
        self._clear_button.clicked.connect(self._clear_history)

        self._history_list: QListWidget = QListWidget()
        self._history_list.setAlternatingRowColors(True)

        self._alarm_sound_available: bool = False

        layout: QVBoxLayout = QVBoxLayout(self)
        layout.addWidget(self._status_label)
        layout.addWidget(self._detail_label)
        layout.addWidget(self._clear_button)
        layout.addWidget(self._history_list, stretch=1)

    def load_alarm_sound(self, sound_path: Path | str | None) -> None:
        from pygame import mixer
        source_path = Path(sound_path)
        print(f"[ALARM] Checking: '{source_path}'")
        print(f"[ALARM] Absolute: '{source_path.resolve()}'")
        print(f"[ALARM] Exists: {source_path.exists()}")
        print(f"[ALARM] Is file: {source_path.is_file()}")
    
        if source_path.is_file():
            mixer.music.load(str(source_path.resolve()))
            self._alarm_sound_available = True
            print("[ALARM] Loaded OK")
        else:
            self._alarm_sound_available = False

    @pyqtSlot(bool, str)
    def set_alarm_state(self, active: bool, message: str) -> None:
        from pygame import mixer
        if active:
            self._status_label.setText("System status: ALERT")
            self._status_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #b71c1c;")
            if self._alarm_sound_available and not mixer.music.get_busy():
                mixer.music.play(-1)
        else:
            self._status_label.setText("System status: Normal")
            self._status_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #1f7a3e;")
            if self._alarm_sound_available and mixer.music.get_busy():
                mixer.music.stop()

        self._detail_label.setText(message)
        self.append_event(message)

    @pyqtSlot(str)
    def append_event(self, message: str) -> None:
        timestamp: str = datetime.now().strftime("%H:%M:%S")
        self._history_list.insertItem(0, f"[{timestamp}] {message}")
        if self._history_list.count() > 200:
            self._history_list.takeItem(self._history_list.count() - 1)

    def _clear_history(self) -> None:
        self._history_list.clear()