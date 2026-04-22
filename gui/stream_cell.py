from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QResizeEvent
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from core.detector import Detection
from gui.alarm_overlay import AlarmOverlay
from utils.draw import draw_detections


class StreamCell(QWidget):
    """Pure view widget for one stream's visual state."""

    def __init__(self, stream_id: str, display_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.stream_id: str = stream_id

        self._title_label: QLabel = QLabel(display_name)
        self._title_label.setStyleSheet("font-size: 13px; font-weight: 700;")

        self._status_label: QLabel = QLabel("Status: idle")
        self._status_label.setStyleSheet("font-size: 12px; color: #5f6368;")

        self._video_label: QLabel = QLabel("Waiting for frames...")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(320, 240)
        self._video_label.setStyleSheet("background: #101418; color: #d7dbe0; border-radius: 8px;")

        layout: QVBoxLayout = QVBoxLayout(self)
        layout.addWidget(self._title_label)
        layout.addWidget(self._status_label)
        layout.addWidget(self._video_label, stretch=1)

        self._overlay: AlarmOverlay = AlarmOverlay(self._video_label)
        self._overlay.setGeometry(self._video_label.rect())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._overlay.setGeometry(self._video_label.rect())

    @pyqtSlot(object, object)
    def update_frame(self, frame: np.ndarray, detections: Sequence[Detection]) -> None:
        rendered: np.ndarray = draw_detections(frame, detections)
        rgb_frame: np.ndarray = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)

        height: int
        width: int
        channels: int
        height, width, channels = rgb_frame.shape
        bytes_per_line: int = channels * width

        image: QImage = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        pixmap: QPixmap = QPixmap.fromImage(image.copy())

        self._video_label.setPixmap(
            pixmap.scaled(
                self._video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    @pyqtSlot(str)
    def set_status(self, status: str) -> None:
        palette: dict[str, str] = {
            "idle": "#5f6368",
            "running": "#1f7a3e",
            "stopped": "#5f6368",
            "error": "#b71c1c",
        }
        color: str = palette.get(status, "#5f6368")
        self._status_label.setText(f"Status: {status}")
        self._status_label.setStyleSheet(f"font-size: 12px; color: {color};")

    @pyqtSlot(object)
    def show_anomaly(self, missing_items: Sequence[str]) -> None:
        if missing_items:
            self._overlay.show_message(f"Missing: {', '.join(missing_items)}")
            return
        self._overlay.clear_message()

    @pyqtSlot()
    def clear_anomaly(self) -> None:
        self._overlay.clear_message()
