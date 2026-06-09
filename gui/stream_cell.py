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
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class StreamCell(QWidget):
    """Pure view widget for one stream's visual state."""

    def __init__(self, stream_id: str, display_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.stream_id: str = stream_id

        # Tracks current missing equipment so draw_detections can color boxes
        self._missing_equipment: tuple[str, ...] = ()

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
        self._overlay.move(12, 12)
        self._overlay.raise_()

        self._normal_video_style: str = "background: #101418; color: #d7dbe0; border-radius: 8px;"
        self._anomaly_video_style: str = "background: #101418; color: #d7dbe0; border-radius: 8px; border: 2px solid #d21919;"

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._overlay.isVisible():
            self._overlay.move(12, 12)

    @pyqtSlot(object, object)
    def update_frame(self, frame: np.ndarray, detections: Sequence[Detection]) -> None:
        # Debug: log incoming detections and current missing equipment
        labels = [d.label for d in detections]
        LOGGER.debug("%s: update_frame - labels=%s missing=%s", self.stream_id, labels, self._missing_equipment)

        # Pass current missing_equipment so person boxes turn red on anomaly
        rendered: np.ndarray = draw_detections(frame, detections, self._missing_equipment)
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
            "idle":    "#5f6368",
            "running": "#1f7a3e",
            "stopped": "#5f6368",
            "error":   "#b71c1c",
        }
        color: str = palette.get(status, "#5f6368")
        self._status_label.setText(f"Status: {status}")
        self._status_label.setStyleSheet(f"font-size: 12px; color: {color};")

    @pyqtSlot(object, object)
    def show_anomaly(self, missing_items: Sequence[str], observed_items: Sequence[str] = ()) -> None:
        # Ensure the missing equipment state is always updated and overlay refreshed
        self._missing_equipment = tuple(missing_items)
        if missing_items:
            normalized_items = [item.strip().lower().replace(" ", "_") for item in missing_items]
            fall_detected = any(item in {"fall-detected", "fall_detected"} for item in normalized_items)
            missing_equipment = [item for item in missing_items if item.strip().lower().replace(" ", "_") not in {"fall-detected", "fall_detected"}]

            message_parts: list[str] = []
            if fall_detected:
                message_parts.append("Fall detected")
            if missing_equipment:
                message_parts.append(f"Missing: {', '.join(missing_equipment)}")

            overlay_message = " | ".join(message_parts) if message_parts else "Anomaly detected"
            self._overlay.show_message(overlay_message)
            self._video_label.setStyleSheet(self._anomaly_video_style)
            self._overlay.move(12, 12)
            # Force repaint of overlay and parent
            self._overlay.update()
            self._video_label.update()
        else:
            self._overlay.clear_message()
            self._video_label.setStyleSheet(self._normal_video_style)
            self._overlay.update()
            self._video_label.update()

    @pyqtSlot()
    def clear_anomaly(self) -> None:
        self._missing_equipment = ()
        self._overlay.clear_message()
        self._video_label.setStyleSheet(self._normal_video_style)