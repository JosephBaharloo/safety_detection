from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from core.event_bus import EventBus
from core.stream_worker import ANOMALY_CLEARED_TOPIC, ANOMALY_DETECTED_TOPIC, AnomalyEvent
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class AlarmManager(QObject):
    """Consumes anomaly events and emits alarm state changes."""

    alarm_state_changed = pyqtSignal(bool, str)

    def __init__(self, event_bus: EventBus, sound_path: Path, cooldown_seconds: float) -> None:
        super().__init__()
        self._event_bus: EventBus = event_bus
        self._sound_path: Path = sound_path
        self._cooldown_seconds: float = cooldown_seconds
        self._active_streams: set[str] = set()

    def start(self) -> None:
        self._event_bus.subscribe(ANOMALY_DETECTED_TOPIC, self._on_anomaly_detected)
        self._event_bus.subscribe(ANOMALY_CLEARED_TOPIC, self._on_anomaly_cleared)

    def stop(self) -> None:
        self._event_bus.unsubscribe(ANOMALY_DETECTED_TOPIC, self._on_anomaly_detected)
        self._event_bus.unsubscribe(ANOMALY_CLEARED_TOPIC, self._on_anomaly_cleared)

    def _on_anomaly_detected(self, event: AnomalyEvent) -> None:
        self._active_streams.add(event.stream_id)
        message: str = (
            f"ALERT: {event.stream_name} missing {', '.join(event.missing_equipment)}"
            if event.missing_equipment
            else f"ALERT: {event.stream_name}"
        )
        self.alarm_state_changed.emit(True, message)

    def _on_anomaly_cleared(self, event: AnomalyEvent) -> None:
        if event.stream_id in self._active_streams:
            self._active_streams.remove(event.stream_id)

        if not self._active_streams:
            self.alarm_state_changed.emit(False, "All streams compliant")
