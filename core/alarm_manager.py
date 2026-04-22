from __future__ import annotations

from pathlib import Path
from time import monotonic

from PyQt6.QtCore import QObject, pyqtSignal
from pygame import mixer

from core.event_bus import EventBus
from core.stream_worker import ANOMALY_CLEARED_TOPIC, ANOMALY_DETECTED_TOPIC, AnomalyEvent
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class AlarmManager(QObject):
    """Consumes anomaly events and controls alarm playback state."""

    alarm_state_changed = pyqtSignal(bool, str)

    def __init__(self, event_bus: EventBus, sound_path: Path, cooldown_seconds: float) -> None:
        super().__init__()
        self._event_bus: EventBus = event_bus
        self._sound_path: Path = sound_path
        self._cooldown_seconds: float = cooldown_seconds
        self._active_streams: set[str] = set()
        self._last_play_time: float = 0.0
        self._mixer_ready: bool = False

    def start(self) -> None:
        self._event_bus.subscribe(ANOMALY_DETECTED_TOPIC, self._on_anomaly_detected)
        self._event_bus.subscribe(ANOMALY_CLEARED_TOPIC, self._on_anomaly_cleared)

    def stop(self) -> None:
        self._event_bus.unsubscribe(ANOMALY_DETECTED_TOPIC, self._on_anomaly_detected)
        self._event_bus.unsubscribe(ANOMALY_CLEARED_TOPIC, self._on_anomaly_cleared)
        if self._mixer_ready:
            mixer.music.stop()
            mixer.quit()
            self._mixer_ready = False

    def _on_anomaly_detected(self, event: AnomalyEvent) -> None:
        self._active_streams.add(event.stream_id)
        message: str = (
            f"ALERT: {event.stream_name} missing {', '.join(event.missing_equipment)}"
            if event.missing_equipment
            else f"ALERT: {event.stream_name}"
        )
        self.alarm_state_changed.emit(True, message)
        self._play_alarm_if_needed()

    def _on_anomaly_cleared(self, event: AnomalyEvent) -> None:
        if event.stream_id in self._active_streams:
            self._active_streams.remove(event.stream_id)

        if not self._active_streams:
            if self._mixer_ready:
                mixer.music.stop()
            self.alarm_state_changed.emit(False, "All streams compliant")

    def _play_alarm_if_needed(self) -> None:
        if not self._ensure_mixer_ready():
            return

        now: float = monotonic()
        if now - self._last_play_time < self._cooldown_seconds:
            return

        self._last_play_time = now
        if not mixer.music.get_busy():
            mixer.music.play(-1)

    def _ensure_mixer_ready(self) -> bool:
        if self._mixer_ready:
            return True

        if not self._sound_path.exists():
            LOGGER.warning("Alarm sound file not found: %s", self._sound_path)
            return False

        try:
            mixer.init()
            mixer.music.load(str(self._sound_path))
            self._mixer_ready = True
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Could not initialize alarm mixer: %s", exc)
            return False
