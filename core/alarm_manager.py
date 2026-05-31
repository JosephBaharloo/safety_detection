from __future__ import annotations

from pathlib import Path
from time import monotonic

from PyQt6.QtCore import QObject, pyqtSignal

from core.event_bus import EventBus
from core.stream_worker import ANOMALY_CLEARED_TOPIC, ANOMALY_DETECTED_TOPIC, AnomalyEvent
from utils.logger import EventLogWriter, get_logger
from utils.screenshot import ScreenshotManager
from core.stream_worker import ANOMALY_CLEARED_TOPIC, ANOMALY_DETECTED_TOPIC, STREAM_ERROR_TOPIC, AnomalyEvent, StreamStateEvent

LOGGER = get_logger(__name__)

# Default directories — created automatically at runtime
_DEFAULT_LOG_DIR: Path = Path("logs")
_DEFAULT_SCREENSHOT_DIR: Path = Path("logs") / "screenshots"


class AlarmManager(QObject):
    """Consumes anomaly events, emits alarm state changes, and persists records.

    Responsibilities
    ----------------
    - Track which streams are currently in anomaly state
    - Emit alarm_state_changed signal to the GUI
    - Write every anomaly transition to CSV + JSON log (FR-4.2)
    - Capture and save a screenshot for each new anomaly (FR-4.3)
    - Enforce a cooldown so the same camera cannot re-trigger within
      cooldown_seconds (FR-3.4)
    """

    alarm_state_changed = pyqtSignal(bool, str)

    def __init__(
        self,
        event_bus: EventBus,
        sound_path: Path,
        cooldown_seconds: float,
        log_dir: Path | None = None,
        screenshot_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self._event_bus: EventBus = event_bus
        self._sound_path: Path = sound_path
        self._cooldown_seconds: float = cooldown_seconds
        self._active_streams: set[str] = set()

        # Cooldown tracking: stream_id → last alert timestamp
        self._last_alert_time: dict[str, float] = {}

        # FR-4.2 — structured event log
        self._event_log: EventLogWriter = EventLogWriter(
            log_dir=log_dir or _DEFAULT_LOG_DIR,
            base_name="events",
            write_json=True,
        )
        LOGGER.info("Event log → %s", self._event_log.csv_path)

        # FR-4.3 — anomaly frame capture
        self._screenshot_manager: ScreenshotManager = ScreenshotManager(
            screenshot_dir=screenshot_dir or _DEFAULT_SCREENSHOT_DIR,
        )
        LOGGER.info("Screenshots → %s", self._screenshot_manager.screenshot_dir)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._event_bus.subscribe(ANOMALY_DETECTED_TOPIC, self._on_anomaly_detected)
        self._event_bus.subscribe(ANOMALY_CLEARED_TOPIC, self._on_anomaly_cleared)
        self._event_bus.subscribe(STREAM_ERROR_TOPIC, self._on_stream_error)
        LOGGER.info("AlarmManager started - subscribed to anomaly and stream error topics")

    def stop(self) -> None:
        self._event_bus.unsubscribe(ANOMALY_DETECTED_TOPIC, self._on_anomaly_detected)
        self._event_bus.unsubscribe(ANOMALY_CLEARED_TOPIC, self._on_anomaly_cleared)
        self._event_bus.unsubscribe(STREAM_ERROR_TOPIC, self._on_stream_error)
        LOGGER.info("AlarmManager stopped")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_anomaly_detected(self, event: AnomalyEvent) -> None:
        self._active_streams.add(event.stream_id)

        # FR-3.4 — cooldown check
        now: float = monotonic()
        last: float = self._last_alert_time.get(event.stream_id, 0.0)
        within_cooldown: bool = (now - last) < self._cooldown_seconds

        message: str = (
            f"ALERT: {event.stream_name} missing {', '.join(event.missing_equipment)}"
            if event.missing_equipment
            else f"ALERT: {event.stream_name}"
        )
        self.alarm_state_changed.emit(True, message)

        if not within_cooldown:
            self._last_alert_time[event.stream_id] = now

            # FR-4.3 — save screenshot if frame is attached to the event
            screenshot_path: str = ""
            frame = getattr(event, "frame", None)
            if frame is not None:
                saved = self._screenshot_manager.capture(
                    frame=frame,
                    stream_id=event.stream_id,
                    anomaly_type="missing_ppe",
                )
                screenshot_path = str(saved) if saved else ""

            # FR-4.2 — write to log files
            self._event_log.log_anomaly(
                camera_id=event.stream_id,
                camera_name=event.stream_name,
                anomaly_type="missing_ppe",
                missing_equipment=event.missing_equipment,
                observed_equipment=event.observed_equipment,
                confidence=0.0,
                screenshot_path=screenshot_path or None,
            )

    def _on_anomaly_cleared(self, event: AnomalyEvent) -> None:
        if event.stream_id in self._active_streams:
            self._active_streams.remove(event.stream_id)

        if not self._active_streams:
            self.alarm_state_changed.emit(False, "All streams compliant")

        # Log the cleared state
        self._event_log.log_state_change(
            camera_id=event.stream_id,
            camera_name=event.stream_name,
            state="anomaly_cleared",
            message="compliance restored",
        )

    def _on_stream_error(self, event: StreamStateEvent) -> None:
        message = f"ALERT: {event.stream_name} — {event.message}"
        self.alarm_state_changed.emit(True, message)
        LOGGER.warning("Stream error: %s", message)