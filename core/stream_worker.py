from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event
from typing import Callable

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config.settings import StreamConfig
from core.detector import Detection, DetectorStrategy
from core.event_bus import EventBus
from utils.logger import get_logger
from utils.video_source import VideoSource

LOGGER = get_logger(__name__)

ANOMALY_DETECTED_TOPIC = "anomaly.detected"
ANOMALY_CLEARED_TOPIC = "anomaly.cleared"
STREAM_STARTED_TOPIC = "stream.started"
STREAM_STOPPED_TOPIC = "stream.stopped"
STREAM_ERROR_TOPIC = "stream.error"

# Target display frame rate — keeps CPU usage reasonable
_TARGET_FPS: float = 30.0
_FRAME_INTERVAL: float = 1.0 / _TARGET_FPS

# Label that must appear in detections before equipment checks run.
# Adjust to match your model's class name for a person/worker.
_PERSON_LABELS: frozenset[str] = frozenset({"person", "worker", "human"})


@dataclass(frozen=True)
class AnomalyEvent:
    stream_id: str
    stream_name: str
    missing_equipment: tuple[str, ...]
    observed_equipment: tuple[str, ...]
    timestamp_utc: str


@dataclass(frozen=True)
class StreamStateEvent:
    stream_id: str
    stream_name: str
    state: str
    message: str
    timestamp_utc: str


class StreamWorker(QThread):
    """Captures frames and runs detection for one stream on its own thread."""

    frame_ready = pyqtSignal(str, object, object)
    status_changed = pyqtSignal(str, str)
    source_failed = pyqtSignal(str, str)

    def __init__(
        self,
        stream_config: StreamConfig,
        detector: DetectorStrategy,
        event_bus: EventBus,
    ) -> None:
        super().__init__()
        self._stream_config: StreamConfig = stream_config
        self._detector: DetectorStrategy = detector
        self._event_bus: EventBus = event_bus
        self._stop_requested: Event = Event()
        # FIX: track previous anomaly state to publish only on transitions
        self._anomaly_active: bool = False

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._stop_requested.clear()
        source: VideoSource = VideoSource(self._stream_config.source)

        if not source.open():
            message: str = f"Cannot open source: {self._stream_config.source}"
            self.source_failed.emit(self._stream_config.stream_id, message)
            self.status_changed.emit(self._stream_config.stream_id, "error")
            self._event_bus.publish(
                STREAM_ERROR_TOPIC,
                self._build_state_event("error", message),
            )
            return

        self.status_changed.emit(self._stream_config.stream_id, "running")
        self._event_bus.publish(
            STREAM_STARTED_TOPIC,
            self._build_state_event("running", "stream started"),
        )

        try:
            while not self._stop_requested.is_set():
                t0: float = time.monotonic()

                frame: np.ndarray | None = source.read()
                if frame is None:
                    # Source temporarily empty — short sleep and retry
                    time.sleep(0.01)
                    continue

                detections: list[Detection] = self._detector.detect(frame)
                self.frame_ready.emit(self._stream_config.stream_id, frame, detections)

                self._evaluate_anomaly(detections)

                # FIX: throttle loop to ~TARGET_FPS to avoid 100 % CPU usage
                elapsed: float = time.monotonic() - t0
                sleep_for: float = _FRAME_INTERVAL - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "Unexpected worker error for %s", self._stream_config.stream_id
            )
            self.status_changed.emit(self._stream_config.stream_id, "error")
            self._event_bus.publish(
                STREAM_ERROR_TOPIC,
                self._build_state_event("error", str(exc)),
            )
        finally:
            source.release()
            self.status_changed.emit(self._stream_config.stream_id, "stopped")
            self._event_bus.publish(
                STREAM_STOPPED_TOPIC,
                self._build_state_event("stopped", "stream stopped"),
            )

    def stop(self) -> None:
        self._stop_requested.set()

    # ------------------------------------------------------------------
    # Anomaly evaluation — publish only on state transitions
    # ------------------------------------------------------------------

    def _evaluate_anomaly(self, detections: list[Detection]) -> None:
        """Publish anomaly events only when the alarm state actually changes.

        Previous implementation published on *every frame* that had missing
        equipment, flooding the event bus and the alarm manager.  The fix:
        check the previous state and publish only when it flips.
        """
        missing: tuple[str, ...] = self._get_missing_equipment(detections)
        observed: tuple[str, ...] = tuple(sorted({d.label for d in detections}))

        if missing:
            if not self._anomaly_active:
                # TRANSITION: compliant → anomaly  — publish once
                self._anomaly_active = True
                self._event_bus.publish(
                    ANOMALY_DETECTED_TOPIC,
                    AnomalyEvent(
                        stream_id=self._stream_config.stream_id,
                        stream_name=self._stream_config.display_name,
                        missing_equipment=missing,
                        observed_equipment=observed,
                        timestamp_utc=self._timestamp_utc(),
                    ),
                )
        else:
            if self._anomaly_active:
                # TRANSITION: anomaly → compliant — publish once
                self._anomaly_active = False
                self._event_bus.publish(
                    ANOMALY_CLEARED_TOPIC,
                    AnomalyEvent(
                        stream_id=self._stream_config.stream_id,
                        stream_name=self._stream_config.display_name,
                        missing_equipment=(),
                        observed_equipment=observed,
                        timestamp_utc=self._timestamp_utc(),
                    ),
                )

    def _get_missing_equipment(self, detections: list[Detection]) -> tuple[str, ...]:
        """Return required labels absent from detections.

        FIX: if no person is present in the frame, return an empty tuple —
        there is nothing to enforce PPE rules against, so no alarm should fire.
        """
        # Guard: only run equipment checks when at least one person is visible
        person_present: bool = any(
            d.label.lower() in _PERSON_LABELS for d in detections
        )
        if not person_present:
            return ()

        observed_labels: set[str] = {d.label for d in detections}
        return tuple(
            required
            for required in self._stream_config.required_equipment
            if required not in observed_labels
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_state_event(self, state: str, message: str) -> StreamStateEvent:
        return StreamStateEvent(
            stream_id=self._stream_config.stream_id,
            stream_name=self._stream_config.display_name,
            state=state,
            message=message,
            timestamp_utc=self._timestamp_utc(),
        )

    @staticmethod
    def _timestamp_utc() -> str:
        return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

DetectorBuilder = Callable[[StreamConfig], DetectorStrategy]


class StreamWorkerFactory:
    """Factory pattern for creating stream workers with swappable detectors."""

    def __init__(self, event_bus: EventBus, detector_builder: DetectorBuilder) -> None:
        self._event_bus: EventBus = event_bus
        self._detector_builder: DetectorBuilder = detector_builder

    def create(self, stream_config: StreamConfig) -> StreamWorker:
        detector: DetectorStrategy = self._detector_builder(stream_config)
        return StreamWorker(
            stream_config=stream_config,
            detector=detector,
            event_bus=self._event_bus,
        )