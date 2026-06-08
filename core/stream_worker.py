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

_TARGET_FPS: float = 30.0
_FRAME_INTERVAL: float = 1.0 / _TARGET_FPS

_VIOLATION_LABELS: frozenset[str] = frozenset({
    "NO-Hardhat",
    "no_safety_vest",
    "Fall-Detected",
})


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
    frame_ready = pyqtSignal(str, object, object)
    status_changed = pyqtSignal(str, str)
    source_failed = pyqtSignal(str, str)

    def __init__(self, stream_config: StreamConfig, detector: DetectorStrategy, event_bus: EventBus) -> None:
        super().__init__()
        self._stream_config = stream_config
        self._detector = detector
        self._event_bus = event_bus
        self._stop_requested: Event = Event()
        self._anomaly_active: bool = False

    def run(self) -> None:
        self._stop_requested.clear()
        LOGGER.info("StreamWorker.run() started for %s", self._stream_config.stream_id)
        source: VideoSource = VideoSource(self._stream_config.source)

        if not source.open():
            message = f"Cannot open source: {self._stream_config.source}"
            LOGGER.error("Cannot open video source: %s", self._stream_config.source)
            self.source_failed.emit(self._stream_config.stream_id, message)
            self.status_changed.emit(self._stream_config.stream_id, "error")
            self._event_bus.publish(STREAM_ERROR_TOPIC, self._build_state_event("error", message))
            return

        LOGGER.info("Video source opened successfully: %s", self._stream_config.source)
        self.status_changed.emit(self._stream_config.stream_id, "running")
        self._event_bus.publish(STREAM_STARTED_TOPIC, self._build_state_event("running", "stream started"))

        try:
            frame_count = 0
            while not self._stop_requested.is_set():
                t0 = time.monotonic()

                frame: np.ndarray | None = source.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                detections: list[Detection] = self._detector.detect(frame)
                frame_count += 1
                if frame_count % 30 == 0:
                    LOGGER.debug("Frame %d: %d detections found", frame_count, len(detections))

                self.frame_ready.emit(self._stream_config.stream_id, frame, detections)
                self._evaluate_anomaly(detections)

                elapsed = time.monotonic() - t0
                sleep_for = _FRAME_INTERVAL - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Unexpected worker error for %s", self._stream_config.stream_id)
            self.status_changed.emit(self._stream_config.stream_id, "error")
            self._event_bus.publish(STREAM_ERROR_TOPIC, self._build_state_event("error", str(exc)))
        finally:
            source.release()
            self.status_changed.emit(self._stream_config.stream_id, "stopped")
            self._event_bus.publish(STREAM_STOPPED_TOPIC, self._build_state_event("stopped", "stream stopped"))

    def stop(self) -> None:
        self._stop_requested.set()

    def _evaluate_anomaly(self, detections: list[Detection]) -> None:
        missing: tuple[str, ...] = self._get_missing_equipment(detections)
        observed: tuple[str, ...] = tuple(sorted({d.label for d in detections}))

        if missing:
            if not self._anomaly_active:
                self._anomaly_active = True
                LOGGER.warning("%s: ANOMALY DETECTED - violations: %s", self._stream_config.stream_id, missing)
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
                self._anomaly_active = False
                LOGGER.info("%s: Anomaly cleared", self._stream_config.stream_id)
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
        return tuple(d.label for d in detections if d.label in _VIOLATION_LABELS)

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


DetectorBuilder = Callable[[StreamConfig], DetectorStrategy]


class StreamWorkerFactory:
    def __init__(self, event_bus: EventBus, detector_builder: DetectorBuilder) -> None:
        self._event_bus = event_bus
        self._detector_builder = detector_builder

    def create(self, stream_config: StreamConfig) -> StreamWorker:
        detector = self._detector_builder(stream_config)
        return StreamWorker(stream_config=stream_config, detector=detector, event_bus=self._event_bus)
    def _evaluate_anomaly(self, detections: list[Detection]) -> None:
    # GEÇİCİ DEBUG
        if detections:
            print(f"[DEBUG] {self._stream_config.stream_id}: {[(d.label, round(d.confidence,2)) for d in detections]}")