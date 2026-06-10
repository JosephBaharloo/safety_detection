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

_MISSING_DEBOUNCE_FRAMES: int = 8
_FALL_DEBOUNCE_FRAMES: int = 2
_CLEAR_DEBOUNCE_FRAMES: int = 10

_PERSON_LABELS: frozenset[str] = frozenset({"Person", "person"})
_HARDHAT_LABELS: frozenset[str] = frozenset({"Hardhat", "hardhat"})
_VEST_LABELS: frozenset[str] = frozenset({"Safety_Vest", "safety_vest", "Safety Vest", "safety vest"})
_FALL_LABELS: frozenset[str] = frozenset({"Fall-Detected", "fall-detected", "Fall_Detected"})


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
        self._missing_streak: int = 0
        self._fall_streak: int = 0
        self._clear_streak: int = 0
        self._last_missing: tuple[str, ...] = ()

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

        # Use the video's native FPS so playback matches real-time speed.
        native_fps: float = source.fps
        frame_interval: float = 1.0 / native_fps
        LOGGER.info("Source FPS=%.1f  frame_interval=%.4fs for %s", native_fps, frame_interval, self._stream_config.source)

        try:
            frame_count: int = 0
            wall_start: float = time.monotonic()
            video_frame_index: int = 0  # counts every frame (including skipped)

            while not self._stop_requested.is_set():
                # --- determine how many frames the video *should* have
                # advanced by now, based on wall-clock time ---------------
                wall_elapsed: float = time.monotonic() - wall_start
                target_frame: int = int(wall_elapsed * native_fps)

                # Skip (grab without decode) frames we've fallen behind on
                frames_behind: int = target_frame - video_frame_index
                if frames_behind > 1:
                    skips: int = frames_behind - 1  # keep 1 to actually decode
                    for _ in range(skips):
                        if not source.grab():
                            break
                        video_frame_index += 1

                # Read & decode the current frame
                frame: np.ndarray | None = source.read()
                if frame is None:
                    time.sleep(0.005)
                    continue
                video_frame_index += 1

                detections: list[Detection] = self._detector.detect(frame)
                frame_count += 1
                if frame_count % 30 == 0:
                    LOGGER.debug("Frame %d: %d detections found", frame_count, len(detections))

                self.frame_ready.emit(self._stream_config.stream_id, frame, detections)
                self._evaluate_anomaly(detections)

                # Sleep only if we're ahead of real-time
                next_frame_time: float = wall_start + video_frame_index * frame_interval
                sleep_for: float = next_frame_time - time.monotonic()
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
        detected_labels = {d.label for d in detections}
        observed: tuple[str, ...] = tuple(sorted(detected_labels))

        is_person_present = any(label in _PERSON_LABELS for label in detected_labels)
        has_hardhat = any(label in _HARDHAT_LABELS for label in detected_labels)
        has_vest = any(label in _VEST_LABELS for label in detected_labels)
        has_fall = any(label in _FALL_LABELS for label in detected_labels)

        missing_items: list[str] = []
        if is_person_present:
            if not has_hardhat:
                missing_items.append("Hardhat")
            if not has_vest:
                missing_items.append("Safety_Vest")

        fall_missing: tuple[str, ...] = ()
        if is_person_present and has_fall:
            fall_missing = ("Fall-Detected",)

        missing: tuple[str, ...] = tuple(missing_items) or fall_missing

        if fall_missing:
            self._missing_streak = 0
            self._last_missing = fall_missing
            self._fall_streak += 1
            if not self._anomaly_active and self._fall_streak >= _FALL_DEBOUNCE_FRAMES:
                self._anomaly_active = True
                LOGGER.warning("%s: ANOMALY DETECTED - violations: %s", self._stream_config.stream_id, fall_missing)
                self._event_bus.publish(
                    ANOMALY_DETECTED_TOPIC,
                    AnomalyEvent(
                        stream_id=self._stream_config.stream_id,
                        stream_name=self._stream_config.display_name,
                        missing_equipment=fall_missing,
                        observed_equipment=observed,
                        timestamp_utc=self._timestamp_utc(),
                    ),
                )
            return

        self._fall_streak = 0

        if missing:
            self._clear_streak = 0
            if missing == self._last_missing:
                self._missing_streak += 1
            else:
                self._last_missing = missing
                self._missing_streak = 1

            if not self._anomaly_active and self._missing_streak >= _MISSING_DEBOUNCE_FRAMES:
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
            return

        self._missing_streak = 0
        self._last_missing = ()
        if self._anomaly_active:
            self._clear_streak += 1
            if self._clear_streak >= _CLEAR_DEBOUNCE_FRAMES:
                self._anomaly_active = False
                self._clear_streak = 0
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