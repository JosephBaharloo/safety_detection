from __future__ import annotations

import sys

from PyQt6.QtCore import QObject, pyqtSlot
from PyQt6.QtWidgets import QApplication

from config.settings import AppSettings, StreamConfig, build_default_settings
from core.alarm_manager import AlarmManager
from core.detector import DetectorStrategy, NullDetector, YoloV8Detector
from core.event_bus import EventBus
from core.stream_worker import (
    ANOMALY_CLEARED_TOPIC,
    ANOMALY_DETECTED_TOPIC,
    STREAM_ERROR_TOPIC,
    AnomalyEvent,
    StreamStateEvent,
    StreamWorker,
    StreamWorkerFactory,
)
from gui.main_window import MainWindow
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class SafetyDetectionController(QObject):
    """Controller layer that wires model/services to GUI views."""

    def __init__(self, app: QApplication, settings: AppSettings) -> None:
        super().__init__()
        self._app: QApplication = app
        self._settings: AppSettings = settings
        self._event_bus: EventBus = EventBus()
        self._window: MainWindow = MainWindow(alarm_sound_path=self._settings.alarm.sound_path)
        self._workers: dict[str, StreamWorker] = {}
        self._is_shutting_down: bool = False

        self._alarm_manager: AlarmManager = AlarmManager(
            event_bus=self._event_bus,
            sound_path=self._settings.alarm.sound_path,
            cooldown_seconds=self._settings.alarm.cooldown_seconds,
        )
        self._alarm_manager.alarm_state_changed.connect(self._window.set_alarm_state)

        self._worker_factory: StreamWorkerFactory = StreamWorkerFactory(
            event_bus=self._event_bus,
            detector_builder=self._build_detector_for_stream,
        )

        self._wire_window()
        self._wire_event_bus()
        self._create_workers()

    def start(self) -> None:
        self._alarm_manager.start()
        self._window.show()

    @pyqtSlot()
    def start_streams(self) -> None:
        LOGGER.info("start_streams() called")
        for worker in self._workers.values():
            if not worker.isRunning():
                LOGGER.info("Starting worker: %s", worker)
                worker.start()
        self._window.append_alarm_log("Monitoring started")

    @pyqtSlot()
    def stop_streams(self) -> None:
        LOGGER.info("stop_streams() called")
        for worker in self._workers.values():
            if worker.isRunning():
                LOGGER.info("Stopping worker: %s", worker)
                worker.stop()

        for worker in self._workers.values():
            if worker.isRunning():
                LOGGER.info("Waiting for worker to finish: %s", worker)
                worker.wait(2000)

        self._window.append_alarm_log("Monitoring stopped")

    @pyqtSlot()
    def shutdown(self) -> None:
        if self._is_shutting_down:
            return

        self._is_shutting_down = True
        self.stop_streams()
        self._alarm_manager.stop()

    def _wire_window(self) -> None:
        self._window.start_requested.connect(self.start_streams)
        self._window.stop_requested.connect(self.stop_streams)
        self._window.close_requested.connect(self.shutdown)

    def _wire_event_bus(self) -> None:
        self._event_bus.subscribe(ANOMALY_DETECTED_TOPIC, self._on_anomaly_detected)
        self._event_bus.subscribe(ANOMALY_CLEARED_TOPIC, self._on_anomaly_cleared)
        self._event_bus.subscribe(STREAM_ERROR_TOPIC, self._on_stream_error)

    def _create_workers(self) -> None:
        for stream in self._settings.streams:
            self._window.register_stream(stream.stream_id, stream.display_name)

            worker: StreamWorker = self._worker_factory.create(stream)
            worker.frame_ready.connect(self._window.update_stream_frame)
            worker.status_changed.connect(self._window.update_stream_status)
            worker.source_failed.connect(self._on_source_failed)

            self._workers[stream.stream_id] = worker

    def _build_detector_for_stream(self, stream_config: StreamConfig) -> DetectorStrategy:
        _ = stream_config
        try:
            detector: DetectorStrategy = YoloV8Detector(
                model_path=str(self._settings.detector.model_path),
                class_map=self._settings.class_map,
                confidence_threshold=self._settings.detector.confidence_threshold,
                iou_threshold=self._settings.detector.iou_threshold,
                image_size=self._settings.detector.image_size,
            )
            return detector
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Falling back to NullDetector because YOLO failed to load: %s", exc)
            return NullDetector()

    def _on_anomaly_detected(self, event: AnomalyEvent) -> None:
        self._window.show_anomaly(event.stream_id, event.missing_equipment)
        message: str = (
            f"{event.stream_name}: missing {', '.join(event.missing_equipment)}"
            if event.missing_equipment
            else f"{event.stream_name}: anomaly detected"
        )
        self._window.append_alarm_log(message)

    def _on_anomaly_cleared(self, event: AnomalyEvent) -> None:
        self._window.clear_anomaly(event.stream_id)
        self._window.append_alarm_log(f"{event.stream_name}: compliance restored")

    def _on_stream_error(self, event: StreamStateEvent) -> None:
        if event.state != "error":
            return
        self._window.update_stream_status(event.stream_id, "error")
        self._window.append_alarm_log(f"{event.stream_name}: {event.message}")

    @pyqtSlot(str, str)
    def _on_source_failed(self, stream_id: str, message: str) -> None:
        self._window.update_stream_status(stream_id, "error")
        self._window.append_alarm_log(f"{stream_id}: {message}")


def main() -> int:
    app: QApplication = QApplication(sys.argv)
    settings: AppSettings = build_default_settings()

    controller: SafetyDetectionController = SafetyDetectionController(app=app, settings=settings)
    app.aboutToQuit.connect(controller.shutdown)
    controller.start()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
