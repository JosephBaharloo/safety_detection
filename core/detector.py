from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


@runtime_checkable
class DetectorStrategy(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...


class NullDetector(DetectorStrategy):
    """Fallback detector used when model loading fails."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        _ = frame
        return []


class YoloV8Detector(DetectorStrategy):
    """YOLOv8-based strategy implementation."""

    def __init__(
        self,
        model_path: str,
        class_map: dict[int, str],
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        image_size: int = 640,
    ) -> None:
        from ultralytics import YOLO

        self._class_map: dict[int, str] = class_map
        self._confidence_threshold: float = confidence_threshold
        self._iou_threshold: float = iou_threshold
        self._image_size: int = image_size
        self._model: YOLO = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        predictions = self._model.predict(
            source=frame,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            imgsz=self._image_size,
            verbose=False,
        )

        if not predictions:
            return []

        result = predictions[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None or boxes.cls is None or boxes.conf is None:
            return []

        coordinates: np.ndarray = boxes.xyxy.cpu().numpy()
        class_ids: np.ndarray = boxes.cls.cpu().numpy().astype(int)
        confidences: np.ndarray = boxes.conf.cpu().numpy()

        detections: list[Detection] = []
        for index, coord in enumerate(coordinates):
            x1, y1, x2, y2 = coord.tolist()
            class_id: int = int(class_ids[index])
            label: str = self._class_map.get(class_id, str(class_id))
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidences[index]),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        return detections
