"""pose_detector.py — YOLOv8-pose based fall / faint detection (FR-2.2).

Strategy pattern implementation that mirrors detector.py so it can be
swapped into StreamWorker or run in parallel alongside the PPE detector.

Algorithm
---------
1. Run YOLOv8-pose inference on the frame → get skeleton keypoints per person.
2. For each detected person compute two signals:
   - **Aspect ratio** of the bounding box (width / height).
     A standing person has ratio < 1.  A fallen person has ratio ≥ 1.
   - **Vertical keypoint spread** (shoulder–hip–knee column height
     relative to bbox height).  A collapsed posture has a very small
     vertical spread even when the bbox is wide.
3. Combine the two signals with configurable thresholds to emit a
   ``PoseDetection`` result whose ``fallen`` flag is True when a fall
   or faint is suspected.

This is a lightweight heuristic — no secondary classifier is needed.
It works reliably for clear camera angles and can be tuned via
``aspect_ratio_threshold`` and ``vertical_spread_threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# COCO keypoint indices used for the heuristic
_KP_LEFT_SHOULDER: int = 5
_KP_RIGHT_SHOULDER: int = 6
_KP_LEFT_HIP: int = 11
_KP_RIGHT_HIP: int = 12
_KP_LEFT_KNEE: int = 13
_KP_RIGHT_KNEE: int = 14

# Minimum confidence for a keypoint to be considered visible
_KP_CONFIDENCE_MIN: float = 0.3

# Default thresholds (can be overridden via settings)
_DEFAULT_ASPECT_RATIO_THRESHOLD: float = 1.1   # bbox width/height ≥ this → suspicious
_DEFAULT_VERTICAL_SPREAD_THRESHOLD: float = 0.55  # relative vertical span < this → suspicious
_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.40
_DEFAULT_IOU_THRESHOLD: float = 0.45
_DEFAULT_IMAGE_SIZE: int = 640


@dataclass(frozen=True)
class PoseDetection:
    """Result for a single detected person."""

    # Bounding box in pixel coordinates (x1, y1, x2, y2)
    bbox: tuple[int, int, int, int]
    # Confidence score of the person detection
    confidence: float
    # True when the heuristic suspects a fall or faint
    fallen: bool
    # Raw signals for debugging / logging
    aspect_ratio: float
    vertical_spread: float


@runtime_checkable
class PoseDetectorStrategy(Protocol):
    def detect(self, frame: np.ndarray) -> list[PoseDetection]:
        ...


class NullPoseDetector:
    """Fallback when YOLOv8-pose model cannot be loaded."""

    def detect(self, frame: np.ndarray) -> list[PoseDetection]:
        _ = frame
        return []


class YoloV8PoseDetector:
    """YOLOv8-pose detector implementing the PoseDetectorStrategy protocol.

    Parameters
    ----------
    model_path:
        Path to ``yolov8n-pose.pt`` (or any *-pose variant).
    aspect_ratio_threshold:
        bbox width/height ratio above which the pose is considered horizontal.
    vertical_spread_threshold:
        Fraction of bbox height covered by the shoulder→knee keypoint column.
        Values below this indicate a collapsed posture.
    confidence_threshold:
        Minimum detection confidence to keep a bounding box.
    iou_threshold:
        NMS IoU threshold.
    image_size:
        Inference image size (pixels).
    """

    def __init__(
        self,
        model_path: str,
        aspect_ratio_threshold: float = _DEFAULT_ASPECT_RATIO_THRESHOLD,
        vertical_spread_threshold: float = _DEFAULT_VERTICAL_SPREAD_THRESHOLD,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = _DEFAULT_IOU_THRESHOLD,
        image_size: int = _DEFAULT_IMAGE_SIZE,
    ) -> None:
        from ultralytics import YOLO  # lazy import — keeps startup fast

        self._model: YOLO = YOLO(model_path)
        self._aspect_ratio_threshold: float = aspect_ratio_threshold
        self._vertical_spread_threshold: float = vertical_spread_threshold
        self._confidence_threshold: float = confidence_threshold
        self._iou_threshold: float = iou_threshold
        self._image_size: int = image_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[PoseDetection]:
        """Run pose inference and return one PoseDetection per visible person."""
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
        keypoints_data = result.keypoints  # shape: (N, 17, 3) — x, y, conf

        if boxes is None or boxes.xyxy is None:
            return []

        coords: np.ndarray = boxes.xyxy.cpu().numpy()        # (N, 4)
        confidences: np.ndarray = boxes.conf.cpu().numpy()   # (N,)

        # keypoints may be absent if the model does not support them
        kp_array: np.ndarray | None = (
            keypoints_data.data.cpu().numpy()  # (N, 17, 3)
            if keypoints_data is not None
            else None
        )

        detections: list[PoseDetection] = []
        for idx in range(len(coords)):
            x1, y1, x2, y2 = coords[idx].tolist()
            bbox = (int(x1), int(y1), int(x2), int(y2))
            conf = float(confidences[idx])

            kp_row: np.ndarray | None = kp_array[idx] if kp_array is not None else None
            aspect_ratio, vertical_spread = self._compute_signals(bbox, kp_row)
            fallen: bool = self._is_fallen(aspect_ratio, vertical_spread)

            detections.append(
                PoseDetection(
                    bbox=bbox,
                    confidence=conf,
                    fallen=fallen,
                    aspect_ratio=aspect_ratio,
                    vertical_spread=vertical_spread,
                )
            )

        return detections

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------

    def _compute_signals(
        self,
        bbox: tuple[int, int, int, int],
        keypoints: np.ndarray | None,  # shape (17, 3): x, y, confidence
    ) -> tuple[float, float]:
        """Return (aspect_ratio, vertical_spread) for one person."""
        x1, y1, x2, y2 = bbox
        bbox_w: float = max(float(x2 - x1), 1.0)
        bbox_h: float = max(float(y2 - y1), 1.0)

        aspect_ratio: float = bbox_w / bbox_h

        if keypoints is None:
            # No keypoints available — fall back to aspect ratio only
            return aspect_ratio, 1.0

        # Collect y-coordinates of reliable torso/leg keypoints
        indices: list[int] = [
            _KP_LEFT_SHOULDER, _KP_RIGHT_SHOULDER,
            _KP_LEFT_HIP, _KP_RIGHT_HIP,
            _KP_LEFT_KNEE, _KP_RIGHT_KNEE,
        ]
        y_values: list[float] = []
        for kp_idx in indices:
            if kp_idx < len(keypoints):
                kp_x, kp_y, kp_conf = keypoints[kp_idx]
                if float(kp_conf) >= _KP_CONFIDENCE_MIN:
                    y_values.append(float(kp_y))

        if len(y_values) < 2:
            # Too few visible keypoints — trust aspect ratio only
            return aspect_ratio, 1.0

        keypoint_vertical_span: float = max(y_values) - min(y_values)
        vertical_spread: float = keypoint_vertical_span / bbox_h

        return aspect_ratio, vertical_spread

    def _is_fallen(self, aspect_ratio: float, vertical_spread: float) -> bool:
        """Apply combined threshold rule.

        A person is considered fallen when the bounding box is wider than
        tall AND the keypoint column is compressed — both signals must fire
        to reduce false positives (e.g., someone leaning close to the camera).
        """
        ratio_positive: bool = aspect_ratio >= self._aspect_ratio_threshold
        spread_positive: bool = vertical_spread < self._vertical_spread_threshold
        return ratio_positive and spread_positive