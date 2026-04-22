from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from core.detector import Detection


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    output: np.ndarray = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label_text: str = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            output,
            label_text,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    return output
