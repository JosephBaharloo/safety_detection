from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from core.detector import Detection

_COLOR_NORMAL: tuple[int, int, int] = (0, 220, 0)   # green
_COLOR_ANOMALY: tuple[int, int, int] = (0, 50, 220)  # red

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_THICKNESS = 2

_KNOWN_LABELS: frozenset[str] = frozenset({
    "person", "Hardhat", "NO-Hardhat",
    "safety_vest", "no_safety_vest", "Fall-Detected",
})

_VIOLATION_LABELS: frozenset[str] = frozenset({
    "NO-Hardhat", "no_safety_vest", "Fall-Detected",
})


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    missing_equipment: Sequence[str] = (),
) -> np.ndarray:
    output: np.ndarray = frame.copy()

    for detection in detections:
        if detection.label not in _KNOWN_LABELS:
            continue

        x1, y1, x2, y2 = detection.bbox

        # Violation labels → red, everything else → green
        color = _COLOR_ANOMALY if detection.label in _VIOLATION_LABELS else _COLOR_NORMAL

        cv2.rectangle(output, (x1, y1), (x2, y2), color, _THICKNESS)

        label_text: str = f"{detection.label} {detection.confidence:0%}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, _FONT, _FONT_SCALE, _THICKNESS
        )
        label_y: int = max(text_h + 8, y1 - 4)
        cv2.rectangle(
            output,
            (x1, label_y - text_h - baseline - 4),
            (x1 + text_w + 4, label_y),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            output,
            label_text,
            (x1 + 2, label_y - baseline - 2),
            _FONT,
            _FONT_SCALE,
            (255, 255, 255),
            _THICKNESS,
            cv2.LINE_AA,
        )

    return output