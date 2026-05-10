from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from core.detector import Detection

# Normal detection — green
_COLOR_NORMAL: tuple[int, int, int] = (0, 220, 0)
# Anomaly / missing equipment — red
_COLOR_ANOMALY: tuple[int, int, int] = (0, 50, 220)
# Text background opacity
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_THICKNESS = 2

# Labels that indicate PPE — if detected, box stays green
_PPE_LABELS: frozenset[str] = frozenset({"helmet", "vest", "gloves", "goggles"})
# Labels that indicate a person — box turns red if PPE is missing
_PERSON_LABELS: frozenset[str] = frozenset({"person", "worker", "human"})


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    missing_equipment: Sequence[str] = (),
) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of *frame*.

    Parameters
    ----------
    frame:
        Original BGR frame from OpenCV.
    detections:
        List of Detection objects from the detector.
    missing_equipment:
        If non-empty, person boxes are drawn in red to signal a violation.
    """
    output: np.ndarray = frame.copy()
    anomaly_active: bool = len(missing_equipment) > 0

    # Only draw labels that belong to the PPE / person vocabulary
    _KNOWN_LABELS: frozenset[str] = frozenset({"person", "worker", "human", "helmet", "vest", "gloves", "goggles"})

    for detection in detections:
        if detection.label.lower() not in _KNOWN_LABELS:
            continue
        x1, y1, x2, y2 = detection.bbox

        # Choose box color
        if detection.label.lower() in _PERSON_LABELS and anomaly_active:
            color = _COLOR_ANOMALY  # red — person without required PPE
        else:
            color = _COLOR_NORMAL   # green — PPE item or compliant person

        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, _THICKNESS)

        # Build label text
        label_text: str = f"{detection.label} {detection.confidence:.2f}"

        # Draw label background for readability
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

        # Draw label text in white over colored background
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