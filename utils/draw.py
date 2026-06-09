from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from core.detector import Detection

_COLOR_PERSON: tuple[int, int, int] = (220, 180, 0)   # blue-ish
_COLOR_HARDHAT: tuple[int, int, int] = (0, 200, 0)    # green
_COLOR_VEST: tuple[int, int, int] = (0, 180, 220)     # yellow/cyan
_COLOR_FALL: tuple[int, int, int] = (0, 50, 220)      # red
_COLOR_UNKNOWN: tuple[int, int, int] = (180, 180, 180) # gray

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_THICKNESS = 2

_PERSON_LABELS: frozenset[str] = frozenset({"person"})
_FALL_LABELS: frozenset[str] = frozenset({"fall-detected", "fall_detected"})


def _normalize_label(label: str) -> str:
    return label.strip().replace(" ", "_").lower()


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    missing_equipment: Sequence[str] = (),
) -> np.ndarray:
    output: np.ndarray = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        normalized_label = _normalize_label(detection.label)

        if normalized_label in _PERSON_LABELS:
            color = _COLOR_PERSON
        elif normalized_label in {"hardhat"}:
            color = _COLOR_HARDHAT
        elif normalized_label in {"safety_vest"}:
            color = _COLOR_VEST
        elif normalized_label in _FALL_LABELS:
            color = _COLOR_FALL
        else:
            color = _COLOR_UNKNOWN

        cv2.rectangle(output, (x1, y1), (x2, y2), color, _THICKNESS)

        # Display confidence as percentage (no decimals)
        label_text: str = f"{detection.label} {detection.confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, _FONT, _FONT_SCALE, _THICKNESS)
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