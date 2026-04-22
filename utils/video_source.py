from __future__ import annotations

from typing import Final

import cv2
import numpy as np

from utils.logger import get_logger

LOGGER = get_logger(__name__)


class VideoSource:
    """Thin wrapper around OpenCV capture for cleaner worker logic."""

    def __init__(self, source: str | int) -> None:
        self._source: Final[str | int] = source
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self._capture = cv2.VideoCapture(self._source)
        is_open: bool = bool(self._capture.isOpened())
        if not is_open:
            LOGGER.error("Unable to open video source: %s", self._source)
        return is_open

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            return None

        success: bool
        frame: np.ndarray
        success, frame = self._capture.read()
        if not success:
            return None
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
