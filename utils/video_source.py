from __future__ import annotations

import time
from typing import Final

import cv2
import numpy as np

from utils.logger import get_logger

LOGGER = get_logger(__name__)

# How many consecutive failed reads before a reconnect is attempted
_MAX_FAILED_READS: int = 30
# Seconds to wait between reconnect attempts
_RECONNECT_DELAY: float = 3.0
# Maximum reconnect attempts before giving up (0 = unlimited)
_MAX_RECONNECT_ATTEMPTS: int = 10


class VideoSource:
    """OpenCV capture wrapper with automatic reconnect support (NFR-3.1).

    For webcam (int) or file sources, reconnect is attempted but usually
    not useful.  For RTSP streams, reconnect retries up to
    _MAX_RECONNECT_ATTEMPTS times with a _RECONNECT_DELAY between each try.
    """

    def __init__(
        self,
        source: str | int,
        max_failed_reads: int = _MAX_FAILED_READS,
        reconnect_delay: float = _RECONNECT_DELAY,
        max_reconnect_attempts: int = _MAX_RECONNECT_ATTEMPTS,
    ) -> None:
        self._source: Final[str | int] = source
        self._capture: cv2.VideoCapture | None = None
        self._max_failed_reads: int = max_failed_reads
        self._reconnect_delay: float = reconnect_delay
        self._max_reconnect_attempts: int = max_reconnect_attempts
        self._failed_reads: int = 0
        self._reconnect_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the video source. Returns True on success."""
        self._capture = cv2.VideoCapture(self._source)
        is_open: bool = bool(self._capture.isOpened())
        if not is_open:
            LOGGER.error("Unable to open video source: %s", self._source)
        else:
            LOGGER.info("Video source opened: %s", self._source)
            self._failed_reads = 0
            self._reconnect_count = 0
        return is_open

    def read(self) -> np.ndarray | None:
        """Read the next frame.

        Returns None when:
        - The capture is not open.
        - A single read fails (caller should retry next cycle).
        - Reconnect failed after _MAX_RECONNECT_ATTEMPTS tries.
        """
        if self._capture is None:
            return None

        success: bool
        frame: np.ndarray
        success, frame = self._capture.read()

        if success:
            self._failed_reads = 0  # reset counter on good read
            return frame

        # Failed read
        self._failed_reads += 1
        LOGGER.warning(
            "Failed read #%d from source: %s",
            self._failed_reads,
            self._source,
        )

        if self._failed_reads >= self._max_failed_reads:
            LOGGER.warning(
                "Too many failed reads (%d) — attempting reconnect for %s",
                self._failed_reads,
                self._source,
            )
            reconnected: bool = self._reconnect()
            if not reconnected:
                return None  # caller (StreamWorker) will handle error
            self._failed_reads = 0

        return None

    def release(self) -> None:
        """Release the capture device."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            LOGGER.info("Video source released: %s", self._source)

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    # ------------------------------------------------------------------
    # Reconnect logic — NFR-3.1
    # ------------------------------------------------------------------

    def _reconnect(self) -> bool:
        """Try to reopen the source after a connection loss.

        Returns True if reconnection succeeded, False if all attempts
        were exhausted.
        """
        self.release()

        while True:
            self._reconnect_count += 1

            if (
                self._max_reconnect_attempts > 0
                and self._reconnect_count > self._max_reconnect_attempts
            ):
                LOGGER.error(
                    "Reconnect failed after %d attempts for source: %s",
                    self._max_reconnect_attempts,
                    self._source,
                )
                return False

            LOGGER.info(
                "Reconnect attempt %d for source: %s",
                self._reconnect_count,
                self._source,
            )
            time.sleep(self._reconnect_delay)

            self._capture = cv2.VideoCapture(self._source)
            if self._capture.isOpened():
                LOGGER.info(
                    "Reconnected successfully to source: %s (attempt %d)",
                    self._source,
                    self._reconnect_count,
                )
                self._reconnect_count = 0
                return True

            LOGGER.warning(
                "Reconnect attempt %d failed for source: %s",
                self._reconnect_count,
                self._source,
            )