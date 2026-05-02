from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from utils.logger import get_logger

_LOGGER = get_logger(__name__)

# Default directory where anomaly screenshots are saved
_DEFAULT_SCREENSHOT_DIR: Path = Path("logs") / "screenshots"


class ScreenshotManager:
    """Captures and saves annotated frames when anomalies are detected.

    Implements FR-4.3: the system must automatically capture and save a
    screenshot of the frame in which the anomaly was detected.

    Parameters
    ----------
    screenshot_dir:
        Directory where screenshots are stored.  Created automatically.
    filename_prefix:
        Optional prefix for every file name.
    image_format:
        Output format — ``"jpg"`` (default, smaller) or ``"png"`` (lossless).
    jpeg_quality:
        JPEG quality 0–100 (only used when image_format is ``"jpg"``).
    max_screenshots:
        Maximum number of files to keep per camera.  Oldest files are
        deleted when the limit is exceeded.  ``0`` means unlimited.
    """

    def __init__(
        self,
        screenshot_dir: Path | str = _DEFAULT_SCREENSHOT_DIR,
        filename_prefix: str = "anomaly",
        image_format: str = "jpg",
        jpeg_quality: int = 85,
        max_screenshots: int = 500,
    ) -> None:
        self._screenshot_dir: Path = Path(screenshot_dir)
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._prefix: str = filename_prefix
        self._format: str = image_format.lower().lstrip(".")
        self._jpeg_quality: int = jpeg_quality
        self._max_screenshots: int = max_screenshots
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(
        self,
        frame: np.ndarray,
        stream_id: str,
        anomaly_type: str = "anomaly",
    ) -> Path | None:
        """Save *frame* to disk and return the file path.

        Parameters
        ----------
        frame:        BGR numpy array as returned by OpenCV.
        stream_id:    Camera identifier used in the filename.
        anomaly_type: Short label included in the filename for quick
                      identification (e.g. ``"missing_ppe"``).

        Returns
        -------
        Path to the saved file, or ``None`` if saving failed.
        """
        if frame is None or frame.size == 0:
            _LOGGER.warning("capture() received an empty frame for %s", stream_id)
            return None

        timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safe_id: str = stream_id.replace(" ", "_").replace("/", "-")
        safe_type: str = anomaly_type.replace(" ", "_")
        filename: str = f"{self._prefix}_{safe_id}_{safe_type}_{timestamp}.{self._format}"
        file_path: Path = self._screenshot_dir / filename

        with self._lock:
            saved: bool = self._write_image(frame, file_path)
            if saved and self._max_screenshots > 0:
                self._enforce_limit(stream_id)

        return file_path if saved else None

    @property
    def screenshot_dir(self) -> Path:
        return self._screenshot_dir

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_image(self, frame: np.ndarray, path: Path) -> bool:
        """Write frame to disk using OpenCV."""
        try:
            encode_params: list[int] = []
            if self._format in ("jpg", "jpeg"):
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            elif self._format == "png":
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

            success: bool = cv2.imwrite(str(path), frame, encode_params)
            if not success:
                _LOGGER.error("cv2.imwrite returned False for %s", path)
            return success
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("Failed to save screenshot %s: %s", path, exc)
            return False

    def _enforce_limit(self, stream_id: str) -> None:
        """Delete the oldest screenshots for this stream if limit exceeded."""
        safe_id: str = stream_id.replace(" ", "_").replace("/", "-")
        pattern: str = f"{self._prefix}_{safe_id}_*"
        files: list[Path] = sorted(
            self._screenshot_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
        )
        excess: int = len(files) - self._max_screenshots
        for old_file in files[:excess]:
            try:
                old_file.unlink()
                _LOGGER.debug("Deleted old screenshot: %s", old_file)
            except OSError as exc:
                _LOGGER.warning("Could not delete screenshot %s: %s", old_file, exc)