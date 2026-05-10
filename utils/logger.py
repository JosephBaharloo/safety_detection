from __future__ import annotations

import csv
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Console logger (existing functionality preserved)
# ---------------------------------------------------------------------------

_FORMATTER: logging.Formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    logger: logging.Logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler: logging.StreamHandler = logging.StreamHandler()
    handler.setFormatter(_FORMATTER)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Event log writer 
# ---------------------------------------------------------------------------

_CSV_HEADERS: list[str] = [
    "timestamp_utc",
    "camera_id",
    "camera_name",
    "anomaly_type",
    "missing_equipment",
    "observed_equipment",
    "confidence",
    "screenshot_path",
]

_LOGGER = get_logger(__name__)


class EventLogWriter:
    """Writes anomaly events to a CSV file and optionally a JSON file.

    Both formats are appended in real-time so the files remain readable
    even if the application crashes mid-session.

    Parameters
    ----------
    log_dir:
        Directory where log files are written.  Created automatically if
        it does not exist.
    base_name:
        File stem used for both formats, e.g. ``"events"`` produces
        ``events.csv`` and ``events.json``.
    write_json:
        When True, every event is also appended to a JSON-lines file
        (one JSON object per line — easy to parse and stream).
    """

    def __init__(
        self,
        log_dir: Path | str = Path("logs"),
        base_name: str = "events",
        write_json: bool = True,
    ) -> None:
        self._log_dir: Path = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path: Path = self._log_dir / f"{base_name}.csv"
        self._json_path: Path = self._log_dir / f"{base_name}.jsonl"
        self._write_json: bool = write_json
        self._lock: threading.Lock = threading.Lock()

        self._ensure_csv_header()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_anomaly(
        self,
        camera_id: str,
        camera_name: str,
        anomaly_type: str,
        missing_equipment: tuple[str, ...] | list[str] = (),
        observed_equipment: tuple[str, ...] | list[str] = (),
        confidence: float = 0.0,
        screenshot_path: str | Path | None = None,
    ) -> None:
        """Append one anomaly event to the log files.

        Parameters
        ----------
        camera_id:      Unique stream identifier (e.g. ``"cam_1"``).
        camera_name:    Human-readable name (e.g. ``"Entrance Camera"``).
        anomaly_type:   Short label such as ``"missing_ppe"`` or ``"fall_detected"``.
        missing_equipment: Equipment items that were not found in the frame.
        observed_equipment: All labels that were detected in the frame.
        confidence:     Highest detection confidence for the event.
        screenshot_path: Path to the saved anomaly screenshot, if any.
        """
        record: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "camera_id": camera_id,
            "camera_name": camera_name,
            "anomaly_type": anomaly_type,
            "missing_equipment": ", ".join(missing_equipment),
            "observed_equipment": ", ".join(observed_equipment),
            "confidence": round(confidence, 4),
            "screenshot_path": str(screenshot_path) if screenshot_path else "",
        }

        with self._lock:
            self._write_csv_row(record)
            if self._write_json:
                self._write_jsonl_row(record)

    def log_state_change(
        self,
        camera_id: str,
        camera_name: str,
        state: str,
        message: str = "",
    ) -> None:
        """Log non-anomaly state transitions such as stream start / stop / error."""
        record: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "camera_id": camera_id,
            "camera_name": camera_name,
            "anomaly_type": state,
            "missing_equipment": "",
            "observed_equipment": message,
            "confidence": 0.0,
            "screenshot_path": "",
        }

        with self._lock:
            self._write_csv_row(record)
            if self._write_json:
                self._write_jsonl_row(record)

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    @property
    def json_path(self) -> Path:
        return self._json_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_csv_header(self) -> None:
        """Write header row only if the file is new / empty."""
        if self._csv_path.exists() and self._csv_path.stat().st_size > 0:
            return
        try:
            with self._csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS)
                writer.writeheader()
        except OSError as exc:
            _LOGGER.error("Cannot create CSV log file %s: %s", self._csv_path, exc)

    def _write_csv_row(self, record: dict[str, Any]) -> None:
        try:
            with self._csv_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS)
                writer.writerow(record)
        except OSError as exc:
            _LOGGER.error("CSV write failed: %s", exc)

    def _write_jsonl_row(self, record: dict[str, Any]) -> None:
        try:
            with self._json_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            _LOGGER.error("JSON write failed: %s", exc)