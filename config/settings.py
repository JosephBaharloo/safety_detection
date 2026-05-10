from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "config"
MODELS_DIR: Path = PROJECT_ROOT / "models"

_MIN_MODEL_SIZE_BYTES: int = 100_000  # 100 KB


@dataclass(frozen=True)
class StreamConfig:
    stream_id: str
    source: str | int
    display_name: str
    required_equipment: tuple[str, ...]


@dataclass(frozen=True)
class DetectorSettings:
    model_path: Path
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.5
    image_size: int = 640


@dataclass(frozen=True)
class AlarmSettings:
    sound_path: Path
    cooldown_seconds: float = 2.0


@dataclass(frozen=True)
class AppSettings:
    detector: DetectorSettings
    alarm: AlarmSettings
    streams: tuple[StreamConfig, ...]
    class_map: dict[int, str]


def load_equipment_classes(path: Path | None = None) -> dict[int, str]:
    yaml_path: Path = path or (CONFIG_DIR / "equipment_classes.yaml")
    with yaml_path.open("r", encoding="utf-8") as handle:
        raw_data: dict[str, Any] = yaml.safe_load(handle) or {}

    names_raw: Any = raw_data.get("names", {})
    if not isinstance(names_raw, dict):
        raise ValueError("equipment_classes.yaml must contain a mapping named 'names'.")

    class_map: dict[int, str] = {int(key): str(value) for key, value in names_raw.items()}
    if not class_map:
        raise ValueError("At least one class must be defined in equipment_classes.yaml.")
    return class_map


def _resolve_model_path() -> Path:
    best_pt: Path = MODELS_DIR / "best.pt"
    if best_pt.exists() and best_pt.stat().st_size > _MIN_MODEL_SIZE_BYTES:
        return best_pt

    yolov8n_local: Path = MODELS_DIR / "yolov8n.pt"
    if yolov8n_local.exists() and yolov8n_local.stat().st_size > _MIN_MODEL_SIZE_BYTES:
        return yolov8n_local

    return Path("yolov8n.pt")


def build_default_settings() -> AppSettings:
    class_map: dict[int, str] = load_equipment_classes()
    model_path: Path = _resolve_model_path()

    # TEST MODE — switch to PRODUCTION when best.pt is ready
    required_defaults: tuple[str, ...] = ("helmet", "vest", "goggles", "gloves")

    # PRODUCTION:
    # using_custom_model: bool = model_path.name == "best.pt"
    # required_defaults: tuple[str, ...] = (
    #     tuple(v for v in ("helmet", "vest") if v in set(class_map.values()))
    #     if using_custom_model
    #     else ()
    # )

    streams: tuple[StreamConfig, ...] = (
        StreamConfig(
            stream_id="cam_1",
            source=0,
            display_name="Camera 1",
            required_equipment=required_defaults,
        ),
        StreamConfig(
            stream_id="cam_2",
            source="videos/cam2.mp4",
            display_name="Camera 2",
            required_equipment=required_defaults,
        ),
        StreamConfig(
            stream_id="cam_3",
            source="videos/cam3.mp4",
            display_name="Camera 3",
            required_equipment=required_defaults,
        ),
        StreamConfig(
            stream_id="cam_4",
            source="videos/cam4.mp4",
            display_name="Camera 4",
            required_equipment=required_defaults,
        ),
        StreamConfig(
            stream_id="cam_5",
            source="videos/cam5.mp4",
            display_name="Camera 5",
            required_equipment=required_defaults,
        ),
        StreamConfig(
            stream_id="cam_6",
            source="videos/cam6.mp4",
            display_name="Camera 6 ",
            required_equipment=required_defaults,
        ),
    )

    detector_settings: DetectorSettings = DetectorSettings(
        model_path=model_path,
        confidence_threshold=0.45,
        iou_threshold=0.5,
        image_size=640,
    )
    alarm_settings: AlarmSettings = AlarmSettings(
        sound_path=CONFIG_DIR / "alarm.mp3",
        cooldown_seconds=2.0,
    )

    return AppSettings(
        detector=detector_settings,
        alarm=alarm_settings,
        streams=streams,
        class_map=class_map,
    )