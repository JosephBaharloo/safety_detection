# Safety Detection

A desktop safety equipment detection system built with PyQt6, OpenCV, and YOLOv8.

This project monitors multiple video streams and highlights safety anomalies such as:
- missing hardhats
- missing safety vests
- fall detection events

The application includes a GUI dashboard, stream status reporting, alarm logging, and an audio alarm panel.

## Project Structure

- `safety_detection-main/`
  - `main.py` — application entrypoint
  - `config/` — runtime settings and class mapping definitions
  - `core/` — detection, stream processing, alarm logic, and event handling
  - `gui/` — PyQt6 user interface components
  - `models/` — model weight files (not committed)
  - `videos/` — sample video stream sources
  - `utils/` — helper utilities for logging, video input, and screenshot handling

## Features

- Multi-stream monitoring with independent worker threads
- YOLOv8-based object detection strategy
- Missing safety equipment detection for hardhats and safety vests
- Fall detection alert support
- Live GUI with per-stream status, frame preview, and alarm events
- Sound alarm and event log panel

## Requirements

- Python 3.11
- Windows is the supported development environment in this repository

Dependencies are listed in `safety_detection-main/requirements.txt` and `safety_detection-main/pyproject.toml`.

## Setup

```powershell
cd "safety_detection-main"
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Model Weights

The repository does not include YOLO model weights.

Copy or download a YOLOv8 `.pt` file into `safety_detection-main/models/`.

The application will resolve the model path in this order:
1. `safety_detection-main/models/my_best_model.pt`
2. `safety_detection-main/models/yolov8n.pt`
3. fallback to the system package path for `yolov8n.pt`

If you do not have a custom model, download an official YOLOv8 model and save it to `safety_detection-main/models/yolov8n.pt`.

## Configuration

The main runtime settings are defined in `safety_detection-main/config/settings.py`.

### Equipment labels

The label mapping and default required equipment are configured in `safety_detection-main/config/equipment_classes.yaml`.

Example labels:
- `Fall-Detected`
- `Hardhat`
- `Person`
- `Safety_Vest`

The file also defines the default equipment required for all streams.

### Stream sources

Default streams are configured in `safety_detection-main/config/settings.py` to use files in `safety_detection-main/videos/`.

To monitor a different source, update the `source` field for a stream in `build_default_settings()`.

## Run the Application

From `safety_detection-main/`:

```powershell
python main.py
```

## Notes

- `safety_detection-main/config/alarm.mp3` is used as the alarm audio file.
- The GUI includes Start/Stop controls and a log panel for anomaly events.
- If YOLO fails to load, the app falls back to a `NullDetector` and continues without alerts.

## Recommended GitHub Additions

- Add a `.gitignore` to exclude `safety_detection-main/.venv/`, `safety_detection-main/models/*.pt`, and generated logs/screenshots.
- Keep model weights out of source control.

## License

No license is specified in the repository. Add a license file if you want to clarify reuse terms.
