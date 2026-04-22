from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train a YOLOv8 model for safety equipment detection."
    )
    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--data", type=Path, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", type=str, default="safety_detector")
    return parser


def train_model(
    weights: Path,
    data_yaml: Path,
    epochs: int,
    image_size: int,
    batch_size: int,
    project: Path,
    run_name: str,
) -> None:
    model: YOLO = YOLO(str(weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project=str(project),
        name=run_name,
    )


def main() -> int:
    parser: argparse.ArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse_args()

    train_model(
        weights=args.weights,
        data_yaml=args.data,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        project=args.project,
        run_name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
