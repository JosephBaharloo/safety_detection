from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.request import urlopen


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, dest.open("wb") as out_file:
        CHUNK = 8192
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            out_file.write(chunk)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download model weights into ./models")
    p.add_argument("--url", type=str, help="URL to the model file (weights).")
    p.add_argument("--output", type=Path, default=Path("models/best.pt"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    url = args.url or os.environ.get("MODEL_URL")
    if not url:
        print("Please provide --url or set MODEL_URL environment variable.")
        return 2

    out_path: Path = args.output
    print(f"Downloading model from {url} to {out_path}")
    try:
        download_file(url, out_path)
    except Exception as exc:
        print(f"Download failed: {exc}")
        return 1

    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
