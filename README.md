# Safety Detection — Setup and Run

This repository contains a scaffold for a safety equipment detection system using PyQt6 and YOLOv8.

Quick setup (Windows PowerShell):

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
# Download model weights (set MODEL_URL or pass --url)
python scripts/get_model.py --url "<MODEL_URL>"
python main.py
```

Notes:
- Do not commit model binaries to the repository. Use `scripts/get_model.py` or GitHub Releases / LFS.
- The `.gitignore` excludes `models/*.pt` by default.
- For reproducible environments consider adding a Dockerfile or lock file.
