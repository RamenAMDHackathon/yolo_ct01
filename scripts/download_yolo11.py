#!/usr/bin/env python3
"""
Download YOLOv11 weights into the repo as yolov11<variant>.pt.

Usage:
  python scripts/download_yolo11.py            # downloads yolov11n.pt
  python scripts/download_yolo11.py s          # downloads yolov11s.pt
  python scripts/download_yolo11.py m|l|x      # other variants

Tries Ultralytics downloader first, then falls back to GitHub release URLs.
"""
import sys
import shutil
from pathlib import Path

VARIANT = (sys.argv[1] if len(sys.argv) > 1 else "n").strip().lower()
if VARIANT not in {"n", "s", "m", "l", "x"}:
    print(f"Invalid variant '{VARIANT}'. Use one of: n/s/m/l/x.")
    sys.exit(1)

model_name = f"yolo11{VARIANT}.pt"
dst = Path.cwd() / f"yolov11{VARIANT}.pt"

def try_ultralytics() -> Path | None:
    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except Exception:
        return None
    try:
        p = Path(attempt_download_asset(model_name))
        return p if p.exists() else None
    except Exception:
        return None

def try_urls() -> Path | None:
    import tempfile, urllib.request
    urls = [
        f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}",
        f"https://github.com/ultralytics/assets/releases/latest/download/{model_name}",
    ]
    for url in urls:
        try:
            print(f"Trying: {url}")
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                with urllib.request.urlopen(url) as r:
                    shutil.copyfileobj(r, tmp)
                return Path(tmp.name)
        except Exception:
            continue
    return None

src = try_ultralytics()
if src is None:
    print("Ultralytics downloader unavailable or offline. Falling back to direct URLs...")
    src = try_urls()

if src is None:
    print("Failed to download YOLOv11 weights. Please ensure internet access and try again.")
    print("Manual fallback (example): curl -L -o yolov11n.pt https://github.com/ultralytics/assets/releases/latest/download/yolo11n.pt")
    sys.exit(2)

shutil.copyfile(src, dst)
print(f"Downloaded -> {dst}")

