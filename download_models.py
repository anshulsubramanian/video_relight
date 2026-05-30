"""
Download model checkpoints for the video relighting pipeline.

Models downloaded:
  - RVM MobileNetV3 ONNX (15 MB)  → checkpoints/rvm_mobilenetv3_fp32.onnx
  - Depth-Anything-V2-Small (~100 MB) → checkpoints/depth_anything_v2_vits.pth

Total disk footprint: ~115 MB (under the 150 MB budget).
"""
import os
import urllib.request
from pathlib import Path

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"

MODELS = {
    "rvm_mobilenetv3_fp32.onnx": (
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"
    ),
    "depth_anything_v2_vits.pth": (
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
    ),
}


class _ProgressBar:
    def __init__(self, filename: str):
        self._filename = filename
        self._last = 0

    def __call__(self, block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            mb_done = downloaded / 1e6
            mb_total = total_size / 1e6
            print(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)
        else:
            print(f"\r  Downloaded {downloaded / 1e6:.1f} MB", end="", flush=True)


def download_models(force: bool = False) -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        dest = CHECKPOINTS_DIR / filename
        if dest.exists() and not force:
            size_mb = dest.stat().st_size / 1e6
            print(f"  [skip] {filename} already exists ({size_mb:.1f} MB)")
            continue
        print(f"Downloading {filename} ...")
        tmp = dest.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(url, tmp, reporthook=_ProgressBar(filename))
            print()  # newline after progress bar
            tmp.rename(dest)
            size_mb = dest.stat().st_size / 1e6
            print(f"  [done] {filename} ({size_mb:.1f} MB)")
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            raise RuntimeError(f"Failed to download {filename}: {exc}") from exc

    total_mb = sum((CHECKPOINTS_DIR / f).stat().st_size for f in MODELS) / 1e6
    print(f"\nAll models ready. Total checkpoint size: {total_mb:.1f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download model checkpoints")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()
    download_models(force=args.force)
