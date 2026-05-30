"""
CLI entrypoint for the video relighting pipeline.

Example
-------
python run.py \\
    --input  input.mp4 \\
    --output relit.mp4 \\
    --light-dir  0.5 0.8 0.3 \\
    --light-color 1.0 0.95 0.85 \\
    --light-intensity 3.0 \\
    --roughness 0.5 \\
    --metallic  0.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time video relighting via RVM + DA-V2 + Cook-Torrance BRDF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--input",       required=True, help="Input video file path")
    p.add_argument("--output",      default=None,  help="Output video file path (auto-named if omitted)")
    p.add_argument("--background",  default=None,  help="Background image or video for compositing")

    # Light / material
    p.add_argument("--light-dir",   nargs=3, type=float, default=[0.5, 0.8, 0.5],
                   metavar=("X", "Y", "Z"), help="Light direction vector (unnormalised)")
    p.add_argument("--light-color", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                   metavar=("R", "G", "B"), help="Light colour, each in [0,1]")
    p.add_argument("--light-intensity", type=float, default=2.0,
                   help="Scalar light intensity multiplier")
    p.add_argument("--roughness",   type=float, default=0.5,  help="GGX roughness in [0,1]")
    p.add_argument("--metallic",    type=float, default=0.0,  help="Metalness in [0,1]")
    p.add_argument("--ambient",     type=float, default=0.05, help="Ambient fill intensity")
    p.add_argument("--view-dir",    nargs=3, type=float, default=[0.0, 0.0, 1.0],
                   metavar=("X", "Y", "Z"), help="Camera/view direction")

    # Temporal
    p.add_argument("--ema-alpha",   type=float, default=0.85,
                   help="EMA smoothing for normals (0=off, closer to 1=more smoothing)")

    # Model / runtime
    p.add_argument("--downsample-ratio", type=float, default=0.25,
                   help="RVM downsample ratio (lower = faster matting)")
    p.add_argument("--device",      default=None,
                   help="Device to use: cuda or cpu (default: cuda if available)")
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   help="Directory containing model checkpoint files")

    # Debug
    p.add_argument("--debug",       action="store_true",
                   help="Write side-by-side debug video with alpha/normals/depth overlay")
    p.add_argument("--max-frames",  type=int, default=None,
                   help="Process at most N frames (useful for testing)")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# background source
# ---------------------------------------------------------------------------

class _BackgroundSource:
    """
    Yields background frames from a static image or a looping video.
    Returns None if no background was provided.
    """

    def __init__(self, path: Optional[str], target_hw: tuple[int, int]) -> None:
        self._frame: Optional[np.ndarray] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._h, self._w = target_hw

        if path is None:
            return

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Background not found: {path}")

        ext = p.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
            img = cv2.imread(str(p))
            self._frame = cv2.resize(img, (self._w, self._h))
        else:
            self._cap = cv2.VideoCapture(str(p))

    def next(self) -> Optional[np.ndarray]:
        if self._frame is not None:
            return self._frame
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if ok:
            return cv2.resize(frame, (self._w, self._h))
        return None


# ---------------------------------------------------------------------------
# debug tile helper
# ---------------------------------------------------------------------------

def _build_debug_tile(
    output: np.ndarray,
    alpha_u8: np.ndarray,
    normals_u8: np.ndarray,
    depth_u8: np.ndarray,
) -> np.ndarray:
    """
    2×2 tile: [output | alpha_rgb] / [normals | depth_rgb].
    All panels are the same size as `output`.
    """
    h, w = output.shape[:2]

    def ensure_3ch(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    alpha_bgr   = ensure_3ch(alpha_u8)
    normals_bgr = ensure_3ch(normals_u8)
    depth_bgr   = ensure_3ch(depth_u8)

    top    = np.concatenate([output, alpha_bgr],   axis=1)
    bottom = np.concatenate([normals_bgr, depth_bgr], axis=1)
    return np.concatenate([top, bottom], axis=0)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)

    # Late imports so --help works without heavy deps
    import torch
    from pipeline import RelightingPipeline, LightConfig

    # ---- device -----------------------------------------------------------
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run.py] Using device: {device}")

    # ---- open input video -------------------------------------------------
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {input_path}", file=sys.stderr)
        sys.exit(1)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[run.py] Input: {width}×{height} @ {fps:.2f} FPS, {n_frames} frames")

    # ---- output path ------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = input_path.parent / f"relit_{input_path.stem}.mp4"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.debug:
        debug_path = out_path.with_stem(out_path.stem + "_debug")

    # ---- video writer(s) --------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    debug_writer: Optional[cv2.VideoWriter] = None
    if args.debug:
        debug_writer = cv2.VideoWriter(
            str(debug_path), fourcc, fps, (width * 2, height * 2)
        )

    # ---- background -------------------------------------------------------
    bg_source = _BackgroundSource(args.background, (height, width))

    # ---- build pipeline ---------------------------------------------------
    pipeline = RelightingPipeline(
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        downsample_ratio=args.downsample_ratio,
        ema_alpha=args.ema_alpha,
    )

    light = LightConfig(
        light_dir=args.light_dir,
        light_color=args.light_color,
        light_intensity=args.light_intensity,
        roughness=args.roughness,
        metallic=args.metallic,
        ambient=args.ambient,
        view_dir=args.view_dir,
    )

    # ---- processing loop --------------------------------------------------
    frame_idx = 0
    t_start = time.perf_counter()
    t_last_report = t_start
    fps_window: list[float] = []

    print(f"[run.py] Processing …")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if args.max_frames and frame_idx >= args.max_frames:
                break

            background = bg_source.next()

            t0 = time.perf_counter()

            if args.debug:
                result = pipeline.process_frame_with_debug(frame_bgr, light, background)
                output_bgr  = result["output"]
                debug_tile  = _build_debug_tile(
                    output_bgr,
                    result["alpha"],
                    result["normals_smooth"],
                    result["depth"],
                )
                debug_writer.write(debug_tile)
            else:
                output_bgr = pipeline.process_frame(frame_bgr, light, background)

            writer.write(output_bgr)

            # FPS reporting
            elapsed = time.perf_counter() - t0
            fps_window.append(elapsed)
            if len(fps_window) > 30:
                fps_window.pop(0)

            frame_idx += 1
            now = time.perf_counter()
            if now - t_last_report >= 2.0:
                cur_fps = 1.0 / (sum(fps_window) / len(fps_window))
                pct = frame_idx / n_frames * 100 if n_frames > 0 else 0
                print(
                    f"  frame {frame_idx:>5}/{n_frames}  ({pct:5.1f}%)  "
                    f"{cur_fps:.1f} FPS",
                    flush=True,
                )
                t_last_report = now

    finally:
        cap.release()
        writer.release()
        if debug_writer is not None:
            debug_writer.release()

    total_time = time.perf_counter() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    print(
        f"\n[run.py] Done. {frame_idx} frames in {total_time:.1f}s "
        f"({avg_fps:.1f} FPS avg).\n"
        f"  Output:  {out_path}"
    )
    if args.debug:
        print(f"  Debug:   {debug_path}")


if __name__ == "__main__":
    main()
