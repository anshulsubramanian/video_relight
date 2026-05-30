"""
RVM (Robust Video Matting) MobileNetV3 wrapper.

Runs via ONNX Runtime with CUDA execution provider.
Manages the four recurrent hidden states (r1–r4) that give RVM
its temporal consistency across frames.

Input:  BGR uint8 frame (H, W, 3) as NumPy array
Output: alpha matte (H, W, 1) float32 in [0, 1]
        foreground RGBA (H, W, 4) float32 in [0, 1]
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_ONNX_FILENAME = "rvm_mobilenetv3_fp32.onnx"


class RVMMatting:
    """
    Wraps the RVM MobileNetV3 ONNX model.

    The model has five inputs:
      src         – (1, 3, H, W) float32 RGB in [0, 1]
      r1i, r2i, r3i, r4i – recurrent hidden states (None on first frame)

    And five outputs:
      fgr  – (1, 3, H, W) float32 foreground RGB
      pha  – (1, 1, H, W) float32 alpha
      r1o, r2o, r3o, r4o – updated recurrent states
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        device: str = "cuda",
        downsample_ratio: float = 0.25,
    ) -> None:
        self._downsample_ratio = float(downsample_ratio)
        self._recurrent: list[Optional[np.ndarray]] = [None, None, None, None]
        self._session = self._load_session(Path(checkpoint_dir) / _ONNX_FILENAME, device)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_session(onnx_path: Path, device: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime-gpu is required. Install with: pip install onnxruntime-gpu"
            ) from exc

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"RVM checkpoint not found at {onnx_path}. "
                "Run `python download_models.py` first."
            )

        providers: list[str] = []
        if device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)

    @staticmethod
    def _bgr_to_tensor(frame_bgr: np.ndarray) -> np.ndarray:
        """Convert (H, W, 3) BGR uint8 → (1, 3, H, W) float32 RGB in [0, 1]."""
        rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
        return rgb.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset recurrent states (call between independent video clips)."""
        self._recurrent = [None, None, None, None]

    def __call__(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process one frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR uint8 image of shape (H, W, 3).

        Returns
        -------
        alpha : np.ndarray
            Float32 alpha matte (H, W, 1) in [0, 1].
        fgr : np.ndarray
            Float32 foreground RGB (H, W, 3) in [0, 1].
        """
        src = self._bgr_to_tensor(frame_bgr)
        r1i, r2i, r3i, r4i = self._recurrent

        feed: dict = {
            "src": src,
            "downsample_ratio": np.array([self._downsample_ratio], dtype=np.float32),
        }
        if r1i is not None:
            feed.update({"r1i": r1i, "r2i": r2i, "r3i": r3i, "r4i": r4i})

        # Fetch only the outputs that exist in the model
        output_names = [o.name for o in self._session.get_outputs()]
        outputs = self._session.run(output_names, feed)
        out_map = dict(zip(output_names, outputs))

        fgr = out_map["fgr"]   # (1, 3, H, W)
        pha = out_map["pha"]   # (1, 1, H, W)

        # Preserve recurrent states for next frame
        self._recurrent = [
            out_map.get("r1o"),
            out_map.get("r2o"),
            out_map.get("r3o"),
            out_map.get("r4o"),
        ]

        # (1, C, H, W) → (H, W, C)
        alpha = pha[0].transpose(1, 2, 0)          # (H, W, 1)
        foreground = fgr[0].transpose(1, 2, 0)     # (H, W, 3)
        return alpha, foreground
