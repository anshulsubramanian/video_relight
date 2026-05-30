"""
Frame-by-frame video relighting pipeline orchestrator.

Data flow per frame
-------------------
BGR frame
  ├─── RVMMatting          → alpha (H,W,1), fgr (H,W,3)
  └─── DepthNormalEstimator → depth (H,W), normals_raw (H,W,3)
                                └─ EMANormalSmoother → normals (H,W,3)

albedo (frame_rgb) + normals + alpha ──► CookTorranceBRDF ──► relit (H,W,3)

relit   ──► composite over background ──► output (H,W,3)

All numpy ↔ torch conversions are done here so individual modules stay clean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from models.matting import RVMMatting
from models.depth import DepthNormalEstimator
from models.brdf import CookTorranceBRDF
from temporal import EMANormalSmoother


# ---------------------------------------------------------------------------
# Light / material config dataclass
# ---------------------------------------------------------------------------

@dataclass
class LightConfig:
    """
    Lighting and material parameters for one frame (or the whole video).

    All fields can be updated between frames for animated relighting.
    """
    light_dir: Union[list, tuple, np.ndarray] = field(
        default_factory=lambda: [0.5, 0.8, 0.5]
    )
    light_color: Union[list, tuple, np.ndarray] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
    )
    light_intensity: float = 2.0
    roughness: float = 0.5
    metallic: float = 0.0
    ambient: float = 0.05
    view_dir: Union[list, tuple, np.ndarray] = field(
        default_factory=lambda: [0.0, 0.0, 1.0]
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RelightingPipeline:
    """
    End-to-end video relighting pipeline.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory containing rvm_mobilenetv3_fp32.onnx and
        depth_anything_v2_vits.pth.
    device : str
        Torch/ONNX device: "cuda" or "cpu".
    downsample_ratio : float
        RVM downsampling ratio (0.25 = fast, 0.375 = balanced, 1.0 = full).
    ema_alpha : float
        EMA smoothing coefficient for surface normals [0, 1).
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        device: str = "cuda",
        downsample_ratio: float = 0.25,
        ema_alpha: float = 0.85,
    ) -> None:
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        ckpt_dir = Path(checkpoint_dir)

        print(f"[Pipeline] Loading RVM MobileNetV3 …", flush=True)
        self._matting = RVMMatting(
            checkpoint_dir=ckpt_dir,
            device=device,
            downsample_ratio=downsample_ratio,
        )

        print(f"[Pipeline] Loading Depth-Anything-V2-Small …", flush=True)
        self._depth_normal = DepthNormalEstimator(
            checkpoint_dir=ckpt_dir,
            device=device,
        )

        self._brdf = CookTorranceBRDF(device=device)
        self._normal_smoother = EMANormalSmoother(alpha=ema_alpha)

        print(f"[Pipeline] Ready on {self._device}.", flush=True)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all temporal state (call between independent video clips)."""
        self._matting.reset()
        self._normal_smoother.reset()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        light: LightConfig,
        background: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Relight one frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Input frame as BGR uint8 (H, W, 3).
        light : LightConfig
            Lighting and material parameters for this frame.
        background : np.ndarray | None
            Optional BGR uint8 background plate (H, W, 3).
            If None, a black background is used.

        Returns
        -------
        output_bgr : np.ndarray
            Relit frame as BGR uint8 (H, W, 3).
        """
        # -- 1. Alpha matting -----------------------------------------------
        alpha, _ = self._matting(frame_bgr)  # (H,W,1) float32

        # -- 2. Depth + surface normals --------------------------------------
        _, normals_raw = self._depth_normal(frame_bgr)   # (H,W,3) float32

        # -- 3. Temporal smoothing of normals --------------------------------
        normals = self._normal_smoother(normals_raw)      # (H,W,3) float32

        # -- 4. Prepare albedo (treat frame as base colour) ------------------
        albedo_rgb = _bgr_to_rgb_float(frame_bgr)         # (H,W,3) float32

        # -- 5. Background plate --------------------------------------------
        bg_tensor: Optional[torch.Tensor] = None
        if background is not None:
            bg_rgb = _bgr_to_rgb_float(background)
            bg_tensor = torch.from_numpy(bg_rgb)

        # -- 6. Cook-Torrance BRDF -------------------------------------------
        albedo_t  = torch.from_numpy(albedo_rgb)
        normals_t = torch.from_numpy(normals)
        alpha_t   = torch.from_numpy(alpha)

        relit_t = self._brdf(
            albedo=albedo_t,
            normals=normals_t,
            alpha=alpha_t,
            light_dir=light.light_dir,
            light_color=light.light_color,
            light_intensity=light.light_intensity,
            roughness=light.roughness,
            metallic=light.metallic,
            view_dir=light.view_dir,
            background=bg_tensor,
            ambient=light.ambient,
        )   # (H,W,3) float32 in [0,1]

        # -- 7. Back to BGR uint8 -------------------------------------------
        relit_rgb = relit_t.cpu().numpy()
        output_bgr = (relit_rgb[:, :, ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        return output_bgr

    def process_frame_with_debug(
        self,
        frame_bgr: np.ndarray,
        light: LightConfig,
        background: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Same as process_frame but also returns intermediate outputs.

        Returns
        -------
        dict with keys: output, alpha, depth, normals_raw, normals_smooth
        """
        alpha, _ = self._matting(frame_bgr)
        depth, normals_raw = self._depth_normal(frame_bgr)
        normals = self._normal_smoother(normals_raw.copy())

        albedo_rgb = _bgr_to_rgb_float(frame_bgr)

        bg_tensor: Optional[torch.Tensor] = None
        if background is not None:
            bg_tensor = torch.from_numpy(_bgr_to_rgb_float(background))

        relit_t = self._brdf(
            albedo=torch.from_numpy(albedo_rgb),
            normals=torch.from_numpy(normals),
            alpha=torch.from_numpy(alpha),
            light_dir=light.light_dir,
            light_color=light.light_color,
            light_intensity=light.light_intensity,
            roughness=light.roughness,
            metallic=light.metallic,
            view_dir=light.view_dir,
            background=bg_tensor,
            ambient=light.ambient,
        )

        relit_rgb = relit_t.cpu().numpy()
        output_bgr = (relit_rgb[:, :, ::-1] * 255.0).clip(0, 255).astype(np.uint8)

        return {
            "output":         output_bgr,
            "alpha":          (alpha[:, :, 0] * 255).astype(np.uint8),
            "depth":          (depth * 255).astype(np.uint8),
            "normals_raw":    ((normals_raw + 1) / 2 * 255).astype(np.uint8),
            "normals_smooth": ((normals + 1) / 2 * 255).astype(np.uint8),
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bgr_to_rgb_float(frame_bgr: np.ndarray) -> np.ndarray:
    """(H,W,3) BGR uint8 → (H,W,3) RGB float32 in [0,1]."""
    return frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
