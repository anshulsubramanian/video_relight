"""
Depth-Anything-V2-Small depth estimator + analytical depth-to-normals.

Depth estimation
----------------
Runs DA-V2-Small (24.8 M params, ~100 MB checkpoint) in PyTorch on the
specified device.  Output is a relative depth map normalised to [0, 1].

Depth → Surface Normals
-----------------------
Conversion uses fixed Sobel kernels applied with torch.nn.functional.conv2d
entirely on the GPU.  No extra model weights are involved.

  dz/dx, dz/dy  = Sobel(depth, kernel_size=7)
  normal_raw    = (-dz/dx, -dz/dy, z_scale)
  normal        = L2-normalize(normal_raw)

Input:  BGR uint8 frame (H, W, 3) as NumPy array
Output: depth (H, W)        float32 in [0, 1]
        normals (H, W, 3)   float32 unit vectors, camera-space
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_CKPT_FILENAME = "depth_anything_v2_vits.pth"
_INPUT_SIZE = 518          # DA-V2-Small default inference size
_Z_SCALE = 1.0             # weight of the Z component in the normal cross-product


def _build_sobel_kernels(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return fixed 7×7 Sobel kernels for X and Y as (1,1,7,7) float32 tensors.
    Using size-7 reduces noise sensitivity versus the standard 3×3 kernel.
    """
    # Build separable Sobel via convolution of [1,2,1] smoothing with [-1,0,1] diff
    smooth = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    diff   = np.array([-1, -2, 0, 2, 1], dtype=np.float64)

    # Outer products → 5×5 Sobel kernels
    kx5 = np.outer(smooth, diff)
    ky5 = np.outer(diff, smooth)

    # Pad to 7×7 with zeros to use a single size throughout
    def pad_to_7(k: np.ndarray) -> np.ndarray:
        k7 = np.zeros((7, 7), dtype=np.float32)
        k7[1:6, 1:6] = k.astype(np.float32)
        return k7

    kx = torch.from_numpy(pad_to_7(kx5)).unsqueeze(0).unsqueeze(0).to(device)
    ky = torch.from_numpy(pad_to_7(ky5)).unsqueeze(0).unsqueeze(0).to(device)
    return kx, ky


class DepthNormalEstimator:
    """
    Combines Depth-Anything-V2-Small depth estimation with on-GPU
    Sobel-based surface normal extraction.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        device: str = "cuda",
        z_scale: float = _Z_SCALE,
    ) -> None:
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._z_scale = float(z_scale)
        self._model = self._load_model(Path(checkpoint_dir) / _CKPT_FILENAME)
        self._kx, self._ky = _build_sobel_kernels(self._device)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, ckpt_path: Path):
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Depth-Anything-V2-Small checkpoint not found at {ckpt_path}. "
                "Run `python download_models.py` first."
            )
        # DA-V2 is distributed as a standalone script; we import it from the
        # standard huggingface transformers interface which bundles the arch.
        try:
            from transformers import AutoModelForDepthEstimation
            model = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            )
        except Exception:
            # Fall back to loading the raw .pth with the DA-V2 architecture
            model = self._load_raw_checkpoint(ckpt_path)

        model.eval().to(self._device)
        return model

    def _load_raw_checkpoint(self, ckpt_path: Path):
        """
        Load DA-V2-Small from a raw .pth checkpoint using the DPT architecture
        bundled inside the depth_anything_v2 package (if installed) or fetched
        via torch.hub.
        """
        try:
            # Try the official DA-V2 package first
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

            cfg = {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            }
            model = DepthAnythingV2(**cfg)
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            return model
        except ImportError:
            pass

        # Last resort: torch.hub
        model = torch.hub.load(
            "DepthAnything/Depth-Anything-V2",
            "DepthAnythingV2",
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
        )
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Resize to _INPUT_SIZE on the long edge, convert BGR→RGB,
        normalise to ImageNet stats, and return (1, 3, H', W') float32.
        """
        h, w = frame_bgr.shape[:2]
        scale = _INPUT_SIZE / max(h, w)
        new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
        # Ensure dimensions are multiples of 14 (DINOv2 patch size)
        new_h = (new_h // 14) * 14
        new_w = (new_w // 14) * 14

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb  = (rgb - mean) / std

        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self._device)

    def _estimate_depth(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Run DA-V2-Small and return a (H, W) float32 depth map on self._device,
        upsampled back to the original resolution, normalised to [0, 1].
        """
        orig_h, orig_w = frame_bgr.shape[:2]
        inp = self._preprocess(frame_bgr)

        with torch.no_grad():
            # Support both HF transformers API and raw DPT API
            if hasattr(self._model, "config"):
                # HuggingFace Transformers model
                out = self._model(pixel_values=inp)
                depth = out.predicted_depth  # (1, H', W')
            else:
                # Raw DepthAnythingV2 DPT model
                depth = self._model(inp)     # (1, H', W') or (H', W')
                if depth.dim() == 2:
                    depth = depth.unsqueeze(0)

        # Upsample to original resolution
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()  # (H, W)

        # Normalise relative depth to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        depth = (depth - d_min) / (d_max - d_min + 1e-8)
        return depth

    def _depth_to_normals(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Convert a (H, W) depth map to (H, W, 3) unit surface normals using
        fixed Sobel kernels applied on the GPU.
        """
        d = depth.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

        dz_dx = F.conv2d(d, self._kx, padding=3)  # (1, 1, H, W)
        dz_dy = F.conv2d(d, self._ky, padding=3)

        nx = -dz_dx.squeeze()                  # (H, W)
        ny = -dz_dy.squeeze()
        nz = torch.full_like(nx, self._z_scale)

        normals = torch.stack([nx, ny, nz], dim=-1)  # (H, W, 3)

        # L2 normalise
        norm = torch.norm(normals, dim=-1, keepdim=True).clamp(min=1e-6)
        return normals / norm

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def __call__(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth and surface normals for one frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR uint8 image of shape (H, W, 3).

        Returns
        -------
        depth : np.ndarray
            Float32 depth map (H, W) in [0, 1] (relative, farther = smaller).
        normals : np.ndarray
            Float32 surface normals (H, W, 3), unit vectors in camera space.
        """
        depth_t = self._estimate_depth(frame_bgr)
        normals_t = self._depth_to_normals(depth_t)

        depth_np   = depth_t.cpu().float().numpy()
        normals_np = normals_t.cpu().float().numpy()
        return depth_np, normals_np
