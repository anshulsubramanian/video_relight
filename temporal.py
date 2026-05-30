"""
Temporal smoothers for the video relighting pipeline.

EMANormalSmoother
-----------------
Applies an exponential moving average (EMA) on per-pixel surface normals
to suppress high-frequency temporal flicker introduced by the depth
estimator running frame-independently.

Update rule:
    normals_smooth ← α · normals_prev + (1 - α) · normals_cur
then L2-renormalise to maintain unit vectors.

α = 0 → no memory (pass-through)
α = 1 → frozen first frame
α ∈ (0.8, 0.9) works well in practice for 24-30 FPS video.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class EMANormalSmoother:
    """
    Per-pixel EMA smoother for surface normal maps.

    Parameters
    ----------
    alpha : float
        Smoothing coefficient in [0, 1).
        Higher = more temporal memory, slower to react to scene changes.
    """

    def __init__(self, alpha: float = 0.85) -> None:
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self._alpha = float(alpha)
        self._prev: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset state (call between independent clips)."""
        self._prev = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not 0.0 <= value < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {value}")
        self._alpha = float(value)

    def __call__(self, normals: np.ndarray) -> np.ndarray:
        """
        Smooth one normal map.

        Parameters
        ----------
        normals : np.ndarray
            (H, W, 3) float32 unit surface normals for the current frame.

        Returns
        -------
        smoothed : np.ndarray
            (H, W, 3) float32 EMA-smoothed unit normals.
        """
        if self._prev is None or self._prev.shape != normals.shape:
            # First frame or resolution change — initialise with current
            self._prev = normals.copy()
            return normals

        blended = self._alpha * self._prev + (1.0 - self._alpha) * normals

        # Re-normalise to maintain unit length
        norm = np.linalg.norm(blended, axis=-1, keepdims=True)
        smoothed = blended / np.maximum(norm, 1e-6)

        self._prev = smoothed
        return smoothed


class EMADepthSmoother:
    """
    Optional EMA smoother for depth maps (same logic as normals, no renorm).

    Can be used in addition to the normal smoother to further stabilise
    the depth-to-normals conversion on noisy inputs.
    """

    def __init__(self, alpha: float = 0.7) -> None:
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self._alpha = float(alpha)
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        depth : np.ndarray
            (H, W) float32 depth map in [0, 1].

        Returns
        -------
        smoothed : np.ndarray
            (H, W) float32 EMA-smoothed depth.
        """
        if self._prev is None or self._prev.shape != depth.shape:
            self._prev = depth.copy()
            return depth

        smoothed = self._alpha * self._prev + (1.0 - self._alpha) * depth
        self._prev = smoothed
        return smoothed
