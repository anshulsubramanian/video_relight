"""
Cook-Torrance microfacet BRDF renderer.

Implements the standard GGX/Schlick PBR model as fully vectorised
PyTorch operations — no model weights, pure math on GPU tensors.

BRDF = specular + diffuse
     = (D · G · F) / (4 · NdotV · NdotL)  ·  NdotL · L
     + (1 - F) · albedo/π                  ·  NdotL · L

where:
  D  = GGX Normal Distribution Function
  G  = Smith-Schlick-GGX Geometry term (view + light masking)
  F  = Fresnel-Schlick  (F0 = lerp(0.04, albedo, metallic))

All inputs are per-pixel tensors of shape (H, W, 3) or (H, W, 1).
The renderer is intentionally single-bounce / direct-lighting only for
real-time performance.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# scalar / vector helpers
# ---------------------------------------------------------------------------

def _normalise(v: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return v / (v.norm(dim=dim, keepdim=True) + eps)


def _dot(a: torch.Tensor, b: torch.Tensor, clamp_min: float = 0.0) -> torch.Tensor:
    """Per-pixel dot product → (..., 1)."""
    return (a * b).sum(dim=-1, keepdim=True).clamp(min=clamp_min)


# ---------------------------------------------------------------------------
# BRDF components
# ---------------------------------------------------------------------------

def _ggx_ndf(n_dot_h: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    """
    GGX/Trowbridge-Reitz Normal Distribution Function.

    D(h) = α² / (π · ((N·H)² · (α²-1) + 1)²)

    Parameters
    ----------
    n_dot_h   : (..., 1) clamped to [0, 1]
    roughness : scalar or (..., 1) in [0, 1]
    """
    a  = roughness * roughness            # α = roughness²
    a2 = a * a
    denom = (n_dot_h * n_dot_h * (a2 - 1.0) + 1.0)
    denom = math.pi * denom * denom + 1e-8
    return a2 / denom


def _geometry_schlick_ggx(n_dot_v: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    """
    Schlick-GGX single-side geometry masking term.

    k = ((roughness + 1)² / 8)   for direct lighting
    G1 = NdotV / (NdotV · (1-k) + k)
    """
    r  = roughness + 1.0
    k  = (r * r) / 8.0
    return n_dot_v / (n_dot_v * (1.0 - k) + k + 1e-8)


def _geometry_smith(
    n_dot_v: torch.Tensor, n_dot_l: torch.Tensor, roughness: torch.Tensor
) -> torch.Tensor:
    """Smith two-sided geometry: G = G1(v) · G1(l)."""
    g_v = _geometry_schlick_ggx(n_dot_v, roughness)
    g_l = _geometry_schlick_ggx(n_dot_l, roughness)
    return g_v * g_l


def _fresnel_schlick(
    cos_theta: torch.Tensor, f0: torch.Tensor
) -> torch.Tensor:
    """
    Fresnel-Schlick approximation.

    F(θ) = F0 + (1 - F0) · (1 - cosθ)⁵
    """
    return f0 + (1.0 - f0) * (1.0 - cos_theta).clamp(min=0.0).pow(5)


# ---------------------------------------------------------------------------
# Main BRDF class
# ---------------------------------------------------------------------------

class CookTorranceBRDF:
    """
    Per-pixel Cook-Torrance BRDF renderer.

    Usage
    -----
    brdf = CookTorranceBRDF(device="cuda")
    relit = brdf(
        albedo    = frame_rgb_tensor,      # (H, W, 3) in [0,1]
        normals   = normals_tensor,        # (H, W, 3) unit vectors
        alpha     = alpha_tensor,          # (H, W, 1) in [0,1]
        light_dir = torch.tensor([0.5, 0.8, 0.3]),
        light_color     = torch.tensor([1.0, 0.95, 0.85]),
        light_intensity = 3.0,
        roughness       = 0.5,
        metallic        = 0.0,
        background      = bg_tensor,       # (H, W, 3) or None
    )
    """

    def __init__(self, device: str = "cuda") -> None:
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        albedo: torch.Tensor,
        normals: torch.Tensor,
        alpha: torch.Tensor,
        light_dir: Union[torch.Tensor, np.ndarray, list],
        light_color: Union[torch.Tensor, np.ndarray, list] = (1.0, 1.0, 1.0),
        light_intensity: float = 2.0,
        roughness: float = 0.5,
        metallic: float = 0.0,
        view_dir: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        background: Optional[torch.Tensor] = None,
        ambient: float = 0.05,
    ) -> torch.Tensor:
        """
        Compute per-pixel Cook-Torrance shading and composite over background.

        Parameters
        ----------
        albedo    : (H, W, 3) float32 tensor in [0, 1]
        normals   : (H, W, 3) float32 unit-vector tensor
        alpha     : (H, W, 1) float32 tensor in [0, 1]
        light_dir : (3,) direction *towards* the light (will be normalised)
        light_color      : (3,) RGB in [0, 1]
        light_intensity  : scalar multiplier
        roughness        : GGX roughness in [0, 1]
        metallic         : metalness factor in [0, 1]
        view_dir         : (3,) camera direction (default: [0, 0, 1])
        background       : (H, W, 3) background plate; black if None
        ambient          : small ambient fill to avoid fully dark shadows

        Returns
        -------
        output : (H, W, 3) float32 tensor in [0, 1], composited result
        """
        dev = self._device

        albedo  = albedo.to(dev)
        normals = _normalise(normals.to(dev))
        alpha   = alpha.to(dev)

        H, W = albedo.shape[:2]

        # ---- vectorise scalar / direction inputs --------------------------
        L = _to_unit_vec(light_dir, dev)            # (3,)
        V = _to_unit_vec(view_dir or [0, 0, 1], dev)
        lc = _to_color(light_color, dev) * light_intensity  # (3,)
        r = torch.tensor(roughness, dtype=torch.float32, device=dev).clamp(0.05, 1.0)
        m = torch.tensor(metallic,  dtype=torch.float32, device=dev).clamp(0.0, 1.0)

        # Broadcast direction vectors to (H, W, 3)
        L_map = L.view(1, 1, 3).expand(H, W, 3)
        V_map = V.view(1, 1, 3).expand(H, W, 3)

        # Half vector
        H_map = _normalise(L_map + V_map)           # (H, W, 3)

        # Dot products clamped to (0.001, 1)
        NdotL = _dot(normals, L_map, clamp_min=0.001)  # (H, W, 1)
        NdotV = _dot(normals, V_map, clamp_min=0.001)
        NdotH = _dot(normals, H_map, clamp_min=0.0)
        HdotV = _dot(H_map,   V_map, clamp_min=0.0)

        # ---- F0: base reflectance -----------------------------------------
        # Dielectric F0 = 0.04, metal F0 = albedo
        f0_dielectric = torch.full_like(albedo, 0.04)
        f0 = f0_dielectric + m * (albedo - f0_dielectric)  # (H, W, 3)

        # ---- specular term ------------------------------------------------
        D = _ggx_ndf(NdotH, r)                             # (H, W, 1)
        G = _geometry_smith(NdotV, NdotL, r)               # (H, W, 1)
        F = _fresnel_schlick(HdotV, f0)                    # (H, W, 3)

        specular = (D * G * F) / (4.0 * NdotV * NdotL + 1e-8)  # (H, W, 3)

        # ---- diffuse term (Lambertian, metal has no diffuse) ---------------
        k_diffuse = (1.0 - F) * (1.0 - m)
        diffuse = k_diffuse * albedo / math.pi             # (H, W, 3)

        # ---- direct lighting ----------------------------------------------
        Lo = (specular + diffuse) * NdotL * lc             # (H, W, 3)

        # ---- ambient fill -------------------------------------------------
        Lo = Lo + ambient * albedo

        Lo = Lo.clamp(0.0, 1.0)

        # ---- composite over background ------------------------------------
        if background is None:
            background = torch.zeros_like(albedo)
        else:
            background = background.to(dev).clamp(0.0, 1.0)

        output = alpha * Lo + (1.0 - alpha) * background   # (H, W, 3)
        return output.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# utility
# ---------------------------------------------------------------------------

def _to_unit_vec(
    v: Union[torch.Tensor, np.ndarray, list], device: torch.device
) -> torch.Tensor:
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)
    v = v.float().to(device)
    return v / (v.norm() + 1e-8)


def _to_color(
    c: Union[torch.Tensor, np.ndarray, list, tuple], device: torch.device
) -> torch.Tensor:
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(list(c), dtype=torch.float32)
    return c.float().to(device)
