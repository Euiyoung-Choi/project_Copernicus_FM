from __future__ import annotations

import numpy as np


def nan_inf_to_num(x: np.ndarray, value: float = 0.0) -> np.ndarray:
    return np.nan_to_num(x, nan=value, posinf=value, neginf=value)


def normalize_channels(img7: np.ndarray) -> np.ndarray:
    """
    img7: (7, H, W) float32/float64

    Channel convention:
      0..3: S2 (B4,B3,B2,B8) 0..10000 -> /10000
      4..5: S1 (VV,VH) dB -> (x + 20) / 20  (approx; clipping decided later)
      6:    Fmask 0..100 -> /100
    """
    if img7.ndim != 3 or img7.shape[0] != 7:
        raise ValueError(f"Expected (7,H,W), got {img7.shape}")

    out = img7.astype(np.float32, copy=True)
    out = nan_inf_to_num(out, value=0.0)

    out[:4] = out[:4] / 10000.0
    out[4:6] = (out[4:6] + 20.0) / 20.0
    out[6] = out[6] / 100.0
    return out


def valid_pixel_mask_from_fmask(fmask: np.ndarray, threshold: float = 60.0) -> np.ndarray:
    """
    fmask: (H, W) in [0,100] (raw, not normalized) or [0,1] (normalized).
    Returns boolean valid mask where cloud prob is below threshold.
    """
    if fmask.ndim != 2:
        raise ValueError(f"Expected (H,W), got {fmask.shape}")
    # Heuristic: if values are <= 1.5, treat as normalized probability.
    if float(np.nanmax(fmask)) <= 1.5:
        thr = threshold / 100.0
    else:
        thr = threshold
    return fmask < thr


def cloud_pixel_mask_from_fmask(fmask: np.ndarray, threshold: float = 60.0) -> np.ndarray:
    return ~valid_pixel_mask_from_fmask(fmask, threshold=threshold)

