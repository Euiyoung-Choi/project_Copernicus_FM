from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class WindowSpec:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class SceneMeta:
    path: Path
    width: int
    height: int
    band_count: int
    dtypes: Tuple[str, ...]


def _require_rasterio():
    try:
        import rasterio  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Reading BigTIFF windows requires `rasterio`.\n"
            "Install: python -m pip install rasterio\n"
            f"Original import error: {e}"
        ) from e


def read_scene_meta(scene_path: str | Path) -> SceneMeta:
    _require_rasterio()
    import rasterio

    p = Path(scene_path).expanduser().resolve()
    with rasterio.open(p) as src:
        return SceneMeta(
            path=p,
            width=int(src.width),
            height=int(src.height),
            band_count=int(src.count),
            dtypes=tuple(src.dtypes),
        )


def read_window(scene_path: str | Path, window: WindowSpec, bands: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """
    Returns array with shape (C, h, w).
    bands: 1-based band indices for rasterio. If None, reads all bands.
    """
    _require_rasterio()
    import rasterio
    from rasterio.windows import Window

    p = Path(scene_path).expanduser().resolve()
    w = Window(window.x, window.y, window.w, window.h)
    with rasterio.open(p) as src:
        if bands is None:
            arr = src.read(window=w)
        else:
            arr = src.read(indexes=list(bands), window=w)
    return arr.astype(np.float32, copy=False)

