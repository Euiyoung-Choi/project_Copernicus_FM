from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from loader.preprocess import nan_inf_to_num
from loader.tiff_window import WindowSpec, read_scene_meta, read_window


@dataclass(frozen=True)
class PatchIndexRecord:
    patch_id: str
    scene_path: str
    x: int
    y: int
    w: int
    h: int
    mean_fmask: float
    cloud_ratio: float
    valid_ratio: float
    nan_ratio: float


def iter_windows(scene_width: int, scene_height: int, patch_size: int, stride: int) -> Iterator[WindowSpec]:
    for y in range(0, scene_height - patch_size + 1, stride):
        for x in range(0, scene_width - patch_size + 1, stride):
            yield WindowSpec(x=x, y=y, w=patch_size, h=patch_size)


def _compute_nan_ratio(img: np.ndarray) -> float:
    # img: (C,H,W)
    total = img.size
    finite = np.isfinite(img).sum()
    return float(1.0 - (finite / max(total, 1)))


def _compute_fmask_stats(fmask: np.ndarray, cloud_threshold: float = 60.0) -> Dict[str, float]:
    """
    fmask: (H,W) raw 0..100 (preferred) or normalized 0..1.
    """
    f = fmask.astype(np.float32, copy=False)
    finite = f[np.isfinite(f)]
    if finite.size == 0:
        return {"mean_fmask": float("nan"), "cloud_ratio": float("nan"), "valid_ratio": float("nan")}

    # If normalized probability, convert threshold accordingly.
    if float(np.nanmax(finite)) <= 1.5:
        thr = cloud_threshold / 100.0
    else:
        thr = cloud_threshold

    cloud = finite >= thr
    cloud_ratio = float(cloud.mean())
    return {
        "mean_fmask": float(finite.mean()),
        "cloud_ratio": cloud_ratio,
        "valid_ratio": float(1.0 - cloud_ratio),
    }


def build_patch_index_for_scene(
    scene_path: str | Path,
    patch_size: int = 256,
    stride: int = 256,
    cloud_threshold: float = 60.0,
    fmask_band_1based: int = 7,
    limit_patches: Optional[int] = None,
) -> List[PatchIndexRecord]:
    meta = read_scene_meta(scene_path)
    if meta.band_count < fmask_band_1based:
        raise ValueError(f"Scene has {meta.band_count} bands but expected at least {fmask_band_1based}: {meta.path}")

    records: List[PatchIndexRecord] = []
    base = Path(meta.path).stem
    for i, win in enumerate(iter_windows(meta.width, meta.height, patch_size=patch_size, stride=stride)):
        if limit_patches is not None and i >= limit_patches:
            break

        # Read only fmask band for stats + a minimal read for nan_ratio (all bands).
        img = read_window(meta.path, win, bands=None)
        nan_ratio = _compute_nan_ratio(img)

        fmask = img[fmask_band_1based - 1]  # 0-based indexing
        fmask = nan_inf_to_num(fmask, value=0.0)
        stats = _compute_fmask_stats(fmask, cloud_threshold=cloud_threshold)

        patch_id = f"{base}_x{win.x}_y{win.y}"
        records.append(
            PatchIndexRecord(
                patch_id=patch_id,
                scene_path=str(meta.path),
                x=win.x,
                y=win.y,
                w=win.w,
                h=win.h,
                mean_fmask=stats["mean_fmask"],
                cloud_ratio=stats["cloud_ratio"],
                valid_ratio=stats["valid_ratio"],
                nan_ratio=nan_ratio,
            )
        )

    return records


def write_index_jsonl(records: List[PatchIndexRecord], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def summarize_records(records: List[PatchIndexRecord]) -> Dict[str, Dict[str, float]]:
    def _summ(vals: np.ndarray) -> Dict[str, float]:
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"count": 0.0}
        return {
            "count": float(vals.size),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
        }

    mean_fmask = np.array([r.mean_fmask for r in records], dtype=np.float32)
    cloud_ratio = np.array([r.cloud_ratio for r in records], dtype=np.float32)
    nan_ratio = np.array([r.nan_ratio for r in records], dtype=np.float32)
    valid_ratio = np.array([r.valid_ratio for r in records], dtype=np.float32)

    return {
        "mean_fmask": _summ(mean_fmask),
        "cloud_ratio": _summ(cloud_ratio),
        "valid_ratio": _summ(valid_ratio),
        "nan_ratio": _summ(nan_ratio),
    }

