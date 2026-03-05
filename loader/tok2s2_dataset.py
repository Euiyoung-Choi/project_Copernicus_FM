from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from loader.tiff_window import WindowSpec, read_window


def zscore_per_band(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {x.shape}")
    out = x.astype(np.float32, copy=True)
    for channel_index in range(out.shape[0]):
        channel = out[channel_index]
        valid = np.isfinite(channel)
        if not np.any(valid):
            out[channel_index] = 0.0
            continue
        mean = float(channel[valid].mean())
        std = float(channel[valid].std())
        if std < eps:
            out[channel_index] = 0.0
        else:
            out[channel_index] = (channel - mean) / (std + eps)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _read_csv_rows(csv_path: str | Path) -> List[Dict[str, str]]:
    path = Path(csv_path).expanduser().resolve()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _rows_from_index_jsonl(index_path: Path) -> List[Dict[str, str]]:
    """
    Convert stage index jsonl rows into pair-like rows.
    Assumes one fused 7-band GeoTIFF per scene where S1/S2 are in the same file.
    """
    rows: List[Dict[str, str]] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            scene_path = str(item["scene_path"])
            rows.append(
                {
                    "s1_path": scene_path,
                    "s2_path": scene_path,
                    "tile_row": str(int(item["y"])),
                    "tile_col": str(int(item["x"])),
                    "tile_size": str(int(item.get("w", 256))),
                    "s1_date": "",
                    "s2_date": "",
                }
            )
    return rows


def _read_rows(csv_or_jsonl_path: str | Path) -> List[Dict[str, str]]:
    path = Path(csv_or_jsonl_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input manifest not found: {path}")
    if path.suffix.lower() == ".jsonl":
        return _rows_from_index_jsonl(path)
    return _read_csv_rows(path)


@dataclass(frozen=True)
class Tok2S2Spec:
    s2_band_indices_1based: Sequence[int]
    tok_key: str = "s1_tok_path"
    s2_norm: str = "zscore"
    tok_dtype: torch.dtype = torch.float32


class Tok2S2Dataset(Dataset):
    def __init__(self, csv_path: str | Path, spec: Tok2S2Spec):
        self.rows = _read_rows(csv_path)
        self.spec = spec
        if len(self.rows) == 0:
            raise RuntimeError(f"No rows found in csv: {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        token_path = row[self.spec.tok_key]
        token = np.load(token_path).astype(np.float32, copy=False)
        if token.ndim != 3:
            raise ValueError(f"Expected token shape (Hp,Wp,D) or (D,Hp,Wp), got {token.shape}")
        if token.shape[0] in {768, 1024}:
            token = np.transpose(token, (1, 2, 0))
        token_tensor = torch.from_numpy(token).to(self.spec.tok_dtype)

        tile_row = int(row["tile_row"])
        tile_col = int(row["tile_col"])
        tile_size = int(row.get("tile_size", 256))
        win = WindowSpec(x=tile_col, y=tile_row, w=tile_size, h=tile_size)
        bands = tuple(int(v) for v in self.spec.s2_band_indices_1based)
        s2 = read_window(row["s2_path"], win, bands=bands)
        if self.spec.s2_norm == "zscore":
            s2 = zscore_per_band(s2)
        s2_tensor = torch.from_numpy(s2.astype(np.float32, copy=False))
        return token_tensor, s2_tensor


def _parse_date_or_none(value: str | None):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%d").date()


def _days_since_unix_epoch(value: date | None) -> float:
    if value is None:
        return float("nan")
    return float((value - date(1970, 1, 1)).days)


def window_center_lonlat_and_gsd(scene_path: str | Path, win: WindowSpec):
    import rasterio
    from rasterio.warp import transform

    p = Path(scene_path).expanduser().resolve()
    with rasterio.open(p) as src:
        cx = win.x + win.w / 2.0
        cy = win.y + win.h / 2.0
        px, py = rasterio.transform.xy(src.transform, cy, cx, offset="center")
        if src.crs is not None and str(src.crs).upper() != "EPSG:4326":
            lon, lat = transform(src.crs, "EPSG:4326", [px], [py])
            lon = float(lon[0])
            lat = float(lat[0])
        else:
            lon = float(px)
            lat = float(py)
        gsd_x = float(abs(src.transform.a))
        gsd_y = float(abs(src.transform.e))
        gsd_m = (gsd_x + gsd_y) / 2.0
    return lon, lat, gsd_m


@dataclass(frozen=True)
class Tok2S2OnTheFlySpec:
    s1_band_indices_1based: Sequence[int]
    s2_band_indices_1based: Sequence[int]
    s1_norm: str = "zscore"
    s2_norm: str = "zscore"
    meta_patch_pixels: int = 16


class Tok2S2OnTheFlyDataset(Dataset):
    """
    Returns:
      - s1: (C,H,W) normalized
      - meta: (4,) [lon,lat,delta_days,patch_area_km2]
      - s2: (C,H,W) normalized
    """

    def __init__(self, csv_path: str | Path, spec: Tok2S2OnTheFlySpec):
        self.rows = _read_rows(csv_path)
        self.spec = spec
        if len(self.rows) == 0:
            raise RuntimeError(f"No rows found in csv: {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        tile_row = int(row["tile_row"])
        tile_col = int(row["tile_col"])
        tile_size = int(row.get("tile_size", 256))
        win = WindowSpec(x=tile_col, y=tile_row, w=tile_size, h=tile_size)

        s1 = read_window(row["s1_path"], win, bands=tuple(int(v) for v in self.spec.s1_band_indices_1based))
        if self.spec.s1_norm == "zscore":
            s1 = zscore_per_band(s1)
        s1_tensor = torch.from_numpy(s1.astype(np.float32, copy=False))

        s2 = read_window(row["s2_path"], win, bands=tuple(int(v) for v in self.spec.s2_band_indices_1based))
        if self.spec.s2_norm == "zscore":
            s2 = zscore_per_band(s2)
        s2_tensor = torch.from_numpy(s2.astype(np.float32, copy=False))

        lon, lat, gsd_m = window_center_lonlat_and_gsd(row["s1_path"], win)
        s1_date = _parse_date_or_none(row.get("s1_date"))
        delta_days = _days_since_unix_epoch(s1_date)
        patch_area_km2 = float((self.spec.meta_patch_pixels * gsd_m) ** 2 / 1_000_000.0)
        meta = torch.tensor([lon, lat, delta_days, patch_area_km2], dtype=torch.float32)

        return s1_tensor, meta, s2_tensor
