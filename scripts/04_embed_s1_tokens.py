from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loader.tiff_window import WindowSpec, read_window
from loader.tok2s2_dataset import zscore_per_band
from model.copernicus_fm import build_copernicus_fm
from scripts.common import ensure_dir, load_config


CONFIG_PATH = "config/tok2s2_embed.yaml"


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


def _window_center_lonlat_and_gsd(scene_path: str | Path, win: WindowSpec):
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import transform

    p = Path(scene_path).expanduser().resolve()
    w = Window(win.x, win.y, win.w, win.h)
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


def _load_rows(csv_path: str | Path) -> List[Dict[str, str]]:
    p = Path(csv_path).expanduser().resolve()
    with p.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_rows(csv_path: str | Path, rows: List[Dict[str, str]]) -> None:
    p = Path(csv_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with p.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _feature_index_for_variant(variant: str) -> int:
    return 23 if "large" in variant else 11


def main() -> int:
    cfg = load_config(CONFIG_PATH)
    input_csv = cfg["input_csv"]
    output_csv = cfg["output_csv"]
    token_dir = ensure_dir(cfg["token_dir"])
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    limit_rows = cfg.get("limit_rows", None)

    cfm_cfg = cfg["copernicus_fm"]
    variant = cfm_cfg.get("variant", "vit_base_varlang_e100")
    checkpoint_path = cfm_cfg.get("checkpoint_path", None)
    strict_load = bool(cfm_cfg.get("strict_load", False))

    model_kwargs = {
        "return_intermediate": True,
        "intermediate_indices": [_feature_index_for_variant(variant)],
    }
    backbone, msg, ckpt = build_copernicus_fm(
        repo_root=REPO_ROOT,
        variant=variant,
        checkpoint_path=checkpoint_path,
        strict=strict_load,
        model_kwargs=model_kwargs,
    )
    backbone.to(device)
    backbone.eval()

    rows = _load_rows(input_csv)
    if isinstance(limit_rows, int) and limit_rows > 0:
        rows = rows[:limit_rows]

    s1_bands = tuple(int(v) for v in cfg.get("s1_band_indices_1based", [1, 2]))
    kernel_size = int(cfg.get("kernel_size", 16))
    meta_patch_pixels = int(cfg.get("meta_patch_pixels", 16))
    spectral_wv = float(cfg.get("spectral_wave", 5e7))
    spectral_bw = float(cfg.get("spectral_bandwidth", 1e9))

    updated_rows: List[Dict[str, str]] = []
    with torch.no_grad():
        for row_index, row in enumerate(rows):
            tile_row = int(row["tile_row"])
            tile_col = int(row["tile_col"])
            tile_size = int(row.get("tile_size", 256))
            win = WindowSpec(x=tile_col, y=tile_row, w=tile_size, h=tile_size)

            s1 = read_window(row["s1_path"], win, bands=s1_bands)
            s1 = zscore_per_band(s1)
            s1_tensor = torch.from_numpy(s1).unsqueeze(0).to(device=device, dtype=torch.float32)

            lon, lat, gsd_m = _window_center_lonlat_and_gsd(row["s1_path"], win)
            s1_date = _parse_date_or_none(row.get("s1_date"))
            delta_days = _days_since_unix_epoch(s1_date)
            patch_area_km2 = float((meta_patch_pixels * gsd_m) ** 2 / 1_000_000.0)
            patch_area_km2 = float(min(510000000.0, max(0.001, patch_area_km2)))
            meta = torch.tensor([[lon, lat, delta_days, patch_area_km2]], dtype=torch.float32, device=device)

            channels = int(s1_tensor.shape[1])
            wvs = [spectral_wv] * channels
            bws = [spectral_bw] * channels

            _, intermediate = backbone(
                s1_tensor,
                meta,
                wvs,
                bws,
                language_embed=None,
                input_mode="spectral",
                kernel_size=kernel_size,
            )
            feature = intermediate[-1][0].detach().cpu().numpy()
            token = np.transpose(feature, (1, 2, 0)).astype(np.float32, copy=False)

            token_name = f"tok_{row_index:07d}.npy"
            token_path = Path(token_dir) / token_name
            np.save(token_path, token)

            out_row = dict(row)
            out_row["s1_tok_path"] = str(token_path.resolve())
            updated_rows.append(out_row)

    _write_rows(output_csv, updated_rows)
    print(f"Loaded checkpoint: {ckpt}")
    print(f"load_state_dict_msg: {msg}")
    print(f"Wrote tokens: {len(updated_rows)} to {Path(token_dir).resolve()}")
    print(f"Wrote csv: {Path(output_csv).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
