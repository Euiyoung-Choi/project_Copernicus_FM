from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from loader.preprocess import normalize_channels, valid_pixel_mask_from_fmask
from loader.tiff_window import WindowSpec, read_window


@dataclass(frozen=True)
class Stage1DatasetSpec:
    input_channels: Sequence[int]
    target_channels: Sequence[int]
    cloud_threshold: float = 30.0
    min_valid_ratio: float = 0.95
    # Allow NaN/Inf-containing patches by default; preprocessing replaces them with 0.
    max_nan_ratio: float = 1.0
    top_n: int | None = None


def read_index_jsonl(index_path: str | Path) -> List[Dict]:
    p = Path(index_path).expanduser().resolve()
    rows: List[Dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_index_rows(rows: List[Dict], spec: Stage1DatasetSpec) -> List[Dict]:
    kept = []
    for row in rows:
        nan_ratio = float(row.get("nan_ratio", 0.0))
        valid_ratio = float(row.get("valid_ratio", 0.0))
        if nan_ratio > spec.max_nan_ratio:
            continue
        if valid_ratio < spec.min_valid_ratio:
            continue
        kept.append(row)
    if spec.top_n is not None:
        kept = kept[: spec.top_n]
    return kept


class Stage1PatchDataset(Dataset):
    """
    Returns dict with:
      - input: [Cin,H,W] float32
      - target: [Cout,H,W] float32
      - valid_mask: [1,H,W] float32 (1=valid)
      - patch_id: str
    """

    def __init__(self, rows: List[Dict], spec: Stage1DatasetSpec):
        self.rows = rows
        self.spec = spec

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        win = WindowSpec(x=int(row["x"]), y=int(row["y"]), w=int(row["w"]), h=int(row["h"]))
        img_raw = read_window(row["scene_path"], win, bands=None)  # (7,H,W), raw scale
        fmask_raw = img_raw[6].astype(np.float32, copy=False)
        valid_mask = valid_pixel_mask_from_fmask(fmask_raw, threshold=self.spec.cloud_threshold).astype(np.float32)

        img = normalize_channels(img_raw)  # (7,H,W), normalized
        input_tensor = torch.from_numpy(img[list(self.spec.input_channels)])
        target_tensor = torch.from_numpy(img[list(self.spec.target_channels)])
        mask_tensor = torch.from_numpy(valid_mask[None, ...])

        return {
            "input": input_tensor,
            "target": target_tensor,
            "valid_mask": mask_tensor,
            "patch_id": row["patch_id"],
        }
