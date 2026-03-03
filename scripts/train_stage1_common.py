from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch

from loader.dataset import Stage1DatasetSpec, Stage1PatchDataset, filter_index_rows, read_index_jsonl
from model.losses import masked_psnr, masked_ssim
from scripts.common import dump_config, ensure_dir, load_config


def make_exp_id(stage: str, model_name: str, patch_n: int, seed: int) -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{now}_{stage}_{model_name}_file1_patch{patch_n}_seed{seed}"


def resolve_configs(dataset_config_path: str, train_config_path: str) -> Tuple[Dict, Dict]:
    dataset_cfg = load_config(dataset_config_path)
    train_cfg = load_config(train_config_path)
    return dataset_cfg, train_cfg


def build_stage1_datasets(
    index_path: str | Path,
    input_channels,
    target_channels,
    cloud_threshold: float,
    min_valid_ratio: float,
    top_n: int,
    seed: int,
):
    rows = read_index_jsonl(index_path)
    spec = Stage1DatasetSpec(
        input_channels=input_channels,
        target_channels=target_channels,
        cloud_threshold=cloud_threshold,
        min_valid_ratio=min_valid_ratio,
        max_nan_ratio=0.0,
        top_n=top_n,
    )
    rows = filter_index_rows(rows, spec)
    if len(rows) < 2:
        raise RuntimeError(f"Not enough rows after filtering: {len(rows)}")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(rows), generator=g).tolist()
    split_idx = max(1, int(0.9 * len(rows)))
    train_rows = [rows[i] for i in perm[:split_idx]]
    val_rows = [rows[i] for i in perm[split_idx:]]
    if len(val_rows) == 0:
        val_rows = train_rows[-1:]
        train_rows = train_rows[:-1]

    return Stage1PatchDataset(train_rows, spec), Stage1PatchDataset(val_rows, spec), len(rows)


def eval_metrics(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, float]:
    psnr = float(masked_psnr(pred, target, valid_mask).item())
    ssim = float(masked_ssim(pred, target, valid_mask).item())
    return {"psnr": psnr, "ssim": ssim}


def save_sample_rgb(output_dir: Path, sample_name: str, inp, target, pred, valid_mask):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    def to_np(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    inp = to_np(inp)
    target = to_np(target)
    pred = to_np(pred)
    valid_mask = to_np(valid_mask)

    # target/pred: (4,H,W), RGB uses first 3 channels in this ordering.
    gt_rgb = np.clip(np.transpose(target[:3], (1, 2, 0)), 0, 1)
    pd_rgb = np.clip(np.transpose(pred[:3], (1, 2, 0)), 0, 1)
    diff = np.clip(np.abs(pd_rgb - gt_rgb), 0, 1)
    mask = valid_mask[0]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(gt_rgb)
    axes[0].set_title("GT RGB")
    axes[1].imshow(pd_rgb)
    axes[1].set_title("Pred RGB")
    axes[2].imshow(diff)
    axes[2].set_title("|diff|")
    axes[3].imshow(mask, cmap="gray")
    axes[3].set_title("valid_mask(Fmask<30)")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"{sample_name}.png", dpi=140)
    plt.close(fig)


def write_metrics_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_output_dirs(base_output: str | Path, exp_id: str) -> Dict[str, Path]:
    root = ensure_dir(Path(base_output) / exp_id)
    paths = {
        "root": root,
        "logs": ensure_dir(root / "logs"),
        "samples": ensure_dir(root / "samples"),
        "figures": ensure_dir(root / "figures"),
    }
    return paths


def save_resolved_configs(out_root: Path, dataset_cfg: Dict, train_cfg: Dict):
    dump_config(out_root / "config_resolved.yaml", {"dataset": dataset_cfg, "train": train_cfg})

