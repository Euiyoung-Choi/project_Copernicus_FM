from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.copernicus_fm import build_copernicus_fm
from model.losses import masked_l1_loss
from scripts.train_stage1_common import (
    build_stage1_datasets,
    eval_metrics,
    make_exp_id,
    prepare_output_dirs,
    resolve_configs,
    save_resolved_configs,
    save_sample_rgb,
    write_metrics_json,
)


class UpDecoder(nn.Module):
    def __init__(self, in_channels: int = 768, out_channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 16->32
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 32->64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 64->128
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 128->256
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def _feature_index_for_variant(variant: str) -> int:
    return 23 if "large" in variant else 11


def _build_wave_bw(train_cfg):
    # Placeholder for SAR VV/VH channel metadata (2 channels).
    spectral = train_cfg.get("spectral", {})
    wave = spectral.get("wave_list", [5600.0, 5600.0])
    bw = spectral.get("bandwidth", [10.0, 10.0])
    return wave, bw


def train_one_epoch(backbone, decoder, loader, optimizer, device, wave_list, bandwidth):
    backbone.eval()
    decoder.train()
    meter = {"l1": 0.0, "n": 0}

    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        mask = batch["valid_mask"].to(device)
        meta = torch.full((inp.shape[0], 4), float("nan"), device=device)

        _, intermediate = backbone(
            inp,
            meta,
            wave_list,
            bandwidth,
            language_embed=None,
            input_mode="spectral",
            kernel_size=16,
        )
        feat = intermediate[-1]
        pred = decoder(feat)

        loss = masked_l1_loss(pred, tgt, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meter["l1"] += float(loss.item())
        meter["n"] += 1

    n = max(1, meter["n"])
    return {"l1": meter["l1"] / n}


@torch.no_grad()
def evaluate(backbone, decoder, loader, device, wave_list, bandwidth):
    backbone.eval()
    decoder.eval()
    metrics = {"psnr": 0.0, "ssim": 0.0, "n": 0}
    sample = None
    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        mask = batch["valid_mask"].to(device)
        meta = torch.full((inp.shape[0], 4), float("nan"), device=device)
        _, intermediate = backbone(
            inp,
            meta,
            wave_list,
            bandwidth,
            language_embed=None,
            input_mode="spectral",
            kernel_size=16,
        )
        pred = decoder(intermediate[-1])
        m = eval_metrics(pred, tgt, mask)
        metrics["psnr"] += m["psnr"]
        metrics["ssim"] += m["ssim"]
        metrics["n"] += 1
        if sample is None:
            sample = (
                batch["patch_id"][0],
                inp[0].cpu(),
                tgt[0].cpu(),
                pred[0].cpu(),
                mask[0].cpu(),
            )
    n = max(1, metrics["n"])
    metrics["psnr"] /= n
    metrics["ssim"] /= n
    return metrics, sample


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage1 Tiny Copernicus-FM training (decoder fine-tune).")
    ap.add_argument("--dataset-config", default="config/dataset.yaml")
    ap.add_argument("--train-config", default="config/stage1_copfm.yaml")
    ap.add_argument("--index-path", default=None)
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dataset_cfg, train_cfg = resolve_configs(args.dataset_config, args.train_config)
    index_path = args.index_path or dataset_cfg["index"]["out_path"]

    input_channels = train_cfg["io"]["input_channels"]
    target_channels = train_cfg["io"]["target_channels"]
    cloud_threshold = float(train_cfg["masking"].get("cloud_threshold", 30.0))
    min_valid_ratio = float(train_cfg.get("subset_min_valid_ratio", 0.0))
    max_nan_ratio = float(train_cfg.get("subset_max_nan_ratio", 1.0))
    top_n = int(train_cfg["train"]["subset"].get("n_patches", 32))
    seed = int(train_cfg["train"]["subset"].get("seed", args.seed))

    ds_train, ds_val, kept_n = build_stage1_datasets(
        index_path=index_path,
        input_channels=input_channels,
        target_channels=target_channels,
        cloud_threshold=cloud_threshold,
        min_valid_ratio=min_valid_ratio,
        max_nan_ratio=max_nan_ratio,
        top_n=top_n,
        seed=seed,
    )
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    exp_id = make_exp_id(stage="stage1", model_name="copfm", patch_n=top_n, seed=seed)
    out = prepare_output_dirs(args.output_root, exp_id)
    save_resolved_configs(out["root"], dataset_cfg, train_cfg)

    cfm_cfg = train_cfg.get("copernicus_fm", {})
    variant = cfm_cfg.get("variant", "vit_base_varlang_e100")
    ckpt_path = cfm_cfg.get("checkpoint_path", None)
    strict = bool(cfm_cfg.get("strict_load", False))
    freeze_backbone = bool(cfm_cfg.get("freeze_backbone", True))

    model_kwargs = {
        "return_intermediate": True,
        "intermediate_indices": [_feature_index_for_variant(variant)],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, load_msg, loaded_ckpt = build_copernicus_fm(
        repo_root=REPO_ROOT,
        variant=variant,
        checkpoint_path=ckpt_path,
        strict=strict,
        model_kwargs=model_kwargs,
    )
    backbone.to(device)
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    decoder = UpDecoder(in_channels=1024 if "large" in variant else 768, out_channels=len(target_channels)).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    wave_list, bandwidth = _build_wave_bw(train_cfg)

    final_train = {}
    for _ in range(args.epochs):
        final_train = train_one_epoch(backbone, decoder, train_loader, optimizer, device, wave_list, bandwidth)

    val_metrics, sample = evaluate(backbone, decoder, val_loader, device, wave_list, bandwidth)
    if sample is not None:
        patch_id, inp, tgt, pred, mask = sample
        save_sample_rgb(out["samples"], f"{patch_id}_copfm", inp, tgt, pred, mask)

    metrics_payload = {
        "model": "copernicus_fm_decoder",
        "device": str(device),
        "kept_patches": kept_n,
        "train_size": len(ds_train),
        "val_size": len(ds_val),
        "cloud_threshold_valid_pixel": cloud_threshold,
        "checkpoint": str(loaded_ckpt),
        "load_state_dict_msg": str(load_msg),
        "train": final_train,
        "val": val_metrics,
    }
    write_metrics_json(out["root"] / "metrics.json", metrics_payload)
    torch.save(decoder.state_dict(), out["root"] / "copfm_decoder.pt")
    print(f"Saved outputs: {out['root']}")
    print(f"Val metrics: {val_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
