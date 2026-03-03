from __future__ import annotations

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

# -----------------------------------------------------------------------------
# Runtime settings (edit here, no argparse)
# -----------------------------------------------------------------------------
DATASET_CONFIG_PATH = "config/dataset.yaml"
TRAIN_CONFIG_PATH = "config/stage1_copfm.yaml"
INDEX_PATH_OVERRIDE = None
OUTPUT_ROOT = "output"
EPOCHS = 2
BATCH_SIZE = 4
NUM_WORKERS = 0
LR = 1e-3
SEED = 0
MIN_VALID_PIXEL_RATIO_FOR_METRICS = 0.01


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


class TransHybridDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 768,
        out_channels: int = 4,
        trans_dim: int = 384,
        trans_heads: int = 8,
        trans_layers: int = 4,
        trans_ffn_ratio: int = 4,
    ):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, trans_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim,
            nhead=trans_heads,
            dim_feedforward=trans_dim * trans_ffn_ratio,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
        self.proj_out = nn.Sequential(
            nn.Conv2d(trans_dim, 256, kernel_size=3, padding=1),
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
        # x: [B, C, 16, 16]
        x = self.proj_in(x)  # [B, D, H, W]
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, D]
        tokens = self.transformer(tokens)
        x = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        return self.proj_out(x)


def build_decoder(decoder_cfg, in_channels: int, out_channels: int):
    decoder_type = str(decoder_cfg.get("type", "upconv")).lower()
    if decoder_type == "upconv":
        return UpDecoder(in_channels=in_channels, out_channels=out_channels), decoder_type
    if decoder_type == "trans_hybrid":
        return (
            TransHybridDecoder(
                in_channels=in_channels,
                out_channels=out_channels,
                trans_dim=int(decoder_cfg.get("trans_dim", 384)),
                trans_heads=int(decoder_cfg.get("trans_heads", 8)),
                trans_layers=int(decoder_cfg.get("trans_layers", 4)),
                trans_ffn_ratio=int(decoder_cfg.get("trans_ffn_ratio", 4)),
            ),
            decoder_type,
        )
    raise ValueError(f"Unknown decoder.type: {decoder_type}")


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
def evaluate(backbone, decoder, loader, device, wave_list, bandwidth, min_valid_pixel_ratio_for_metrics: float):
    backbone.eval()
    decoder.eval()
    metrics = {"psnr": 0.0, "ssim": 0.0, "valid_pixel_ratio": 0.0, "eligible_batches": 0, "total_batches": 0}
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
        m = eval_metrics(pred, tgt, mask, min_valid_pixel_ratio=min_valid_pixel_ratio_for_metrics)
        metrics["total_batches"] += 1
        metrics["valid_pixel_ratio"] += float(m["valid_pixel_ratio"])
        if bool(m["eligible"]):
            metrics["psnr"] += float(m["psnr"])
            metrics["ssim"] += float(m["ssim"])
            metrics["eligible_batches"] += 1
        if sample is None:
            sample = (
                batch["patch_id"][0],
                inp[0].cpu(),
                tgt[0].cpu(),
                pred[0].cpu(),
                mask[0].cpu(),
            )
    total_batches = max(1, metrics["total_batches"])
    eligible = metrics["eligible_batches"]
    metrics["valid_pixel_ratio"] /= total_batches
    if eligible > 0:
        metrics["psnr"] /= eligible
        metrics["ssim"] /= eligible
    else:
        metrics["psnr"] = None
        metrics["ssim"] = None
    return metrics, sample


def main() -> int:
    torch.manual_seed(SEED)
    dataset_cfg, train_cfg = resolve_configs(DATASET_CONFIG_PATH, TRAIN_CONFIG_PATH)
    runtime = train_cfg.get("runtime", {})
    index_path = INDEX_PATH_OVERRIDE or dataset_cfg["index"]["out_path"]

    input_channels = train_cfg["io"]["input_channels"]
    target_channels = train_cfg["io"]["target_channels"]
    cloud_threshold = float(train_cfg["masking"].get("cloud_threshold", 30.0))
    min_valid_ratio = float(train_cfg.get("subset_min_valid_ratio", 0.0))
    max_nan_ratio = float(train_cfg.get("subset_max_nan_ratio", 1.0))
    top_n = int(train_cfg["train"]["subset"].get("n_patches", 32))
    seed = int(train_cfg["train"]["subset"].get("seed", runtime.get("seed", SEED)))
    epochs = int(runtime.get("epochs", EPOCHS))
    batch_size = int(runtime.get("batch_size", BATCH_SIZE))
    num_workers = int(runtime.get("num_workers", NUM_WORKERS))
    lr = float(runtime.get("lr", LR))
    min_valid_pixel_ratio_for_metrics = float(
        train_cfg.get("eval", {}).get("min_valid_pixel_ratio_for_metrics", MIN_VALID_PIXEL_RATIO_FOR_METRICS)
    )
    save_val_png_each_epoch = bool(train_cfg.get("eval", {}).get("save_val_png_each_epoch", True))

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
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    exp_id = make_exp_id(stage="stage1", model_name="copfm", patch_n=top_n, seed=seed)
    out = prepare_output_dirs(OUTPUT_ROOT, exp_id)
    save_resolved_configs(out["root"], dataset_cfg, train_cfg)

    cfm_cfg = train_cfg.get("copernicus_fm", {})
    decoder_cfg = train_cfg.get("decoder", {})
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

    decoder, decoder_type = build_decoder(
        decoder_cfg,
        in_channels=1024 if "large" in variant else 768,
        out_channels=len(target_channels),
    )
    decoder = decoder.to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    wave_list, bandwidth = _build_wave_bw(train_cfg)

    final_train = {}
    history = []
    last_sample = None
    last_val_metrics = None
    for epoch_idx in range(1, epochs + 1):
        final_train = train_one_epoch(backbone, decoder, train_loader, optimizer, device, wave_list, bandwidth)
        val_metrics, sample = evaluate(
            backbone,
            decoder,
            val_loader,
            device,
            wave_list,
            bandwidth,
            min_valid_pixel_ratio_for_metrics=min_valid_pixel_ratio_for_metrics,
        )
        if save_val_png_each_epoch and sample is not None:
            patch_id, inp, tgt, pred, mask = sample
            save_sample_rgb(out["samples"], f"epoch{epoch_idx:03d}_{patch_id}_copfm", inp, tgt, pred, mask)
        history.append({"epoch": epoch_idx, "train": final_train, "val": val_metrics})
        last_sample = sample
        last_val_metrics = val_metrics
        print(
            f"[epoch {epoch_idx}/{epochs}] "
            f"l1={final_train['l1']:.6f} val_psnr={val_metrics['psnr']} val_ssim={val_metrics['ssim']}"
        )

    if (not save_val_png_each_epoch) and last_sample is not None:
        patch_id, inp, tgt, pred, mask = last_sample
        save_sample_rgb(out["samples"], f"{patch_id}_copfm", inp, tgt, pred, mask)

    metrics_payload = {
        "model": "copernicus_fm_decoder",
        "decoder_type": decoder_type,
        "device": str(device),
        "kept_patches": kept_n,
        "train_size": len(ds_train),
        "val_size": len(ds_val),
        "cloud_threshold_valid_pixel": cloud_threshold,
        "checkpoint": str(loaded_ckpt),
        "load_state_dict_msg": str(load_msg),
        "train": final_train,
        "val": last_val_metrics,
        "history": history,
    }
    write_metrics_json(out["root"] / "metrics.json", metrics_payload)
    torch.save(decoder.state_dict(), out["root"] / "copfm_decoder.pt")
    print(f"Saved outputs: {out['root']}")
    print(f"Decoder: {decoder_type}")
    print(f"Val metrics: {val_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
