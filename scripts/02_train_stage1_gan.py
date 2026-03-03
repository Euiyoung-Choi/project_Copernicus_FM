from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.gan_pix2pix import build_patchgan_discriminator, build_pix2pix_generator
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
TRAIN_CONFIG_PATH = "config/stage1_gan.yaml"
INDEX_PATH_OVERRIDE = None  # e.g. "output/index/first_scene.index.jsonl"
OUTPUT_ROOT = "output"
EPOCHS = 2
BATCH_SIZE = 4
NUM_WORKERS = 0
LR = 2e-4
LAMBDA_L1 = 100.0
SEED = 0
MIN_VALID_PIXEL_RATIO_FOR_METRICS = 0.01


def train_one_epoch(generator, discriminator, loader, opt_g, opt_d, device, lambda_l1: float):
    generator.train()
    discriminator.train()
    bce = nn.BCEWithLogitsLoss()
    meter = {"g_total": 0.0, "g_l1": 0.0, "g_adv": 0.0, "d_total": 0.0, "n": 0}

    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        mask = batch["valid_mask"].to(device)

        # D-step
        with torch.no_grad():
            fake = generator(inp)
        d_real = discriminator(inp, tgt)
        d_fake = discriminator(inp, fake)
        d_loss = 0.5 * (bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake)))
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # G-step
        pred = generator(inp)
        g_adv = bce(discriminator(inp, pred), torch.ones_like(d_real))
        g_l1 = masked_l1_loss(pred, tgt, mask)
        g_loss = g_adv + lambda_l1 * g_l1
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        meter["g_total"] += float(g_loss.item())
        meter["g_l1"] += float(g_l1.item())
        meter["g_adv"] += float(g_adv.item())
        meter["d_total"] += float(d_loss.item())
        meter["n"] += 1

    n = max(1, meter["n"])
    return {k: (v / n if k != "n" else v) for k, v in meter.items()}


@torch.no_grad()
def evaluate(generator, loader, device, min_valid_pixel_ratio_for_metrics: float):
    generator.eval()
    metrics = {"psnr": 0.0, "ssim": 0.0, "valid_pixel_ratio": 0.0, "eligible_batches": 0, "total_batches": 0}
    sample = None
    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        mask = batch["valid_mask"].to(device)
        pred = generator(inp)
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
    lambda_l1 = float(runtime.get("lambda_l1", LAMBDA_L1))
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

    exp_id = make_exp_id(stage="stage1", model_name="gan", patch_n=top_n, seed=seed)
    out = prepare_output_dirs(OUTPUT_ROOT, exp_id)
    save_resolved_configs(out["root"], dataset_cfg, train_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = build_pix2pix_generator(in_channels=len(input_channels), out_channels=len(target_channels)).to(device)
    discriminator = build_patchgan_discriminator(in_channels=len(input_channels), out_channels=len(target_channels)).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    final_train = {}
    history = []
    last_sample = None
    last_val_metrics = None
    for epoch_idx in range(1, epochs + 1):
        final_train = train_one_epoch(generator, discriminator, train_loader, opt_g, opt_d, device, lambda_l1)
        val_metrics, sample = evaluate(
            generator,
            val_loader,
            device,
            min_valid_pixel_ratio_for_metrics=min_valid_pixel_ratio_for_metrics,
        )
        if save_val_png_each_epoch and sample is not None:
            patch_id, inp, tgt, pred, mask = sample
            save_sample_rgb(out["samples"], f"epoch{epoch_idx:03d}_{patch_id}_gan", inp, tgt, pred, mask)
        history.append({"epoch": epoch_idx, "train": final_train, "val": val_metrics})
        last_sample = sample
        last_val_metrics = val_metrics
        print(
            f"[epoch {epoch_idx}/{epochs}] "
            f"g_total={final_train['g_total']:.5f} g_l1={final_train['g_l1']:.5f} d={final_train['d_total']:.5f} "
            f"val_psnr={val_metrics['psnr']} val_ssim={val_metrics['ssim']}"
        )

    if (not save_val_png_each_epoch) and last_sample is not None:
        patch_id, inp, tgt, pred, mask = last_sample
        save_sample_rgb(out["samples"], f"{patch_id}_gan", inp, tgt, pred, mask)

    metrics_payload = {
        "model": "gan",
        "device": str(device),
        "kept_patches": kept_n,
        "train_size": len(ds_train),
        "val_size": len(ds_val),
        "cloud_threshold_valid_pixel": cloud_threshold,
        "train": final_train,
        "val": last_val_metrics,
        "history": history,
    }
    write_metrics_json(out["root"] / "metrics.json", metrics_payload)
    torch.save(generator.state_dict(), out["root"] / "gan_generator.pt")
    torch.save(discriminator.state_dict(), out["root"] / "gan_discriminator.pt")

    print(f"Saved outputs: {out['root']}")
    print(f"Val metrics: {val_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
