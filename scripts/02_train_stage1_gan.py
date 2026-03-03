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
def evaluate(generator, loader, device):
    generator.eval()
    metrics = {"psnr": 0.0, "ssim": 0.0, "n": 0}
    sample = None
    for batch in loader:
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        mask = batch["valid_mask"].to(device)
        pred = generator(inp)
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
    ap = argparse.ArgumentParser(description="Stage1 Tiny GAN training (VV/VH -> B4/B3/B2/B8)")
    ap.add_argument("--dataset-config", default="config/dataset.yaml")
    ap.add_argument("--train-config", default="config/stage1_gan.yaml")
    ap.add_argument("--index-path", default=None)
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda-l1", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dataset_cfg, train_cfg = resolve_configs(args.dataset_config, args.train_config)
    index_path = args.index_path or dataset_cfg["index"]["out_path"]

    input_channels = train_cfg["io"]["input_channels"]
    target_channels = train_cfg["io"]["target_channels"]
    cloud_threshold = float(train_cfg["masking"].get("cloud_threshold", 30.0))
    min_valid_ratio = float(train_cfg.get("subset_min_valid_ratio", 0.0))
    top_n = int(train_cfg["train"]["subset"].get("n_patches", 32))
    seed = int(train_cfg["train"]["subset"].get("seed", args.seed))

    ds_train, ds_val, kept_n = build_stage1_datasets(
        index_path=index_path,
        input_channels=input_channels,
        target_channels=target_channels,
        cloud_threshold=cloud_threshold,
        min_valid_ratio=min_valid_ratio,
        top_n=top_n,
        seed=seed,
    )

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    exp_id = make_exp_id(stage="stage1", model_name="gan", patch_n=top_n, seed=seed)
    out = prepare_output_dirs(args.output_root, exp_id)
    save_resolved_configs(out["root"], dataset_cfg, train_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = build_pix2pix_generator(in_channels=len(input_channels), out_channels=len(target_channels)).to(device)
    discriminator = build_patchgan_discriminator(in_channels=len(input_channels), out_channels=len(target_channels)).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    final_train = {}
    for _ in range(args.epochs):
        final_train = train_one_epoch(generator, discriminator, train_loader, opt_g, opt_d, device, args.lambda_l1)

    val_metrics, sample = evaluate(generator, val_loader, device)
    if sample is not None:
        patch_id, inp, tgt, pred, mask = sample
        save_sample_rgb(out["samples"], f"{patch_id}_gan", inp, tgt, pred, mask)

    metrics_payload = {
        "model": "gan",
        "device": str(device),
        "kept_patches": kept_n,
        "train_size": len(ds_train),
        "val_size": len(ds_val),
        "cloud_threshold_valid_pixel": cloud_threshold,
        "train": final_train,
        "val": val_metrics,
    }
    write_metrics_json(out["root"] / "metrics.json", metrics_payload)
    torch.save(generator.state_dict(), out["root"] / "gan_generator.pt")
    torch.save(discriminator.state_dict(), out["root"] / "gan_discriminator.pt")

    print(f"Saved outputs: {out['root']}")
    print(f"Val metrics: {val_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
