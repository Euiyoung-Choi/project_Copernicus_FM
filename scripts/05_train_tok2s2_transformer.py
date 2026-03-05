from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loader.tok2s2_dataset import Tok2S2OnTheFlyDataset, Tok2S2OnTheFlySpec
from model.copernicus_fm import build_copernicus_fm
from scripts.common import ensure_dir, load_config


CONFIG_PATH = "config/tok2s2_transformer.yaml"


def to_rgb(x_chw: torch.Tensor, eps: float = 1e-6):
    x = x_chw.detach().float().cpu().numpy()
    rgb = x[[2, 1, 0], :, :]
    lo = np.percentile(rgb, 2)
    hi = np.percentile(rgb, 98)
    rgb = (rgb - lo) / (hi - lo + eps)
    rgb = np.clip(rgb, 0, 1)
    return np.transpose(rgb, (1, 2, 0))


def save_viz_triplet(out_dir: Path, epoch: int, step: int, gt_chw: torch.Tensor, pred_chw: torch.Tensor):
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_rgb = to_rgb(gt_chw)
    pred_rgb = to_rgb(pred_chw)
    diff = np.abs(gt_rgb - pred_rgb).mean(axis=2)

    fig = plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title("GT (RGB)")
    ax1.imshow(gt_rgb)
    ax1.axis("off")
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("Pred (RGB)")
    ax2.imshow(pred_rgb)
    ax2.axis("off")
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title("Abs diff (mean RGB)")
    image = ax3.imshow(diff)
    ax3.axis("off")
    plt.colorbar(image, fraction=0.046, pad=0.04)

    output_file = out_dir / f"e{epoch:03d}_s{step:06d}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=140)
    plt.close(fig)


class SimpleTransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        nhead: int = 8,
        depth: int = 6,
        c_out: int = 4,
        patch: int = 16,
        hp: int = 16,
        wp: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        refine_channels: int = 64,
        refine_depth: int = 3,
    ):
        super().__init__()
        self.c_out = c_out
        self.patch = patch
        self.hp = hp
        self.wp = wp
        self.out_dim = c_out * patch * patch

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, self.out_dim)

        convs = [nn.Conv2d(c_out, refine_channels, 3, padding=1), nn.GELU()]
        for _ in range(max(0, refine_depth - 2)):
            convs += [nn.Conv2d(refine_channels, refine_channels, 3, padding=1), nn.GELU()]
        convs += [nn.Conv2d(refine_channels, c_out, 3, padding=1)]
        self.refine = nn.Sequential(*convs)

    def unpatchify(self, x: torch.Tensor):
        batch_size, token_count, patch_dim = x.shape
        p = self.patch
        c = self.c_out
        if patch_dim != c * p * p:
            raise ValueError(f"Unexpected patch_dim={patch_dim}, expected {c * p * p}")
        if token_count != self.hp * self.wp:
            raise ValueError(f"Unexpected token_count={token_count}, expected {self.hp * self.wp}")

        x = x.view(batch_size, self.hp, self.wp, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(batch_size, c, self.hp * p, self.wp * p)

    def forward(self, token: torch.Tensor):
        if token.dim() == 4:
            token = token.view(token.shape[0], -1, token.shape[-1])
        x = self.blocks(token)
        y = self.head(x)
        image0 = self.unpatchify(y)
        return image0 + self.refine(image0)


def _feature_index_for_variant(variant: str) -> int:
    return 23 if "large" in variant else 11


def main() -> int:
    cfg = load_config(CONFIG_PATH)
    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["train"]
    mz_cfg = cfg["model"]
    cfm_cfg = cfg["copernicus_fm"]
    spc_cfg = cfg["spectral"]

    device_name = tr_cfg.get("device", "cuda")
    device = torch.device(device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu")

    dataset = Tok2S2OnTheFlyDataset(
        csv_path=ds_cfg["csv"],
        spec=Tok2S2OnTheFlySpec(
            s1_band_indices_1based=[int(v) for v in ds_cfg.get("s1_band_indices_1based", [1, 2])],
            s2_band_indices_1based=[int(v) for v in ds_cfg.get("s2_band_indices_1based", [1, 2, 3, 4])],
            s1_norm=ds_cfg.get("s1_norm", "zscore"),
            s2_norm=ds_cfg.get("s2_norm", "zscore"),
            meta_patch_pixels=int(ds_cfg.get("meta_patch_pixels", 16)),
        ),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(tr_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(tr_cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    variant = cfm_cfg.get("variant", "vit_base_varlang_e100")
    checkpoint_path = cfm_cfg.get("checkpoint_path", None)
    strict_load = bool(cfm_cfg.get("strict_load", False))
    freeze_backbone = bool(cfm_cfg.get("freeze_backbone", True))
    model_kwargs = {
        "return_intermediate": True,
        "intermediate_indices": [_feature_index_for_variant(variant)],
    }
    backbone, load_msg, loaded_ckpt = build_copernicus_fm(
        repo_root=REPO_ROOT,
        variant=variant,
        checkpoint_path=checkpoint_path,
        strict=strict_load,
        model_kwargs=model_kwargs,
    )
    backbone = backbone.to(device)
    backbone.eval()
    if freeze_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad = False

    model = SimpleTransformerDecoder(
        embed_dim=int(mz_cfg.get("embed_dim", 1024 if "large" in variant else 768)),
        nhead=int(mz_cfg.get("nhead", 8)),
        depth=int(mz_cfg.get("depth", 6)),
        c_out=int(mz_cfg.get("c_out", 4)),
        patch=int(mz_cfg.get("patch", 16)),
        hp=int(mz_cfg.get("hp", 16)),
        wp=int(mz_cfg.get("wp", 16)),
        mlp_ratio=float(mz_cfg.get("mlp_ratio", 4.0)),
        dropout=float(mz_cfg.get("dropout", 0.0)),
        refine_channels=int(mz_cfg.get("refine_channels", 64)),
        refine_depth=int(mz_cfg.get("refine_depth", 3)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr_cfg.get("lr", 1e-4)),
        weight_decay=float(tr_cfg.get("weight_decay", 1e-4)),
    )
    loss_fn = nn.L1Loss()
    epochs = int(tr_cfg.get("epochs", 10))
    viz_dir = ensure_dir(tr_cfg.get("viz_dir", "output/viz_tok2s2_transformer"))
    checkpoint_path = Path(tr_cfg.get("checkpoint_path", "output/tok2s2_transformer.pt")).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    wave_list = [float(spc_cfg.get("wave", 5e7))] * int(len(ds_cfg.get("s1_band_indices_1based", [1, 2])))
    bandwidth = [float(spc_cfg.get("bandwidth", 1e9))] * int(len(ds_cfg.get("s1_band_indices_1based", [1, 2])))

    global_step = 0
    model.train()
    for epoch_idx in range(epochs):
        total_loss = 0.0
        for step_in_epoch, (s1, meta, s2) in enumerate(dataloader):
            s1 = s1.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            s2 = s2.to(device, non_blocking=True)

            with torch.no_grad():
                _, intermediate = backbone(
                    s1,
                    meta,
                    wave_list,
                    bandwidth,
                    language_embed=None,
                    input_mode="spectral",
                    kernel_size=16,
                )
                feature = intermediate[-1]
                token = feature.permute(0, 2, 3, 1).contiguous()
            pred = model(token)

            if step_in_epoch == 0:
                save_viz_triplet(viz_dir, epoch_idx + 1, global_step, s2[0], pred[0])

            loss = loss_fn(pred, s2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            global_step += 1

        print(f"epoch {epoch_idx + 1}/{epochs} loss={total_loss / max(1, len(dataloader)):.6f}")

    torch.save({"model": model.state_dict()}, checkpoint_path)
    print(f"Loaded checkpoint: {loaded_ckpt}")
    print(f"load_state_dict_msg: {load_msg}")
    print(f"Saved: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
