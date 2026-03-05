from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loader.tok2s2_dataset import Tok2S2OnTheFlyDataset, Tok2S2OnTheFlySpec
from model.copernicus_fm import build_copernicus_fm
from model.losses import masked_l1_loss
from scripts.common import ensure_dir, load_config
from scripts.train_stage1_common import save_sample_rgb


CONFIG_PATH = "config/tok2s2_transformer.yaml"


def save_viz_triplet(
    out_dir: Path,
    epoch: int,
    step: int,
    inp_chw: torch.Tensor,
    gt_chw: torch.Tensor,
    pred_chw: torch.Tensor,
    valid_mask_chw: torch.Tensor,
):
    sample_name = f"e{epoch:03d}_s{step:06d}"
    save_sample_rgb(out_dir, sample_name, inp_chw, gt_chw, pred_chw, valid_mask_chw)


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
            fmask_band_1based=int(ds_cfg.get("fmask_band_1based", 7)),
            cloud_threshold=float(ds_cfg.get("cloud_threshold", 30.0)),
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
        for step_in_epoch, (s1, meta, s2, valid_mask) in enumerate(dataloader):
            s1 = s1.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            s2 = s2.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)

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
                save_viz_triplet(viz_dir, epoch_idx + 1, global_step, s1[0], s2[0], pred[0], valid_mask[0])

            loss = masked_l1_loss(pred, s2, valid_mask)
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
