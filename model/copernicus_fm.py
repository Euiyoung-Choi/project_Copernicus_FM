from __future__ import annotations

from pathlib import Path


class CopernicusFMNotAvailable(RuntimeError):
    pass


def find_local_weights(repo_root: str | Path) -> dict:
    root = Path(repo_root).resolve()
    weights_dir = root / "Copernicus-FM"
    return {
        "vit_base_varlang_e100": str(weights_dir / "CopernicusFM_ViT_base_varlang_e100.pth"),
        "vit_large_varlang_e100": str(weights_dir / "CopernicusFM_ViT_large_varlang_e100.pth"),
    }


def build_copernicus_fm(*args, **kwargs):
    """
    Placeholder.

    This repository currently contains Copernicus-FM weight files under `Copernicus-FM/`,
    but does not include the Python implementation needed to instantiate the model.
    """
    raise CopernicusFMNotAvailable(
        "Copernicus-FM model code is not present in this repo (only .pth weights found).\n"
        "Add the upstream Copernicus-FM/DOFA implementation as a submodule or vendor it, "
        "then implement this builder to return a torch.nn.Module."
    )

