from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional


class CopernicusFMNotAvailable(RuntimeError):
    pass


def find_local_weights(repo_root: str | Path) -> Dict[str, str]:
    root = Path(repo_root).resolve()
    candidate_dirs = [
        root / "Copernicus-FM",
        root / "Copernicus-FM" / "Copernicus-FM",
        root / "Copernicus-FM" / "weights",
        root / "Copernicus-FM" / "Copernicus-FM" / "weights",
        root / "Copernicus-FM" / "deprecated",
        root / "Copernicus-FM" / "Copernicus-FM" / "deprecated",
    ]
    weights: Dict[str, str] = {}
    for d in candidate_dirs:
        if not d.exists():
            continue
        base = d / "CopernicusFM_ViT_base_varlang_e100.pth"
        large = d / "CopernicusFM_ViT_large_varlang_e100.pth"
        base_full = d / "CopernicusFM_ViT_base_varlang_e100_fullmodel.pth"
        large_full = d / "CopernicusFM_ViT_large_varlang_e100_fullmodel.pth"
        if base.exists():
            weights["vit_base_varlang_e100"] = str(base)
        if large.exists():
            weights["vit_large_varlang_e100"] = str(large)
        if base_full.exists():
            weights["vit_base_varlang_e100_fullmodel"] = str(base_full)
        if large_full.exists():
            weights["vit_large_varlang_e100_fullmodel"] = str(large_full)

    return weights


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise CopernicusFMNotAvailable(f"Failed to create import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _import_upstream_model_vit(repo_root: Path) -> ModuleType:
    """
    Load upstream Copernicus-FM `model_vit.py` without requiring package installation.
    Handles relative imports by creating a runtime package namespace.
    """
    src_dir = repo_root / "Copernicus-FM" / "Copernicus-FM" / "src"
    model_vit_path = src_dir / "model_vit.py"
    if not model_vit_path.exists():
        raise CopernicusFMNotAvailable(
            f"Upstream model code not found: {model_vit_path}\n"
            "Expected repo structure: Copernicus-FM/Copernicus-FM/src/model_vit.py"
        )

    package_name = "copernicusfm_src_runtime"
    package_spec = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
    if package_spec is None:
        raise CopernicusFMNotAvailable("Failed to initialize runtime package for Copernicus-FM source.")
    package = importlib.util.module_from_spec(package_spec)
    package.__path__ = [str(src_dir)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    # Preload dependency modules used via relative import in model_vit.py
    _load_module(f"{package_name}.dynamic_hypernetwork", src_dir / "dynamic_hypernetwork.py")
    _load_module(f"{package_name}.aurora.fourier", src_dir / "aurora" / "fourier.py")
    _load_module(f"{package_name}.flexivit.utils", src_dir / "flexivit" / "utils.py")

    return _load_module(f"{package_name}.model_vit", model_vit_path)


def resolve_weight_path(
    repo_root: str | Path,
    variant: str = "vit_base_varlang_e100",
    explicit_weight_path: Optional[str | Path] = None,
) -> Path:
    if explicit_weight_path is not None:
        p = Path(explicit_weight_path).expanduser().resolve()
        if not p.exists():
            raise CopernicusFMNotAvailable(f"Checkpoint not found: {p}")
        return p

    available = find_local_weights(repo_root)
    if variant not in available:
        raise CopernicusFMNotAvailable(
            f"Checkpoint variant '{variant}' not found under Copernicus-FM.\n"
            f"Available: {sorted(available.keys()) if available else 'none'}"
        )
    return Path(available[variant]).resolve()


def load_checkpoint_state_dict(checkpoint_path: str | Path):
    try:
        import torch
    except Exception as e:
        raise CopernicusFMNotAvailable(
            "PyTorch is required to load Copernicus-FM checkpoints.\n"
            "Install: python -m pip install torch"
        ) from e

    p = Path(checkpoint_path).expanduser().resolve()
    if not p.exists():
        raise CopernicusFMNotAvailable(f"Checkpoint not found: {p}")

    ckpt = torch.load(str(p), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def build_copernicus_fm(
    repo_root: str | Path,
    variant: str = "vit_base_varlang_e100",
    checkpoint_path: Optional[str | Path] = None,
    num_classes: int = 0,
    strict: bool = False,
):
    """
    Build Copernicus-FM ViT backbone and load local checkpoint.

    - variant: 'vit_base_varlang_e100' | 'vit_large_varlang_e100'
               and optional *_fullmodel variants if provided.
    - num_classes: default 0 for feature extraction/backbone usage.
    - strict: False by default to tolerate head key mismatches.
    """
    repo_root_path = Path(repo_root).resolve()
    model_vit = _import_upstream_model_vit(repo_root_path)

    if "large" in variant:
        model = model_vit.vit_large_patch16(num_classes=num_classes, global_pool=False)
    else:
        model = model_vit.vit_base_patch16(num_classes=num_classes, global_pool=False)

    checkpoint = resolve_weight_path(repo_root_path, variant=variant, explicit_weight_path=checkpoint_path)
    state_dict = load_checkpoint_state_dict(checkpoint)
    msg = model.load_state_dict(state_dict, strict=strict)
    return model, msg, checkpoint


def encode_spectral(
    model,
    image_tensor,
    wave_list,
    bandwidth,
    meta_info=None,
    kernel_size: int = 16,
):
    """
    Thin helper for spectral forward.
    image_tensor: [B, C, H, W]
    meta_info: [B, 4] -> [lon, lat, delta_time, area], NaN allowed.
    """
    try:
        import torch
    except Exception as e:
        raise CopernicusFMNotAvailable("PyTorch is required for forward pass.") from e

    if meta_info is None:
        meta_info = torch.full((image_tensor.shape[0], 4), float("nan"), device=image_tensor.device)
    return model(
        image_tensor,
        meta_info,
        wave_list,
        bandwidth,
        language_embed=None,
        input_mode="spectral",
        kernel_size=kernel_size,
    )


def get_recommended_variant() -> str:
    return "vit_base_varlang_e100"


def get_variant_priority() -> list[str]:
    return [
        "vit_base_varlang_e100",
        "vit_base_varlang_e100_fullmodel",
        "vit_large_varlang_e100",
        "vit_large_varlang_e100_fullmodel",
    ]
