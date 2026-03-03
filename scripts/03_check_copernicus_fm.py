from __future__ import annotations

from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.copernicus_fm import (
    CopernicusFMNotAvailable,
    build_copernicus_fm,
    find_local_weights,
    get_recommended_variant,
    get_variant_priority,
)


# -----------------------------------------------------------------------------
# Edit these variables directly (no argparse).
# -----------------------------------------------------------------------------
REPO_ROOT_OVERRIDE = REPO_ROOT
VARIANT = get_recommended_variant()  # e.g. "vit_base_varlang_e100"
CHECKPOINT_PATH = None  # e.g. "/home/ey/.../CopernicusFM_ViT_base_varlang_e100.pth"
DO_LOAD = False         # False: list only, True: instantiate + load checkpoint
STRICT_LOAD = False


def main() -> int:
    repo_root = Path(REPO_ROOT_OVERRIDE).resolve()
    available = find_local_weights(repo_root)
    print("repo_root:", repo_root)
    print("recommended_variant:", get_recommended_variant())
    print("variant_priority:", get_variant_priority())
    print("available_checkpoints:", sorted(available.keys()) if available else "none")
    for key in sorted(available.keys()):
        print(f"  - {key}: {available[key]}")

    if not DO_LOAD:
        return 0

    try:
        model, msg, checkpoint = build_copernicus_fm(
            repo_root=repo_root,
            variant=VARIANT,
            checkpoint_path=CHECKPOINT_PATH,
            strict=STRICT_LOAD,
        )
        print("loaded_variant:", VARIANT)
        print("checkpoint:", checkpoint)
        print("load_state_dict_msg:", msg)
        print("model_type:", type(model).__name__)
    except CopernicusFMNotAvailable as e:
        print("ERROR:", e)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
