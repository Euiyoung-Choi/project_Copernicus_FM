from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Check Copernicus-FM local checkpoint wiring.")
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--variant", default=None, help="Checkpoint variant key.")
    ap.add_argument("--checkpoint", default=None, help="Explicit checkpoint path.")
    ap.add_argument(
        "--load",
        action="store_true",
        help="Actually instantiate model and load checkpoint (requires torch/timm/etc).",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    available = find_local_weights(repo_root)
    print("repo_root:", repo_root)
    print("recommended_variant:", get_recommended_variant())
    print("variant_priority:", get_variant_priority())
    print("available_checkpoints:", sorted(available.keys()) if available else "none")
    for key in sorted(available.keys()):
        print(f"  - {key}: {available[key]}")

    if not args.load:
        return 0

    variant = args.variant or get_recommended_variant()
    try:
        model, msg, checkpoint = build_copernicus_fm(
            repo_root=repo_root,
            variant=variant,
            checkpoint_path=args.checkpoint,
            strict=False,
        )
        print("loaded_variant:", variant)
        print("checkpoint:", checkpoint)
        print("load_state_dict_msg:", msg)
        print("model_type:", type(model).__name__)
    except CopernicusFMNotAvailable as e:
        print("ERROR:", e)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

