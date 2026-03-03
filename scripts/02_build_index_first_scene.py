from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import sys

# Allow running as a script without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loader.indexing import build_patch_index_for_scene, summarize_records, write_index_jsonl
from loader.scenes import discover_scenes, pick_first_scene
from scripts.common import dump_config, ensure_dir


def _maybe_write_histograms(out_dir: Path, records):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        (out_dir / "figures" / "SKIPPED_matplotlib.txt").write_text(
            f"matplotlib not available; skipping histograms. error={e}\n", encoding="utf-8"
        )
        return

    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    def hist(values, title, fname, bins=50):
        arr = np.array(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        plt.figure()
        plt.hist(arr, bins=bins)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figs / fname, dpi=140)
        plt.close()

    hist([r.mean_fmask for r in records], "mean_fmask", "mean_fmask_hist.png")
    hist([r.cloud_ratio for r in records], "cloud_ratio (Fmask>=T)", "cloud_ratio_hist.png")
    hist([r.valid_ratio for r in records], "valid_ratio (Fmask<T)", "valid_ratio_hist.png")
    hist([r.nan_ratio for r in records], "nan_ratio", "nan_ratio_hist.png")


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 3: Build patch index for first scene (256x256 windows).")
    ap.add_argument("--data-root", default="/home/ey/data_2/SARtoRGB/Korea/", help="Dataset root directory.")
    ap.add_argument("--glob", default="*.tif", help="Scene glob pattern.")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--cloud-threshold", type=float, default=60.0, help="Cloud pixel if Fmask>=T.")
    ap.add_argument("--limit-patches", type=int, default=None, help="Optional patch limit for quick runs.")
    ap.add_argument("--index-out", default="output/index/first_scene.index.jsonl", help="Index jsonl output path.")
    ap.add_argument("--stats-out-dir", default="output/step03_index_stats", help="Stats output directory.")
    args = ap.parse_args()

    scenes = discover_scenes(args.data_root, glob=args.glob).scenes
    if not scenes:
        raise SystemExit("No scenes found.")
    first_scene = pick_first_scene(scenes)

    stats_dir = ensure_dir(args.stats_out_dir)
    dump_config(
        stats_dir / "config_resolved.yaml",
        {
            "step": "step02_build_index_first_scene",
            "data_root": args.data_root,
            "glob": args.glob,
            "first_scene": str(first_scene),
            "patch_size": args.patch_size,
            "stride": args.stride,
            "cloud_threshold_T": args.cloud_threshold,
            "limit_patches": args.limit_patches,
            "index_out": args.index_out,
            "stats_out_dir": str(stats_dir),
        },
    )

    records = build_patch_index_for_scene(
        first_scene,
        patch_size=args.patch_size,
        stride=args.stride,
        cloud_threshold=args.cloud_threshold,
        limit_patches=args.limit_patches,
    )
    write_index_jsonl(records, args.index_out)

    summary = summarize_records(records)
    summary_path = stats_dir / "summary.json"
    dump_config(summary_path, summary)

    decision_md = [
        f"date: {datetime.now().isoformat(timespec='seconds')}",
        f"first_scene: {first_scene}",
        f"patch_size: {args.patch_size}",
        f"stride: {args.stride}",
        f"cloud_threshold_T: {args.cloud_threshold}",
        f"limit_patches: {args.limit_patches}",
        f"patch_count: {len(records)}",
        "",
        "selection rules (initial):",
        "- drop: nan_ratio > 0 (conservative start)",
        "- stage1_train_candidate: valid_ratio >= 0.95",
        "- stage2_infer_candidate: cloud_ratio >= 0.10",
        "",
        "notes:",
        "- Patch counts are derived from index generation; do not assume a fixed number a priori.",
        "",
    ]
    (stats_dir / "decision.md").write_text("\n".join(decision_md), encoding="utf-8")

    _maybe_write_histograms(stats_dir, records)

    print(f"Wrote index: {Path(args.index_out).resolve()}")
    print(f"Wrote stats: {stats_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
