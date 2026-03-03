from __future__ import annotations

from datetime import datetime
from pathlib import Path

import sys

# Allow running as a script without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loader.indexing import build_patch_index_for_scene, summarize_records, write_index_jsonl
from loader.scenes import discover_scenes, pick_first_scene
from scripts.common import dump_config, ensure_dir, load_config

# -----------------------------------------------------------------------------
# Runtime settings (edit here, no argparse)
# -----------------------------------------------------------------------------
DATASET_CONFIG_PATH = "config/dataset.yaml"
INDEX_SCOPE = "all"  # "first" | "all"
INDEX_OUT = "output/index/all_scenes.index.jsonl"
STATS_OUT_DIR = "output/step03_index_stats"
LIMIT_PATCHES = None  # e.g. 64 for quick run, None for all


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
    dataset_cfg = load_config(DATASET_CONFIG_PATH)
    data_root = dataset_cfg["data_root"]
    glob = dataset_cfg.get("scene_glob", "*.tif")
    patch_size = int(dataset_cfg["patching"]["patch_size"])
    stride = int(dataset_cfg["patching"]["stride"])
    cloud_threshold = float(dataset_cfg.get("thresholds", {}).get("cloud_pixel", 30.0))

    scenes = discover_scenes(data_root, glob=glob).scenes
    if not scenes:
        raise SystemExit("No scenes found.")
    first_scene = pick_first_scene(scenes)
    target_scenes = [first_scene] if INDEX_SCOPE == "first" else list(scenes)

    stats_dir = ensure_dir(STATS_OUT_DIR)
    dump_config(
        stats_dir / "config_resolved.yaml",
        {
            "step": "step02_build_index_first_scene",
            "data_root": data_root,
            "glob": glob,
            "index_scope": INDEX_SCOPE,
            "first_scene": str(first_scene),
            "num_target_scenes": len(target_scenes),
            "patch_size": patch_size,
            "stride": stride,
            "cloud_threshold_T": cloud_threshold,
            "limit_patches": LIMIT_PATCHES,
            "index_out": INDEX_OUT,
            "stats_out_dir": str(stats_dir),
        },
    )

    records = []
    for scene in target_scenes:
        scene_records = build_patch_index_for_scene(
            scene,
            patch_size=patch_size,
            stride=stride,
            cloud_threshold=cloud_threshold,
            limit_patches=LIMIT_PATCHES,
        )
        records.extend(scene_records)
    write_index_jsonl(records, INDEX_OUT)

    summary = summarize_records(records)
    summary_path = stats_dir / "summary.json"
    dump_config(summary_path, summary)

    decision_md = [
        f"date: {datetime.now().isoformat(timespec='seconds')}",
        f"index_scope: {INDEX_SCOPE}",
        f"first_scene: {first_scene}",
        f"num_target_scenes: {len(target_scenes)}",
        f"patch_size: {patch_size}",
        f"stride: {stride}",
        f"cloud_threshold_T: {cloud_threshold}",
        f"limit_patches: {LIMIT_PATCHES}",
        f"patch_count: {len(records)}",
        "",
        "selection rules (initial):",
        "- drop: nan_ratio filtering is deferred to training config (default relaxed)",
        "- stage1_train_candidate: valid_ratio >= 0.95",
        "- stage2_infer_candidate: cloud_ratio >= 0.10",
        "",
        "notes:",
        "- Patch counts are derived from index generation; do not assume a fixed number a priori.",
        "",
    ]
    (stats_dir / "decision.md").write_text("\n".join(decision_md), encoding="utf-8")

    _maybe_write_histograms(stats_dir, records)

    print(f"Wrote index: {Path(INDEX_OUT).resolve()}")
    print(f"Wrote stats: {stats_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
