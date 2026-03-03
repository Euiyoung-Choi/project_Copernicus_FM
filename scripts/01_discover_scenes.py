from __future__ import annotations

from datetime import datetime
from pathlib import Path

import sys

# Allow running as a script without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loader.scenes import discover_scenes, pick_first_scene
from scripts.common import dump_config, ensure_dir, load_config

# -----------------------------------------------------------------------------
# Runtime settings (edit here, no argparse)
# -----------------------------------------------------------------------------
DATASET_CONFIG_PATH = "config/dataset.yaml"
OUT_DIR = "output/step01_discovery"


def main() -> int:
    dataset_cfg = load_config(DATASET_CONFIG_PATH)
    data_root = dataset_cfg["data_root"]
    glob = dataset_cfg.get("scene_glob", "*.tif")

    out_dir = ensure_dir(OUT_DIR)
    dump_config(
        out_dir / "config_resolved.yaml",
        {
            "step": "step01_discovery",
            "data_root": data_root,
            "glob": glob,
            "out_dir": str(out_dir),
        },
    )
    res = discover_scenes(data_root, glob=glob)
    if not res.scenes:
        raise SystemExit(f"No scenes found under {res.data_root} with glob={glob}")

    first_scene = pick_first_scene(res.scenes)
    (out_dir / "scene_list.txt").write_text(
        "\n".join(str(p) for p in res.scenes) + "\n", encoding="utf-8"
    )
    (out_dir / "excluded_list.txt").write_text(
        "\n".join(str(p) for p in res.excluded) + "\n", encoding="utf-8"
    )
    notes = [
        f"date: {datetime.now().isoformat(timespec='seconds')}",
        f"data_root: {res.data_root}",
        f"glob: {glob}",
        f"scene_count: {len(res.scenes)}",
        f"excluded_count: {len(res.excluded)}",
        f"first_scene: {first_scene}",
        "",
        "policy:",
        "- Always exclude macOS sidecars '._*.tif'.",
        "- First experiments must use only first_scene (sorted order).",
        "",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"Wrote {out_dir / 'scene_list.txt'}")
    print(f"First scene: {first_scene}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
