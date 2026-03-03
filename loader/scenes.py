from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class SceneDiscoveryResult:
    data_root: Path
    scenes: List[Path]
    excluded: List[Path]


def discover_scenes(data_root: str | Path, glob: str = "*.tif") -> SceneDiscoveryResult:
    """
    Discover GeoTIFF scenes under data_root, excluding macOS '._' sidecar files.
    Returns sorted paths for determinism.
    """
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"data_root is not a directory: {root}")

    all_candidates = sorted(root.glob(glob))
    scenes: List[Path] = []
    excluded: List[Path] = []
    for p in all_candidates:
        if p.name.startswith("._"):
            excluded.append(p)
            continue
        if not p.is_file():
            excluded.append(p)
            continue
        scenes.append(p)

    return SceneDiscoveryResult(data_root=root, scenes=scenes, excluded=excluded)


def pick_first_scene(scenes: Iterable[Path]) -> Path:
    scenes = list(scenes)
    if not scenes:
        raise ValueError("No scenes found.")
    return sorted(scenes)[0]

