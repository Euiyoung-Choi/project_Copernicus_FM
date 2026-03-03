from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _try_import_yaml():
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    return yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".json"}:
        return json.loads(text)
    if p.suffix.lower() in {".yml", ".yaml"}:
        yaml = _try_import_yaml()
        if yaml is None:
            raise RuntimeError("YAML config requires PyYAML. Install: python -m pip install PyYAML")
        return yaml.safe_load(text) or {}
    raise ValueError(f"Unsupported config extension: {p.suffix}")


def dump_config(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".json"}:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    if p.suffix.lower() in {".yml", ".yaml"}:
        yaml = _try_import_yaml()
        if yaml is None:
            # fallback to json-in-yaml extension
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return
        p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return
    raise ValueError(f"Unsupported output extension: {p.suffix}")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

