from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULTS: Dict[str, Optional[str]] = {
    "image_root": None,
    "models_root": "models",
    "weights_root": "weights",
    "cache_root": "cache",
    "runs_root": "runs",
}


@dataclass
class Paths:
    image_root: Optional[str]
    models_root: str
    weights_root: str
    cache_root: str
    runs_root: str

    def model(self, name_or_path: str) -> str:
        return _under(self.models_root, name_or_path)

    def weights(self, name_or_path: str) -> str:
        return _under(self.weights_root, name_or_path)


def _under(root: str, name_or_path: str) -> str:
    p = Path(name_or_path)
    if p.is_absolute() or name_or_path.startswith(("./", "../")):
        return str(p)
    return str(Path(root) / name_or_path)


def get_paths(overrides: Optional[Dict[str, str]] = None) -> Paths:
    base: Dict[str, Optional[str]] = dict(_DEFAULTS)
    for k in list(base):
        env = os.environ.get(f"MEDTRACE_{k.upper()}")
        if env:
            base[k] = env
    for k, v in (overrides or {}).items():
        if v is not None and k in base:
            base[k] = str(v)
    for k, v in base.items():
        if v is not None and not Path(v).is_absolute():
            base[k] = str(Path.cwd() / v)
    return Paths(**base)
