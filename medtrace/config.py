from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import yaml

PathLike = Union[str, os.PathLike]


def deep_merge(base: Dict, top: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (top or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def load_config(path: Optional[PathLike] = None,
                overrides: Iterable[str] = (),
                _seen: Optional[List[Path]] = None) -> Dict:
    cfg: Dict = {}
    if path is not None:
        p = Path(path).resolve()
        seen = list(_seen or [])
        if p in seen:
            raise ValueError(f"_base_ cycle: {' -> '.join(map(str, seen + [p]))}")
        data = _load_yaml(p)
        bases = data.pop("_base_", []) or []
        if isinstance(bases, str):
            bases = [bases]
        for b in bases:
            cfg = deep_merge(cfg, load_config(p.parent / b, (), seen + [p]))
        cfg = deep_merge(cfg, data)
    for ov in overrides:
        set_dotted(cfg, *parse_override(ov))
    return cfg


def parse_override(s: str):
    if "=" not in s:
        raise ValueError(f"override {s!r} must look like key.path=value")
    key, raw = s.split("=", 1)
    return key.strip(), yaml.safe_load(raw) if raw.strip() != "" else ""


def set_dotted(cfg: Dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def get_dotted(cfg: Dict, key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def require(cfg: Dict, key: str) -> Any:
    sentinel = object()
    v = get_dotted(cfg, key, sentinel)
    if v is sentinel or v is None:
        raise KeyError(f"config key {key!r} is required")
    return v


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def config_hash(obj: Any, n: int = 12) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:n]
