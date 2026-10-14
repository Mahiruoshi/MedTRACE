from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..mappings.base import Frames

VIEWS = ("premerger", "projected")


@dataclass
class FeatureStore:
    root: Path

    @classmethod
    def open(cls, cache_root: str, encoder_tag: str) -> "FeatureStore":
        root = Path(cache_root) / "features" / encoder_tag
        if not (root / "meta.json").is_file():
            raise FileNotFoundError(
                f"no feature cache at {root}; run `python -m medtrace.features.extract_vlm` for this encoder first")
        return cls(root)

    @property
    def meta(self) -> Dict:
        with open(self.root / "meta.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def has(self, pid: str) -> bool:
        return (self.root / "frames" / f"{pid}.npz").is_file()

    def load(self, pid: str, views: Sequence[str] = ("premerger",), x_view: str = "premerger") -> Frames:
        p = self.root / "frames" / f"{pid}.npz"
        if not p.is_file():
            raise FileNotFoundError(f"{pid}: features not extracted ({p})")
        with np.load(p, allow_pickle=False) as z:
            keys = [str(k) for k in z["frame_ids"]]
            vs = {}
            for v in set(views) | {x_view}:
                if v not in z.files:
                    raise KeyError(f"{pid}: view {v!r} missing from cache (have {z.files})")
                vs[v] = z[v].astype(np.float64)
        return Frames(X=vs[x_view], keys=keys, views=vs)

    def load_many(self, pids: Sequence[str], **kw) -> Dict[str, Frames]:
        return {pid: self.load(pid, **kw) for pid in pids}

    def write_frames(self, pid: str, frame_ids: List[str], arrays: Dict[str, np.ndarray]) -> None:
        d = self.root / "frames"
        d.mkdir(parents=True, exist_ok=True)
        for k, a in arrays.items():
            if k not in VIEWS:
                raise KeyError(f"unknown view {k!r}; declare it in store.VIEWS")
            if a.shape[0] != len(frame_ids):
                raise ValueError(f"{pid}/{k}: {a.shape[0]} rows for {len(frame_ids)} frames")
        tmp = d / f".{pid}.tmp.npz"
        np.savez(tmp, frame_ids=np.asarray(frame_ids), **{k: a.astype(np.float32) for k, a in arrays.items()})
        os.replace(tmp, d / f"{pid}.npz")

    def write_meta(self, meta: Dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


class CachedTextEmbedder:
    def __init__(self, store_root: Path):
        from tokenizers import Tokenizer

        tdir = Path(store_root) / "text"
        self._tok = Tokenizer.from_file(str(tdir / "tokenizer.json"))
        self._table = np.load(tdir / "embed_table.npy", mmap_mode="r")
        self.d_h = int(self._table.shape[1])

    def token_ids(self, text: str) -> List[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    def rows(self, ids: Sequence[int]) -> np.ndarray:
        return np.asarray(self._table[np.asarray(list(ids), dtype=np.int64)], dtype=np.float64)


def encoder_tag(model_path: str, pixel_cfg: Dict) -> str:
    import hashlib

    h = hashlib.sha256(json.dumps(pixel_cfg, sort_keys=True).encode()).hexdigest()[:8]
    return f"{Path(model_path).name}__{h}"
