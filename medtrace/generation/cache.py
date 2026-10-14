from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import canonical_json
from .vllm_client import GenRequest, GenResult, image_sha256, normalize_sampling

KEY_VERSION = 1


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def request_identity(served_model: str, req: GenRequest) -> Dict[str, Any]:
    return {"v": KEY_VERSION, "model": served_model,
            "system": _sha(req.system), "question": _sha(req.question),
            "images": [image_sha256(p, req.key, i) for i, p in enumerate(req.image_paths)],
            "sampling": normalize_sampling(req.sampling),
            "prefix": None if req.prefix is None else _sha(req.prefix)}


def request_key(served_model: str, req: GenRequest) -> str:
    return _sha(canonical_json(request_identity(served_model, req)))


class GenerationCache:
    def __init__(self, root: os.PathLike | str):
        self.root = Path(root)

    @classmethod
    def for_tag(cls, cache_root: os.PathLike | str, gen_tag: str) -> "GenerationCache":
        return cls(Path(cache_root) / "gen" / gen_tag)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[GenResult]:
        p = self.path(key)
        if not p.is_file():
            return None
        with open(p, "r", encoding="utf-8") as f:
            blob = json.load(f)
        if blob.get("key") != key:
            raise ValueError(f"cache file {p} holds key {blob.get('key')!r}")
        return GenResult.from_json(blob["result"])

    def put(self, key: str, result: GenResult, identity: Optional[Dict[str, Any]] = None) -> None:
        if not result.ok or result.text is None:
            raise ValueError(f"refusing to cache a failed generation ({result.key}: {result.error})")
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = {"key": key, "request": identity, "result": result.to_json()}
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                json.dump(blob, f, ensure_ascii=False)
            os.replace(tmp, p)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def __contains__(self, key: str) -> bool:
        return self.path(key).is_file()
