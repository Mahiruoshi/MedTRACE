from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .tasks import TASK_TYPES, SurvivalTarget, Target, Task, survival_bin, survival_edges

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
Manifest = Dict[str, Any]


def natural_key(s: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", os.path.basename(s))]


def frame_id(pid: str, idx: int) -> str:
    return f"{pid}/s{idx:03d}"


def list_frames(seq_dir: Path) -> List[str]:
    if not seq_dir.is_dir():
        return []
    files = [str(seq_dir / f) for f in os.listdir(seq_dir) if f.lower().endswith(IMAGE_EXTS)]
    return sorted(files, key=natural_key)


@dataclass
class Patient:
    pid: str
    frames: List[str] = field(repr=False)
    split: str = ""
    age: str = ""

    @property
    def T(self) -> int:
        return len(self.frames)

    @property
    def frame_ids(self) -> List[str]:
        return [frame_id(self.pid, i) for i in range(self.T)]

    def public(self) -> Dict:
        return {"pid": self.pid, "age": self.age, "T": self.T, "split": self.split}


SPLITS = ("train", "test")


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _bit(v: Any) -> bool:
    return v in (0, 1) and not isinstance(v, bool) and isinstance(v, int)


def load_manifest(path: Union[str, os.PathLike, Manifest]) -> Manifest:
    if isinstance(path, Mapping):
        man = dict(path)
        man.setdefault("_dir", os.getcwd())
    else:
        with open(path, "r", encoding="utf-8-sig") as f:
            man = json.load(f)
        if not isinstance(man, dict):
            raise ValueError(f"{path}: a manifest is a JSON object")
        man["_dir"] = str(Path(path).resolve().parent)
    kind = man.get("task_type")
    if kind not in TASK_TYPES:
        raise ValueError(f"manifest task_type must be one of {TASK_TYPES}, got {kind!r}")
    pats = man.get("patients")
    if not isinstance(pats, list) or not pats:
        raise ValueError("manifest: 'patients' must be a non-empty list")
    seen = set()
    for n, r in enumerate(pats):
        if not isinstance(r, dict):
            raise ValueError(f"manifest patients[{n}] must be an object")
        for k in ("pid", "split", "frames_dir"):
            if r.get(k) is None or r.get(k) == "":
                raise ValueError(f"manifest patients[{n}]: missing {k!r}")
        if isinstance(r["pid"], bool) or not isinstance(r["pid"], (str, int)):
            raise ValueError(f"manifest patients[{n}]: pid must be a string")
        pid = str(r["pid"])
        if pid in seen:
            raise ValueError(f"manifest: duplicate pid {pid!r}")
        seen.add(pid)
        if "/" in pid or "\\" in pid:
            raise ValueError(f"pid {pid!r} must not contain '/' or '\\' (frame ids are '<pid>/s<idx>')")
        if r["split"] not in SPLITS:
            raise ValueError(f"{pid}: split must be one of {SPLITS}, got {r['split']!r}")
        if r.get("n_frames") is not None and (not isinstance(r["n_frames"], int) or isinstance(r["n_frames"], bool)
                                              or r["n_frames"] < 1):
            raise ValueError(f"{pid}: n_frames must be a positive integer")
        if kind == "binary":
            if not _bit(r.get("label")):
                raise ValueError(f"{pid}: binary label must be 0 or 1, got {r.get('label')!r}")
        else:
            if not _number(r.get("survival_time")) or r["survival_time"] < 0:
                raise ValueError(f"{pid}: survival_time must be a finite number >= 0, got {r.get('survival_time')!r}")
            if not _bit(r.get("event")):
                raise ValueError(f"{pid}: event must be 0 (censored) or 1 (observed), got {r.get('event')!r}")
    return man


def manifest_sha256(manifest: Union[str, os.PathLike, Manifest]) -> str:
    man = manifest if isinstance(manifest, dict) and "_dir" in manifest else load_manifest(manifest)
    body = {k: v for k, v in man.items() if k != "_dir"}
    body["patients"] = [{k: v for k, v in r.items() if k != "meta"} for r in man["patients"]]
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def splits(man: Manifest) -> List[str]:
    return sorted({r["split"] for r in man["patients"]})


def _records(man: Manifest, split: str) -> List[Dict[str, Any]]:
    if split not in SPLITS:
        raise KeyError(f"split must be one of {SPLITS}, got {split!r}")
    rec = [r for r in man["patients"] if r["split"] == split]
    if not rec:
        raise KeyError(f"split {split!r} not in manifest; have {splits(man)}")
    return sorted(rec, key=lambda r: natural_key(str(r["pid"])))


def _frames_dir(man: Manifest, rec: Mapping[str, Any], image_root: Optional[str]) -> Path:
    p = Path(rec["frames_dir"])
    if p.is_absolute():
        return p
    root = image_root or man.get("image_root")
    if root:
        root_p = Path(root)
        return (root_p if root_p.is_absolute() else Path(man["_dir"]) / root_p) / p
    return Path(man["_dir"]) / p


def load_patients(manifest: Union[str, os.PathLike, Manifest], split: str, image_root: Optional[str] = None,
                  pids: Optional[Sequence[str]] = None, strict: bool = True) -> List[Patient]:
    man = manifest if isinstance(manifest, dict) and "_dir" in manifest else load_manifest(manifest)
    keep = set(pids) if pids is not None else None
    out: List[Patient] = []
    for rec in _records(man, split):
        pid = str(rec["pid"])
        if keep is not None and pid not in keep:
            continue
        try:
            frames = list_frames(_frames_dir(man, rec, image_root))
        except OSError as e:
            raise OSError(f"{pid}: frames_dir unreadable ({type(e).__name__}); path withheld") from None
        if not frames:
            raise FileNotFoundError(f"{pid}: frames_dir is missing or holds no image (path withheld)")
        if strict and rec.get("n_frames") is not None and len(frames) != int(rec["n_frames"]):
            raise FileNotFoundError(f"{pid}: manifest says {rec['n_frames']} frames, found {len(frames)}")
        age = rec.get("age")
        out.append(Patient(pid=pid, frames=frames, split=split, age="" if age is None else str(age)))
    if keep is not None and len(out) != len(keep):
        missing = sorted(keep - {p.pid for p in out})
        raise KeyError(f"requested pids not loaded: {missing}")
    return out


def task_type(manifest: Union[str, os.PathLike, Manifest]) -> str:
    man = manifest if isinstance(manifest, dict) and "_dir" in manifest else load_manifest(manifest)
    return man["task_type"]


def modality(manifest: Union[str, os.PathLike, Manifest]) -> str:
    man = manifest if isinstance(manifest, dict) and "_dir" in manifest else load_manifest(manifest)
    return str(man.get("modality") or "")


def edges(manifest: Union[str, os.PathLike, Manifest], task: Task) -> Tuple[float, ...]:
    man = manifest if isinstance(manifest, dict) and "_dir" in manifest else load_manifest(manifest)
    if task.kind != "survival":
        return ()
    tr = _records(man, "train")
    return survival_edges([r["survival_time"] for r in tr], [r["event"] for r in tr], task.K)


def load_targets(manifest: Union[str, os.PathLike, Manifest], split: str, task: Task,
                 pids: Optional[Sequence[str]] = None) -> Dict[str, Target]:
    man = manifest if isinstance(manifest, dict) and "_dir" in manifest else load_manifest(manifest)
    if man["task_type"] != task.kind:
        raise ValueError(f"manifest task_type {man['task_type']!r} != task {task.kind!r}")
    keep = set(pids) if pids is not None else None
    e = edges(man, task)
    out: Dict[str, Target] = {}
    for r in _records(man, split):
        pid = str(r["pid"])
        if keep is not None and pid not in keep:
            continue
        if task.kind == "binary":
            out[pid] = int(r["label"])
        else:
            out[pid] = SurvivalTarget(time=float(r["survival_time"]), event=int(r["event"]),
                                      bin=survival_bin(r["survival_time"], e))
    return out
