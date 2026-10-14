from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..config import config_hash
from ..paths import REPO_ROOT

PathLike = Union[str, os.PathLike]

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

STAGES = ("distill", "generate", "score", "report")
PER_PRM = ("score", "report")

_ENV_PACKAGES = ("numpy", "scipy", "pyyaml", "pillow", "requests", "tokenizers", "safetensors", "torch",
                 "torchvision", "transformers", "accelerate", "timm", "einops", "open_clip_torch", "vllm")


class StageConflict(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: PathLike, obj: Any, indent: Optional[int] = 1) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, sort_keys=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def write_text(path: PathLike, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    os.replace(tmp, path)
    return path


def read_json(path: PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_state(repo: PathLike = REPO_ROOT) -> Dict[str, Any]:
    def _git(*args) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                 text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    rev = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no") if rev else None
    return {"git_rev": rev, "git_dirty": (bool(status) if status is not None else None)}


def code_sha256(root: PathLike = PACKAGE_ROOT) -> str:
    h = hashlib.sha256()
    root = Path(root)
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def env_versions() -> Dict[str, Any]:
    from importlib import metadata

    out: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in _ENV_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"):
        if os.environ.get(k) is not None:
            out[k] = os.environ[k]
    return out


def diff_keys(a: Any, b: Any, prefix: str = "", limit: int = 12) -> List[str]:
    out: List[str] = []
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for k in sorted(set(a) | set(b), key=str):
            key = f"{prefix}.{k}" if prefix else str(k)
            if k not in a or k not in b:
                out.append(key)
            else:
                out.extend(diff_keys(a[k], b[k], key, limit))
            if len(out) >= limit:
                break
    elif json.dumps(a, sort_keys=True, default=str) != json.dumps(b, sort_keys=True, default=str):
        out.append(prefix or "<root>")
    return out[:limit]


class RunDir:
    def __init__(self, runs_root: PathLike, exp: str, gen_tag: str, variant: str, seed: int,
                 cache_root: Optional[PathLike] = None):
        for name, v in (("exp", exp), ("gen_tag", gen_tag), ("variant", variant)):
            if not v or any(c in str(v) for c in "/\\:") or str(v).startswith("."):
                raise ValueError(f"{name} {v!r} is not a valid directory component")
        self.runs_root = Path(runs_root)
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.exp, self.gen_tag, self.variant, self.seed = str(exp), str(gen_tag), str(variant), int(seed)
        self.root = self.runs_root / self.exp / self.gen_tag / self.variant / f"seed{self.seed}"

    def __repr__(self) -> str:
        return f"RunDir({self.exp}/{self.gen_tag}/{self.variant}/seed{self.seed})"

    @property
    def key(self) -> str:
        return f"{self.variant}/seed{self.seed}"

    def stage_dir(self, stage: str, prm_tag: Optional[str] = None) -> Path:
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}; stages are {STAGES}")
        if stage in PER_PRM:
            if not prm_tag:
                raise ValueError(f"stage {stage!r} is per-PRM and needs a prm_tag")
            return self.root / stage / prm_tag
        return self.root / stage

    @property
    def signals(self) -> Path:
        return self.root / "distill" / "signals.json"

    @property
    def prompts(self) -> Path:
        return self.root / "generate" / "prompts.json"

    @property
    def candidates(self) -> Path:
        return self.root / "generate" / "candidates.jsonl"

    def scores(self, prm_tag: str) -> Path:
        return self.stage_dir("score", prm_tag) / "scores.json"

    def metrics(self, prm_tag: str) -> Path:
        return self.stage_dir("report", prm_tag) / "metrics.json"

    def summary(self, prm_tag: str) -> Path:
        return self.stage_dir("report", prm_tag) / "summary.txt"

    def manifest_path(self, stage: str, prm_tag: Optional[str] = None) -> Path:
        return self.stage_dir(stage, prm_tag) / "manifest.json"

    def _bases(self) -> List[Tuple[Path, str]]:
        out = [(self.root, "")]
        if self.cache_root is not None:
            out.append((self.cache_root, "cache:"))
        out.append((REPO_ROOT, "repo:"))
        return out

    def rel(self, path: PathLike) -> str:
        p = Path(path).resolve()
        for base, prefix in self._bases():
            try:
                return prefix + p.relative_to(base.resolve()).as_posix()
            except ValueError:
                pass
        return p.as_posix()

    def resolve(self, key: str) -> Path:
        for base, prefix in self._bases()[1:]:
            if key.startswith(prefix):
                return base / key[len(prefix):]
        if key.startswith("cache:"):
            raise ValueError(f"{self}: manifest key {key!r} names the cache root, which is not set")
        return Path(key) if Path(key).is_absolute() else self.root / key

    def read_manifest(self, stage: str, prm_tag: Optional[str] = None) -> Optional[Dict]:
        p = self.manifest_path(stage, prm_tag)
        return read_json(p) if p.is_file() else None

    def outputs_problem(self, m: Mapping[str, Any]) -> str:
        for key, sha in sorted((m.get("outputs") or {}).items()):
            p = self.resolve(key)
            if not p.is_file():
                return f"output {key} is missing"
            if sha256_file(p) != sha:
                return f"output {key} changed since the stage completed"
        return ""

    def stale_reason(self, stage: str, cfg: Dict, inputs: Sequence[PathLike] = (),
                     prm_tag: Optional[str] = None) -> str:
        m = self.read_manifest(stage, prm_tag)
        where = f"{stage}{'/' + prm_tag if prm_tag else ''}"
        if m is None:
            return f"{where} has not run"
        if m.get("status") != "complete":
            return f"{where} is {m.get('status')!r}, not complete"
        try:
            inp = self.hash_inputs(inputs)
        except FileNotFoundError as e:
            return f"{where}: {e}"
        why = self._mismatch(m, cfg, inp)
        if why:
            return f"{where} is stale: {why}"
        why = self.outputs_problem(m)
        return f"{where} is incomplete: {why}" if why else ""

    def hash_inputs(self, inputs: Iterable[PathLike]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for p in inputs:
            if not Path(p).is_file():
                raise FileNotFoundError(f"stage input {p} does not exist; run the upstream stage first")
            out[self.rel(p)] = sha256_file(p)
        return out

    def write_manifest(self, stage: str, cfg: Dict, seeds: Optional[Dict] = None,
                       inputs: Optional[Dict[str, str]] = None, prm_tag: Optional[str] = None,
                       status: str = "running", extra: Optional[Dict] = None) -> Dict:
        prev = self.read_manifest(stage, prm_tag) or {}
        m = {
            "stage": stage,
            "run": {"exp": self.exp, "gen_tag": self.gen_tag, "variant": self.variant,
                    "seed": self.seed, "prm_tag": prm_tag},
            "status": status,
            "created_utc": prev.get("created_utc") or utc_now(),
            "updated_utc": utc_now(),
            "config_hash": config_hash(cfg),
            "config": cfg,
            "seeds": seeds or {},
            **git_state(),
            "code_sha256": code_sha256(),
            "env": env_versions(),
            "inputs": dict(inputs or {}),
        }
        if extra:
            m.update(extra)
        write_json(self.manifest_path(stage, prm_tag), m)
        return m

    def begin(self, stage: str, cfg: Dict, seeds: Optional[Dict] = None,
              inputs: Sequence[PathLike] = (), prm_tag: Optional[str] = None,
              force: bool = False) -> str:
        inp = self.hash_inputs(inputs)
        d = self.stage_dir(stage, prm_tag)
        old = self.read_manifest(stage, prm_tag)
        state = "new"
        if old is not None:
            why = self._mismatch(old, cfg, inp)
            if why and not force:
                raise StageConflict(
                    f"{self} {stage}{'/' + prm_tag if prm_tag else ''}: {why}. "
                    f"Refusing to mix outputs; use a new exp name, or --force to supersede "
                    f"(the old directory is renamed aside, not deleted).")
            if why or force:
                self._supersede(d)
            elif old.get("status") == "complete":
                broken = self.outputs_problem(old)
                if not broken:
                    return "done"
                moved = self._supersede(d)
                print(f"[{stage}] {self}: {broken}; recomputing (old directory renamed to {moved.name})",
                      flush=True)
            else:
                state = "resume"
        elif d.is_dir() and any(d.iterdir()):
            if not force:
                raise StageConflict(f"{d} has content but no manifest; refusing to reuse it "
                                    f"(--force renames it aside)")
            self._supersede(d)
        self.write_manifest(stage, cfg, seeds, inp, prm_tag, status="running")
        return state

    def finish(self, stage: str, outputs: Sequence[PathLike] = (), prm_tag: Optional[str] = None,
               extra: Optional[Dict] = None) -> Dict:
        m = self.read_manifest(stage, prm_tag)
        if m is None:
            raise RuntimeError(f"{self} {stage}: finish() without begin()")
        m["status"] = "complete"
        m["completed_utc"] = utc_now()
        m["outputs"] = self.hash_inputs(outputs)
        if extra:
            m.update(extra)
        write_json(self.manifest_path(stage, prm_tag), m)
        return m

    @staticmethod
    def _mismatch(old: Dict, cfg: Dict, inputs: Dict[str, str]) -> str:
        if old.get("config_hash") != config_hash(cfg):
            keys = diff_keys(old.get("config", {}), cfg)
            return (f"recorded config_hash {old.get('config_hash')} != {config_hash(cfg)} "
                    f"(differs at: {', '.join(keys) or 'unknown'})")
        rec = old.get("inputs", {})
        changed = sorted(k for k in set(rec) | set(inputs) if rec.get(k) != inputs.get(k))
        if changed:
            return f"inputs changed since this stage ran: {', '.join(changed)}"
        return ""

    @staticmethod
    def _supersede(d: Path) -> Optional[Path]:
        if not d.exists():
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        dst = d.with_name(f"{d.name}.superseded-{stamp}")
        os.replace(d, dst)
        return dst
