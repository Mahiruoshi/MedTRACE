from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..data.cohort import Manifest, Patient, load_manifest, load_patients
from ..data.tasks import Task
from ..mappings.base import Frames
from ..paths import get_paths
from .extract_vlm import env_task, read_frames, select_patients
from .store import FeatureStore, encoder_tag

VIEW = "clip_img"
KIND = "dual_encoder"
QUERIES = "queries.json"


def default_text_config(model_path: str, hf_id: str) -> Optional[str]:
    cand = Path(model_path).parent / hf_id.rstrip("/").split("/")[-1]
    return str(cand) if (cand / "config.json").is_file() else None


class OpenCLIPEncoder:
    name = "open_clip"
    CONFIG = "open_clip_config.json"
    WEIGHTS = ("*.safetensors", "*.bin", "*.pth")

    @classmethod
    def read_config(cls, model_path: str) -> Dict:
        p = Path(model_path) / cls.CONFIG
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found; is {model_path} an open_clip model directory?")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def read_preprocess(cls, model_path: str) -> Dict:
        cfg = cls.read_config(model_path)
        return {"image_size": cfg["model_cfg"]["vision_cfg"].get("image_size"), **(cfg.get("preprocess_cfg") or {})}

    def __init__(self, model_path: str, device: str = "cpu", text_config: Optional[str] = None):
        cfg = self.read_config(model_path)
        if not any(p.is_file() for g in self.WEIGHTS for p in Path(model_path).glob(g)):
            raise FileNotFoundError(f"{model_path}: no CLIP weights ({', '.join(self.WEIGHTS)}); "
                                    f"is the download complete?")
        text_cfg = dict(cfg["model_cfg"].get("text_cfg") or {})
        kw: Dict = {}
        self.text_config = None
        hf_id = text_cfg.get("hf_model_name")
        if text_config and not hf_id:
            raise ValueError(f"{model_path}: text_config given, but the text tower is not an HF model")
        import open_clip
        import torch

        self.torch, self.device = torch, device
        if hf_id:
            self.text_config = text_config or default_text_config(model_path, hf_id) or hf_id
            if self.text_config != hf_id:
                kw["text_cfg"] = {**text_cfg, "hf_model_name": self.text_config, "hf_model_pretrained": False}
        spec = f"local-dir:{model_path}"
        try:
            self.model, self.preprocess = open_clip.create_model_from_pretrained(spec, device=device, **kw)
        except (OSError, RuntimeError, ValueError) as exc:
            hint = (f"; the text tower is named by the hub id {hf_id!r}: pass --text-config <local dir with its "
                    f"config.json>" if hf_id and self.text_config == hf_id else "")
            raise RuntimeError(f"open_clip could not load {model_path} ({type(exc).__name__}: {exc}){hint}") from exc
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(spec)
        self.d_c = int(cfg["model_cfg"]["embed_dim"])
        self.versions = {"open_clip": open_clip.__version__, "torch": torch.__version__}

    def encode_images(self, images: Sequence) -> np.ndarray:
        x = self.torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with self.torch.no_grad():
            return self.model.encode_image(x).float().cpu().numpy()

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        tok = self.tokenizer(list(texts)).to(self.device)
        with self.torch.no_grad():
            return self.model.encode_text(tok).float().cpu().numpy()

    def describe(self) -> Dict:
        return {"model_class": type(self.model).__name__, "tokenizer": type(self.tokenizer).__name__,
                "text_config": self.text_config, "precision": "fp32", "device": str(self.device), **self.versions}


def clip_tag(model_path: str, preprocess: Dict) -> str:
    return encoder_tag(model_path, {"backend": OpenCLIPEncoder.name, "preprocess": preprocess})


def l2_normalize(a: np.ndarray, what: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise RuntimeError(f"{what}: expected finite (n, d) embeddings, got shape {a.shape}")
    n = np.linalg.norm(a, axis=1, keepdims=True)
    if (n <= 0).any():
        raise RuntimeError(f"{what}: {int((n <= 0).sum())} zero embedding(s)")
    return (a / n).astype(np.float32)


def write_frames(root: Path, pid: str, frame_ids: Sequence[str], emb: np.ndarray) -> None:
    if emb.shape[0] != len(frame_ids):
        raise ValueError(f"{pid}/{VIEW}: {emb.shape[0]} rows for {len(frame_ids)} frames")
    d = Path(root) / "frames"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{pid}.tmp.npz"
    np.savez(tmp, frame_ids=np.asarray(list(frame_ids)), **{VIEW: np.asarray(emb, dtype=np.float32)})
    os.replace(tmp, d / f"{pid}.npz")


def is_cached(root: Path, p: Patient) -> bool:
    f = Path(root) / "frames" / f"{p.pid}.npz"
    if not f.is_file():
        return False
    try:
        with np.load(f, allow_pickle=False) as z:
            return [str(k) for k in z["frame_ids"]] == p.frame_ids and VIEW in z.files and z[VIEW].shape[0] == p.T
    except (OSError, ValueError, KeyError):
        return False


def query_key(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def read_queries(root: Path) -> Dict[str, Dict]:
    p = Path(root) / "text" / QUERIES
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict, **dump) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, **dump)
        f.write("\n")
    os.replace(tmp, path)


def write_queries(root: Path, entries: Dict[str, Dict]) -> None:
    _write_json(Path(root) / "text" / QUERIES, entries, indent=1, sort_keys=True)


def write_meta(root: Path, meta: Dict) -> None:
    _write_json(Path(root) / "meta.json", meta, indent=2)


def _read_meta(root: Path) -> Dict:
    with open(Path(root) / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_cache(encoder: Optional[str], feature_root: Union[str, os.PathLike]) -> Path:
    if encoder:
        root = Path(encoder) if Path(encoder).is_absolute() else Path(feature_root) / encoder
    else:
        metas = sorted(Path(feature_root).glob("*/meta.json")) if Path(feature_root).is_dir() else []
        hits = [m.parent for m in metas if _read_meta(m.parent).get("kind") == KIND]
        if len(hits) != 1:
            raise FileNotFoundError(
                f"{len(hits)} dual-encoder caches under {feature_root} ({[h.name for h in hits]}); extract one with "
                f"`python -m medtrace.features.extract_clip --model <dir>` or name it (scorer param `encoder`)")
        root = hits[0]
    if not (root / "meta.json").is_file():
        raise FileNotFoundError(f"no dual-encoder cache at {root}; run `python -m medtrace.features.extract_clip` first")
    if _read_meta(root).get("kind") != KIND:
        raise ValueError(f"{root} is a feature cache but not a dual-encoder one (meta.kind != {KIND!r})")
    return root


class DualEncoderCache:
    def __init__(self, root: Union[str, os.PathLike]):
        self.root = Path(root)
        self.meta = _read_meta(self.root)
        self.d_c = int(self.meta["d_c"])
        self._frames = FeatureStore(self.root)

    @classmethod
    def open(cls, encoder: Optional[str], feature_root: Union[str, os.PathLike]) -> "DualEncoderCache":
        return cls(resolve_cache(encoder, feature_root))

    @property
    def tag(self) -> str:
        return self.root.name

    def query(self, text: str) -> Tuple[Dict, np.ndarray]:
        key = query_key(text)
        qs = read_queries(self.root)
        if key not in qs or qs[key].get("query") != text:
            raise KeyError(f"{self.tag}: no embedding of this Q_bio (key {key}; have {sorted(qs)}); re-run "
                           f"`python -m medtrace.features.extract_clip` with the current environment JSON")
        e = np.asarray(qs[key]["embedding"], dtype=np.float64)
        if e.shape != (self.d_c,):
            raise ValueError(f"{self.tag}: query embedding {key} has shape {e.shape}, d_c={self.d_c}")
        return {"query": text, "query_key": key}, e

    def frames(self, pid: str) -> Frames:
        return self._frames.load(pid, views=(VIEW,), x_view=VIEW)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[extract_clip {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def encode_patient(enc, p: Patient, batch: int) -> np.ndarray:
    parts = []
    ids = p.frame_ids
    for i in range(0, p.T, batch):
        parts.append(np.asarray(enc.encode_images(read_frames(p.frames[i:i + batch], ids[i:i + batch])),
                                dtype=np.float64))
    X = np.concatenate(parts)
    if X.shape != (p.T, enc.d_c):
        raise RuntimeError(f"{p.pid}: image tower returned {X.shape}, expected ({p.T}, d_c={enc.d_c})")
    return l2_normalize(X, p.pid)


def build_meta(root: Path, desc: Dict, d_c: int, model_path: str, preprocess: Dict, tag: str,
               cohort: Dict[str, List[Patient]], run: Dict) -> Dict:
    old = _read_meta(root) if (root / "meta.json").is_file() else {}
    if old and int(old.get("d_c", d_c)) != d_c:
        raise RuntimeError(f"{root}: existing meta has d_c={old.get('d_c')}, this encoder {d_c}")
    queries = read_queries(root)
    splits = dict(old.get("splits", {}))
    for s, ps in cohort.items():
        done = [p for p in ps if is_cached(root, p)]
        done_ids = {p.pid for p in done}
        splits[s] = {"n_patients": len(done), "n_manifest": len(ps), "n_frames": int(sum(p.T for p in done)),
                     "views": [VIEW] if done else [], "complete": len(done) == len(ps),
                     "missing": [p.pid for p in ps if p.pid not in done_ids]}
    return {"kind": KIND, "encoder_tag": tag, "model_path": model_path, "backend": OpenCLIPEncoder.name,
            "d_c": d_c, "views": [VIEW],
            "normalized": "L2 per row in float64, stored float32 (image rows and queries)",
            "preprocess": preprocess, "encoder": desc, "splits": splits,
            "text": {"file": f"text/{QUERIES}",
                     "queries": {k: {"query": e["query"]} for k, e in sorted(queries.items())}},
            "created_utc": old.get("created_utc", run["utc"]), "updated_utc": run["utc"],
            "runs": list(old.get("runs", [])) + [run]}


def extract(model: str, manifest: Union[str, os.PathLike, Manifest], splits: Sequence[str] = ("train",),
            pids: Optional[Sequence[str]] = None, task: Optional[Task] = None,
            batch: int = 64, force: bool = False, device: str = "cpu", cache_root: Optional[str] = None,
            text_config: Optional[str] = None) -> Path:
    paths = get_paths()
    model_path = paths.model(model)
    preprocess = OpenCLIPEncoder.read_preprocess(model_path)
    tag = clip_tag(model_path, preprocess)
    root = Path(cache_root or paths.cache_root) / "features" / tag
    man = load_manifest(manifest)
    task = task or env_task(man, need=("prompts.biomarker_query",))
    q_text = task.biomarker_query
    if not q_text.strip():
        raise ValueError("the biomarker query Q_bio (prompts.biomarker_query) is empty")
    cohort = {s: load_patients(man, s, image_root=paths.image_root) for s in splits}
    todo = {s: [p for p in ps if force or not is_cached(root, p)] for s, ps in select_patients(cohort, pids).items()}
    key = query_key(q_text)
    have = read_queries(root)
    new_q = [key] if force or (have.get(key) or {}).get("query") != q_text else []
    n_todo = sum(len(v) for v in todo.values())
    _log(f"tag={tag} root={root} backend={OpenCLIPEncoder.name} device={device}")
    _log("to extract: " + ", ".join(f"{s} {len(todo[s])}/{len(cohort[s])} patients" for s in cohort)
        + f"; queries {new_q or 'cached'}")
    if n_todo == 0 and not new_q and (root / "meta.json").is_file():
        _log("cache up to date")
        return root

    t0 = time.time()
    enc = OpenCLIPEncoder(model_path, device=device, text_config=text_config)
    t_load = time.time() - t0
    _log(f"loaded {OpenCLIPEncoder.name} in {t_load:.0f}s: d_c={enc.d_c}")
    root.mkdir(parents=True, exist_ok=True)
    if new_q:
        E = l2_normalize(enc.encode_texts([q_text]), "queries")
        if E.shape[1] != enc.d_c:
            raise RuntimeError(f"text tower returned {E.shape[1]}-d embeddings, d_c={enc.d_c}")
        write_queries(root, {**have, key: {"query": q_text, "embedding": [float(x) for x in E[0]]}})

    t1, n_frames = time.time(), 0
    for s, ps in todo.items():
        for i, p in enumerate(ps):
            tp = time.time()
            write_frames(root, p.pid, p.frame_ids, encode_patient(enc, p, batch))
            n_frames += p.T
            dt = time.time() - tp
            _log(f"[{s} {i + 1}/{len(ps)}] {p.pid} T={p.T} {dt:.1f}s ({p.T / max(dt, 1e-9):.1f} fr/s)")
    t_frames = time.time() - t1
    run = {"utc": _utc(), "splits": {s: len(ps) for s, ps in todo.items()}, "frames": n_frames, "queries": new_q,
           "load_s": round(t_load, 1), "frames_s": round(t_frames, 1),
           "frames_per_s": round(n_frames / max(t_frames, 1e-9), 2), "batch": batch, "device": device,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    meta = build_meta(root, enc.describe(), enc.d_c, model_path, preprocess, tag, cohort, run)
    write_meta(root, meta)
    _log(f"done: {n_frames} frames in {t_frames:.0f}s; meta at {root / 'meta.json'}")
    return root


def _csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m medtrace.features.extract_clip",
                                 description="Cache g^V(I_t) (view clip_img) and g^T(Q_bio) of a dual encoder "
                                             "for the ADA relevance (eq:ada_relevance); Q_bio = prompts.biomarker_query "
                                             "of the environment JSON ($MEDTRACE_ENV).")
    ap.add_argument("--model", required=True, help="model dir name under models_root, or a path")
    ap.add_argument("--manifest", default=None, help="dataset manifest JSON (medtrace.data.cohort); required "
                                                     "unless --print-tag")
    ap.add_argument("--splits", type=_csv, default=["train"],
                    help="comma list (default train: ADA samples the PRM training pool only)")
    ap.add_argument("--pids", nargs="+", default=None, help="restrict to these pids (space or comma separated)")
    ap.add_argument("--batch", type=int, default=64, help="images per image-tower forward")
    ap.add_argument("--force", action="store_true", help="re-extract cached patients and queries")
    ap.add_argument("--device", default="cpu", help="torch device (ViT-B/16 runs fine on CPU)")
    ap.add_argument("--cache-root", default=None, help="default: medtrace.paths cache_root")
    ap.add_argument("--text-config", default=None,
                    help="local dir with the HF text tower's config.json (default: <models dir>/<hub id basename>)")
    ap.add_argument("--print-tag", action="store_true", help="print the cache tag and dir, then exit")
    return ap


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.pids is not None:
        a.pids = [x for tok in a.pids for x in _csv(tok)]
    if not a.print_tag and not a.manifest:
        ap.error("--manifest is required (the dataset manifest JSON)")
    if not a.splits:
        ap.error("--splits is empty")
    if a.batch < 1:
        ap.error("--batch must be >= 1")
    return a


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    if a.print_tag:
        paths = get_paths()
        model_path = paths.model(a.model)
        pre = OpenCLIPEncoder.read_preprocess(model_path)
        tag = clip_tag(model_path, pre)
        print(json.dumps({"encoder_tag": tag, "root": str(Path(a.cache_root or paths.cache_root) / "features" / tag),
                          "model_path": model_path, "backend": OpenCLIPEncoder.name, "preprocess": pre}, indent=2))
        return 0
    extract(a.model, a.manifest, splits=a.splits, pids=a.pids, batch=a.batch,
            force=a.force, device=a.device, cache_root=a.cache_root, text_config=a.text_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
