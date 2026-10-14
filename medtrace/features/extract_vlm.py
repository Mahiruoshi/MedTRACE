from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .. import envjson
from ..data.cohort import Manifest, Patient, load_manifest, load_patients, task_type
from ..data.tasks import Task, load_task
from ..paths import get_paths
from .store import VIEWS, CachedTextEmbedder, FeatureStore, encoder_tag

TEXT_FILES = ("embed_table.npy", "tokenizer.json")
PROBE_SIDES = (512, 256)
NUMERIC_PROBE = "-0.12 3.4e-05 17 0.0081 -2.5"


def env_task(manifest: Union[str, os.PathLike, Manifest], need: Sequence[str] = ()) -> Task:
    return load_task(envjson.load_env(), task_type(manifest), need=need)


def read_preprocessor_config(model_path: str) -> Dict:
    p = Path(model_path) / "preprocessor_config.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found; is {model_path} a Qwen-VL checkpoint directory?")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def pixel_config(model_path: str, min_pixels: Optional[int] = None,
                 max_pixels: Optional[int] = None) -> Dict:
    pp = read_preprocessor_config(model_path)
    size = pp.get("size") or {}
    lo = pp.get("min_pixels", size.get("shortest_edge"))
    hi = pp.get("max_pixels", size.get("longest_edge"))
    if lo is None or hi is None:
        raise ValueError(f"{model_path}: preprocessor config names no min/max pixel budget")
    cfg = {
        "min_pixels": int(lo if min_pixels is None else min_pixels),
        "max_pixels": int(hi if max_pixels is None else max_pixels),
        "patch_size": int(pp["patch_size"]),
        "merge_size": int(pp["merge_size"]),
        "temporal_patch_size": int(pp["temporal_patch_size"]),
        "image_mean": [float(x) for x in pp["image_mean"]],
        "image_std": [float(x) for x in pp["image_std"]],
        "color": "RGB",
    }
    if cfg["min_pixels"] > cfg["max_pixels"]:
        raise ValueError(f"min_pixels {cfg['min_pixels']} > max_pixels {cfg['max_pixels']}")
    return cfg


def resolve_encoder_tag(model: str, min_pixels: Optional[int] = None,
                        max_pixels: Optional[int] = None) -> Tuple[str, Dict, str]:
    model_path = get_paths().model(model)
    pc = pixel_config(model_path, min_pixels, max_pixels)
    return model_path, pc, encoder_tag(model_path, pc)


def grid_for(height: int, width: int, pc: Dict) -> Tuple[int, int, int]:
    factor = pc["patch_size"] * pc["merge_size"]
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > pc["max_pixels"]:
        beta = math.sqrt(height * width / pc["max_pixels"])
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < pc["min_pixels"]:
        beta = math.sqrt(pc["min_pixels"] / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return 1, h_bar // pc["patch_size"], w_bar // pc["patch_size"]


def token_counts(grids: Sequence[Sequence[int]], merge: int) -> Tuple[List[int], List[int]]:
    pre = [int(t) * int(h) * int(w) for t, h, w in grids]
    for n in pre:
        if n % (merge * merge):
            raise ValueError(f"{n} patch tokens do not fold into {merge}x{merge} groups")
    return pre, [n // (merge * merge) for n in pre]


def pool_per_image(x, sizes: Sequence[int]):
    if sum(sizes) != x.shape[0]:
        raise RuntimeError(f"{x.shape[0]} rows for per-image sizes summing to {sum(sizes)}")
    out, off = [], 0
    for n in sizes:
        out.append(x[off:off + n].mean(0))
        off += n
    if isinstance(x, np.ndarray):
        return np.stack(out)
    import torch
    return torch.stack(out)


def load_vlm(model_path: str, device: str):
    import torch
    from transformers import AutoModelForImageTextToText

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map={"": device})
    except (ValueError, KeyError) as exc:
        import transformers
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)
        raise RuntimeError(
            f"{model_path} declares model_type={cfg.model_type!r} (architectures="
            f"{getattr(cfg, 'architectures', None)}), which transformers "
            f"{transformers.__version__} cannot load as an image-text-to-text model") from exc
    model.eval()
    return model


def find_vision_tower(model):
    for holder in (model, getattr(model, "model", None)):
        visual = getattr(holder, "visual", None) if holder is not None else None
        if visual is not None:
            return visual
    raise RuntimeError(f"{type(model).__name__} exposes no `visual` submodule")


def premerger_width(visual) -> int:
    merger = getattr(visual, "merger", None)
    if merger is None:
        raise RuntimeError(f"{type(visual).__name__} has no `merger`")
    for _name, mod in merger.named_children():
        shape = getattr(mod, "normalized_shape", None)
        if shape:
            return int(shape[0])
        weight = getattr(mod, "weight", None)
        if weight is not None and weight.ndim == 1 and "norm" in type(mod).__name__.lower():
            return int(weight.shape[0])
    raise RuntimeError(f"{type(merger).__name__} exposes no normalisation layer to read d_v from")


def merge_size(visual) -> int:
    m = getattr(visual, "spatial_merge_size", None)
    return int(m if m is not None else visual.config.spatial_merge_size)


def _read_image(path: str):
    from PIL import Image

    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception as exc:
        err = type(exc).__name__
    raise OSError(f"unreadable slice image ({err}); path withheld")


def read_frames(paths: Sequence[str], ids: Sequence[str]) -> List:
    if len(paths) != len(ids):
        raise ValueError(f"{len(paths)} frame paths for {len(ids)} frame ids")
    out = []
    for p, fid in zip(paths, ids):
        try:
            out.append(_read_image(p))
        except OSError as exc:
            raise OSError(f"{fid}: {exc}") from None
    return out


class VLMFeatureExtractor:
    def __init__(self, model_path: str, pixel_cfg: Dict, device: str = "cuda:0", batch_size: int = 8):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.model_path = model_path
        self.pixel_cfg = pixel_cfg
        self.device = device
        self.batch_size = int(batch_size)
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=pixel_cfg["min_pixels"], max_pixels=pixel_cfg["max_pixels"])
        self.model = load_vlm(model_path, device)
        self.visual = find_vision_tower(self.model)
        self.merger = self.visual.merger
        self.d_v = premerger_width(self.visual)
        self.d_h = int(self.model.get_input_embeddings().weight.shape[1])
        self.merge = merge_size(self.visual)
        if self.merge != pixel_cfg["merge_size"]:
            raise RuntimeError(f"vision tower merges {self.merge}x{self.merge}, preprocessor config says "
                               f"{pixel_cfg['merge_size']}")
        self.grids = {s: self.processor_grid(s, s) for s in PROBE_SIDES}
        for s, g in self.grids.items():
            if g != grid_for(s, s, pixel_cfg):
                raise RuntimeError(f"processor grid {g} for {s}x{s} != smart_resize port "
                                   f"{grid_for(s, s, pixel_cfg)}; update grid_for()")

    def _preprocess(self, imgs):
        return self.processor.image_processor(images=imgs, return_tensors="pt")

    def processor_grid(self, height: int, width: int) -> Tuple[int, int, int]:
        from PIL import Image

        vi = self._preprocess([Image.new("L", (width, height), 128).convert("RGB")])
        return tuple(int(v) for v in vi["image_grid_thw"][0].tolist())

    def encode(self, paths: Sequence[str], ids: Sequence[str],
               views: Sequence[str] = VIEWS) -> Dict[str, np.ndarray]:
        torch = self.torch
        bad = set(views) - set(VIEWS)
        if bad:
            raise KeyError(f"unknown views {sorted(bad)}; known {VIEWS}")
        cap: Dict[str, object] = {}

        def hook(_mod, args, kwargs, out):
            x = args[0] if args else next(v for v in kwargs.values() if torch.is_tensor(v))
            cap["in"], cap["out"] = x.detach(), out.detach()

        acc: Dict[str, List[np.ndarray]] = {v: [] for v in views}
        handle = self.merger.register_forward_hook(hook, with_kwargs=True)
        try:
            for i in range(0, len(paths), self.batch_size):
                vi = self._preprocess(read_frames(paths[i:i + self.batch_size], ids[i:i + self.batch_size]))
                pv = vi["pixel_values"].to(self.device, torch.bfloat16)
                grid = vi["image_grid_thw"]
                cap.clear()
                with torch.no_grad():
                    self.visual(pv, grid_thw=grid.to(self.device))
                g = [tuple(int(v) for v in row) for row in grid.tolist()]
                n_pre, n_post = token_counts(g, self.merge)
                x_in, x_out = cap["in"].float(), cap["out"].float()
                if x_in.shape != (sum(n_pre), self.d_v):
                    raise RuntimeError(f"merger input {tuple(x_in.shape)} != ({sum(n_pre)}, d_v={self.d_v}); "
                                       f"the merger normalises something other than its input here")
                if x_out.shape != (sum(n_post), self.d_h):
                    raise RuntimeError(f"merger output {tuple(x_out.shape)} != ({sum(n_post)}, d_h={self.d_h})")
                if "premerger" in acc:
                    acc["premerger"].append(pool_per_image(x_in, n_pre).cpu().numpy())
                if "projected" in acc:
                    acc["projected"].append(pool_per_image(x_out, n_post).cpu().numpy())
        finally:
            handle.remove()
        return {v: np.concatenate(a).astype(np.float32) for v, a in acc.items()}

    def write_text(self, tdir: Path, chunk: int = 8192) -> Dict:
        torch = self.torch
        tdir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            W = self.model.get_input_embeddings().weight.detach()
            V, d = int(W.shape[0]), int(W.shape[1])
            table = W.to(torch.float16).cpu().numpy()
            if not np.isfinite(table).all():
                raise RuntimeError("embedding table overflows float16")
            err = 0.0
            for i in range(0, V, chunk):
                w = W[i:i + chunk]
                err = max(err, float((w.float() - w.to(torch.float16).float()).abs().max()))
        _save_npy(tdir / "embed_table.npy", table)
        src = Path(self.model_path) / "tokenizer.json"
        if src.is_file():
            shutil.copyfile(src, tdir / "tokenizer.json")
            tok_source = "copied from model dir"
        else:
            self.processor.tokenizer.backend_tokenizer.save(str(tdir / "tokenizer.json"))
            tok_source = "saved from the fast tokenizer"
        return {"vocab_rows": V, "d_h": d, "fp16_max_abs_err": err, "tokenizer_source": tok_source}

    def verify_text(self, store_root: Path, instruction: str) -> Dict:
        if not instruction:
            raise ValueError("verify_text needs the task instruction Q (prompts.instruction)")
        torch = self.torch
        te = CachedTextEmbedder(store_root)
        emb = self.model.get_input_embeddings()
        tok = self.processor.tokenizer
        worst = 0.0
        probes = [instruction, NUMERIC_PROBE]
        for text in probes:
            ids = te.token_ids(text)
            ref = list(tok(text, add_special_tokens=False)["input_ids"])
            if ids != ref:
                raise RuntimeError(f"cached tokenizer.json disagrees with the HF tokenizer on {text!r}: "
                                   f"{ids} vs {ref}")
            with torch.no_grad():
                rows_t = emb(torch.tensor(ids, device=emb.weight.device)).float().cpu().numpy()
            worst = max(worst, float(np.abs(te.rows(ids) - rows_t).max()))
        if worst > 1e-5:
            raise RuntimeError(f"cached embedding rows differ from G's by {worst:.3g}")
        return {"probes": len(probes), "token_ids_equal": True, "rows_max_abs_diff": worst,
                "instruction_tokens": len(te.token_ids(instruction)),
                "hf_tokenizer_len": len(tok)}

    def describe(self) -> Dict:
        import transformers

        tokens = {}
        for s, g in self.grids.items():
            (pre,), (post,) = token_counts([g], self.merge)
            tokens[f"{s}x{s}"] = {"premerger": pre, "projected": post}
        return {"model_class": type(self.model).__name__,
                "vision_class": type(self.visual).__name__,
                "merger_class": type(self.merger).__name__,
                "image_processor": type(self.processor.image_processor).__name__,
                "attn_implementation": getattr(self.model.config, "_attn_implementation", None),
                "dtype": "bfloat16", "d_v": self.d_v, "d_h": self.d_h, "merge_size": self.merge,
                "grid": {f"{s}x{s}": list(g) for s, g in self.grids.items()}, "tokens": tokens,
                "transformers": transformers.__version__, "torch": self.torch.__version__}


def _save_npy(path: Path, a: np.ndarray) -> None:
    tmp = path.with_name(f".{path.stem}.tmp.npy")
    np.save(tmp, a)
    os.replace(tmp, path)


def cached_views(store: FeatureStore, p: Patient) -> Optional[List[str]]:
    f = store.root / "frames" / f"{p.pid}.npz"
    if not f.is_file():
        return None
    try:
        with np.load(f, allow_pickle=False) as z:
            if [str(k) for k in z["frame_ids"]] != p.frame_ids:
                return None
            vs = [v for v in VIEWS if v in z.files]
            if any(z[v].shape[0] != p.T for v in vs):
                return None
            return vs
    except (OSError, ValueError, KeyError):
        return None


def is_cached(store: FeatureStore, p: Patient, views: Sequence[str]) -> bool:
    vs = cached_views(store, p)
    return vs is not None and set(views) <= set(vs)


def text_cached(store: FeatureStore) -> bool:
    return all((store.root / "text" / n).is_file() for n in TEXT_FILES)


def select_patients(cohort: Dict[str, List[Patient]], pids: Optional[Sequence[str]]) -> Dict[str, List[Patient]]:
    if not pids:
        return {s: list(ps) for s, ps in cohort.items()}
    keep = set(pids)
    missing = sorted(keep - {p.pid for ps in cohort.values() for p in ps})
    if missing:
        raise KeyError(f"pids not in splits {list(cohort)}: {missing}")
    return {s: [p for p in ps if p.pid in keep] for s, ps in cohort.items()}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[extract_vlm {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def extract(model: str, manifest: Union[str, os.PathLike, Manifest], splits: Sequence[str] = ("test", "train"),
            pids: Optional[Sequence[str]] = None, views: Sequence[str] = VIEWS, batch: int = 8,
            force: bool = False, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None,
            device: str = "cuda:0", cache_root: Optional[str] = None,
            task: Optional[Task] = None) -> FeatureStore:
    bad = set(views) - set(VIEWS)
    if not views or bad:
        raise KeyError(f"views must be a non-empty subset of {VIEWS}, got {list(views)}")
    paths = get_paths()
    model_path, pc, tag = resolve_encoder_tag(model, min_pixels, max_pixels)
    root = Path(cache_root or paths.cache_root) / "features" / tag
    store = FeatureStore(root)
    man = load_manifest(manifest)
    cohort = {s: load_patients(man, s, image_root=paths.image_root) for s in splits}
    todo = {s: [p for p in ps if force or not is_cached(store, p, views)]
            for s, ps in select_patients(cohort, pids).items()}
    need_text = force or not text_cached(store)
    n_todo = sum(len(v) for v in todo.values())
    _log(f"tag={tag} root={root} views={list(views)} pixel_cfg(min,max)=({pc['min_pixels']},{pc['max_pixels']})")
    _log("to extract: " + ", ".join(f"{s} {len(todo[s])}/{len(cohort[s])} patients" for s in cohort)
        + f"; text {'yes' if need_text else 'cached'}")
    if n_todo == 0 and not need_text and (root / "meta.json").is_file():
        _log("cache up to date")
        return store
    task = task or env_task(man, need=("prompts.instruction",))
    if not task.instruction:
        raise ValueError("the task instruction Q (prompts.instruction) is empty; the cached tokenizer "
                         "is verified on it")

    t0 = time.time()
    ex = VLMFeatureExtractor(model_path, pc, device=device, batch_size=batch)
    t_load = time.time() - t0
    _log(f"loaded {type(ex.model).__name__} in {t_load:.0f}s: d_v={ex.d_v} d_h={ex.d_h} "
        f"grid512={ex.grids[512]} grid256={ex.grids[256]}")

    text_info = {}
    t1 = time.time()
    if need_text:
        text_info = ex.write_text(root / "text")
        _log(f"text table V={text_info['vocab_rows']} x {text_info['d_h']} written "
            f"(bf16->fp16 max err {text_info['fp16_max_abs_err']:.2e})")
    text_check = ex.verify_text(root, task.instruction)
    t_text = time.time() - t1

    t2 = time.time()
    n_frames = 0
    for s, ps in todo.items():
        for i, p in enumerate(ps):
            tp = time.time()
            keep = cached_views(store, p) or []
            arrays = ex.encode(p.frames, p.frame_ids, [v for v in VIEWS if v in views or v in keep])
            for v, a in arrays.items():
                if not np.isfinite(a).all():
                    raise RuntimeError(f"{p.pid}: non-finite values in view {v}")
            store.write_frames(p.pid, p.frame_ids, arrays)
            n_frames += p.T
            dt = time.time() - tp
            _log(f"[{s} {i + 1}/{len(ps)}] {p.pid} T={p.T} {dt:.1f}s ({p.T / max(dt, 1e-9):.1f} fr/s)")
    t_frames = time.time() - t2

    run = {"utc": _utc(), "splits": {s: len(ps) for s, ps in todo.items()}, "frames": n_frames,
           "load_s": round(t_load, 1), "text_s": round(t_text, 1), "frames_s": round(t_frames, 1),
           "frames_per_s": round(n_frames / max(t_frames, 1e-9), 2), "batch": batch, "device": device,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    meta = build_meta(store, ex.describe(), model_path, pc, tag, cohort,
                      pixel_source="model_default" if min_pixels is None and max_pixels is None else "explicit",
                      text_info=text_info, text_check=text_check, run=run)
    store.write_meta(meta)
    _log(f"done: {n_frames} frames in {t_frames:.0f}s; meta at {root / 'meta.json'}")
    return store


def build_meta(store: FeatureStore, desc: Dict, model_path: str, pc: Dict, tag: str,
               cohort: Dict[str, List[Patient]], pixel_source: str, text_info: Dict,
               text_check: Dict, run: Dict) -> Dict:
    old = store.meta if (store.root / "meta.json").is_file() else {}
    for k in ("d_v", "d_h"):
        if old and old.get(k) != desc[k]:
            raise RuntimeError(f"{store.root}: existing meta has {k}={old.get(k)}, this model {desc[k]}")
    splits = dict(old.get("splits", {}))
    for s, ps in cohort.items():
        have = {p.pid: cached_views(store, p) for p in ps}
        done = [p for p in ps if have[p.pid] is not None]
        vs = [v for v in VIEWS if done and all(v in have[p.pid] for p in done)]
        splits[s] = {"n_patients": len(done), "n_manifest": len(ps),
                     "n_frames": int(sum(p.T for p in done)), "views": vs,
                     "complete": len(done) == len(ps),
                     "missing": [p.pid for p in ps if have[p.pid] is None]}
    present = [set(e["views"]) for e in splits.values() if e["n_patients"]]
    return {
        "encoder_tag": tag, "model_path": model_path, "model_class": desc["model_class"],
        "d_v": desc["d_v"], "d_h": desc["d_h"],
        "views": [v for v in VIEWS if present and all(v in p for p in present)],
        "pixel_cfg": pc, "pixel_source": pixel_source,
        "grid_512x512": desc["grid"]["512x512"], "grid_256x256": desc["grid"]["256x256"],
        "tokens_per_image": desc["tokens"],
        "encoder": {k: desc[k] for k in ("vision_class", "merger_class", "image_processor",
                                         "attn_implementation", "dtype", "merge_size")},
        "pooling": {"premerger": "fp32 mean over merger-input patch rows (t*h*w per image)",
                    "projected": "fp32 mean over merger-output rows (t*h*w/merge^2 per image); "
                                 "main merger only (Qwen3-VL deepstack mergers ignored)"},
        "text": {**old.get("text", {}), **text_info, "check": text_check},
        "splits": splits,
        "transformers": desc["transformers"], "torch": desc["torch"],
        "created_utc": old.get("created_utc", run["utc"]), "updated_utc": run["utc"],
        "runs": list(old.get("runs", [])) + [run],
    }


def _csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m medtrace.features.extract_vlm",
                                 description="Extract F_t, Pi_vlm(F_t) and the f_LM table of G into the feature "
                                             "cache; the tokenizer is verified on prompts.instruction of the "
                                             "environment JSON ($MEDTRACE_ENV).")
    ap.add_argument("--model", required=True, help="model dir name under models_root, or a path")
    ap.add_argument("--manifest", default=None, help="dataset manifest JSON (medtrace.data.cohort); required "
                                                     "unless --print-tag")
    ap.add_argument("--splits", type=_csv, default=["test", "train"], help="comma list (default test,train)")
    ap.add_argument("--pids", nargs="+", default=None, help="restrict to these pids (space or comma separated)")
    ap.add_argument("--views", type=_csv, default=list(VIEWS), help=f"comma list from {','.join(VIEWS)}")
    ap.add_argument("--batch", type=int, default=8, help="images per vision-tower forward")
    ap.add_argument("--force", action="store_true", help="re-extract cached patients and the text table")
    ap.add_argument("--min-pixels", type=int, default=None, help="override the preprocessor's min_pixels")
    ap.add_argument("--max-pixels", type=int, default=None, help="override the preprocessor's max_pixels")
    ap.add_argument("--device", default="cuda:0", help="torch device (pick the card with CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--cache-root", default=None, help="default: medtrace.paths cache_root")
    ap.add_argument("--print-tag", action="store_true", help="print the encoder tag and cache dir, then exit")
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
    if not a.views or set(a.views) - set(VIEWS):
        ap.error(f"--views must be a non-empty subset of {','.join(VIEWS)}, got {a.views}")
    a.views = [v for v in VIEWS if v in a.views]
    if a.batch < 1:
        ap.error("--batch must be >= 1")
    return a


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = parse_args(argv)
    if a.print_tag:
        model_path, pc, tag = resolve_encoder_tag(a.model, a.min_pixels, a.max_pixels)
        root = Path(a.cache_root or get_paths().cache_root) / "features" / tag
        print(json.dumps({"encoder_tag": tag, "root": str(root), "model_path": model_path, "pixel_cfg": pc,
                          "grid_512x512": list(grid_for(512, 512, pc))}, indent=2))
        return 0
    extract(a.model, a.manifest, splits=a.splits, pids=a.pids, views=a.views, batch=a.batch, force=a.force,
            min_pixels=a.min_pixels, max_pixels=a.max_pixels, device=a.device, cache_root=a.cache_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
