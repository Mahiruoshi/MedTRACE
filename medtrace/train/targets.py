from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..data.cohort import load_targets, manifest_sha256, task_type
from ..data.tasks import Target, Task, load_task
from ..features.store import CachedTextEmbedder, FeatureStore
from ..generation.candidates import Candidate, load_candidates
from ..generation.parsing import PAD_MARK
from ..mappings.base import EPS, Frames, TextEmbedder, embed_text
from ..paths import get_paths
from ..pipeline.rundir import code_sha256, env_versions, git_state, read_json, sha256_file, utc_now, write_json
from ..prm.layouts import NEG_TOKEN, POS_TOKEN
from .dataset import target_to_json
from .mc import MANIFEST as MC_MANIFEST
from .mc import ROLLOUTS, check_task, load_rollouts
from .split import load_partition

PathLike = Union[str, os.PathLike]

DEFAULT_BETA = 0.5
CANDIDATES = "generate/candidates.jsonl"
CONFIG_DEFAULTS: Dict[str, Any] = {"beta": None, "manifest": None, "cache_root": None}


def sim_plus(H: np.ndarray, P: np.ndarray) -> np.ndarray:
    H = np.atleast_2d(np.asarray(H, dtype=np.float64))
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    if H.shape[1] != P.shape[1]:
        raise ValueError(f"dimension mismatch: steps are {H.shape[1]}-d, frames are {P.shape[1]}-d "
                         f"(sim+ compares f_LM(y_i) with Pi_vlm(F_t), both in R^d_h)")
    den = np.outer(np.linalg.norm(H, axis=1), np.linalg.norm(P, axis=1))
    cos = (H @ P.T) / np.maximum(den, EPS)
    return 0.5 * (1.0 + np.clip(cos, -1.0, 1.0))


def step_embeddings(steps: Sequence[str], text: TextEmbedder) -> np.ndarray:
    if not steps:
        raise ValueError("a trajectory has at least one step")
    return np.stack([embed_text(text, s) for s in steps])


def match_matrix(steps: Sequence[str], frames: Frames, text: TextEmbedder) -> np.ndarray:
    if frames.T == 0:
        raise ValueError("s^match needs at least one retained frame")
    P = frames.view("projected")
    if P.shape[1] != int(text.d_h):
        raise ValueError(f"projected view is {P.shape[1]}-d, f_LM is {text.d_h}-d: they must come from the "
                         f"same generator G")
    return sim_plus(step_embeddings(steps, text), P)


def apply_gates(S: np.ndarray, gates: Sequence[float]) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    g = np.asarray(gates, dtype=np.float64).ravel()
    if S.ndim != 2 or g.shape != (S.shape[1],):
        raise ValueError(f"{g.size} gates for a sim+ matrix of shape {S.shape}")
    if not (np.all(np.isfinite(g)) and np.all(g > 0) and np.all(g <= 1)):
        raise ValueError("gates must lie in (0, 1]: alpha in [alpha_min, alpha_max], a subset of (0, 1]")
    return (S * g[None, :]).max(axis=1)


def s_match(steps: Sequence[str], frames: Frames, gates: Sequence[float], text: TextEmbedder) -> np.ndarray:
    return apply_gates(match_matrix(steps, frames, text), gates)


def signal_gates(signal: Mapping[str, Any]) -> np.ndarray:
    sel = [int(i) for i in signal["selected"]]
    if "gates" not in signal:
        raise KeyError(f"{signal['pid']}: signals.json entry without gates; rerun the pool's distill stage")
    g = np.asarray(signal["gates"], dtype=np.float64)
    if g.shape != (len(sel),):
        raise ValueError(f"{signal['pid']}: {g.size} recorded gates for {len(sel)} selected frames")
    return g


def resolve_beta(beta: Optional[float] = None) -> float:
    b = DEFAULT_BETA if beta is None else float(beta)
    if not 0.0 <= b <= 1.0:
        raise ValueError(f"beta must lie in [0, 1], got {b}")
    return b


def fuse(s_match: Sequence[float], p_mc: Sequence[float], beta: float) -> np.ndarray:
    s = np.asarray(s_match, dtype=np.float64)
    p = np.asarray(p_mc, dtype=np.float64)
    if s.shape != p.shape:
        raise ValueError(f"s_match {s.shape} and p_mc {p.shape} differ in shape")
    b = float(beta)
    if not 0.0 <= b <= 1.0:
        raise ValueError(f"beta must lie in [0, 1], got {b}")
    for name, v in (("s_match", s), ("p_mc", p)):
        if not np.all((v >= 0.0) & (v <= 1.0)):
            raise ValueError(f"{name} must lie in [0, 1] (the common scale of eq:fused_signal)")
    return b * s + (1.0 - b) * p


def markers(p_tilde: Sequence[float], theta: float) -> List[str]:
    return [POS_TOKEN if v > theta else NEG_TOKEN for v in np.asarray(p_tilde, dtype=np.float64)]


def _classes(p: Sequence[float], correct: Sequence[bool]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64).ravel()
    c = np.asarray(correct, dtype=bool).ravel()
    if p.shape != c.shape:
        raise ValueError(f"{p.size} values for {c.size} correctness labels")
    if p.size == 0 or not np.all(np.isfinite(p)):
        raise ValueError("theta calibration needs finite values on at least one step")
    return p, np.sort(p[c]), np.sort(p[~c])


def _above(sorted_x: np.ndarray, v: np.ndarray) -> np.ndarray:
    return sorted_x.size - np.searchsorted(sorted_x, v, side="right")


def youden_theta(p: Sequence[float], correct: Sequence[bool]) -> float:
    p, pos, neg = _classes(p, correct)
    if pos.size == 0 or neg.size == 0:
        raise ValueError(f"Youden's J needs correct and incorrect trajectories on the calibration split "
                         f"(have {pos.size} / {neg.size} steps)")
    u = np.unique(p)
    j = _above(pos, u) * neg.size - _above(neg, u) * pos.size
    return float(u[int(np.argmax(j))])


def separation(p: Sequence[float], correct: Sequence[bool], theta: float) -> Dict[str, Any]:
    p, pos, neg = _classes(p, correct)
    tp, fp = int(_above(pos, np.array([theta]))[0]), int(_above(neg, np.array([theta]))[0])
    tpr = tp / pos.size if pos.size else None
    fpr = fp / neg.size if neg.size else None
    return {"n_steps": int(p.size), "n_correct_steps": int(pos.size), "n_incorrect_steps": int(neg.size),
            "pos_rate": (tp + fp) / p.size, "tpr": tpr, "fpr": fpr,
            "J": None if tpr is None or fpr is None else tpr - fpr}


@dataclass
class Pool:
    root: Path
    key: str
    seed: int
    task: Any
    split: str
    encoder_tag: str
    signals: Dict[str, Dict[str, Any]]
    prompts: Dict[str, Dict[str, Any]]
    candidates: List[Candidate]
    distill_inputs: Dict[str, str]
    distill_config: Dict[str, Any]
    sha: Dict[str, str]
    calib: Optional[Dict[str, Any]] = None


def _complete(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: the stage has not run")
    m = read_json(path)
    if m.get("status") != "complete":
        raise RuntimeError(f"{path}: stage is {m.get('status')!r}, not complete")
    return m


def open_pool(root: PathLike) -> Pool:
    root = Path(root)
    dm = _complete(root / "distill" / "manifest.json")
    gm = _complete(root / "generate" / "manifest.json")
    rel = {"distill/signals.json": root / "distill" / "signals.json",
           "generate/prompts.json": root / "generate" / "prompts.json",
           CANDIDATES: root / CANDIDATES}
    sha = {k: sha256_file(p) for k, p in rel.items()}
    recorded = {"distill/signals.json": [(dm.get("outputs") or {}), (gm.get("inputs") or {})],
                "generate/prompts.json": [gm.get("outputs") or {}],
                CANDIDATES: [gm.get("outputs") or {}]}
    for k, books in recorded.items():
        for book in books:
            if book.get(k) != sha[k]:
                raise RuntimeError(f"{root}: {k} differs from what the manifests recorded "
                                   f"(rewritten or regenerated since); rerun the stage")
    blob = read_json(rel["distill/signals.json"])
    signals = {s["pid"]: s for s in blob["signals"]}
    prompts = read_json(rel["generate/prompts.json"])
    if set(prompts) != set(signals):
        raise RuntimeError(f"{root}: prompts.json and signals.json cover different patients")
    order = {pid: i for i, pid in enumerate(prompts)}
    cands = load_candidates(rel[CANDIDATES])
    stray = sorted({c.pid for c in cands} - set(order))
    if stray:
        raise RuntimeError(f"{root}: candidates of patients without a prompt, e.g. {stray[:3]}")
    cands.sort(key=lambda c: (order[c.pid], c.cot_idx))
    run = gm["run"]
    return Pool(root=root, key=f"{run['variant']}/seed{int(run['seed'])}", seed=int(run["seed"]),
                task=(gm.get("config") or {}).get("task"), split=str(dm["config"]["split"]),
                encoder_tag=str(blob["meta"]["encoder_tag"]),
                signals=signals, prompts=prompts, candidates=cands,
                distill_inputs=dict(dm.get("inputs") or {}), distill_config=dict(dm.get("config") or {}),
                sha=sha, calib=blob["meta"].get("calib"))


def load_mc(mc_dir: PathLike, run: str) -> Dict[str, List[float]]:
    p = Path(mc_dir) / ROLLOUTS
    if not p.is_file():
        raise FileNotFoundError(f"{mc_dir}: no {ROLLOUTS}; run medtrace.train.mc on this pool")
    out: Dict[str, List[float]] = {}
    for rec in load_rollouts(p):
        where = f"{p}: {rec['id']}"
        if rec.get("run") != run:
            raise ValueError(f"{where}: rolled out from pool {rec.get('run')!r}, not {run!r}")
        vals = rec.get("p_mc")
        if not isinstance(vals, list) or not vals:
            raise KeyError(f"{where}: no p_mc list")
        vals = [float(v) for v in vals]
        if any(not 0.0 <= v <= 1.0 for v in vals):
            raise ValueError(f"{where}: p_mc outside [0, 1]")
        out[str(rec["id"])] = vals
    return out


def assemble_p_mc(p_mc: Sequence[float], n: int, correct: bool, cid: str) -> List[float]:
    if len(p_mc) != n:
        raise ValueError(f"{cid}: {len(p_mc)} MC values for {n} steps")
    last = 1.0 if correct else 0.0
    if float(p_mc[-1]) != last:
        raise ValueError(f"{cid}: MC value of the final step ({p_mc[-1]}) != 1[ans(y) = y] = {last}; "
                         f"the MC module and the pool disagree on the label or the parse")
    return [float(v) for v in p_mc]


def _mc_tables(mc: Sequence[PathLike], pools: Sequence[Pool]) -> List[Dict[str, List[float]]]:
    dirs = [Path(m) for m in mc]
    if len(dirs) != len(pools):
        raise ValueError(f"{len(dirs)} MC source(s) for {len(pools)} pool(s): give one per pool, in pool order")
    for m, p in zip(dirs, pools):
        man = m / MC_MANIFEST
        if not man.is_file():
            raise FileNotFoundError(f"MC source {m} has no {MC_MANIFEST}: not a medtrace.train.mc output dir")
        if (read_json(man).get("inputs") or {}).get(CANDIDATES) != p.sha[CANDIDATES]:
            raise RuntimeError(f"MC source {m} was drawn from another pool than {p.key} (or the pool changed since)")
    return [load_mc(m, p.key) for m, p in zip(dirs, pools)]


@dataclass
class Matched:
    pool: Pool
    cand: Candidate
    gates: np.ndarray
    s: np.ndarray


def match_pools(pools: Sequence[Pool], store: FeatureStore, text: TextEmbedder,
                skip: Collection[str] = ()) -> Iterator[Matched]:
    full: Dict[str, Frames] = {}
    skip = set(skip)
    for pool in pools:
        for c in pool.candidates:
            if c.pid in skip:
                continue
            sig = pool.signals[c.pid]
            sel = [int(i) for i in sig["selected"]]
            ids = [sig["frame_ids"][i] for i in sel]
            if list(c.frame_ids) != ids or list(pool.prompts[c.pid]["frame_ids"]) != ids:
                raise RuntimeError(f"{pool.key} {c.id}: candidate/prompt frame ids differ from the signal's I*")
            if c.pid not in full:
                full[c.pid] = store.load(c.pid, views=("projected",), x_view="projected")
            if full[c.pid].keys != list(sig["frame_ids"]):
                raise RuntimeError(f"{pool.key} {c.pid}: the feature cache enumerates other frames than the "
                                   f"signal (T={full[c.pid].T} vs {len(sig['frame_ids'])})")
            g = signal_gates(sig)
            yield Matched(pool=pool, cand=c, gates=g, s=s_match(c.steps, full[c.pid].take(sel), g, text))


def _open_store(pools: Sequence[Pool], cache_root: Optional[PathLike]) -> Tuple[FeatureStore, CachedTextEmbedder]:
    tags = sorted({p.encoder_tag for p in pools})
    if len(tags) != 1:
        raise ValueError(f"pools were distilled on different feature caches {tags}; f_LM and Pi_vlm must be "
                         f"those of the one generator G")
    store = FeatureStore.open(str(cache_root or get_paths().cache_root), tags[0])
    meta_sha = sha256_file(store.root / "meta.json")
    stale = [p.key for p in pools if meta_sha not in p.distill_inputs.values()]
    if stale:
        raise RuntimeError(f"feature cache {store.root.name} is not the one pool(s) {stale} were distilled on "
                           f"(its meta.json changed since); rerun their distill stage")
    text = CachedTextEmbedder(store.root)
    d_h = store.meta.get("d_h")
    if d_h is not None and int(d_h) != text.d_h:
        raise ValueError(f"feature cache meta says d_h={d_h}, the embedding table is {text.d_h}-d")
    return store, text


def _open_pools(pools: Sequence[PathLike]) -> List[Pool]:
    if not pools:
        raise ValueError("no pool run dirs given")
    out = [open_pool(p) for p in pools]
    keys = [p.key for p in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"the same pool (variant/seed) is given twice: {keys}")
    return out


def resolve_config(cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    c = dict(CONFIG_DEFAULTS)
    unknown = sorted(set(cfg or {}) - set(c))
    if unknown:
        raise KeyError(f"unknown targets config keys {unknown}; allowed: {sorted(c)}")
    c.update(cfg or {})
    c["beta"] = resolve_beta(c["beta"])
    return c


def target_tag(cfg: Optional[Mapping[str, Any]] = None) -> str:
    return f"fused-b{resolve_config(cfg)['beta']:g}-youden"


def _check_calibration(pool: Pool, train: Mapping[str, Any]) -> None:
    foreign = sorted(set((pool.calib or {}).get("pids") or ()) - set(train))
    if foreign:
        raise ValueError(f"{pool.key}: its calibration set holds patients outside the manifest's train split "
                         f"(e.g. {foreign[:3]}); PRM training never touches the test split, not even its "
                         f"unlabeled frames")


def _built_on(pool: Pool, sha: str) -> bool:
    return sha in pool.distill_inputs.values() or sha in json.dumps(pool.distill_config, sort_keys=True)


def _targets(manifest: Path, task: Task, pools: Sequence[Pool]) -> Dict[str, Target]:
    sha = manifest_sha256(manifest)
    for p in pools:
        if not _built_on(p, sha):
            raise RuntimeError(f"{p.key}: the dataset manifest {manifest} is not the one this pool was built on")
    return load_targets(manifest, "train", task)


def build_rows(pools: Sequence[PathLike], mc: Sequence[PathLike], split: PathLike,
               cfg: Optional[Mapping[str, Any]] = None, out: Optional[PathLike] = None,
               force: bool = False, task: Optional[Task] = None) -> Dict[str, Any]:
    c = resolve_config(cfg)
    beta = c["beta"]
    if not c["manifest"]:
        raise ValueError("targets read the labels of the dataset manifest: set config 'manifest' (--manifest)")
    manifest = Path(c["manifest"])
    if not mc:
        raise ValueError("the fused target needs p^MC: give one medtrace.train.mc output dir per pool (--mc)")
    if task is None:
        from .. import envjson

        task = load_task(envjson.load_env(), task_type(manifest))
    runs = _open_pools(pools)
    parts = load_partition(split, manifest)
    assign = {pid: part for part in ("fit", "meta") for pid in parts[part]}
    recorded = sorted({json.dumps(p.task, sort_keys=True) for p in runs})
    if len(recorded) != 1:
        raise ValueError(f"pools of different tasks {recorded}")
    for p in runs:
        check_task(p.task, task, p.key)
    not_train = [p.key for p in runs if p.split != "train"]
    if not_train:
        raise ValueError(f"pools {not_train} were not built on the train split; PRM training never touches the "
                         f"test split")
    targets = _targets(manifest, task, runs)
    undetermined = {pid for pid, t in targets.items() if not task.determined(t)}
    for p in runs:
        _check_calibration(p, targets)
        unknown = sorted(pid for pid in p.prompts if pid not in targets)
        if unknown:
            raise ValueError(f"{p.key}: patients {unknown[:5]} are not in the manifest's train split")
        outside = sorted(pid for pid in p.prompts if pid not in assign and pid not in undetermined)
        if outside:
            raise ValueError(f"{p.key}: patients {outside[:5]} are in neither fit nor meta")
        listed = sorted(pid for pid in p.prompts if pid in assign and pid in undetermined)
        if listed:
            raise ValueError(f"{p.key}: undetermined patients {listed[:5]} are in the partition (drawn under "
                             f"another task); re-partition with medtrace.train.split")
    tables = dict(zip((p.key for p in runs), _mc_tables(mc, runs)))
    store, text = _open_store(runs, c["cache_root"])
    left_out = {"undetermined_patients": len({pid for p in runs for pid in p.prompts} & undetermined),
                "undetermined_candidates": sum(cand.pid in undetermined for p in runs for cand in p.candidates)}

    rows: List[Dict[str, Any]] = []
    seen: set = set()
    n_duplicates = 0
    for m in match_pools(runs, store, text, skip=undetermined):
        pool, cand = m.pool, m.cand
        rid = f"{pool.key}/{cand.id}"
        dkey = (cand.pid, tuple(cand.frame_ids), cand.raw)
        if dkey in seen:
            n_duplicates += 1
            continue
        seen.add(dkey)
        target = targets[cand.pid]
        correct = task.is_correct(cand.pred, target)
        table = tables[pool.key]
        if cand.id not in table:
            raise KeyError(f"{pool.key}: no MC estimates for candidate {cand.id}")
        p_mc = assemble_p_mc(table[cand.id], len(cand.steps), bool(correct), rid)
        pt = fuse(m.s, p_mc, beta)
        prompt = pool.prompts[cand.pid]
        rows.append({
            "id": rid, "cand_id": cand.id, "run": pool.key, "pid": cand.pid, "split": assign[cand.pid],
            "seed": pool.seed, "decode_seed": cand.seed, "cot_idx": cand.cot_idx,
            "system": prompt["system"], "question": prompt["question"], "frame_ids": list(cand.frame_ids),
            "gates": [float(g) for g in m.gates], "steps": list(cand.steps),
            "padded": [PAD_MARK in s for s in cand.steps], "truncated": cand.meta.get("truncated"),
            "answer": cand.answer, "pred": cand.pred, "target": target_to_json(target),
            "y": task.stratum(target), "correct": int(correct),
            "s_match": [float(v) for v in m.s], "p_mc": p_mc, "p_tilde": [float(v) for v in pt], "markers": None,
            "beta": beta})

    meta_rows = [r for r in rows if r["split"] == "meta"]
    if not meta_rows:
        raise ValueError("theta is calibrated on the meta split, and the pools hold no meta patient")
    theta = youden_theta([v for r in meta_rows for v in r["p_tilde"]],
                         [r["correct"] for r in meta_rows for _ in r["p_tilde"]])
    for r in rows:
        r["markers"] = markers(r["p_tilde"], theta)

    stats = _stats(c, theta, rows, n_duplicates, left_out)
    result = {"rows": rows, "theta": theta, "stats": stats, "config": {**c, "target_tag": target_tag(c)}}
    if out is not None:
        inputs = {(p.root / rel).resolve().as_posix(): h for p in runs for rel, h in p.sha.items()}
        inputs[manifest.resolve().as_posix()] = sha256_file(manifest)
        inputs[(store.root / "meta.json").resolve().as_posix()] = sha256_file(store.root / "meta.json")
        inputs[Path(split).resolve().as_posix()] = sha256_file(split)
        for src in mc:
            f = Path(src) / ROLLOUTS
            inputs[f.resolve().as_posix()] = sha256_file(f)
        _write(Path(out), result, inputs, [p.key for p in runs], store.root.name, task, force)
    return result


def _stats(c: Mapping[str, Any], theta: float, rows: Sequence[Mapping[str, Any]],
           n_duplicates: int, left_out: Mapping[str, int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"theta": theta, "method": "youden", "calibrated_on": "meta",
                           "beta": c["beta"], "target_tag": target_tag(c),
                           "counts": {"rows": len(rows), "duplicates_dropped": int(n_duplicates),
                                      **{k: int(v) for k, v in left_out.items()}}}
    for part in ("fit", "meta"):
        R = [r for r in rows if r["split"] == part]
        if not R:
            out[part] = None
            continue
        ypid = {r["pid"]: r["y"] for r in R}
        out[part] = {
            "patients": len(ypid),
            "patients_by_stratum": {str(s): sum(v == s for v in ypid.values()) for s in sorted(set(ypid.values()))},
            "trajectories": len(R), "correct_trajectories": sum(int(r["correct"]) for r in R),
            "markers": separation([v for r in R for v in r["p_tilde"]],
                                  [r["correct"] for r in R for _ in r["p_tilde"]], theta),
            "mean": {name: float(np.mean([x for r in R for x in r[name]]))
                     for name in ("s_match", "p_mc", "p_tilde")}}
    warnings = []
    sep_meta = out["meta"]["markers"]
    if sep_meta["J"] is None or sep_meta["J"] <= 0:
        warnings.append("J <= 0 on meta: the markers carry no information about correctness there")
    for part in ("fit", "meta"):
        if out[part] is not None and not 0.02 <= out[part]["markers"]["pos_rate"] <= 0.98:
            warnings.append(f"positive rate {out[part]['markers']['pos_rate']:.3f} on {part}: "
                            f"markers are nearly constant")
    out["warnings"] = warnings
    return out


def _write(out: Path, result: Mapping[str, Any], inputs: Mapping[str, str], pools: Sequence[str],
           encoder_tag: str, task: Task, force: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"{out} is not empty; use a new --out or --force (the old dir is renamed aside)")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        os.replace(out, out.with_name(f"{out.name}.superseded-{stamp}"))
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    tmp = out / f".rows.jsonl.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in result["rows"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, rows_path)
    write_json(out / "stats.json", result["stats"])
    outputs = {n: sha256_file(out / n) for n in ("rows.jsonl", "stats.json")}
    write_json(out / "manifest.json", {
        "stage": "targets", "status": "complete", "created_utc": utc_now(), "config": result["config"],
        "task": task.public(), "pools": list(pools), "encoder_tag": encoder_tag, "n_rows": len(result["rows"]),
        "inputs": dict(inputs), "outputs": outputs, **git_state(), "code_sha256": code_sha256(),
        "env": env_versions()})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m medtrace.train.targets", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pools", nargs="+", required=True, help="pipeline run dirs (<runs>/<exp>/<gen>/<variant>/seed<S>)")
    p.add_argument("--mc", nargs="+", required=True,
                   help="one medtrace.train.mc output dir per pool, in pool order")
    p.add_argument("--split-json", required=True,
                   help="fit/meta partition of the train split (python -m medtrace.train split)")
    p.add_argument("--out", required=True, help="output dir (targets/<target_tag>/)")
    p.add_argument("--beta", type=float, default=None, help=f"default {DEFAULT_BETA}")
    p.add_argument("--manifest", required=True, help="dataset manifest the pools were built on (train-split labels)")
    p.add_argument("--cache-root", default=None, help="feature cache root (default: MEDTRACE_CACHE_ROOT or ./cache)")
    p.add_argument("--force", action="store_true", help="rename a non-empty --out aside and rebuild")
    return p


def _fmt(d: Optional[Mapping[str, Any]], keys: Sequence[str]) -> str:
    if not d:
        return "-"
    return " ".join(f"{k}={d[k]:.4f}" if isinstance(d.get(k), float) else f"{k}={d.get(k)}" for k in keys)


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = {"beta": a.beta, "manifest": a.manifest, "cache_root": a.cache_root}
    res = build_rows(a.pools, a.mc, a.split_json, cfg, out=a.out, force=a.force)
    st = res["stats"]
    n = st["counts"]
    print(f"{res['config']['target_tag']}: {n['rows']} rows ({n['duplicates_dropped']} duplicates dropped; "
          f"{n['undetermined_patients']} undetermined patients left out), theta={res['theta']:.6f} "
          f"({st['method']})")
    for part in ("meta", "fit"):
        print(f"{part}: {_fmt((st[part] or {}).get('markers'), ('J', 'tpr', 'fpr', 'pos_rate'))}")
    for w in st["warnings"]:
        print(f"WARNING: {w}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
