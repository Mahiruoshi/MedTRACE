from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..config import config_hash
from ..data.tasks import Target, Task
from ..generation.cache import GenerationCache, request_identity, request_key
from ..generation.candidates import Candidate, cut_partial_tail, load_candidates
from ..generation.parsing import parse
from ..generation.vllm_client import GenRequest, GenResult
from ..pipeline.rundir import (StageConflict, code_sha256, diff_keys, env_versions, git_state, read_json,
                               sha256_file, utc_now, write_json)
from ..seeds import SeedRegistry, sub_seed

MC_VERSION = 2
DEFAULT_R = 4
DEFAULT_MAX_TOKENS = 1024
POOL_SAMPLING = ("temperature", "top_p", "top_k", "repetition_penalty")
ROLLOUTS = "rollouts.jsonl"
MANIFEST = "manifest.json"
POOL_FILES = ("generate/candidates.jsonl", "generate/prompts.json")
PARSE_FIELDS = ("kind", "labels", "aliases", "step_names", "answer_prefix")


def task_fields(task: Task) -> Dict[str, Any]:
    pub = task.public()
    return {k: pub[k] for k in PARSE_FIELDS}


def check_task(recorded: Any, task: Task, where: str) -> None:
    if not isinstance(recorded, Mapping):
        raise ValueError(f"{where}: the generate stage recorded no task (config.task); rerun the pool")
    missing = [k for k in PARSE_FIELDS if k != "kind" and k not in recorded]
    if missing:
        raise ValueError(f"{where}: the generate stage's config.task lacks {missing}; rerun the pool")
    pub = task.public()
    diff = [k for k in PARSE_FIELDS if k in recorded and recorded[k] != pub[k]]
    if diff:
        raise ValueError(f"{where}: the environment JSON's task differs at {', '.join(diff)} from the one the "
                         f"pool was generated with; use the pool's environment JSON")


def rollout_prefix(steps: Sequence[str], i: int) -> str:
    if not 1 <= i < len(steps):
        raise ValueError(f"prefix length {i} outside [1, {len(steps) - 1}] for {len(steps)} steps")
    return "\n\n".join(steps[:i]) + "\n\n"


def rollout_seed(mc_seed: int, candidate_id: str, i: int) -> int:
    return sub_seed(int(mc_seed), candidate_id, int(i))


def rollout_key(candidate_id: str, i: int) -> str:
    return f"{candidate_id}@{i}"


def rollout_requests(cand: Candidate, prompt: Mapping[str, Any], image_paths: Sequence[str],
                     sampling: Mapping[str, Any], mc_seed: int, R: int, max_tokens: int) -> List[GenRequest]:
    base = {k: sampling[k] for k in POOL_SAMPLING}
    return [GenRequest(key=rollout_key(cand.id, i), system=prompt["system"], question=prompt["question"],
                       image_paths=list(image_paths),
                       sampling={**base, "max_tokens": int(max_tokens), "n": int(R),
                                 "seed": rollout_seed(mc_seed, cand.id, i)},
                       prefix=rollout_prefix(cand.steps, i))
            for i in range(1, len(cand.steps))]


def _hit(answer: Optional[str], target: Target, task: Task) -> int:
    hit = task.is_correct(task.parse_label(answer), target)
    if hit is None:
        raise ValueError("an undetermined patient (Task.determined) reached the MC hits")
    return hit


def mc_record(cand: Candidate, results: Sequence[GenResult], target: Target, task: Task, R: int, max_tokens: int,
              mc_seed: int, request_hashes: Sequence[str]) -> Dict[str, Any]:
    n = len(cand.steps)
    if len(results) != n - 1 or len(request_hashes) != n - 1:
        raise ValueError(f"{cand.id}: {len(results)} results / {len(request_hashes)} hashes for {n - 1} prefixes")
    hits: List[List[int]] = []
    answers: List[List[Optional[str]]] = []
    finish: Counter = Counter()
    usage: Counter = Counter()
    for i, res in enumerate(results, start=1):
        texts = [c["text"] for c in res.choices]
        prefix = rollout_prefix(cand.steps, i)
        ans = [parse(prefix + t, task).answer for t in texts]
        answers.append(ans)
        hits.append([_hit(a, target, task) for a in ans])
        finish.update(str(c.get("finish_reason")) for c in res.choices)
        usage.update({k: int(v) for k, v in res.usage.items()})
    answers.append([cand.answer])
    hits.append([_hit(cand.answer, target, task)])
    return {"id": cand.id, "pid": cand.pid, "cot_idx": cand.cot_idx, "seed": cand.seed, "n_steps": n,
            "R": int(R), "max_tokens": int(max_tokens),
            "mc_seeds": [rollout_seed(mc_seed, cand.id, i) for i in range(1, n)],
            "p_mc": [sum(h) / len(h) for h in hits], "hits": hits, "answers": answers,
            "n_unparseable": sum(a is None for row in answers[:-1] for a in row),
            "finish_reasons": dict(sorted(finish.items())), "usage": dict(sorted(usage.items())),
            "request_hashes": list(request_hashes)}


def _check_same_x(cand: Candidate, prompt: Mapping[str, Any], image_paths: Sequence[str], model: str) -> None:
    req = GenRequest(key=cand.id, system=prompt["system"], question=prompt["question"],
                     image_paths=list(image_paths), sampling=cand.meta["sampling"])
    if request_key(model, req) != cand.meta.get("request_hash"):
        raise ValueError(f"{cand.id}: prompt or slices differ from those it was generated from "
                         f"(request hash mismatch); the rollouts must condition on the same x")


def _result_error(res: GenResult, R: int) -> str:
    if not res.ok:
        return res.error or "unknown error"
    if len(res.choices) != R or not all(isinstance(c.get("text"), str) for c in res.choices):
        return f"asked for n={R} rollouts, got {len(res.choices)} choice(s) with text"
    return ""


class RolloutWriter:
    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.repaired = cut_partial_tail(self.path)
        self.done: Set[str] = {r["id"] for r in load_rollouts(self.path)} if self.path.is_file() else set()
        self._f = open(self.path, "a", encoding="utf-8", newline="\n")

    def append(self, rec: Mapping[str, Any]) -> None:
        if rec["id"] in self.done:
            raise ValueError(f"rollouts of {rec['id']} already written")
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())
        self.done.add(rec["id"])

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "RolloutWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_rollouts(path: os.PathLike | str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    for n, ln in enumerate(lines, start=1):
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
            rid = rec["id"]
        except (ValueError, KeyError, TypeError) as e:
            if n == len(lines):
                break
            raise ValueError(f"{path}:{n}: malformed rollout line ({type(e).__name__})") from None
        if rid in seen:
            raise ValueError(f"{path}:{n}: duplicate candidate id {rid}")
        seen.add(rid)
        out.append(rec)
    return out


@dataclass
class Pool:
    root: Path
    candidates: List[Candidate]
    prompts: Dict[str, Dict[str, Any]]
    manifest: Dict[str, Any]

    @property
    def key(self) -> str:
        run = self.manifest["run"]
        return f"{run['variant']}/seed{int(run['seed'])}"

    def model(self) -> str:
        models = {c.meta.get("model") for c in self.candidates}
        if len(models) != 1 or None in models:
            raise ValueError(f"{self.root}: candidates name {len(models)} generator(s) {sorted(map(str, models))}")
        return str(models.pop())

    def sampling(self) -> Dict[str, Any]:
        seen = set()
        for c in self.candidates:
            s = c.meta.get("sampling") or {}
            missing = [k for k in POOL_SAMPLING if k not in s]
            if missing:
                raise ValueError(f"{c.id}: meta.sampling lacks {missing}")
            seen.add(tuple(s[k] for k in POOL_SAMPLING))
        if len(seen) != 1:
            raise ValueError(f"{self.root}: candidates were sampled under {len(seen)} settings")
        return dict(zip(POOL_SAMPLING, seen.pop()))

    def input_hashes(self) -> Dict[str, str]:
        return {rel: sha256_file(self.root / rel) for rel in POOL_FILES}


def load_pool(run_dir: os.PathLike | str) -> Pool:
    root = Path(run_dir)
    mp = root / "generate" / "manifest.json"
    if not mp.is_file():
        raise FileNotFoundError(f"{root}: no generate/manifest.json; not a pipeline run dir with a pool")
    man = read_json(mp)
    if man.get("status") != "complete":
        raise RuntimeError(f"{root}: generate is {man.get('status')!r}, not complete; finish the pool first")
    recorded = man.get("outputs") or {}
    for rel in POOL_FILES:
        if recorded.get(rel) != sha256_file(root / rel):
            raise RuntimeError(f"{root}: {rel} is not the file generate recorded (rewritten since, or no "
                               f"sha256 recorded); rerun generate")
    cands = load_candidates(root / POOL_FILES[0])
    prompts = read_json(root / POOL_FILES[1])
    for c in cands:
        if c.pid not in prompts:
            raise ValueError(f"{c.id}: no prompt for {c.pid} in prompts.json")
        if list(c.frame_ids) != list(prompts[c.pid]["frame_ids"]):
            raise ValueError(f"{c.id}: frame ids differ from its prompt's")
    return Pool(root=root, candidates=cands, prompts=prompts, manifest=man)


def _pool_split(root: Path) -> str:
    mp = root / "distill" / "manifest.json"
    if not mp.is_file():
        raise FileNotFoundError(f"{root}: no distill/manifest.json; not a pipeline run dir with a pool")
    split = (read_json(mp).get("config") or {}).get("split")
    if not split:
        raise ValueError(f"{mp}: no config.split recorded")
    return str(split)


def _begin(out: Path, cfg: Dict[str, Any], inputs: Dict[str, str], pool: Pool, mc_seed: int) -> Dict[str, Any]:
    mp, rp = out / MANIFEST, out / ROLLOUTS
    old = read_json(mp) if mp.is_file() else None
    if old is None and rp.is_file() and rp.stat().st_size:
        raise StageConflict(f"{out} holds {ROLLOUTS} but no {MANIFEST}; refusing to extend it")
    if old is not None:
        if old.get("config_hash") != config_hash(cfg):
            keys = diff_keys(old.get("config", {}), cfg)
            raise StageConflict(f"{out}: rollouts were drawn under another config (differs at: "
                                f"{', '.join(keys) or 'unknown'}); use a new --out")
        changed = sorted(k for k in set(old.get("inputs", {})) | set(inputs)
                         if old.get("inputs", {}).get(k) != inputs.get(k))
        if changed:
            raise StageConflict(f"{out}: the pool changed since its rollouts were drawn ({', '.join(changed)})")
    man = {"stage": "mc", "status": "running",
           "created_utc": (old or {}).get("created_utc") or utc_now(), "updated_utc": utc_now(),
           "config_hash": config_hash(cfg), "config": cfg,
           "pool": {"run_dir": pool.root.resolve().as_posix(), "run": pool.manifest.get("run"),
                    "generate_config_hash": pool.manifest.get("config_hash")},
           "inputs": inputs, "seeds": {"mc": int(mc_seed)},
           **git_state(), "code_sha256": code_sha256(), "env": env_versions()}
    write_json(mp, man)
    return man


def mc_from_pool(run_dir: os.PathLike | str, patients: Any, targets: Mapping[str, Target], task: Task,
                 generator: Any, seeds: Any, R: int, max_tokens: int, out_dir: os.PathLike | str,
                 cache: Optional[GenerationCache] = None, *,
                 provenance: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    t0 = time.time()
    pool = load_pool(run_dir)
    pats = dict(patients) if isinstance(patients, Mapping) else {p.pid: p for p in patients}
    unknown = sorted(set(pats) - {c.pid for c in pool.candidates})
    if unknown:
        raise KeyError(f"patients {unknown[:5]} have no candidates in {pool.root}")
    unlabelled = sorted(set(pats) - set(targets))
    if unlabelled:
        raise KeyError(f"patients {unlabelled[:5]} have no target")
    test = sorted(pid for pid, p in pats.items() if getattr(p, "split", "") == "test")
    if test or _pool_split(pool.root) == "test":
        what = f"{len(test)} test-split patient(s) (e.g. {test[0]})" if test else "a pool built on the test split"
        raise ValueError(f"{what}: PRM training never reads test labels")
    undetermined = sorted(pid for pid in pats if not task.determined(targets[pid]))
    model = pool.model()
    if not task.answer_prefix.strip():
        raise ValueError("the task has no answer_prefix (prompts.answer_prefix): ans() cannot find the answer line")
    check_task((pool.manifest.get("config") or {}).get("task"), task, str(pool.root))
    if generator.served_model != model:
        raise ValueError(f"generator serves {generator.served_model!r}; the pool was generated by {model!r} "
                         f"(eq:mc_score rolls out the SAME G)")
    R, max_tokens, mc_seed = int(R), int(max_tokens), int(seeds["mc"])
    if R < 1:
        raise ValueError("R must be >= 1")
    cfg = {"v": MC_VERSION, "task": task_fields(task), "served_model": model, "sampling": pool.sampling(),
           "R": R, "max_tokens": max_tokens, "mc_seed": mc_seed, "provenance": dict(provenance or {})}
    out = Path(out_dir)
    man = _begin(out, cfg, pool.input_hashes(), pool, mc_seed)

    frame_paths = {fid: path for p in pats.values() for fid, path in zip(p.frame_ids, p.frames)}
    skip = set(undetermined)
    stats: Dict[str, Any] = {"pool_candidates": len(pool.candidates), "undetermined_patients": len(skip),
                             "undetermined_candidates": sum(c.pid in skip for c in pool.candidates),
                             "selected": 0, "skipped": 0,
                             "requests": 0, "cached": 0, "generated": 0, "failed": 0, "errors": [],
                             "written": 0, "rollouts": 0, "unparseable": 0}
    latencies: List[float] = []
    with RolloutWriter(out / ROLLOUTS) as w:
        foreign = sorted(w.done - {c.id for c in pool.candidates})
        if foreign:
            raise StageConflict(f"{out}: {ROLLOUTS} holds ids outside the pool, e.g. {foreign[:3]}")
        chosen = [c for c in pool.candidates if c.pid in pats and c.pid not in skip]
        todo = [c for c in chosen if c.id not in w.done]
        stats.update(selected=len(chosen), skipped=len(chosen) - len(todo), repaired_bytes=w.repaired)

        slot: Dict[str, Tuple[Candidate, int]] = {}
        reqs: Dict[str, List[GenRequest]] = {}
        hashes: Dict[str, str] = {}
        got: Dict[str, Dict[int, GenResult]] = {c.id: {} for c in todo}
        for c in todo:
            missing = [f for f in c.frame_ids if f not in frame_paths]
            if missing:
                raise KeyError(f"{c.id}: frame(s) {missing[:3]} are not slices of patient {c.pid} on disk")
            paths = [frame_paths[f] for f in c.frame_ids]
            _check_same_x(c, pool.prompts[c.pid], paths, model)
            reqs[c.id] = rollout_requests(c, pool.prompts[c.pid], paths, cfg["sampling"], mc_seed, R, max_tokens)
            for i, r in enumerate(reqs[c.id], start=1):
                slot[r.key] = (c, i)
                hashes[r.key] = request_key(model, r)
        stats["requests"] = len(slot)

        def complete(c: Candidate) -> None:
            if len(got[c.id]) != len(c.steps) - 1:
                return
            rec = mc_record(c, [got[c.id][i] for i in range(1, len(c.steps))], targets[c.pid], task, R,
                            max_tokens, mc_seed, [hashes[r.key] for r in reqs[c.id]])
            rec["run"] = pool.key
            w.append(rec)
            stats["written"] += 1
            stats["rollouts"] += R * (len(c.steps) - 1)
            stats["unparseable"] += rec["n_unparseable"]

        def accept(res: GenResult) -> bool:
            err = _result_error(res, R)
            if err:
                stats["failed"] += 1
                if len(stats["errors"]) < 5:
                    stats["errors"].append(f"{res.key}: {err}")
                return False
            c, i = slot[res.key]
            got[c.id][i] = res
            complete(c)
            return True

        misses: List[GenRequest] = []
        for c in todo:
            for r in reqs[c.id]:
                hit = cache.get(hashes[r.key]) if cache is not None else None
                if hit is None:
                    misses.append(r)
                elif accept(dataclasses.replace(hit, key=r.key)):
                    stats["cached"] += 1
            if not reqs[c.id]:
                complete(c)

        by_key = {r.key: r for r in misses}
        last = [time.time()]

        def on_result(res: GenResult) -> None:
            if cache is not None and not _result_error(res, R):
                cache.put(hashes[res.key], res, request_identity(model, by_key[res.key]))
            if accept(res):
                stats["generated"] += 1
                if "latency_s" in res.meta:
                    latencies.append(float(res.meta["latency_s"]))
            if time.time() - last[0] > 60:
                last[0] = time.time()
                print(f"[mc] {stats['generated'] + stats['failed']}/{len(misses)} requests, "
                      f"{len(w.done)}/{len(chosen)} candidates", flush=True)

        if misses:
            generator.generate(misses, on_result=on_result)
        n_done = len(w.done)
        done_all = {c.id for c in pool.candidates if c.pid not in skip} <= w.done

    stats["elapsed_s"] = round(time.time() - t0, 2)
    if latencies:
        lat = sorted(latencies)
        stats["latency_s"] = {"mean": round(sum(lat) / len(lat), 3), "p50": lat[len(lat) // 2], "max": lat[-1]}
    man.update(status="complete" if done_all else "partial", updated_utc=utc_now(),
               done=n_done, last_run=stats)
    write_json(out / MANIFEST, man)
    if stats["failed"]:
        raise RuntimeError(f"{out}: {stats['failed']} rollout request(s) failed, e.g. {stats['errors'][:3]}; "
                           f"{stats['written']} candidate(s) written this run, rerun to resume")
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m medtrace.train.mc",
                                description="p^MC (eq:mc_score) for every step of a pipeline pool.")
    p.add_argument("--run-dir", required=True,
                   help="pipeline run dir <runs_root>/<exp>/<gen_tag>/<variant>/seed<S> with a complete pool")
    p.add_argument("--out", required=True, help="output dir (rollouts.jsonl, manifest.json); one per pool")
    p.add_argument("--R", type=int, default=DEFAULT_R, help="rollouts per prefix")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="new tokens per rollout")
    p.add_argument("--limit-pids", nargs="+", default=None, metavar="PID",
                   help="only these patients (space or comma separated)")
    p.add_argument("--manifest", required=True,
                   help="dataset manifest the pool was built on (frames and the train-split labels of the hits)")
    p.add_argument("--image-root", default=None,
                   help="base of relative frames_dir, relative to the working directory "
                        "(default: MEDTRACE_IMAGE_ROOT, else the manifest's)")
    p.add_argument("--base-urls", default=None,
                   help="comma-separated vLLM base urls (default: the live replicas of the pool's served model)")
    p.add_argument("--serve-root", default="serve",
                   help="scripts/serve_vllm.sh state root (generation.serve.root; relative to the working directory)")
    p.add_argument("--concurrency", type=int, default=64, help="in-flight requests (each samples R rollouts)")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--ready-timeout", type=float, default=60)
    p.add_argument("--cache-root", default=None,
                   help="generation cache root (default: MEDTRACE_CACHE_ROOT or ./cache); namespace gen/<gen_tag>")
    p.add_argument("--no-cache", action="store_true", help="do not read or write the generation cache")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .. import envjson
    from ..data.cohort import load_patients, load_targets, task_type
    from ..data.tasks import load_task
    from ..paths import get_paths
    from ..data.cohort import manifest_sha256
    from ..pipeline.stages import build_generator

    a = build_parser().parse_args(argv)
    run = Path(a.run_dir)
    pool = load_pool(run)
    gm = pool.manifest
    split = "train"
    pool_split = _pool_split(run)
    if pool_split != split:
        raise SystemExit(f"{run}: the pool was distilled on the {pool_split!r} split; "
                         f"MC rollouts are drawn on train-split pools only")
    manifest = a.manifest
    task = load_task(envjson.load_env(), task_type(manifest), need=("prompts.answer_prefix", "prompts.step_names"))
    paths = get_paths({"image_root": a.image_root, "cache_root": a.cache_root})
    pids = sorted({c.pid for c in pool.candidates})
    if a.limit_pids:
        pids = [p for arg in a.limit_pids for p in arg.split(",") if p]
    patients = load_patients(manifest, split, paths.image_root, pids=pids)
    targets = load_targets(manifest, split, task, pids=pids)
    seeds = SeedRegistry(master=int(gm["seeds"]["master"]),
                         overrides={str(k): int(v) for k, v in (gm["seeds"].get("overrides") or {}).items()})
    gen = build_generator({"generation": {
        "backend": "vllm", "served_model": pool.model(), "base_urls": a.base_urls or [],
        "serve": {"root": a.serve_root}, "concurrency": a.concurrency, "timeout": a.timeout,
        "retries": a.retries, "ready_timeout": a.ready_timeout}})
    cache = None if a.no_cache else GenerationCache.for_tag(paths.cache_root, gm["run"]["gen_tag"])
    stats = mc_from_pool(run, patients, targets, task, gen, seeds, a.R, a.max_tokens, a.out, cache,
                         provenance={"split": split, "manifest_sha256": manifest_sha256(manifest)})
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
