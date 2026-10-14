from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import envjson
from ..config import config_hash, get_dotted, require
from ..data import cohort
from ..data.cohort import Patient, load_targets
from ..data.tasks import Task, load_task
from ..paths import Paths, get_paths
from ..registry import GENERATORS
from ..seeds import SeedRegistry
from .rundir import PER_PRM, STAGES, RunDir, StageConflict, read_json, sha256_file, write_json, write_text

OPERATIONAL = {
    "generation": ("base_urls", "concurrency", "cache", "serve", "timeout", "retries", "ready_timeout"),
    "prm": ("batch_size",),
}
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "max_tokens", "repetition_penalty")
SEED_LISTS = ("inference", "pool")
GENERATE_NEEDS = ("prompts.system", "prompts.user", "prompts.frame_line", "prompts.step_names",
                  "prompts.answer_prefix")
GENERATE_TASK_FIELDS = ("system", "user", "frame_line", "step_names", "answer_prefix", "labels", "aliases")
REPORT_TASK_FIELDS = ("kind", "labels", "aliases", "answer_prefix", "step_names")


_ENV_MEMO: Dict[Tuple, Dict[str, Any]] = {}


def _env_key() -> Tuple:
    src = os.environ.get(envjson.ENV_VAR) or ""
    s = src.strip()
    if s and not s.startswith(("{", "[")):
        try:
            return (src, hashlib.sha256(Path(s).expanduser().read_bytes()).hexdigest())
        except OSError:
            pass
    return (src,)


def run_env(cfg: Mapping) -> Dict[str, Any]:
    key = _env_key()
    if key not in _ENV_MEMO:
        env = envjson.load_env(key[0] or None)
        _ENV_MEMO.clear()
        _ENV_MEMO[key] = env
    return copy.deepcopy(_ENV_MEMO[key])


def manifest_path(cfg: Mapping) -> Path:
    m = cfg.get("manifest")
    if not m:
        raise KeyError("config key 'manifest' is required: the dataset manifest JSON (README.md, 'Datasets'), "
                       "e.g. --set manifest=data/binary/manifest.json")
    return Path(str(m))


def run_manifest(cfg: Mapping) -> cohort.Manifest:
    return cohort.load_manifest(manifest_path(cfg))


def manifest_sha256(cfg: Mapping) -> str:
    return cohort.manifest_sha256(run_manifest(cfg))


def run_task(cfg: Mapping, need: Sequence[str] = ()) -> Task:
    return load_task(run_env(cfg), cohort.task_type(run_manifest(cfg)), need=need)


def run_seeds(cfg: Mapping) -> SeedRegistry:
    env = run_env(cfg)
    lst = str(require(cfg, "seed_list"))
    if lst not in SEED_LISTS:
        raise ValueError(f"seed_list must be one of {SEED_LISTS}, got {lst!r}")
    return SeedRegistry(envjson.seed(env, f"seeds.{lst}", int(require(cfg, "seed_index"))),
                        envjson.stage_overrides(env))


def resolve_paths(cfg: Mapping) -> Paths:
    return get_paths(cfg.get("paths") or {})


def load_plugins(cfg: Mapping) -> None:
    for mod in cfg.get("plugins") or []:
        importlib.import_module(mod)


def gen_tag(cfg: Mapping) -> str:
    tag = str(require(cfg, "generation.tag"))
    backend = get_dotted(cfg, "generation.backend", "vllm")
    return tag if backend == "vllm" else f"{tag}-{backend}"


def prm_tag(cfg: Mapping) -> str:
    p = cfg["prm"]
    if p.get("tag"):
        return str(p["tag"])
    w = p.get("weights")
    return Path(str(w)).name if w else f"untrained-{Path(str(require(cfg, 'prm.model'))).name}"


def variant_name(cfg: Mapping) -> str:
    return str(require(cfg, "distill.variant"))


def run_dir(cfg: Mapping) -> RunDir:
    paths = resolve_paths(cfg)
    return RunDir(paths.runs_root, require(cfg, "exp"), gen_tag(cfg), variant_name(cfg), run_seeds(cfg).master,
                  cache_root=paths.cache_root)


def variant_spec(cfg: Mapping, name: Optional[str] = None):
    name = name or variant_name(cfg)
    inline = (cfg.get("variant_defs") or {}).get(name)
    return copy.deepcopy(dict(inline)) if inline is not None else name


def resolve_variant(cfg: Mapping, name: Optional[str] = None) -> Dict:
    from ..distill.presets import resolve_variant as _resolve

    name = name or variant_name(cfg)
    return _resolve(variant_spec(cfg, name), name=name)


def scorer_name(cfg: Mapping) -> str:
    import medtrace.distill
    from ..registry import SCORERS

    return SCORERS.resolve(resolve_variant(cfg)["scorer"]["name"])


def distill_needs(cfg: Mapping) -> Tuple[str, ...]:
    return ("prompts.biomarker_query",) if scorer_name(cfg) == "clip_query" else ("prompts.instruction",)


def generate_needs(task: Task) -> Tuple[str, ...]:
    from ..generation.prompts import required_prompt_fields

    placed = tuple(f if f.startswith("prompts.") else f"prompts.{f}" for f in required_prompt_fields(task))
    return tuple(dict.fromkeys(GENERATE_NEEDS + placed))


def task_fields(cfg: Mapping, stage: str, task: Task) -> Dict[str, Any]:
    pub = task.public()
    if stage == "distill":
        names: Sequence[str] = [f.split(".", 1)[1] for f in distill_needs(cfg)]
    elif stage in ("generate", "score"):
        placed = {f.rsplit(".", 1)[-1] for f in generate_needs(task)}
        names = GENERATE_TASK_FIELDS + tuple(f for f in ("question", "instruction") if f in placed)
    elif stage == "report":
        names = REPORT_TASK_FIELDS
    else:
        raise KeyError(f"unknown stage {stage!r}")
    return {k: pub[k] for k in names}


def resolve_encoder_tag(cfg: Mapping, paths: Optional[Paths] = None) -> str:
    explicit = get_dotted(cfg, "features.encoder_tag")
    if explicit:
        return str(explicit)
    paths = paths or resolve_paths(cfg)
    model = str(require(cfg, "generation.model"))
    model_path = paths.model(model)
    if (Path(model_path) / "preprocessor_config.json").is_file():
        from ..features.extract_vlm import resolve_encoder_tag as _tag

        return _tag(model_path, get_dotted(cfg, "features.min_pixels"),
                    get_dotted(cfg, "features.max_pixels"))[2]
    root = Path(paths.cache_root) / "features"
    hits = []
    for meta in sorted(root.glob("*/meta.json")) if root.is_dir() else []:
        m = read_json(meta)
        if Path(str(m.get("model_path", ""))).name == Path(model).name:
            hits.append(meta.parent.name)
    if len(hits) != 1:
        raise FileNotFoundError(
            f"cannot resolve the feature cache of {model!r}: model dir {model_path} absent and "
            f"{len(hits)} cache(s) under {root} name it {hits}; set features.encoder_tag")
    return hits[0]


def _without(d: Mapping, keys: Sequence[str]) -> Dict:
    return {k: v for k, v in d.items() if k not in keys}


_CKPT_MEMO: Dict[Tuple, str] = {}


def checkpoint_identity(cfg: Mapping) -> Optional[List[Dict[str, str]]]:
    w = cfg["prm"].get("weights")
    if not w:
        return None
    from ..prm.internvl import checkpoint_files

    out = []
    for f in checkpoint_files(resolve_paths(cfg).weights(str(w))):
        st = os.stat(f)
        key = (os.path.abspath(f), st.st_size, st.st_mtime_ns)
        if key not in _CKPT_MEMO:
            _CKPT_MEMO[key] = sha256_file(f)
        out.append({"name": os.path.basename(f), "sha256": _CKPT_MEMO[key]})
    return out


def generation_n(cfg: Mapping) -> int:
    n = require(cfg, "generation.n")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"generation.n must be a positive integer, got {n!r}")
    return n


def stage_config(cfg: Mapping, stage: str) -> Dict:
    seeds = run_seeds(cfg)
    task = run_task(cfg, need=distill_needs(cfg) if stage == "distill"
                    else GENERATE_NEEDS if stage in ("generate", "score") else ())
    run = {"task": task_fields(cfg, stage, task), "seed": seeds.master, "manifest_sha256": manifest_sha256(cfg)}
    if stage == "distill":
        return {**run, "split": cfg["split"], "pids": config_pids(cfg), "calib": cfg["calib"],
                "distill": cfg["distill"], "variant": _without(resolve_variant(cfg), ("note",)),
                "encoder_tag": resolve_encoder_tag(cfg), "seed_overrides": dict(seeds.overrides)}
    if stage == "generate":
        generation_n(cfg)
        return {**run, "modality": cohort.modality(run_manifest(cfg)),
                "generation": _without(cfg["generation"], OPERATIONAL["generation"]),
                "decode_seed": seeds["decode"]}
    if stage == "score":
        return {**run, "prm": _without(cfg["prm"], OPERATIONAL["prm"]), "prm_tag": prm_tag(cfg),
                "checkpoint": checkpoint_identity(cfg)}
    if stage == "report":
        return {**run, "report": cfg["report"], "prm_tag": prm_tag(cfg)}
    raise KeyError(f"unknown stage {stage!r}")


def stage_inputs(cfg: Mapping, stage: str, rd: Optional[RunDir] = None) -> List[Path]:
    rd = rd or run_dir(cfg)
    if stage == "distill":
        paths = resolve_paths(cfg)
        feats = Path(paths.cache_root) / "features"
        out = [feats / resolve_encoder_tag(cfg, paths) / "meta.json"]
        if scorer_name(cfg) == "clip_query":
            from ..features.extract_clip import resolve_cache
            sc = resolve_variant(cfg)["scorer"]
            out.append(Path(resolve_cache(sc.get("encoder"), feats)) / "meta.json")
        return out
    if stage == "generate":
        return [rd.signals]
    if stage == "score":
        return [rd.candidates, rd.prompts]
    if stage == "report":
        return [rd.scores(prm_tag(cfg))]
    raise KeyError(f"unknown stage {stage!r}")


def staleness(cfg: Mapping, stage: str, rd: Optional[RunDir] = None) -> str:
    rd = rd or run_dir(cfg)
    for st in STAGES[:STAGES.index(stage) + 1]:
        try:
            inputs = stage_inputs(cfg, st, rd)
        except FileNotFoundError as e:
            return f"{st}: {e}"
        why = rd.stale_reason(st, stage_config(cfg, st), inputs, prm_tag(cfg) if st in PER_PRM else None)
        if why:
            return why
    return ""


def require_current(cfg: Mapping, stage: str, rd: Optional[RunDir] = None) -> None:
    rd = rd or run_dir(cfg)
    why = staleness(cfg, stage, rd)
    if why:
        raise RuntimeError(f"{rd}: {why}; run (or --force) that stage first")


def _done(rd: RunDir, stage: str, sub: Dict, inputs: Sequence[Path], tag: Optional[str] = None,
          force: bool = False) -> bool:
    return not force and rd.stale_reason(stage, sub, inputs, tag) == ""


def config_pids(cfg: Mapping) -> Optional[List[str]]:
    want = cfg.get("pids")
    if want is None or (isinstance(want, (str, list, tuple)) and len(want) == 0):
        return None
    if isinstance(want, str):
        out = [w.strip() for w in want.split(",") if w.strip()]
    elif isinstance(want, int) and not isinstance(want, bool):
        out = [str(want)]
    elif isinstance(want, (list, tuple)) and all(isinstance(w, (str, int)) and not isinstance(w, bool)
                                                 for w in want):
        out = [str(w) for w in want]
    else:
        raise TypeError(f"pids must be a list of patient ids (e.g. --set pids=[a,b]), got {want!r}")
    return out or None


def split_pids(cfg: Mapping, split: Optional[str] = None) -> List[str]:
    split = split or cfg["split"]
    man = run_manifest(cfg)
    if split not in cohort.splits(man):
        raise KeyError(f"split {split!r} not in {manifest_path(cfg)}; have {cohort.splits(man)}")
    pids = sorted((str(r["pid"]) for r in man["patients"] if r["split"] == split), key=cohort.natural_key)
    want = config_pids(cfg)
    if want:
        missing = sorted(set(want) - set(pids))
        if missing:
            raise KeyError(f"pids {missing} are not in split {split!r}")
        keep = set(want)
        pids = [p for p in pids if p in keep]
    return pids


def load_patients(cfg: Mapping, pids: Sequence[str], split: Optional[str] = None) -> Dict[str, Patient]:
    ps = cohort.load_patients(run_manifest(cfg), split or cfg["split"], image_root=resolve_paths(cfg).image_root,
                              pids=list(pids), strict=bool(cfg.get("strict_cohort", True)))
    return {p.pid: p for p in ps}


def stage_distill(cfg: Mapping, force: bool = False) -> Path:
    from ..distill import Distiller, build_calibration, context_from_store, save_signals
    from ..features.store import FeatureStore
    from ..generation.prompts import check_task_texts

    rd, seeds, paths = run_dir(cfg), run_seeds(cfg), resolve_paths(cfg)
    sub, inputs = stage_config(cfg, "distill"), stage_inputs(cfg, "distill", rd)
    if _done(rd, "distill", sub, inputs, force=force):
        return rd.signals
    needs = distill_needs(cfg)
    task = run_task(cfg, need=needs)
    if "prompts.instruction" in needs:
        check_task_texts(task, names=("instruction",))
    cohort.edges(run_manifest(cfg), task)
    store = FeatureStore.open(paths.cache_root, resolve_encoder_tag(cfg, paths))
    d, c = cfg["distill"], cfg["calib"]
    dist = Distiller(variant_spec(cfg), context_from_store(store, task, seeds), m=int(d["m"]),
                     tau_r=float(d["tau_r"]), alpha_min=float(d["alpha_min"]), alpha_max=float(d["alpha_max"]),
                     name=variant_name(cfg))
    calib = None
    if dist.needs_calibration:
        pool = split_pids({**cfg, "pids": None}, "train")
        max_frames = c.get("max_frames")
        calib = build_calibration(store, pool, seeds=seeds, n_patients=int(c["n_patients"]),
                                  max_frames=None if max_frames is None else int(max_frames))
    dist.fit(calib)
    signals = dist.run(store.load_many(split_pids(cfg)))

    if rd.begin("distill", sub, seeds.as_dict(), inputs, force=force) == "done":
        return rd.signals
    meta = {**dist.describe(), "encoder_tag": store.root.name, "config_hash": config_hash(sub)}
    save_signals(rd.signals, meta, signals)
    degenerate = sum(bool(s.meta.get("degenerate")) for s in signals)
    rd.finish("distill", [rd.signals], extra={"n_patients": len(signals), "n_degenerate": degenerate,
                                              "calib_pids": (meta.get("calib") or {}).get("pids")})
    return rd.signals


def build_generator(cfg: Mapping, task: Optional[Task] = None):
    import medtrace.generation

    g = cfg["generation"]
    backend = str(g.get("backend", "vllm"))
    if backend != "vllm":
        kwargs = dict(g.get("backend_args") or {})
        if "task" in inspect.signature(GENERATORS.get(backend)).parameters:
            kwargs["task"] = task if task is not None else run_task(cfg)
        return GENERATORS.build(backend, served_model=g["served_model"], **kwargs)
    from ..generation.vllm_client import VLLMGenerator, serve_urls

    urls = g.get("base_urls") or []
    if isinstance(urls, str):
        urls = [u for u in urls.split(",") if u.strip()]
    root = str((g.get("serve") or {}).get("root") or "serve")
    if not urls:
        urls = serve_urls(g["served_model"], root)
    if not urls:
        raise RuntimeError(f"no vLLM replicas: generation.base_urls is empty and none are alive under "
                           f"{root}/{g['served_model']} (start them: bash scripts/run_serve.sh <config> start)")
    gen = VLLMGenerator(urls, g["served_model"], concurrency=int(g.get("concurrency", 64)),
                        timeout=float(g.get("timeout", 900)), retries=int(g.get("retries", 3)))
    gen.wait_ready(timeout=float(g.get("ready_timeout", 60)))
    return gen


def sampling_of(cfg: Mapping) -> Dict[str, Any]:
    g = cfg["generation"]
    return {k: g[k] for k in SAMPLING_KEYS}


def stage_generate(cfg: Mapping, force: bool = False, generator=None) -> Path:
    from ..generation.cache import GenerationCache
    from ..generation.candidates import CandidateWriter, make_jobs, run_jobs
    from ..generation.prompts import check_task_texts, prompt_record

    rd, seeds, paths = run_dir(cfg), run_seeds(cfg), resolve_paths(cfg)
    require_current(cfg, "distill", rd)
    sub, inputs = stage_config(cfg, "generate"), stage_inputs(cfg, "generate", rd)
    if _done(rd, "generate", sub, inputs, force=force):
        return rd.candidates
    n = generation_n(cfg)
    task = run_task(cfg, need=GENERATE_NEEDS)
    envjson.require(run_env(cfg), *generate_needs(task))
    check_task_texts(task)
    man = run_manifest(cfg)
    cohort.edges(man, task)
    modality = cohort.modality(man)
    sigs = read_json(rd.signals)["signals"]
    patients = load_patients(cfg, [s["pid"] for s in sigs])
    prompts: Dict[str, Dict] = {}
    jobs = []
    for s in sigs:
        p = patients[s["pid"]]
        if list(s["frame_ids"]) != p.frame_ids:
            raise RuntimeError(f"{p.pid}: frame ids of the feature cache ({len(s['frame_ids'])}) differ from "
                               f"the slices on disk ({p.T}); re-extract features for this image tree")
        sel = [int(i) for i in s["selected"]]
        rec = prompt_record(p.public(), task, modality, sel)
        prompts[p.pid] = rec
        jobs += make_jobs(p.pid, rec["system"], rec["question"], [p.frames[i] for i in sel],
                          rec["frame_ids"], n, seeds["decode"], sampling_of(cfg))

    state = rd.begin("generate", sub, seeds.as_dict(), inputs, force=force)
    if state == "done":
        return rd.candidates
    if state == "resume" and rd.prompts.is_file() and read_json(rd.prompts) != prompts:
        raise StageConflict(f"{rd}: prompts differ from the ones the partial pool was generated with")
    write_json(rd.prompts, prompts)

    gen = generator if generator is not None else build_generator(cfg, task)
    cache = GenerationCache.for_tag(paths.cache_root, gen_tag(cfg)) if cfg["generation"].get("cache", True) else None
    with CandidateWriter(rd.candidates) as w:
        foreign = sorted(w.done - {j.id for j in jobs})
        if foreign:
            raise StageConflict(f"{rd}: candidates.jsonl holds ids outside this job set, e.g. {foreign[:3]}")
        stats = run_jobs(gen, jobs, task, w, cache=cache)
        n_done = len(w.done)
    if stats["failed"] or n_done != len(jobs):
        raise RuntimeError(f"{rd}: {stats['failed']} generation(s) failed ({n_done}/{len(jobs)} on disk), "
                           f"e.g. {stats.get('errors', [])[:3]}; rerun `generate` to resume")
    rd.finish("generate", [rd.prompts, rd.candidates], extra={"generation": stats, "generator": gen.describe()})
    return rd.candidates


def default_device(gpu: Optional[int]) -> str:
    return f"cuda:{0 if gpu is None else int(gpu)}"


def check_device(device: str) -> str:
    if not device.startswith("cuda"):
        return device
    idx = int(device.split(":", 1)[1]) if ":" in device else 0
    if idx < 0:
        raise ValueError(f"{device}: negative device index")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        ids = [x.strip() for x in vis.split(",") if x.strip()]
        if idx >= len(ids):
            raise ValueError(f"{device}: CUDA_VISIBLE_DEVICES={vis!r} exposes {len(ids)} device(s)")
    return device


def prm_build_args(cfg: Mapping) -> Dict[str, Any]:
    p = cfg["prm"]
    paths = resolve_paths(cfg)
    return dict(model_path=paths.model(str(require(cfg, "prm.model"))),
                weights_path=paths.weights(str(p["weights"])) if p.get("weights") else None,
                allow_untrained=bool(p.get("allow_untrained", False)), max_tiles=p.get("max_tiles"),
                max_images=p.get("max_images"), batch_size=int(p.get("batch_size", 1)))


def build_prm_scorer(cfg: Mapping, device: str):
    from ..prm.internvl import PRMScorer

    a = prm_build_args(cfg)
    return PRMScorer(a.pop("model_path"), a.pop("weights_path"), device=check_device(device), **a)


def _score_pool(scorer, cands: Sequence, prompts: Mapping[str, Dict], patients: Mapping[str, Patient],
                label: str = "") -> Tuple[List[Dict], Dict]:
    import time

    from ..prm.aggregate import AGGREGATION, score_record
    from ..prm.internvl import ScoreRequest

    frame_paths = {fid: path for p in patients.values() for fid, path in zip(p.frame_ids, p.frames)}
    reqs = [ScoreRequest(key=c.id, system=prompts[c.pid]["system"], question=prompts[c.pid]["question"],
                         steps=list(c.steps), image_paths=[frame_paths[f] for f in c.frame_ids], group=c.pid)
            for c in cands]
    t0, last = time.time(), [0.0]

    def progress(done: int, total: int) -> None:
        if time.time() - last[0] > 30 or done == total:
            last[0] = time.time()
            print(f"[score] {label} {done}/{total} ({done / max(time.time() - t0, 1e-9):.2f}/s)", flush=True)

    step_scores = scorer.score_candidates(reqs, progress=progress)
    if len(step_scores) != len(cands):
        raise RuntimeError(f"scorer returned {len(step_scores)} results for {len(cands)} candidates")
    out = []
    for c, ps in zip(cands, step_scores):
        if len(ps) != len(c.steps):
            raise RuntimeError(f"{c.id}: {len(ps)} step scores for {len(c.steps)} steps")
        out.append(score_record(c.id, c.pid, c.cot_idx, ps, c.pred, c.answer, truncated=c.meta.get("truncated")))
    meta = dict(scorer.describe())
    meta.update(aggregation=AGGREGATION, n_scored=len(out), elapsed_s=round(time.time() - t0, 2))
    return out, meta


def stage_score(cfg: Mapping, gpu: Optional[int] = None, scorer=None, force: bool = False) -> Path:
    from ..generation.candidates import load_candidates

    rd, seeds, tag = run_dir(cfg), run_seeds(cfg), prm_tag(cfg)
    require_current(cfg, "generate", rd)
    sub, inputs = stage_config(cfg, "score"), stage_inputs(cfg, "score", rd)
    if _done(rd, "score", sub, inputs, tag, force=force):
        return rd.scores(tag)
    prompts = read_json(rd.prompts)
    cands = load_candidates(rd.candidates)
    patients = load_patients(cfg, list(prompts))
    scorer = scorer if scorer is not None else build_prm_scorer(cfg, default_device(gpu))

    if rd.begin("score", sub, seeds.as_dict(), inputs, prm_tag=tag, force=force) == "done":
        return rd.scores(tag)
    scored, meta = _score_pool(scorer, cands, prompts, patients, label=rd.key)
    meta = {**meta, "prm_tag": tag, "config_hash": config_hash(sub)}
    write_json(rd.scores(tag), {"meta": meta, "scored": scored})
    rd.finish("score", [rd.scores(tag)], prm_tag=tag, extra={"n_scored": len(scored)})
    return rd.scores(tag)


def stage_report(cfg: Mapping, force: bool = False) -> Path:
    from ..eval.metrics import evaluate
    from ..eval.report import render_summary

    rd, seeds, tag = run_dir(cfg), run_seeds(cfg), prm_tag(cfg)
    require_current(cfg, "score", rd)
    sub, inputs = stage_config(cfg, "report"), stage_inputs(cfg, "report", rd)
    if _done(rd, "report", sub, inputs, tag, force=force):
        return rd.summary(tag)
    blob = read_json(rd.scores(tag))
    scored = blob["scored"]
    pids = split_pids(cfg)
    pool = {d["pid"] for d in scored}
    if pool != set(pids):
        raise RuntimeError(f"{rd}: the scored pool covers {len(pool)} patient(s), the configured split/pids "
                           f"{len(pids)}; differing: {sorted(pool ^ set(pids))[:5]}")
    n_list = [int(n) for n in require(cfg, "report.n_list")]
    n_gen = generation_n(cfg)
    if not any(n <= n_gen for n in n_list):
        raise ValueError(f"report.n_list {n_list}: no N is <= generation.n = {n_gen}, nothing to report")
    task = run_task(cfg)
    targets = load_targets(run_manifest(cfg), cfg["split"], task, pids=pids)
    metrics = evaluate(scored, targets, task, n_list=n_list)
    if not metrics["meta"]["evaluated_n"]:
        raise RuntimeError(f"{rd}: no N of report.n_list {n_list} could be evaluated "
                           f"({metrics['meta']['skipped_n']})")
    metrics["meta"]["run"] = {"exp": rd.exp, "gen_tag": rd.gen_tag, "variant": rd.variant, "seed": rd.seed,
                              "prm_tag": tag, "prm": blob.get("meta", {})}
    head = (f"MedTRACE report  exp={rd.exp}  task={task.kind}  G={rd.gen_tag}  variant={rd.variant}  "
            f"seed={rd.seed}  PRM={tag}\n{'=' * 78}")
    summary = head + "\n" + render_summary(metrics, task)

    if rd.begin("report", sub, seeds.as_dict(), inputs, prm_tag=tag, force=force) == "done":
        return rd.summary(tag)
    write_json(rd.metrics(tag), metrics)
    write_text(rd.summary(tag), summary)
    rd.finish("report", [rd.metrics(tag), rd.summary(tag)], prm_tag=tag)
    return rd.summary(tag)


def run_all(cfg: Mapping, gpu: Optional[int] = None, force: bool = False) -> Path:
    stage_distill(cfg, force=force)
    stage_generate(cfg, force=force)
    stage_score(cfg, gpu=gpu, force=force)
    return stage_report(cfg, force=force)
