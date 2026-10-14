from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..config import config_hash
from ..data.tasks import Target, Task
from ..paths import get_paths
from ..pipeline.rundir import code_sha256, env_versions, git_state, sha256_file, utc_now, write_json
from ..prm.internvl import PRMScorer
from ..seeds import SeedRegistry, sub_seed
from .dataset import (PRMEncoder, PRMRowsDataset, RowSampler, TrainRow, check_rows, collate, load_rows,
                      require_train, row_targets)

BEST_BY = ("within_mean", "bon_bal_acc", "bon_c_index")
BEST_BY_KIND = {"bon_bal_acc": "binary", "bon_c_index": "survival"}
MARKER = "medtrace_train.json"
LOG_MARKER = "train_config.json"
OPERATIONAL = ("device", "weights_root", "log_dir", "force")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class TrainConfig:
    rows: str
    out_name: str
    manifest: str
    model: str = "InternVL3-1B"
    lr: float = 5e-7
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.0
    batch_size: int = 16
    micro_batch: int = 1
    epochs: float = 2.0
    max_steps: Optional[int] = None
    grad_clip: float = 1.0
    eval_every: int = 25
    best_by: str = "within_mean"
    grad_ckpt: bool = False
    max_tiles: Optional[int] = None
    max_images: Optional[int] = None
    seed: Optional[int] = None
    seed_overrides: Optional[Dict[str, int]] = None
    device: str = "cuda:0"
    weights_root: Optional[str] = None
    log_dir: Optional[str] = None
    image_root: Optional[str] = None
    fit_limit: Optional[int] = None
    meta_limit: Optional[int] = None
    force: bool = False

    def validate(self) -> "TrainConfig":
        check_out_name(self.out_name)
        if not self.manifest:
            raise ValueError("manifest required (frame ids resolve through the dataset manifest)")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError(f"seed must be an int or None (seeds.train of the environment JSON), got {self.seed!r}")
        if self.best_by not in BEST_BY:
            raise ValueError(f"best_by {self.best_by!r} not in {BEST_BY}")
        if self.batch_size < 1 or self.micro_batch < 1 or self.batch_size % self.micro_batch:
            raise ValueError(f"batch_size ({self.batch_size}) must be a positive multiple of micro_batch "
                             f"({self.micro_batch})")
        if not self.lr > 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("lr > 0, weight_decay >= 0 and grad_clip > 0 required")
        if not 0 <= self.warmup_ratio < 1 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("warmup_ratio in [0, 1) and min_lr_ratio in [0, 1] required")
        if self.max_steps is None and not self.epochs > 0:
            raise ValueError("epochs > 0 or max_steps required")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.eval_every < 0:
            raise ValueError("eval_every must be >= 0 (0 = only after the last update)")
        for key in ("fit_limit", "meta_limit"):
            if getattr(self, key) is not None and getattr(self, key) < 1:
                raise ValueError(f"{key} must be >= 1 or None (all rows)")
        return self

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    def train_hash(self) -> str:
        return config_hash({k: v for k, v in asdict(self).items() if k not in OPERATIONAL})


def check_out_name(name: str) -> str:
    if not _NAME_RE.match(str(name)) or re.search(r"_step\d+$", str(name)):
        raise ValueError(f"out name {name!r} must match <letter or digit><letters/digits/._-> and not end "
                         f"in _step<k>")
    return str(name)


def lr_at(k: int, total: int, base: float, warmup: int, min_ratio: float = 0.0) -> float:
    if not 1 <= k <= total:
        raise ValueError(f"update {k} outside 1..{total}")
    if k <= warmup:
        return base * k / warmup
    prog = (k - 1 - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog)))


def sequence_means(per_slot, seq, n_seq: int):
    import torch

    sums = torch.zeros(n_seq, dtype=per_slot.dtype, device=per_slot.device).index_add_(0, seq, per_slot)
    cnt = torch.zeros(n_seq, dtype=per_slot.dtype, device=per_slot.device).index_add_(
        0, seq, torch.ones_like(per_slot))
    if bool((cnt == 0).any()):
        raise RuntimeError("a trajectory without a scoring slot reached the loss")
    return sums / cnt


def head_rows(scorer):
    torch = scorer._torch
    head = scorer.model.language_model.get_output_embeddings()
    idx = torch.tensor([scorer.slots.pos_id, scorer.slots.neg_id], device=head.weight.device)
    b = getattr(head, "bias", None)
    return head.weight.index_select(0, idx).float(), None if b is None else b.index_select(0, idx).float()


def slot_logit_diff(scorer, batch):
    dev = scorer.device
    h = scorer.hidden(batch["input_ids"].to(dev), batch["attention_mask"].to(dev),
                       batch["pixel_values"].to(scorer.dtype).to(dev), batch["image_flags"].to(dev))
    hs = h[batch["b_idx"].to(dev), batch["s_idx"].to(dev)].float()
    w, b = head_rows(scorer)
    z = (hs.unsqueeze(1) * w.unsqueeze(0)).sum(-1)
    if b is not None:
        z = z + b
    return z[:, 0] - z[:, 1]


class MasterAdamW:
    def __init__(self, params: Iterable, lr: float, weight_decay: float = 0.0,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        import torch

        self.work = [p for p in params if p.requires_grad]
        if not self.work:
            raise ValueError("no trainable parameters")
        self.master = [p if p.dtype == torch.float32 else p.detach().float().clone() for p in self.work]
        self.opt = torch.optim.AdamW(self.master, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._torch = torch

    def accumulate(self) -> None:
        for p, m in zip(self.work, self.master):
            if m is p or p.grad is None:
                continue
            if m.grad is None:
                m.grad = p.grad.float()
            else:
                m.grad.add_(p.grad)
            p.grad = None

    def clip(self, max_norm: float) -> float:
        norm = float(self._torch.nn.utils.clip_grad_norm_([m for m in self.master if m.grad is not None],
                                                          max_norm))
        if not math.isfinite(norm):
            raise FloatingPointError(f"non-finite gradient norm ({norm}); stopping instead of stepping")
        return norm

    def step(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = float(lr)
        self.opt.step()
        with self._torch.no_grad():
            for p, m in zip(self.work, self.master):
                if m is not p:
                    p.copy_(m)
        self.opt.zero_grad(set_to_none=True)


def meta_metrics(rows: Sequence[TrainRow], step_scores: Sequence[Sequence[float]], targets: Mapping[str, Target],
                 task: Task) -> Dict[str, Any]:
    from ..eval.metrics import binary_metrics, c_index, group_candidates, trajectory_ranking
    from ..prm.aggregate import best_of_n, score_record

    if len(rows) != len(step_scores):
        raise ValueError(f"{len(step_scores)} score lists for {len(rows)} rows")
    rank: Dict[str, int] = {}
    scored = []
    for r, ps in zip(rows, step_scores):
        if len(ps) != r.n:
            raise RuntimeError(f"{r.id}: {len(ps)} step scores for {r.n} steps")
        k = rank.get(r.pid, 0)
        rank[r.pid] = k + 1
        scored.append(score_record(r.id, r.pid, k, ps, r.pred, None))
    groups, _ = group_candidates(scored, targets, task)
    rk = trajectory_ranking(groups, task=task)
    pids = sorted(groups)
    picks = {p: best_of_n(groups[p]).pred for p in pids}
    out: Dict[str, Any] = {
        "n_rows": len(rows), "n_patients": len(pids), "n_ranked": rk.get("n"),
        "within_mean": rk.get("within_mean"), "auc_within": rk.get("auc_within"),
    }
    if task.kind == "binary":
        bon = binary_metrics([int(targets[p]) for p in pids], [picks[p] for p in pids])
        out.update(bon_acc=bon["acc"], bon_bal_acc=bon["bal_acc"], bon_c_index=None)
    else:
        ok = [p for p in pids if picks[p] is not None]
        hits = [h for h in (task.is_correct(picks[p], targets[p]) for p in pids) if h is not None]
        out.update(bon_acc=sum(hits) / len(hits) if hits else None,
                   bon_bal_acc=None, bon_unparsed=len(pids) - len(ok),
                   bon_c_index=c_index([targets[p].time for p in ok], [targets[p].event for p in ok],
                                       [task.risk(picks[p]) for p in ok]) if ok else None)
    return out


def evaluate_meta(scorer, meta: PRMRowsDataset, task: Task) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model = scorer.model
    was_training = model.training
    model.eval()
    scorer.refresh_head()
    try:
        ps = scorer.score_candidates([meta.request(i) for i in range(len(meta))])
    finally:
        model.train(was_training)
    per_row = [{"id": r.id, "pid": r.pid, "step_scores": [float(x) for x in p]} for r, p in zip(meta.rows, ps)]
    return meta_metrics(meta.rows, ps, meta.targets, task), per_row


def select_best(evals: Sequence[Mapping[str, Any]], by: str) -> Optional[Mapping[str, Any]]:
    if by not in BEST_BY:
        raise ValueError(f"best_by {by!r} not in {BEST_BY}")
    ok = [e for e in evals if e.get("snapshot") and e["metrics"].get(by) is not None]
    if not ok:
        return None
    return max(ok, key=lambda e: (float(e["metrics"][by]), -int(e["step"])))


def snapshot_dir(weights_root: Union[str, os.PathLike], name: str, step: Optional[int] = None) -> Path:
    check_out_name(name)
    return Path(weights_root) / (name if step is None else f"{name}_step{int(step)}")


def _supersede(d: Path) -> Path:
    dst = d.with_name(f"{d.name}.superseded-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}")
    os.replace(d, dst)
    return dst


def claim_outputs(weights_root: Union[str, os.PathLike], name: str, log_dir: Union[str, os.PathLike],
                  force: bool = False) -> List[str]:
    root, log_dir = Path(weights_root), Path(log_dir)
    step_re = re.compile(rf"^{re.escape(check_out_name(name))}_step\d+$")
    found = [root / name] + sorted(p for p in root.glob(f"{name}_step*") if step_re.match(p.name))
    ours: List[Path] = []
    for d in found:
        if d.exists():
            if not (d / MARKER).is_file():
                raise FileExistsError(f"{d} exists and was not written by medtrace.train.trainer; refusing to touch it")
            ours.append(d)
    if log_dir.is_dir() and any(log_dir.iterdir()):
        if not (log_dir / LOG_MARKER).is_file():
            raise FileExistsError(f"log dir {log_dir} holds files this trainer did not write")
        ours.append(log_dir)
    if ours and not force:
        raise FileExistsError(f"outputs of {name!r} exist ({', '.join(d.name for d in ours)}); force/--force "
                              f"renames them aside (never deletes)")
    return [_supersede(d).name for d in ours]


def save_snapshot(model, out: Path, marker: Mapping[str, Any],
                  init_state: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    state = {k: v.detach().to("cpu", copy=True).contiguous() for k, v in model.state_dict().items()}
    info: Dict[str, Any] = {"n_tensors": len(state), "dtypes": sorted({str(v.dtype) for v in state.values()})}
    if init_state is not None:
        changed, n_el, n_changed = 0, 0, 0
        for k, v in state.items():
            diff = int((v != init_state[k]).sum())
            changed += int(diff > 0)
            n_changed += diff
            n_el += v.numel()
        info.update(tensors_changed=changed, elements_changed_frac=n_changed / max(1, n_el))
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / ".model.safetensors.tmp"
    save_file(state, str(tmp), metadata={"format": "pt"})
    os.replace(tmp, out / "model.safetensors")
    del state
    info.update(dir=out.name, bytes=(out / "model.safetensors").stat().st_size,
                sha256=sha256_file(out / "model.safetensors"), torch=torch.__version__)
    write_json(out / MARKER, {**marker, "snapshot": info})
    return info


def publish_best(weights_root: Path, name: str, best: Mapping[str, Any], marker: Mapping[str, Any]) -> Path:
    src = snapshot_dir(weights_root, name, best["step"]) / "model.safetensors"
    dst_dir = snapshot_dir(weights_root, name)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "model.safetensors"
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    write_json(dst_dir / MARKER, {**marker, "selected_step": best["step"], "selected_from": src.parent.name})
    return dst_dir


def _peak_gib(device: str) -> Optional[float]:
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3)
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3)
    except ImportError:
        return None


def _split_stats(ds: PRMRowsDataset) -> Dict[str, Any]:
    strata = sorted(set(ds.strata.values()))
    return {"rows": len(ds), "patients": len(ds.pids),
            "patients_by_stratum": {str(s): sum(1 for v in ds.strata.values() if v == s) for s in strata},
            "correct_rows": sum(r.correct for r in ds.rows),
            "unparseable_rows": sum(1 for r in ds.rows if r.pred is None),
            "marker_pos_share": sum(sum(r.markers) for r in ds.rows) / sum(r.n for r in ds.rows)}


def resolve_run(cfg: TrainConfig, task: Optional[Task] = None) -> Tuple[TrainConfig, Task]:
    from .. import envjson
    from ..data.cohort import task_type
    from ..data.tasks import load_task

    env = envjson.load_env() if cfg.seed is None or task is None else None
    if cfg.seed is None:
        cfg = replace(cfg, seed=envjson.seed(env, "seeds.train"),
                      seed_overrides=(envjson.stage_overrides(env) if cfg.seed_overrides is None
                                      else dict(cfg.seed_overrides)))
    elif cfg.seed_overrides is None:
        cfg = replace(cfg, seed_overrides={})
    kind = task_type(cfg.manifest)
    if task is None:
        task = load_task(env, kind)
    elif task.kind != kind:
        raise ValueError(f"task kind {task.kind!r} != the manifest's task_type {kind!r}")
    need = BEST_BY_KIND.get(cfg.best_by)
    if need is not None and task.kind != need:
        raise ValueError(f"best_by {cfg.best_by!r} is defined only for a {need} task, this one is {task.kind}; "
                         f"use within_mean or bon_{'bal_acc' if task.kind == 'binary' else 'c_index'}")
    return cfg, task


def train(cfg: Union[TrainConfig, Mapping[str, Any]], task: Optional[Task] = None) -> Dict[str, Any]:
    cfg = (cfg if isinstance(cfg, TrainConfig) else TrainConfig(**dict(cfg))).validate()
    cfg, task = resolve_run(cfg, task)
    device = cfg.device
    paths = get_paths({"image_root": cfg.image_root, "weights_root": cfg.weights_root})
    seeds = SeedRegistry(master=int(cfg.seed), overrides=dict(cfg.seed_overrides or {}))
    rows_path = Path(cfg.rows)
    rows = load_rows(rows_path)
    fit_rows = [r for r in rows if r.split == "fit"][:cfg.fit_limit]
    meta_rows = [r for r in rows if r.split == "meta"][:cfg.meta_limit]
    if not fit_rows or not meta_rows:
        raise ValueError(f"{rows_path.name}: {len(fit_rows)} fit and {len(meta_rows)} meta rows; both are needed "
                         f"(selection runs on meta, never on test)")
    shared = sorted({r.pid for r in fit_rows} & {r.pid for r in meta_rows})
    if shared:
        raise ValueError(f"patients in both fit and meta, e.g. {shared[:5]}: meta must be held out")
    require_train(sorted({r.pid for r in fit_rows + meta_rows}), cfg.manifest)
    check_rows(fit_rows + meta_rows, task)
    row_targets(fit_rows + meta_rows)
    wroot = Path(paths.weights_root)
    log_dir = Path(cfg.log_dir) if cfg.log_dir else rows_path.parent / "train" / cfg.out_name
    superseded = claim_outputs(wroot, cfg.out_name, log_dir, cfg.force)

    import torch

    t_start = time.time()
    torch.manual_seed(sub_seed(seeds["train"], "torch"))
    scorer = PRMScorer(paths.model(cfg.model), None, device=device, allow_untrained=True,
                       max_tiles=cfg.max_tiles, max_images=cfg.max_images, batch_size=1)
    model = scorer.model
    encoder = PRMEncoder.for_scorer(scorer)
    common = dict(encoder=encoder, layout=scorer.layout, image_root=paths.image_root, manifest=cfg.manifest)
    fit = PRMRowsDataset(fit_rows, "fit", **common)
    meta = PRMRowsDataset(meta_rows, "meta", **common)
    sampler = RowSampler(fit.rows, fit.strata, seeds["train"])
    total = int(cfg.max_steps or max(1, math.ceil(cfg.epochs * len(fit) / cfg.batch_size)))
    warmup = int(round(cfg.warmup_ratio * total))
    accum = cfg.batch_size // cfg.micro_batch

    for p in model.parameters():
        p.requires_grad_(True)
    if cfg.grad_ckpt:
        model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = MasterAdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    init_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    rows_sha = sha256_file(rows_path)
    marker = {"trainer": "medtrace.train.trainer", "name": cfg.out_name, "config_hash": cfg.train_hash(),
              "config": cfg.to_json(), "rows_sha256": rows_sha, "seeds": seeds.as_dict(), "task": task.public()}
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / LOG_MARKER, {**marker, "created_utc": utc_now(), "superseded": superseded})
    summary: Dict[str, Any] = {
        "status": "running", "name": cfg.out_name, "weights_root": str(wroot), "log_dir": str(log_dir),
        "config": cfg.to_json(), "config_hash": cfg.train_hash(), "seeds": seeds.as_dict(),
        "rows": {"path": str(rows_path.resolve()), "sha256": rows_sha, "fit": _split_stats(fit),
                 "meta": _split_stats(meta)},
        "model": scorer.describe(), "total_steps": total, "warmup_steps": warmup,
        "updates_per_epoch": len(fit) / cfg.batch_size, "evals": [], "best": None,
        **git_state(), "code_sha256": code_sha256(), "env": env_versions(),
    }

    def log(obj: Mapping[str, Any]) -> None:
        with open(log_dir / "train_log.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def run_eval(step: int, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        metrics, per_row = evaluate_meta(scorer, meta, task)
        write_json(log_dir / "evals" / f"step{step}.json", {"step": step, "scored": per_row})
        ev = {"event": "eval", "step": step, "snapshot": snapshot, "metrics": metrics,
              "sec": round(time.time() - t0, 2)}
        log(ev)
        summary["evals"].append(ev)
        print(f"[train] eval step {step}: within_mean={metrics['within_mean']} bon_acc={metrics['bon_acc']} "
              f"bon_bal_acc={metrics['bon_bal_acc']} bon_c_index={metrics['bon_c_index']}", flush=True)
        return ev

    print(f"[train] {cfg.out_name}: fit {len(fit)} rows / meta {len(meta)} rows, {total} updates of "
          f"{cfg.batch_size} ({accum} x {cfg.micro_batch}), warmup {warmup}, lr {cfg.lr}, base {cfg.model}",
          flush=True)

    import torch.nn.functional as F

    model.train()
    stream = sampler.stream()
    for step in range(1, total + 1):
        t0 = time.time()
        loss_sum, n_slots, n_tok, n_correct, epoch = 0.0, 0, 0, 0, 0
        drawn: Dict[str, int] = {}
        for _ in range(accum):
            draws = [next(stream) for _ in range(cfg.micro_batch)]
            epoch = draws[-1][0]
            for _, i in draws:
                r = fit.rows[i]
                drawn[str(r.y)] = drawn.get(str(r.y), 0) + 1
                n_correct += r.correct
            batch = collate([fit[i] for _, i in draws], encoder.pad_id)
            d = slot_logit_diff(scorer, batch)
            per_slot = F.binary_cross_entropy_with_logits(d, batch["markers"].to(d.device), reduction="none")
            per_seq = sequence_means(per_slot, batch["b_idx"].to(d.device), batch["n_seq"])
            loss = per_seq.sum() / cfg.batch_size
            loss.backward()
            opt.accumulate()
            loss_sum += float(loss.detach())
            n_slots += int(d.shape[0])
            n_tok += int(batch["attention_mask"].sum())
        gnorm = opt.clip(cfg.grad_clip)
        lr = lr_at(step, total, cfg.lr, warmup, cfg.min_lr_ratio)
        opt.step(lr)
        log({"event": "step", "step": step, "epoch": epoch, "lr": lr, "loss": loss_sum, "grad_norm": gnorm,
             "slots": n_slots, "tokens": n_tok, "rows_by_stratum": dict(sorted(drawn.items())),
             "correct_rows": n_correct,
             "sec": round(time.time() - t0, 3), "peak_gib": _peak_gib(device)})
        if step % 10 == 0 or step == 1:
            print(f"[train] update {step}/{total} epoch {epoch} loss {loss_sum:.4f} |g| {gnorm:.3g} lr {lr:.3g} "
                  f"{time.time() - t0:.2f}s", flush=True)
        if (cfg.eval_every > 0 and step % cfg.eval_every == 0) or step == total:
            snap = save_snapshot(model, snapshot_dir(wroot, cfg.out_name, step), {**marker, "step": step},
                                 init_state)
            run_eval(step, snap)
            write_json(log_dir / "summary.json", summary)

    best = select_best(summary["evals"], cfg.best_by)
    if best is None:
        best = summary["evals"][-1]
        summary["best_note"] = f"no snapshot had a defined {cfg.best_by}; kept the last one"
    summary["best"] = {"step": best["step"], "by": cfg.best_by, "value": best["metrics"].get(cfg.best_by),
                       "snapshot": best["snapshot"]["dir"], "metrics": best["metrics"]}
    final = publish_best(wroot, cfg.out_name, best, {**marker, "best": summary["best"]})
    summary.update(status="complete", final_dir=str(final), seconds=round(time.time() - t_start, 1),
                   peak_gib=_peak_gib(device), completed_utc=utc_now())
    write_json(log_dir / "summary.json", summary)
    print(f"[train] best {cfg.best_by}={summary['best']['value']} at update {best['step']} -> {final}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    D = TrainConfig
    ap = argparse.ArgumentParser(prog="python -m medtrace.train.trainer",
                                 description="Fine-tune the InternVL3 PRM on rows.jsonl (Alg. 1 lines 17-22); "
                                             "select the GPU with CUDA_VISIBLE_DEVICES / --device. The master "
                                             "seed is seeds.train of the environment JSON ($MEDTRACE_ENV).")
    ap.add_argument("--rows", required=True, help="targets/<target_tag>/rows.jsonl")
    ap.add_argument("--out-name", required=True, help="checkpoint name under weights_root")
    ap.add_argument("--manifest", required=True,
                    help="dataset manifest the rows were built on (frame ids -> slices; its task_type)")
    ap.add_argument("--model", default=D.model, help="base model (name under models_root or path)")
    ap.add_argument("--lr", type=float, default=D.lr)
    ap.add_argument("--weight-decay", type=float, default=D.weight_decay)
    ap.add_argument("--warmup-ratio", type=float, default=D.warmup_ratio)
    ap.add_argument("--min-lr-ratio", type=float, default=D.min_lr_ratio)
    ap.add_argument("--batch-size", type=int, default=D.batch_size, help="trajectories per update")
    ap.add_argument("--micro-batch", type=int, default=D.micro_batch, help="trajectories per forward")
    ap.add_argument("--epochs", type=float, default=D.epochs)
    ap.add_argument("--max-steps", type=int, default=None, help="updates (overrides --epochs)")
    ap.add_argument("--grad-clip", type=float, default=D.grad_clip)
    ap.add_argument("--eval-every", type=int, default=D.eval_every,
                    help="updates between snapshots (each scored on meta; 0 = only after the last)")
    ap.add_argument("--best-by", default=D.best_by, choices=BEST_BY,
                    help="meta selection metric (bon_bal_acc: binary tasks; bon_c_index: survival tasks)")
    ap.add_argument("--grad-ckpt", action="store_true", help="checkpoint the language model too")
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--device", default=D.device, help="cuda:i indexes CUDA_VISIBLE_DEVICES")
    ap.add_argument("--weights-root", default=None)
    ap.add_argument("--log-dir", default=None, help="default <rows dir>/train/<out-name>")
    ap.add_argument("--image-root", default=None,
                    help="base of relative frames_dir, relative to the working directory "
                         "(default: MEDTRACE_IMAGE_ROOT, else the manifest's)")
    ap.add_argument("--fit-limit", type=int, default=None, help="first N fit rows (smoke runs)")
    ap.add_argument("--meta-limit", type=int, default=None, help="first N meta rows (smoke runs)")
    ap.add_argument("--force", action="store_true", help="rename this trainer's earlier outputs of the name aside")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        cfg = TrainConfig(**vars(a)).validate()
    except ValueError as e:
        ap.error(str(e))
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
