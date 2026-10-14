from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from ..data.cohort import load_manifest, load_patients
from ..data.tasks import SurvivalTarget, Target, Task
from ..prm.internvl import IMAGE_SIZE, PRMScorer, ScoreRequest
from ..prm.layouts import NEG_TOKEN, POS_TOKEN, PRMPrompt
from ..seeds import sub_seed

SPLITS = ("fit", "meta")
REQUIRED_FIELDS = ("id", "pid", "split", "system", "question", "frame_ids", "steps", "markers", "p_tilde", "pred",
                   "correct", "target", "y")
_MARKERS = {POS_TOKEN: 1, NEG_TOKEN: 0}


def target_to_json(t: Target) -> Any:
    if isinstance(t, SurvivalTarget):
        return {"time": float(t.time), "event": int(t.event), "bin": int(t.bin)}
    return int(t)


def target_from_json(v: Any, where: str = "?") -> Target:
    if isinstance(v, Mapping):
        if set(v) != {"time", "event", "bin"}:
            raise ValueError(f"{where}: a survival target is {{time, event, bin}}, got {sorted(v)}")
        t, e, b = v["time"], v["event"], v["bin"]
        if (isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(float(t)) or t < 0
                or e not in (0, 1) or isinstance(e, bool) or isinstance(b, bool) or not isinstance(b, int) or b < 0):
            raise ValueError(f"{where}: bad survival target {dict(v)!r}")
        return SurvivalTarget(time=float(t), event=int(e), bin=int(b))
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"{where}: target must be a label 0/1 or a survival {{time, event, bin}}, got {v!r}")
    return int(v)


def _stratum(t: Target) -> int:
    return int(t.bin) if isinstance(t, SurvivalTarget) else int(t)


@dataclass(frozen=True)
class TrainRow:
    id: str
    pid: str
    split: str
    system: str
    question: str
    frame_ids: Tuple[str, ...]
    steps: Tuple[str, ...]
    markers: Tuple[int, ...]
    p_tilde: Tuple[float, ...]
    pred: Optional[int]
    correct: int
    target: Target
    y: int
    cot_idx: Optional[int] = None

    @property
    def n(self) -> int:
        return len(self.steps)


def parse_row(d: Mapping[str, Any]) -> TrainRow:
    rid = d.get("id", "?")
    missing = [k for k in REQUIRED_FIELDS if k not in d]
    if missing:
        raise KeyError(f"row {rid}: missing fields {missing} (schema: medtrace.train.dataset docstring)")
    pid, split = str(d["pid"]), str(d["split"])
    if split not in SPLITS:
        raise ValueError(f"row {rid}: split {split!r} not in {SPLITS}")
    steps = list(d["steps"])
    n = len(steps)
    if n == 0 or any(not isinstance(s, str) or not s.strip() for s in steps):
        raise ValueError(f"row {rid}: steps must be {n or 'one or more'} non-empty strings")
    for key in ("markers", "p_tilde"):
        if len(d[key]) != n:
            raise ValueError(f"row {rid}: {len(d[key])} {key} for {n} steps")
    bad = [m for m in d["markers"] if not isinstance(m, str) or m not in _MARKERS]
    if bad:
        raise ValueError(f"row {rid}: markers must be '{POS_TOKEN}'/'{NEG_TOKEN}', got {bad[:3]}")
    markers = tuple(_MARKERS[m] for m in d["markers"])
    p_tilde = tuple(float(x) for x in d["p_tilde"])
    if any(not (math.isfinite(x) and 0.0 <= x <= 1.0) for x in p_tilde):
        raise ValueError(f"row {rid}: p_tilde must lie in [0, 1]")
    frames = [str(f) for f in d["frame_ids"]]
    if any(f.rpartition("/s")[0] != pid for f in frames):
        raise ValueError(f"row {rid}: frame ids must all be '{pid}/s<idx>'")
    pred = d["pred"]
    if pred is not None and (isinstance(pred, (bool, float)) or not isinstance(pred, int) or pred < 0):
        raise ValueError(f"row {rid}: pred must be a label index >= 0 or null, got {pred!r}")
    if d["correct"] not in (0, 1) or isinstance(d["correct"], float):
        raise ValueError(f"row {rid}: correct must be 0/1, got {d['correct']!r}")
    if d["correct"] and pred is None:
        raise ValueError(f"row {rid}: an unparseable answer cannot be correct")
    target = target_from_json(d["target"], f"row {rid}")
    y = d["y"]
    if isinstance(y, (bool, float)) or not isinstance(y, int) or y != _stratum(target):
        raise ValueError(f"row {rid}: y must be the target's stratum {_stratum(target)}, got {y!r}")
    if not str(d["system"]).strip() or not str(d["question"]).strip():
        raise ValueError(f"row {rid}: empty system prompt or question")
    return TrainRow(id=str(rid), pid=pid, split=split, system=str(d["system"]), question=str(d["question"]),
                    frame_ids=tuple(frames), steps=tuple(steps), markers=markers, p_tilde=p_tilde,
                    pred=pred, correct=int(d["correct"]), target=target, y=int(y),
                    cot_idx=None if d.get("cot_idx") is None else int(d["cot_idx"]))


def load_rows(path: Union[str, os.PathLike], split: Optional[str] = None) -> List[TrainRow]:
    rows, seen = [], set()
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = parse_row(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{Path(path).name}:{n}: not JSON ({e.msg})") from None
            if r.id in seen:
                raise ValueError(f"{Path(path).name}:{n}: duplicate row id {r.id!r}")
            seen.add(r.id)
            rows.append(r)
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"split {split!r} not in {SPLITS}")
        rows = [r for r in rows if r.split == split]
    return rows


def row_targets(rows: Sequence[TrainRow]) -> Tuple[Dict[str, int], Dict[str, Target]]:
    strata: Dict[str, int] = {}
    targets: Dict[str, Target] = {}
    for r in rows:
        if r.pid in targets and (targets[r.pid] != r.target or strata[r.pid] != r.y):
            raise ValueError(f"rows of {r.pid} carry different targets")
        strata[r.pid], targets[r.pid] = r.y, r.target
    return strata, targets


def check_rows(rows: Sequence[TrainRow], task: Task) -> None:
    bad: List[str] = []
    for r in rows:
        if (task.kind == "survival") != isinstance(r.target, SurvivalTarget) or (
                task.kind == "binary" and r.target not in (0, 1)):
            bad.append(f"{r.id} (target {r.target!r} for a {task.kind} task)")
        elif not task.determined(r.target):
            bad.append(f"{r.id} (an undetermined target: censoring leaves its correctness open)")
        elif r.pred is not None and r.pred >= task.K:
            bad.append(f"{r.id} (pred {r.pred} for {task.K} answers)")
        elif r.y != task.stratum(r.target) or r.correct != task.is_correct(r.pred, r.target):
            bad.append(f"{r.id} (y / correct disagree with the task)")
    if bad:
        raise ValueError(f"rows disagree with the task: {bad[:5]}")


def require_train(pids: Sequence[str], manifest: Union[str, os.PathLike, Mapping[str, Any]]) -> None:
    man = load_manifest(manifest)
    train = {str(r["pid"]) for r in man["patients"] if r["split"] == "train"}
    foreign = sorted(set(pids) - train)
    if foreign:
        raise ValueError(f"rows name patients outside the train split, e.g. {foreign[:5]} "
                         f"(PRM training never touches the test split)")


def train_cohort(pids: Sequence[str], manifest: Union[str, os.PathLike], image_root: Optional[str] = None,
                 strict: bool = True) -> Callable[[str], str]:
    from ..paths import get_paths

    man = load_manifest(manifest)
    require_train(pids, man)
    pts = load_patients(man, "train", image_root or get_paths().image_root, pids=sorted(set(pids)), strict=strict)
    paths = {fid: path for p in pts for fid, path in zip(p.frame_ids, p.frames)}

    def resolve(fid: str) -> str:
        try:
            return paths[fid]
        except KeyError:
            raise KeyError(f"frame id {fid} is not a slice of its patient on disk") from None

    return resolve


class PRMEncoder:
    def __init__(self, tokenizer: Any, num_image_token: int, tiles: Callable, slots: Any, pad_id: int):
        self.tokenizer = tokenizer
        self.num_image_token = int(num_image_token)
        self.tiles = tiles
        self.slots = slots
        self.pad_id = int(pad_id)

    @classmethod
    def for_scorer(cls, scorer: PRMScorer) -> "PRMEncoder":
        return cls(scorer.tokenizer, scorer.num_image_token, tiles=scorer.tiles, slots=scorer.slots,
                   pad_id=scorer.pad_id)

    def encode(self, pr: PRMPrompt, key: Optional[str] = None):
        return PRMScorer.encode(self, pr, key)


class PRMRowsDataset:
    def __init__(self, rows: Union[str, os.PathLike, Sequence[TrainRow]], split: str, *,
                 encoder: PRMEncoder, layout: Any, manifest: Union[str, os.PathLike],
                 image_root: Optional[str] = None):
        if split not in SPLITS:
            raise ValueError(f"split {split!r} not in {SPLITS}")
        all_rows = load_rows(rows) if isinstance(rows, (str, os.PathLike)) else list(rows)
        self.rows: List[TrainRow] = [r for r in all_rows if r.split == split]
        if not self.rows:
            raise ValueError(f"no {split!r} rows")
        self.split, self.encoder, self.layout = split, encoder, layout
        self.resolve = train_cohort(sorted({r.pid for r in self.rows}), manifest, image_root)
        self.strata, self.targets = row_targets(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def pids(self) -> List[str]:
        return sorted(self.strata)

    def image_paths(self, i: int) -> List[str]:
        return [self.resolve(f) for f in self.rows[i].frame_ids]

    def fills(self, i: int) -> List[str]:
        return [POS_TOKEN] * self.rows[i].n

    def prompt(self, i: int) -> PRMPrompt:
        r = self.rows[i]
        pr = self.layout.build(r.system, r.question, list(r.steps), self.image_paths(i), self.fills(i))
        if pr.slot_steps != list(range(r.n)):
            raise RuntimeError(f"{r.id}: layout {self.layout.name!r} scores steps {pr.slot_steps}, "
                               f"not every step once in order")
        return pr

    def encode_row(self, i: int):
        ids, pos, tiles = self.encoder.encode(self.prompt(i), self.rows[i].id)
        if len(pos) != self.rows[i].n:
            raise RuntimeError(f"{self.rows[i].id}: {len(pos)} slots for {self.rows[i].n} steps")
        return ids, pos, tiles

    def request(self, i: int) -> ScoreRequest:
        r = self.rows[i]
        return ScoreRequest(key=r.id, system=r.system, question=r.question, steps=list(r.steps),
                            image_paths=self.image_paths(i), group=r.pid)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        import torch

        r = self.rows[i]
        ids, pos, tiles = self.encode_row(i)
        if tiles:
            pv = torch.cat(tiles, 0)
            flags = torch.ones(pv.shape[0], dtype=torch.long)
        else:
            pv = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            flags = torch.zeros(1, dtype=torch.long)
        return {"id": r.id, "pid": r.pid,
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "pixel_values": pv, "image_flags": flags,
                "slots": torch.tensor(pos, dtype=torch.long),
                "markers": torch.tensor(r.markers, dtype=torch.float32)}


def collate(items: Sequence[Mapping[str, Any]], pad_id: int) -> Dict[str, Any]:
    import torch

    B, L = len(items), max(int(it["input_ids"].shape[0]) for it in items)
    ids = torch.full((B, L), int(pad_id), dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.long)
    for b, it in enumerate(items):
        n = int(it["input_ids"].shape[0])
        ids[b, :n] = it["input_ids"]
        mask[b, :n] = it["attention_mask"]
    b_idx = torch.cat([torch.full((int(it["slots"].shape[0]),), b, dtype=torch.long) for b, it in enumerate(items)])
    return {"ids": [it["id"] for it in items], "input_ids": ids, "attention_mask": mask,
            "pixel_values": torch.cat([it["pixel_values"] for it in items], 0),
            "image_flags": torch.cat([it["image_flags"] for it in items], 0),
            "b_idx": b_idx, "s_idx": torch.cat([it["slots"] for it in items]),
            "markers": torch.cat([it["markers"] for it in items]), "n_seq": B}


class RowSampler:
    KIND = "label_balanced"

    def __init__(self, rows: Sequence[TrainRow], strata: Mapping[str, int], seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"sampler seed must be an int, got {seed!r}")
        self.seed, self.n = int(seed), len(rows)
        if not self.n:
            raise ValueError("no rows to sample")
        self.by_pid: Dict[str, List[int]] = {}
        for i, r in enumerate(rows):
            self.by_pid.setdefault(r.pid, []).append(i)
        self.by_stratum: Dict[int, List[str]] = {}
        for pid in sorted(self.by_pid):
            self.by_stratum.setdefault(int(strata[pid]), []).append(pid)
        self.strata: List[int] = sorted(self.by_stratum, reverse=True)
        if len(self.strata) < 2:
            raise ValueError(f"the label-balanced sampler needs at least two strata among the rows' "
                             f"patients (have {self.strata})")

    def epoch(self, e: int) -> List[int]:
        rng = random.Random(sub_seed(self.seed, "sampler", self.KIND, int(e)))
        per = self.n // len(self.strata)
        draws = [s for s in self.strata for _ in range(per)]
        rest = list(self.strata)
        for _ in range(self.n % len(self.strata)):
            draws.append(rest.pop(int(rng.random() * len(rest))))
        rng.shuffle(draws)
        return [rng.choice(self.by_pid[rng.choice(self.by_stratum[s])]) for s in draws]

    def stream(self) -> Iterator[Tuple[int, int]]:
        e = 0
        while True:
            for i in self.epoch(e):
                yield e, i
            e += 1
