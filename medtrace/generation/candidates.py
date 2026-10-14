from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from ..data.tasks import Task
from ..seeds import sub_seed
from .cache import GenerationCache, request_identity, request_key
from .parsing import parse
from .vllm_client import GenRequest, GenResult, normalize_sampling


def candidate_id(pid: str, k: int) -> str:
    return f"{pid}_c{k}"


def candidate_seed(decode_seed: int, pid: str, k: int) -> int:
    return sub_seed(decode_seed, pid, k)


@dataclass
class Candidate:
    id: str
    pid: str
    cot_idx: int
    seed: int
    frame_ids: List[str]
    raw: str
    steps: List[str]
    answer: Optional[str]
    pred: Optional[int]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {"id": self.id, "pid": self.pid, "cot_idx": self.cot_idx, "seed": self.seed,
                "frame_ids": list(self.frame_ids), "raw": self.raw, "steps": list(self.steps),
                "answer": self.answer, "pred": self.pred, "meta": self.meta}

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "Candidate":
        return Candidate(id=d["id"], pid=d["pid"], cot_idx=int(d["cot_idx"]), seed=int(d["seed"]),
                         frame_ids=list(d["frame_ids"]), raw=d["raw"], steps=list(d["steps"]),
                         answer=d.get("answer"), pred=None if d.get("pred") is None else int(d["pred"]),
                         meta=dict(d.get("meta") or {}))


def from_result(result: GenResult, pid: str, k: int, seed: int, frame_ids: Sequence[str], task: Task,
                sampling: Mapping[str, Any], request_hash: Optional[str] = None,
                model: Optional[str] = None) -> Candidate:
    if not result.ok or result.text is None:
        raise ValueError(f"{result.key}: generation failed ({result.error}); not a candidate")
    s = normalize_sampling(sampling)
    if s["seed"] != int(seed):
        raise ValueError(f"{result.key}: sampling seed {s['seed']} != candidate seed {seed}")
    p = parse(result.text, task)
    meta: Dict[str, Any] = {"n_padded": p.n_padded,
                            "truncated": result.finish_reason == "length",
                            "finish_reason": result.finish_reason,
                            "answer_line": p.answer_line,
                            "usage": dict(result.usage),
                            "sampling": s,
                            "request_hash": request_hash}
    if model is not None:
        meta["model"] = model
    return Candidate(id=candidate_id(pid, k), pid=pid, cot_idx=int(k), seed=int(seed),
                     frame_ids=list(frame_ids), raw=result.text, steps=p.steps, answer=p.answer,
                     pred=p.pred, meta=meta)


class CandidateWriter:
    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.repaired = cut_partial_tail(self.path)
        self.done: Set[str] = {c.id for c in load_candidates(self.path)} if self.path.is_file() else set()
        self._f = open(self.path, "a", encoding="utf-8", newline="\n")

    def append(self, c: Candidate) -> None:
        if c.id in self.done:
            raise ValueError(f"candidate {c.id} already written")
        self._f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())
        self.done.add(c.id)

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "CandidateWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def cut_partial_tail(path: Path) -> int:
    if not path.is_file():
        return 0
    size = path.stat().st_size
    if size == 0:
        return 0
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return 0
        pos = size - 1
        block = 1 << 16
        while pos > 0:
            start = max(0, pos - block)
            f.seek(start)
            nl = f.read(pos - start).rfind(b"\n")
            if nl >= 0:
                keep = start + nl + 1
                break
            pos = start
        else:
            keep = 0
        f.truncate(keep)
        return size - keep


def load_candidates(path: os.PathLike | str) -> List[Candidate]:
    out: List[Candidate] = []
    seen: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    for n, ln in enumerate(lines, start=1):
        if not ln.strip():
            continue
        try:
            c = Candidate.from_json(json.loads(ln))
        except (ValueError, KeyError, TypeError) as e:
            if n == len(lines):
                break
            raise ValueError(f"{path}:{n}: malformed candidate line ({type(e).__name__})") from None
        if c.id in seen:
            raise ValueError(f"{path}:{n}: duplicate candidate id {c.id}")
        seen.add(c.id)
        out.append(c)
    return out


@dataclass
class CandidateJob:
    pid: str
    k: int
    frame_ids: List[str]
    request: GenRequest

    @property
    def id(self) -> str:
        return candidate_id(self.pid, self.k)

    @property
    def seed(self) -> int:
        return int(self.request.sampling["seed"])


def make_jobs(pid: str, system: str, question: str, image_paths: Sequence[str], frame_ids: Sequence[str],
              n: int, decode_seed: int, sampling: Mapping[str, Any], start: int = 0) -> List[CandidateJob]:
    if len(image_paths) != len(frame_ids):
        raise ValueError(f"{pid}: {len(image_paths)} images for {len(frame_ids)} frame ids")
    if "seed" in sampling:
        raise ValueError("sampling must not carry a seed; it is derived per candidate")
    jobs = []
    for k in range(start, n):
        s = dict(sampling, seed=candidate_seed(decode_seed, pid, k))
        jobs.append(CandidateJob(pid=pid, k=k, frame_ids=list(frame_ids),
                                 request=GenRequest(key=candidate_id(pid, k), system=system, question=question,
                                                    image_paths=list(image_paths), sampling=s)))
    return jobs


def run_jobs(generator, jobs: Sequence[CandidateJob], task: Task, writer: CandidateWriter,
             cache: Optional[GenerationCache] = None) -> Dict[str, int]:
    model = generator.served_model
    todo = [j for j in jobs if j.id not in writer.done]
    ids = [j.id for j in todo]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate ids among jobs")
    for j in todo:
        if j.request.key != j.id:
            raise ValueError(f"job {j.id}: request key {j.request.key!r} must equal the candidate id")
    stats = {"skipped": len(jobs) - len(todo), "cached": 0, "generated": 0, "failed": 0,
             "errors": []}
    keys = {j.id: request_key(model, j.request) for j in todo}
    by_id = {j.id: j for j in todo}

    def emit(j: CandidateJob, res: GenResult) -> None:
        c = from_result(res, j.pid, j.k, j.seed, j.frame_ids, task, j.request.sampling,
                        request_hash=keys[j.id], model=model)
        writer.append(c)

    misses = []
    for j in todo:
        hit = cache.get(keys[j.id]) if cache is not None else None
        if hit is None:
            misses.append(j)
            continue
        emit(j, dataclasses.replace(hit, key=j.id))
        stats["cached"] += 1

    def on_result(res: GenResult) -> None:
        j = by_id[res.key]
        if not res.ok:
            stats["failed"] += 1
            if len(stats["errors"]) < 5:
                stats["errors"].append(f"{res.key}: {res.error}")
            return
        if cache is not None:
            cache.put(keys[j.id], res, request_identity(model, j.request))
        emit(j, res)
        stats["generated"] += 1

    generator.generate([j.request for j in misses], on_result=on_result)
    return stats
