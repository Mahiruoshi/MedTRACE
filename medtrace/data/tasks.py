from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from .. import envjson

TASK_TYPES = ("binary", "survival")
ANSWER_SEPARATORS = (":", "：", "-", "–")

_SEP = re.compile(r"[\W_]+", re.UNICODE)


def _norm(s: str) -> str:
    return _SEP.sub(" ", str(s).casefold()).strip()


@dataclass(frozen=True)
class SurvivalTarget:
    time: float
    event: int
    bin: int


Target = Union[int, SurvivalTarget]


@dataclass(frozen=True)
class Task:
    kind: str
    labels: Tuple[str, ...]
    aliases: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    instruction: str = ""
    question: str = ""
    biomarker_query: str = ""
    system: str = ""
    user: str = ""
    frame_line: str = ""
    step_names: Tuple[str, ...] = ()
    answer_prefix: str = ""

    def __post_init__(self):
        if self.kind not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {TASK_TYPES}, got {self.kind!r}")
        if self.kind == "binary" and len(self.labels) != 2:
            raise ValueError(f"a binary task needs answer_labels = [negative, positive], got {list(self.labels)}")
        if self.kind == "survival" and len(self.labels) < 2:
            raise ValueError("a survival task needs at least two answer labels (stages)")
        unknown = sorted(set(self.aliases) - set(self.labels))
        if unknown:
            raise ValueError(f"answer_aliases names unknown labels {unknown}")
        table: Dict[str, int] = {}
        for i, lab in enumerate(self.labels):
            for word in (lab, *self.aliases.get(lab, ())):
                key = _norm(word)
                if not key:
                    raise ValueError(f"answer label/alias {word!r} has no letters or digits")
                if table.get(key, i) != i:
                    raise ValueError(f"answer label/alias {word!r} also names {self.labels[table[key]]!r}; "
                                     "labels and aliases must be distinct after case folding")
                table[key] = i
        object.__setattr__(self, "_table", table)
        if self.answer_prefix and not self.answer_prefix.rstrip().endswith(ANSWER_SEPARATORS):
            raise ValueError(f"prompts.answer_prefix must end with one of {' '.join(ANSWER_SEPARATORS)} "
                             f"(the parser reads '<prefix> <label>'), got {self.answer_prefix!r}")

    @property
    def K(self) -> int:
        return len(self.labels)

    @property
    def n_steps(self) -> int:
        return len(self.step_names) + 1

    def parse_label(self, text: Optional[str]) -> Optional[int]:
        if text is None:
            return None
        return self._table.get(_norm(text))

    def label_words(self) -> Tuple[str, ...]:
        return tuple(sorted({_norm(x) for x in self.labels}, key=lambda w: (-len(w), w)))

    def vocabulary(self) -> Tuple[str, ...]:
        return tuple(sorted(self._table, key=lambda w: (-len(w), w)))

    def determined(self, target: Target) -> bool:
        if self.kind == "binary":
            return True
        t: SurvivalTarget = target
        return bool(t.event) or int(t.bin) == self.K - 1

    def is_correct(self, pred: Optional[int], target: Target) -> Optional[int]:
        if pred is None:
            return 0
        pred = int(pred)
        if self.kind == "binary":
            return int(pred == int(target))
        t: SurvivalTarget = target
        if t.event:
            return int(pred == t.bin)
        if pred < t.bin:
            return 0
        return 1 if t.bin == self.K - 1 else None

    def stratum(self, target: Target) -> int:
        return int(target) if self.kind == "binary" else int(target.bin)

    def risk(self, idx: Optional[int]) -> Optional[float]:
        if idx is None:
            return None
        return float(idx) if self.kind == "binary" else float(self.K - 1 - int(idx))

    def public(self) -> Dict[str, Any]:
        return {"kind": self.kind, "labels": list(self.labels),
                "aliases": {k: list(v) for k, v in sorted(self.aliases.items())},
                "instruction": self.instruction, "question": self.question,
                "biomarker_query": self.biomarker_query, "system": self.system, "user": self.user,
                "frame_line": self.frame_line, "step_names": list(self.step_names),
                "answer_prefix": self.answer_prefix}


def load_task(env: Mapping[str, Any], kind: str, need: Sequence[str] = ()) -> Task:
    envjson.require(env, "prompts.answer_labels", *need)
    p = env["prompts"]
    aliases = {str(k): tuple(v) for k, v in (p.get("answer_aliases") or {}).items()}
    return Task(kind=kind, labels=tuple(p["answer_labels"]), aliases=aliases,
                instruction=p.get("instruction") or "", question=p.get("question") or "",
                biomarker_query=p.get("biomarker_query") or "", system=p.get("system") or "",
                user=p.get("user") or "", frame_line=p.get("frame_line") or "",
                step_names=tuple(p.get("step_names") or ()),
                answer_prefix=p.get("answer_prefix") or "")


def survival_edges(times: Sequence[float], events: Sequence[int], K: int) -> Tuple[float, ...]:
    t = sorted(float(x) for x, e in zip(times, events) if int(e) == 1)
    if any(not math.isfinite(x) for x in t):
        raise ValueError("survival times must be finite")
    if len(t) < K:
        raise ValueError(f"need at least {K} uncensored train patients for {K} survival bins, have {len(t)}")
    edges = []
    for k in range(1, K):
        q = k / K * (len(t) - 1)
        lo, hi = int(math.floor(q)), int(math.ceil(q))
        edges.append(t[lo] + (t[hi] - t[lo]) * (q - lo))
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"survival bin edges are not increasing ({edges}); use fewer answer labels")
    counts = [0] * K
    for x in t:
        counts[survival_bin(x, edges)] += 1
    if min(counts) == 0:
        raise ValueError(f"a survival bin holds no uncensored train time (counts {counts}, edges {edges}); "
                         "use fewer answer labels")
    return tuple(edges)


def survival_bin(time: float, edges: Sequence[float]) -> int:
    return int(sum(1 for e in edges if float(time) >= e))
