from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..registry import MAPPINGS
from ..seeds import SeedRegistry

EPS = 1e-8


@dataclass
class Frames:
    X: np.ndarray
    keys: List[str]
    views: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.X.ndim != 2:
            raise ValueError(f"Frames.X must be 2-D, got shape {self.X.shape}")
        if len(self.keys) != self.X.shape[0]:
            raise ValueError(f"{len(self.keys)} keys for {self.X.shape[0]} rows")
        for k, v in self.views.items():
            if np.asarray(v).shape[0] != self.X.shape[0]:
                raise ValueError(f"view {k!r} has {np.asarray(v).shape[0]} rows, X has {self.X.shape[0]}")

    @property
    def T(self) -> int:
        return self.X.shape[0]

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    def with_X(self, X: np.ndarray) -> "Frames":
        return Frames(X=X, keys=list(self.keys), views=self.views, meta=dict(self.meta))

    def view(self, name: str) -> np.ndarray:
        if name not in self.views:
            raise KeyError(f"view {name!r} not available; have {sorted(self.views)}. "
                           f"Extract it with medtrace.features first.")
        return np.asarray(self.views[name], dtype=np.float64)

    def take(self, idx: Sequence[int]) -> "Frames":
        idx = list(idx)
        return Frames(X=self.X[idx], keys=[self.keys[i] for i in idx],
                      views={k: np.asarray(v)[idx] for k, v in self.views.items()}, meta=dict(self.meta))

    @staticmethod
    def concat(parts: Sequence["Frames"]) -> "Frames":
        if not parts:
            raise ValueError("concat of zero Frames")
        views = {}
        common = set.intersection(*[set(p.views) for p in parts])
        for k in common:
            views[k] = np.concatenate([np.asarray(p.views[k]) for p in parts], axis=0)
        return Frames(X=np.concatenate([p.X for p in parts], axis=0),
                      keys=[k for p in parts for k in p.keys], views=views)


class TextEmbedder(Protocol):
    d_h: int

    def token_ids(self, text: str) -> List[int]: ...

    def rows(self, ids: Sequence[int]) -> np.ndarray: ...


def embed_text(te: TextEmbedder, text: str) -> np.ndarray:
    ids = te.token_ids(text)
    if not ids:
        raise ValueError(f"text {text!r} tokenizes to nothing")
    return np.asarray(te.rows(ids), dtype=np.float64).mean(axis=0)


@dataclass
class FitContext:
    d_h: int
    seeds: SeedRegistry
    text: Optional[TextEmbedder] = None
    instruction: str = ""
    cache: Dict[str, Any] = field(default_factory=dict)
    feature_root: Optional[str] = None
    biomarker_query: str = ""

    def need_text(self) -> TextEmbedder:
        if self.text is None:
            raise RuntimeError("this mapping needs f_LM (FitContext.text) but none was provided")
        return self.text

    def probe(self) -> np.ndarray:
        if "probe" not in self.cache:
            if not self.instruction:
                raise RuntimeError("FitContext.instruction is empty; the probe q is undefined")
            self.cache["probe"] = embed_text(self.need_text(), self.instruction)
        return self.cache["probe"]


class Mapping:
    name: str = "mapping"
    needs_fit: bool = False

    def __init__(self, **params):
        self.params: Dict[str, Any] = dict(params)
        self.fitted: bool = not self.needs_fit
        self.stats: Dict[str, Any] = {}

    def fit(self, calib: Frames, ctx: FitContext) -> "Mapping":
        self.fitted = True
        return self

    def transform(self, fr: Frames, ctx: FitContext) -> Frames:
        raise NotImplementedError

    def __call__(self, fr: Frames, ctx: FitContext) -> Frames:
        if not self.fitted:
            raise RuntimeError(f"mapping {self.name!r} must be fit on a calibration set before use "
                               f"(it never self-fits on the frames it transforms)")
        return self.transform(fr, ctx)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "params": jsonable(self.params), "stats": jsonable(self.stats)}

    def __repr__(self) -> str:
        ps = ",".join(f"{k}={_spec_value(v)}" for k, v in self.params.items() if v is not None)
        return f"{self.name}({ps})" if ps else self.name


class Chain(Mapping):
    name = "chain"

    def __init__(self, stages: Sequence[Mapping]):
        super().__init__()
        self.stages: List[Mapping] = list(stages)
        self.fitted = all(s.fitted for s in self.stages)
        self.needs_fit = not self.fitted

    def fit(self, calib: Frames, ctx: FitContext) -> "Chain":
        cur = calib
        for st in self.stages:
            st.fit(cur, ctx)
            cur = st(cur, ctx)
        self.fitted = True
        return self

    def transform(self, fr: Frames, ctx: FitContext) -> Frames:
        cur = fr
        for st in self.stages:
            cur = st(cur, ctx)
        return cur

    @property
    def spec(self) -> str:
        return ">".join(repr(s) for s in self.stages) or "identity"

    def describe(self) -> Dict[str, Any]:
        return {"name": "chain", "spec": self.spec, "stages": [s.describe() for s in self.stages]}

    def __repr__(self) -> str:
        return self.spec


_TOKEN = re.compile(r"^\s*([A-Za-z_][\w\-]*)\s*(?:\((.*)\))?\s*$")


def _parse_value(raw: str) -> Any:
    s = raw.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _split_top(s: str, sep: str) -> List[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def parse_stage(tok: str) -> Tuple[str, Dict[str, Any]]:
    m = _TOKEN.match(tok)
    if not m:
        raise ValueError(f"bad mapping token {tok!r}")
    name, args = m.group(1), m.group(2)
    params: Dict[str, Any] = {}
    if args and args.strip():
        for kv in _split_top(args, ","):
            if "=" not in kv:
                raise ValueError(f"mapping argument {kv!r} in {tok!r} must be key=value")
            k, v = kv.split("=", 1)
            params[k.strip()] = _parse_value(v)
    return name, params


def parse_chain(spec: str | Sequence) -> Chain:
    stages: List[Mapping] = []
    if isinstance(spec, str):
        s = spec.strip()
        if s in ("", "identity"):
            return Chain([])
        for tok in _split_top(s, ">"):
            name, params = parse_stage(tok)
            stages.append(MAPPINGS.build(name, **params))
    else:
        for item in spec:
            if isinstance(item, str):
                stages.extend(parse_chain(item).stages)
            else:
                item = dict(item)
                if not item.get("name"):
                    raise ValueError(f"chain stage {item!r} needs a 'name'")
                stages.append(MAPPINGS.build(item.pop("name"), **item))
    return Chain(stages)


def cosine_rows(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if X.shape[1] != q.shape[0]:
        raise ValueError(f"dimension mismatch: frames are {X.shape[1]}-d, probe is {q.shape[0]}-d "
                         f"(compose a lifting operator Pi: d_v -> d_h first)")
    num = X @ q
    den = np.linalg.norm(X, axis=1) * np.linalg.norm(q)
    return num / np.maximum(den, EPS)


def _spec_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 16 else {"shape": list(obj.shape), "mean": float(obj.mean()),
                                                     "std": float(obj.std())}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


@MAPPINGS.register("identity")
class Identity(Mapping):
    name = "identity"

    def transform(self, fr: Frames, ctx: FitContext) -> Frames:
        return fr
