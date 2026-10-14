from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from ..data.tasks import Task
from ..features.store import CachedTextEmbedder, FeatureStore
from ..mappings.base import FitContext, Frames, jsonable, parse_chain
from ..registry import SCORERS, SELECTORS
from ..seeds import SeedRegistry
from .presets import resolve_variant

DEGENERATE_RANGE = 1e-3
_DISTILLER_ARGS = ("m", "tau_r", "alpha_min", "alpha_max")


@dataclass
class Signal:
    pid: str
    variant: str
    T: int
    frame_ids: List[str]
    s: List[float]
    selected: List[int]
    gates: List[float]
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.frame_ids) != self.T or len(self.s) != self.T:
            raise ValueError(f"{self.pid}: T={self.T} but {len(self.frame_ids)} ids / {len(self.s)} scores")
        if len(self.gates) != len(self.selected):
            raise ValueError(f"{self.pid}: {len(self.gates)} gates for {len(self.selected)} selected frames")
        if any(not 0 <= i < self.T for i in self.selected) or len(set(self.selected)) != len(self.selected):
            raise ValueError(f"{self.pid}: selected indices must be distinct and in [0, T)")

    @property
    def selected_ids(self) -> List[str]:
        return [self.frame_ids[i] for i in self.selected]

    def to_json(self) -> Dict[str, Any]:
        return {"pid": self.pid, "variant": self.variant, "T": self.T, "frame_ids": list(self.frame_ids),
                "s": [float(x) for x in self.s], "selected": [int(i) for i in self.selected],
                "selected_ids": self.selected_ids, "gates": [float(x) for x in self.gates],
                "meta": self.meta}


def context_from_store(store: FeatureStore, task: Task, seeds: SeedRegistry) -> FitContext:
    if not isinstance(task, Task):
        raise TypeError(f"context_from_store needs the run's Task, got {type(task).__name__}")
    te = CachedTextEmbedder(store.root)
    d_h = store.meta.get("d_h")
    if d_h is not None and int(d_h) != te.d_h:
        raise ValueError(f"feature cache meta says d_h={d_h}, embedding table is {te.d_h}-d")
    return FitContext(d_h=te.d_h, seeds=seeds, text=te, instruction=task.instruction,
                      biomarker_query=task.biomarker_query, feature_root=str(store.root.parent))


class Distiller:
    def __init__(self, variant_spec: Union[str, Mapping[str, Any]], ctx: FitContext, m: int = 5,
                 tau_r: float = 0.1, alpha_min: float = 0.1, alpha_max: float = 1.0,
                 name: Optional[str] = None):
        self.spec = resolve_variant(variant_spec, name=name)
        self.ctx = ctx
        self.chain = parse_chain(self.spec["chain"])
        sc = dict(self.spec["scorer"])
        self.scorer = SCORERS.build(sc.pop("name"), chain=self.chain, **sc)
        sel = dict(self.spec["selector"])
        clash = sorted(set(sel) & set(_DISTILLER_ARGS))
        if clash:
            raise ValueError(f"variant {self.name!r}: {clash} are run-level settings (Distiller arguments), "
                             f"not selector parameters")
        self.selector = SELECTORS.build(sel.pop("name"), m=m, tau_r=tau_r, alpha_min=alpha_min,
                                        alpha_max=alpha_max, **sel)
        self.calib_meta: Optional[Dict[str, Any]] = None
        self.fitted = False

    @property
    def name(self) -> str:
        return self.spec["name"]

    @property
    def needs_calibration(self) -> bool:
        return bool(self.scorer.needs_calibration)

    def fit(self, calib: Optional[Frames]) -> "Distiller":
        if calib is None and self.needs_calibration:
            raise ValueError(f"variant {self.name!r} (chain {self.chain.spec}) needs a calibration set")
        self.scorer.fit(calib, self.ctx)
        if calib is not None:
            self.calib_meta = dict(calib.meta.get("calib", {}))
            self.calib_meta.setdefault("n_frames", calib.T)
        self.fitted = True
        return self

    def distill_one(self, pid: str, fr: Frames) -> Signal:
        if not self.fitted:
            raise RuntimeError("Distiller.fit(calib) must run before distillation")
        bad = [k for k in fr.keys if not k.startswith(f"{pid}/")]
        if bad:
            raise ValueError(f"{pid}: frames carry foreign ids, e.g. {bad[0]!r}")
        res = self.scorer.score(fr, self.ctx, pid)
        s = np.asarray(res.s, dtype=np.float64)
        if s.shape != (fr.T,):
            raise RuntimeError(f"{pid}: scorer {self.scorer.name} returned shape {s.shape} for T={fr.T}")
        sel = self.selector.select(s, pid)
        dyn = float(s.max() - s.min())
        meta = {"dyn_range": dyn, "degenerate": bool(dyn < DEGENERATE_RANGE),
                "s_mean": float(s.mean()), "s_std": float(s.std()),
                "selector": jsonable(sel.meta)}
        return Signal(pid=pid, variant=self.name, T=fr.T, frame_ids=list(fr.keys), s=s.tolist(),
                      selected=[int(i) for i in sel.indices], gates=[float(g) for g in sel.gates],
                      meta=meta)

    def run(self, frames_by_pid: Mapping[str, Frames]) -> List[Signal]:
        return [self.distill_one(pid, fr) for pid, fr in frames_by_pid.items()]

    def describe(self) -> Dict[str, Any]:
        return {"variant": self.name, "spec": self.spec, "chain": self.chain.describe(),
                "scorer": self.scorer.describe(), "selector": self.selector.describe(),
                "calib": self.calib_meta,
                "seeds": self.ctx.seeds.as_dict(), "d_h": int(self.ctx.d_h),
                "instruction": self.ctx.instruction, "degenerate_range": DEGENERATE_RANGE}


def save_signals(path: Union[str, os.PathLike], meta: Mapping[str, Any], signals: Sequence[Signal]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"meta": jsonable(dict(meta)), "signals": [sg.to_json() for sg in signals]},
                  f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, p)
