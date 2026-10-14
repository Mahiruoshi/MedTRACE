from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..registry import SELECTORS


def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0:
        raise ValueError(f"temperature must be > 0, got {tau}")
    z = np.asarray(x, dtype=np.float64) / float(tau)
    e = np.exp(z - z.max())
    return e / e.sum()


def raw_gates(s_sel: np.ndarray, tau_r: float) -> np.ndarray:
    s_sel = np.asarray(s_sel, dtype=np.float64)
    return s_sel.size * softmax(s_sel, tau_r)


def gates(s_sel: np.ndarray, tau_r: float = 0.1, alpha_min: float = 0.1, alpha_max: float = 1.0) -> np.ndarray:
    if not 0 < alpha_min <= alpha_max:
        raise ValueError(f"need 0 < alpha_min <= alpha_max, got {alpha_min}, {alpha_max}")
    if np.asarray(s_sel).size == 0:
        return np.zeros(0)
    return np.clip(raw_gates(s_sel, tau_r), alpha_min, alpha_max)


def order_desc(s: np.ndarray, idx: Optional[np.ndarray] = None) -> List[int]:
    s = np.asarray(s, dtype=np.float64)
    idx = np.arange(s.size) if idx is None else np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return []
    return [int(i) for i in idx[np.lexsort((idx, -s[idx]))]]


def normalized_entropy(s: np.ndarray, tau: float) -> float:
    s = np.asarray(s, dtype=np.float64)
    if s.size <= 1:
        return 0.0
    p = softmax(s, tau)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / math.log(s.size))


@dataclass
class Selection:
    indices: List[int]
    gates: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


class Selector:
    name = "selector"
    param_names: Tuple[str, ...] = ()

    def __init__(self, m: int = 5, tau_r: float = 0.1, alpha_min: float = 0.1, alpha_max: float = 1.0,
                 **params):
        unknown = sorted(set(params) - set(self.param_names))
        if unknown:
            raise TypeError(f"selector {self.name!r} takes no parameters {unknown} (accepts {list(self.param_names)})")
        if int(m) < 1:
            raise ValueError(f"budget m must be >= 1, got {m}")
        self.m, self.tau_r = int(m), float(tau_r)
        self.alpha_min, self.alpha_max = float(alpha_min), float(alpha_max)
        self.params: Dict[str, Any] = dict(params)

    def pick(self, s: np.ndarray) -> Tuple[List[int], Dict[str, Any]]:
        raise NotImplementedError

    def select(self, s: np.ndarray, pid: str) -> Selection:
        s = np.asarray(s, dtype=np.float64)
        if s.ndim != 1 or s.size == 0:
            raise ValueError(f"{pid}: relevance must be a non-empty 1-D array, got shape {s.shape}")
        if not np.isfinite(s).all():
            raise ValueError(f"{pid}: relevance contains non-finite values")
        idx, meta = self.pick(s)
        g = gates(s[idx], self.tau_r, self.alpha_min, self.alpha_max)
        return Selection(indices=idx, gates=g, meta=meta)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "m": self.m, "tau_r": self.tau_r, "alpha_min": self.alpha_min,
                "alpha_max": self.alpha_max, "params": dict(self.params)}


@SELECTORS.register("topk")
class TopK(Selector):
    name = "topk"

    def pick(self, s: np.ndarray) -> Tuple[List[int], Dict[str, Any]]:
        return order_desc(s)[: self.m], {}


@SELECTORS.register("ada")
class ADA(Selector):
    name = "ada"
    param_names = ("rho", "max_depth")

    def __init__(self, m: int = 5, tau_r: float = 0.1, alpha_min: float = 0.1, alpha_max: float = 1.0,
                 rho: float = 0.9, max_depth: int = 3):
        super().__init__(m, tau_r, alpha_min, alpha_max, rho=rho, max_depth=max_depth)
        if not 0 < rho < 1:
            raise ValueError(f"rho must lie in (0, 1), got {rho}")
        self.rho, self.max_depth = float(rho), int(max_depth)

    def _segment(self, s: np.ndarray, lo: int, hi: int, budget: int, depth: int, leaves: List) -> List[int]:
        n = hi - lo
        if n <= budget:
            leaves.append([lo, hi, budget, "all"])
            return list(range(lo, hi))
        h = normalized_entropy(s[lo:hi], self.tau_r)
        if h < self.rho or depth >= self.max_depth or budget == 1:
            leaves.append([lo, hi, budget, "focal" if h < self.rho else "leaf"])
            return order_desc(s, np.arange(lo, hi))[:budget]
        mid = lo + n // 2
        return (self._segment(s, lo, mid, budget // 2, depth + 1, leaves)
                + self._segment(s, mid, hi, budget - budget // 2, depth + 1, leaves))

    def pick(self, s: np.ndarray) -> Tuple[List[int], Dict[str, Any]]:
        h = normalized_entropy(s, self.tau_r)
        leaves: List = []
        chosen = self._segment(s, 0, s.size, min(self.m, s.size), 0, leaves)
        if s.size <= self.m:
            mode = "all"
        elif h < self.rho:
            mode = "focal"
        else:
            mode = "axial" if self.max_depth > 0 and self.m > 1 else "leaf"
        return order_desc(s, np.asarray(chosen, dtype=np.int64)), {"mode": mode, "h_norm": h, "leaves": leaves}
