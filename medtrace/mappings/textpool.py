from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .base import FitContext, Frames

RAW_VIEW = "premerger"
_RANK_TOL = 1e-10


@dataclass
class PCABasis:
    mean: np.ndarray
    components: np.ndarray

    @property
    def k(self) -> int:
        return int(self.components.shape[0])

    def coords(self, F: np.ndarray) -> np.ndarray:
        F = np.asarray(F, dtype=np.float64)
        if F.shape[1] != self.mean.shape[0]:
            raise ValueError(f"PCA basis was fitted on {self.mean.shape[0]}-d features, got {F.shape[1]}-d")
        return (F - self.mean) @ self.components.T

    def strings(self, F: np.ndarray, sig_figs: int) -> List[str]:
        return [serialize(z, sig_figs) for z in self.coords(F)]


def serialize(z: Sequence[float], sig_figs: int) -> str:
    return "[" + ", ".join(f"{float(x):.{int(sig_figs)}g}" for x in z) + "]"


def fit_pca(F: np.ndarray, k: int) -> PCABasis:
    F = np.asarray(F, dtype=np.float64)
    n = F.shape[0]
    if n < 2:
        raise ValueError(f"PCA needs at least 2 calibration frames, got {n}")
    mean = F.mean(axis=0)
    _, sv, vt = np.linalg.svd(F - mean, full_matrices=False)
    rank = int((sv > sv[0] * _RANK_TOL).sum()) if sv[0] > 0 else 0
    r = min(int(k), rank)
    if r < 1:
        raise ValueError("calibration features are constant; PCA is undefined")
    comps = vt[:r].copy()
    pivot = np.abs(comps).argmax(axis=1)
    comps *= np.where(comps[np.arange(r), pivot] < 0, -1.0, 1.0)[:, None]
    return PCABasis(mean=mean, components=comps)


def raw_features(fr: Frames) -> np.ndarray:
    return fr.view(RAW_VIEW)


def _digest(*arrays_or_strs) -> str:
    h = hashlib.sha1()
    for a in arrays_or_strs:
        if isinstance(a, np.ndarray):
            h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        else:
            h.update(str(a).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def pca_basis(calib: Frames, ctx: FitContext, k: int) -> PCABasis:
    F = raw_features(calib)
    key = ("pca_basis", int(k), int(F.shape[0]), _digest(F))
    if key not in ctx.cache:
        ctx.cache[key] = fit_pca(F, k)
    return ctx.cache[key]


def instr_pool(calib: Frames, ctx: FitContext) -> np.ndarray:
    if not ctx.instruction:
        raise RuntimeError("text pool 'instr' needs FitContext.instruction")
    te = ctx.need_text()
    ids = te.token_ids(ctx.instruction)
    if not ids:
        raise ValueError("the instruction tokenizes to nothing")
    return np.asarray(te.rows(ids), dtype=np.float64)


def pca_pool(calib: Frames, ctx: FitContext, k: int = 8, sig_figs: int = 2) -> np.ndarray:
    te = ctx.need_text()
    basis = pca_basis(calib, ctx, k)
    ids = set()
    for s in basis.strings(raw_features(calib), sig_figs):
        ids.update(te.token_ids(s))
    if not ids:
        raise ValueError("PCA strings tokenize to nothing")
    return np.asarray(te.rows(sorted(ids)), dtype=np.float64)


def text_pool(kind: str, calib: Frames, ctx: FitContext, **params) -> np.ndarray:
    if kind == "instr":
        build = instr_pool
    elif kind == "pca":
        build = pca_pool
    else:
        raise KeyError(f"unknown text pool {kind!r}; D_text = instr U pca")
    F = raw_features(calib)
    seeds = ctx.seeds.as_dict()
    key = ("text_pool", kind, tuple(sorted((k, repr(v)) for k, v in params.items())),
           _digest(F, ctx.instruction, seeds))
    if key not in ctx.cache:
        arr = np.asarray(build(calib, ctx, **params), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError(f"text pool {kind!r} returned shape {arr.shape}")
        arr.setflags(write=False)
        ctx.cache[key] = arr
    return ctx.cache[key]
