from __future__ import annotations

from typing import Tuple

import numpy as np

from ..registry import MAPPINGS
from .base import FitContext, Frames, Mapping
from .textpool import text_pool

TEXT_POOL_KINDS = ("instr", "pca")


def moments(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    return X.mean(axis=0), X.std(axis=0)


def moment_match(mu_src: np.ndarray, sd_src: np.ndarray, mu_tar: np.ndarray, sd_tar: np.ndarray,
                 eps: float) -> Tuple[np.ndarray, np.ndarray]:
    scale = sd_tar / (sd_src + eps)
    return scale, mu_tar - scale * mu_src


@MAPPINGS.register("ot")
class MomentOT(Mapping):
    name = "ot"
    needs_fit = True

    def __init__(self, eps: float = 1e-6, k: int = 8, sig_figs: int = 2):
        super().__init__(eps=eps, k=k, sig_figs=sig_figs)

    def fit(self, calib: Frames, ctx: FitContext) -> "MomentOT":
        p = self.params
        pools = [text_pool("instr", calib, ctx),
                 text_pool("pca", calib, ctx, k=p["k"], sig_figs=p["sig_figs"])]
        V = np.vstack(pools)
        mu_tar, sd_tar = moments(V)
        mu_src, sd_src = moments(calib.X)
        if mu_tar.shape != mu_src.shape:
            raise ValueError(f"{self.name}: source is {mu_src.shape[0]}-d, target is {mu_tar.shape[0]}-d "
                             f"(compose a lifting operator Pi: d_v -> d_h in front)")
        eps = float(p["eps"])
        self.mu_src, self.sd_src, self.mu_tar, self.sd_tar = mu_src, sd_src, mu_tar, sd_tar
        self.scale, self.shift = moment_match(mu_src, sd_src, mu_tar, sd_tar, eps)
        self.stats.update(n_src=calib.T, dim=int(mu_src.shape[0]),
                          scale_median=float(np.median(self.scale)),
                          src_const_dims=int((sd_src <= eps).sum()),
                          n_tar=int(V.shape[0]),
                          pools={k: int(P.shape[0]) for k, P in zip(TEXT_POOL_KINDS, pools)})
        self.fitted = True
        return self

    def transform(self, fr: Frames, ctx: FitContext) -> Frames:
        if fr.dim != self.scale.shape[0]:
            raise ValueError(f"{self.name}: fitted on {self.scale.shape[0]}-d rows, got {fr.dim}-d")
        return fr.with_X(fr.X * self.scale + self.shift)
