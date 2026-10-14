from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..registry import MAPPINGS
from .base import FitContext, Frames, Mapping

SEED_STAGE = "projection"


def random_lift_matrix(d_in: int, d_out: int, seed: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).normal(0.0, 1.0 / math.sqrt(d_out), size=(int(d_in), int(d_out)))


@MAPPINGS.register("rnd")
class RandomLift(Mapping):
    name = "rnd"

    def __init__(self):
        super().__init__()
        self._W: Optional[np.ndarray] = None
        self._key = None

    def weight(self, d_in: int, ctx: FitContext) -> np.ndarray:
        d_out = int(ctx.d_h)
        seed = ctx.seeds[SEED_STAGE]
        key = (int(d_in), d_out, seed)
        if key != self._key:
            self._W, self._key = random_lift_matrix(d_in, d_out, seed), key
            self.stats.update(d_in=int(d_in), d_out=d_out, seed=seed)
        return self._W

    def fit(self, calib: Frames, ctx: FitContext) -> "RandomLift":
        self.weight(calib.dim, ctx)
        self.fitted = True
        return self

    def transform(self, fr: Frames, ctx: FitContext) -> Frames:
        return fr.with_X(fr.X @ self.weight(fr.dim, ctx))
