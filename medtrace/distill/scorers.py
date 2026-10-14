from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..mappings import Chain, FitContext, Frames, cosine_rows, parse_chain
from ..registry import SCORERS


@dataclass
class ScoreResult:
    s: np.ndarray


ChainSpec = Union[str, Sequence, Chain, None]


class Scorer:
    name = "scorer"
    needs_calibration = False
    uses_chain = False
    param_names: Tuple[str, ...] = ()

    def __init__(self, chain: ChainSpec = None, **params):
        self.chain: Chain = chain if isinstance(chain, Chain) else parse_chain(chain or "identity")
        unknown = sorted(set(params) - set(self.param_names))
        if unknown:
            raise TypeError(f"scorer {self.name!r} takes no parameters {unknown} (accepts {list(self.param_names)})")
        if self.chain.stages and not self.uses_chain:
            raise ValueError(f"scorer {self.name!r} does not read a mapping chain; got {self.chain.spec!r}")
        self.params: Dict[str, Any] = dict(params)

    def fit(self, calib: Optional[Frames], ctx: FitContext) -> "Scorer":
        return self

    def score(self, fr: Frames, ctx: FitContext, pid: str) -> ScoreResult:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}


@SCORERS.register("probe_cosine")
class ProbeCosine(Scorer):
    name = "probe_cosine"
    uses_chain = True

    @property
    def needs_calibration(self) -> bool:
        return self.chain.needs_fit

    def fit(self, calib: Optional[Frames], ctx: FitContext) -> "ProbeCosine":
        if calib is None:
            if self.chain.needs_fit:
                raise ValueError(f"chain {self.chain.spec!r} has data-dependent stages; "
                                 f"fit it on a calibration set")
            return self
        self.chain.fit(calib, ctx)
        return self

    def score(self, fr: Frames, ctx: FitContext, pid: str) -> ScoreResult:
        out = self.chain(fr, ctx)
        return ScoreResult(cosine_rows(out.X, ctx.probe()))

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "chain": self.chain.spec}


@SCORERS.register("clip_query")
class ClipQueryScorer(Scorer):
    name = "clip_query"
    param_names = ("encoder",)

    def __init__(self, chain: ChainSpec = None, encoder: Optional[str] = None):
        super().__init__(chain, encoder=encoder)
        self.cache = None
        self.q: Optional[np.ndarray] = None
        self.query: Dict[str, Any] = {}

    def fit(self, calib: Optional[Frames], ctx: FitContext) -> "ClipQueryScorer":
        from pathlib import Path

        from ..features.extract_clip import DualEncoderCache
        from ..paths import get_paths

        text = getattr(ctx, "biomarker_query", "") or ""
        if not text.strip():
            raise ValueError("clip_query: FitContext.biomarker_query (Q_bio) is empty; fill "
                             "prompts.biomarker_query in the environment JSON and build the context from the Task")
        root = getattr(ctx, "feature_root", None) or Path(get_paths().cache_root) / "features"
        self.cache = DualEncoderCache.open(self.params["encoder"], root)
        self.query, self.q = self.cache.query(text)
        return self

    def score(self, fr: Frames, ctx: FitContext, pid: str) -> ScoreResult:
        if self.q is None:
            raise RuntimeError("clip_query: fit(calib, ctx) must run first (it opens the dual-encoder cache)")
        clip = self.cache.frames(pid)
        if clip.keys != list(fr.keys):
            raise ValueError(f"{pid}: dual-encoder cache {self.cache.tag} holds {clip.T} frame ids that differ from "
                             f"the {fr.T} being distilled; re-extract it for this image tree")
        return ScoreResult(cosine_rows(clip.X, self.q))

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        if self.cache is not None:
            d.update(encoder_tag=self.cache.tag, d_c=self.cache.d_c, **self.query)
        return d
