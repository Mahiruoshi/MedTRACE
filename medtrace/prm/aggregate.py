from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from ..registry import AGGREGATORS

KAPPA = 1e-6
AGGREGATION = "log_sum"


def _as_probs(p: Iterable[float]) -> np.ndarray:
    a = np.asarray(p if isinstance(p, (np.ndarray, list, tuple)) else list(p), dtype=np.float64).reshape(-1)
    if a.size == 0:
        raise ValueError("no step scores to aggregate (a scored trajectory has at least one step)")
    return a


def clip_probs(p: Iterable[float], kappa: float = KAPPA) -> np.ndarray:
    return np.clip(_as_probs(p), kappa, 1.0 - kappa)


def log_odds(p: Iterable[float], kappa: float = KAPPA) -> np.ndarray:
    c = clip_probs(p, kappa)
    return np.log(c) - np.log1p(-c)


@AGGREGATORS.register(AGGREGATION)
def agg_log_sum(p) -> float:
    return float(log_odds(p).sum())


def aggregate(p) -> float:
    return float(AGGREGATORS.get(AGGREGATION)(p))


def score_record(cid: str, pid: str, cot_idx: int, step_scores: Sequence[float],
                 pred: Optional[int], answer: Optional[str], *, truncated: Optional[bool] = None
                 ) -> Dict[str, Any]:
    ps = [float(x) for x in step_scores]
    return {"id": cid, "pid": pid, "cot_idx": int(cot_idx), "step_scores": ps,
            "scores": {AGGREGATION: aggregate(ps)}, "pred": pred, "answer": answer,
            "meta": {"truncated": None if truncated is None else bool(truncated)}}


def _field(c: Any, key: str) -> Any:
    try:
        return c[key] if isinstance(c, Mapping) else getattr(c, key)
    except (KeyError, AttributeError):
        raise KeyError(f"scored candidate has no {key!r} (a scores.json entry carries id, pid, cot_idx, "
                       f"step_scores, scores, pred, answer, meta)") from None


def candidate_score(c: Any) -> float:
    stored = _field(c, "scores")
    if AGGREGATION not in stored:
        raise KeyError(f"scored candidate has no scores[{AGGREGATION!r}]")
    return float(stored[AGGREGATION])


def is_answerable(c: Any) -> bool:
    return _field(c, "pred") is not None


def _rank_key(A: float, cot_idx: int):
    return (-math.inf if A != A else A, -int(cot_idx))


def best_of_n(scored: Sequence[Any]) -> Optional[Any]:
    pool = list(scored)
    if not pool:
        return None
    pool = [c for c in pool if is_answerable(c)] or pool
    return max(pool, key=lambda c: _rank_key(candidate_score(c), _field(c, "cot_idx")))
