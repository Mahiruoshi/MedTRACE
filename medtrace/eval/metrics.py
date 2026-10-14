from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

from ..data.tasks import SurvivalTarget, Task
from ..prm.aggregate import AGGREGATION, KAPPA, best_of_n

N_LIST = (5, 8, 12)


def auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel()
    if s.shape != y.shape:
        raise ValueError(f"{s.size} scores for {y.size} labels")
    if not np.all(np.isfinite(s)):
        raise ValueError("auc: non-finite score")
    pos = y == 1
    n1 = int(pos.sum())
    n0 = int((y == 0).sum())
    if n1 + n0 != y.size:
        raise ValueError("auc: labels must be 0/1")
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def c_index(times: Sequence[float], events: Sequence[int], risks: Sequence[float]) -> Optional[float]:
    t = np.asarray(times, dtype=np.float64).ravel()
    e = np.asarray(events).ravel()
    r = np.asarray(risks, dtype=np.float64).ravel()
    if not t.shape == e.shape == r.shape:
        raise ValueError(f"c_index: {t.size} times, {e.size} events, {r.size} risks")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(r))):
        raise ValueError("c_index: non-finite time or risk")
    if not np.all((e == 0) | (e == 1)):
        raise ValueError("c_index: events must be 0/1")
    later = (t[:, None] < t[None, :]) | ((t[:, None] == t[None, :]) & (e[None, :] == 0))
    comp = later & (e[:, None] == 1)
    n = int(comp.sum())
    if n == 0:
        return None
    conc = int((comp & (r[:, None] > r[None, :])).sum())
    tied = int((comp & (r[:, None] == r[None, :])).sum())
    return float((conc + 0.5 * tied) / n)


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[Optional[int]]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64)
    if y.size != len(y_pred):
        raise ValueError(f"{y.size} labels for {len(y_pred)} predictions")
    res = np.array([p is not None for p in y_pred], dtype=bool)
    p = np.array([-1 if v is None else int(v) for v in y_pred], dtype=np.int64)
    tp = int(np.sum(res & (y == 1) & (p == 1)))
    tn = int(np.sum(res & (y == 0) & (p == 0)))
    fp = int(np.sum(res & (y == 0) & (p == 1)))
    fn = int(np.sum(res & (y == 1) & (p == 0)))
    u_pos = int(np.sum(~res & (y == 1)))
    u_neg = int(np.sum(~res & (y == 0)))
    n = int(y.size)
    P, N = tp + fn + u_pos, tn + fp + u_neg
    sens = tp / P if P else None
    spec = tn / N if N else None
    prec = tp / (tp + fp) if tp + fp else None
    f1_den = 2 * tp + fp + fn + u_pos
    FP, FN = fp + u_neg, fn + u_pos
    mcc_den = float(np.sqrt(float(tp + FP) * (tp + FN) * (tn + FP) * (tn + FN)))
    n_res = n - u_pos - u_neg
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "unresolved": u_pos + u_neg,
        "acc": (tp + tn) / n if n else None,
        "bal_acc": (sens + spec) / 2 if sens is not None and spec is not None else None,
        "sensitivity": sens, "specificity": spec, "precision": prec,
        "f1": 2 * tp / f1_den if f1_den else None,
        "mcc": (tp * tn - FP * FN) / mcc_den if mcc_den > 0 else 0.0,
        "picks_pos": (tp + fp) / n_res if n_res else None,
    }


def mid_risk(task: Task) -> float:
    return (task.K - 1) / 2.0


def survival_metrics(targets: Sequence[SurvivalTarget], preds: Sequence[Optional[int]], task: Task
                     ) -> Dict[str, Any]:
    if len(targets) != len(preds):
        raise ValueError(f"{len(targets)} targets for {len(preds)} predictions")
    n = len(preds)
    res = [int(p) for p in preds if p is not None]
    mid = mid_risk(task)
    risks = [mid if p is None else task.risk(p) for p in preds]
    decided = [v for v in (task.is_correct(p, t) for p, t in zip(preds, targets)) if v is not None]
    return {
        "n": n, "unresolved": n - len(res), "undetermined": n - len(decided),
        "c_index": c_index([t.time for t in targets], [t.event for t in targets], risks),
        "acc": sum(decided) / len(decided) if decided else None,
        "answer_share": [sum(1 for p in res if p == k) / len(res) for k in range(task.K)] if res else None,
    }


def _softmax(a: np.ndarray) -> np.ndarray:
    e = np.exp(a - a.max())
    return e / e.sum()


def _mean(x: Sequence[float]) -> Optional[float]:
    return float(np.mean(x)) if len(x) else None


@dataclass(frozen=True)
class Cand:
    pid: str
    cot_idx: int
    pred: Optional[int]
    step_scores: Tuple[float, ...]
    scores: Mapping[str, float]
    correct: Optional[int]
    truncated: Optional[bool]


def _check_pred(pred: Any, K: int, who: Any) -> Optional[int]:
    if pred is None:
        return None
    try:
        v = int(pred)
    except (TypeError, ValueError):
        v = None
    if v is None or v != pred or not 0 <= v < K:
        raise ValueError(f"{who}: pred must be an answer index in 0..{K - 1} or None, got {pred!r}")
    return v


def _entry(d: Mapping[str, Any], K: int) -> Dict[str, Any]:
    who = d.get("id", d.get("pid"))
    for k in ("pid", "cot_idx", "pred", "step_scores", "scores", "meta"):
        if k not in d:
            raise KeyError(f"{who}: scored entry without {k!r} (written by aggregate.score_record)")
    pred = _check_pred(d["pred"], K, who)
    steps = tuple(float(x) for x in d["step_scores"])
    if not steps:
        raise ValueError(f"{who}: empty step_scores (a scored trajectory has >= 1 step)")
    if AGGREGATION not in d["scores"]:
        raise KeyError(f"{who}: scored entry without scores[{AGGREGATION!r}]")
    meta = d["meta"]
    if not isinstance(meta, Mapping) or "truncated" not in meta:
        raise KeyError(f"{who}: scored entry without meta.truncated")
    trunc = meta["truncated"]
    return dict(pid=str(d["pid"]), cot_idx=int(d["cot_idx"]), pred=pred, step_scores=steps,
                scores={k: float(v) for k, v in d["scores"].items()},
                truncated=None if trunc is None else bool(trunc))


def group_candidates(scored: Sequence[Mapping[str, Any]], targets: Mapping[str, Any], task: Task
                     ) -> Tuple[Dict[str, List[Cand]], int]:
    groups: Dict[str, List[Cand]] = {}
    checked: Dict[str, Any] = {}
    dropped = 0
    for d in scored:
        f = _entry(d, task.K)
        pid = f["pid"]
        if pid not in targets:
            dropped += 1
            continue
        if pid not in checked:
            t = targets[pid]
            checked[pid] = as_survival_target(t, task.K, pid) if task.kind == "survival" else t
        groups.setdefault(pid, []).append(Cand(**f, correct=task.is_correct(f["pred"], checked[pid])))
    for pid, cs in groups.items():
        cs.sort(key=lambda c: c.cot_idx)
        idx = [c.cot_idx for c in cs]
        if len(set(idx)) != len(idx):
            raise ValueError(f"{pid}: duplicated cot_idx in the scored pool")
    return {p: groups[p] for p in sorted(groups)}, dropped


Selection = Tuple[Optional[int], Optional[int]]


def sel_bon(cands: Sequence[Cand]) -> Selection:
    c = best_of_n(cands)
    return c.cot_idx, c.pred


def sel_majority(cands: Sequence[Cand]) -> Selection:
    tally: Dict[int, int] = {}
    for c in cands:
        if c.pred is not None:
            tally[c.pred] = tally.get(c.pred, 0) + 1
    if not tally:
        return None, None
    best = max(tally.values())
    c = next(c for c in cands if c.pred is not None and tally[c.pred] == best)
    return c.cot_idx, c.pred


def sel_pass1(cands: Sequence[Cand]) -> Selection:
    return cands[0].cot_idx, cands[0].pred


def sel_oracle(cands: Sequence[Cand]) -> Selection:
    c = next((c for c in cands if c.correct == 1), None)
    if c is None:
        c = next((c for c in cands if c.correct is None), cands[0])
    return c.cot_idx, c.pred


STRATEGIES: Dict[str, Callable[[Sequence[Cand]], Selection]] = {
    "prm_bon": sel_bon, "majority_vote": sel_majority, "pass1": sel_pass1, "oracle": sel_oracle}


def evaluate_strategy(groups: Mapping[str, Sequence[Cand]], targets: Mapping[str, Any], task: Task,
                      fn: Callable[[Sequence[Cand]], Selection]) -> Dict[str, Any]:
    pids = sorted(groups)
    sel = {pid: fn(groups[pid]) for pid in pids}
    y = [targets[p] for p in pids]
    pred = [sel[p][1] for p in pids]
    out = binary_metrics(y, pred) if task.kind == "binary" else survival_metrics(y, pred, task)
    out["selected"] = {p: sel[p][0] for p in pids}
    out["pred"] = {p: sel[p][1] for p in pids}
    return out


def _oracle(cs: Sequence[Cand]) -> Optional[int]:
    if any(c.correct == 1 for c in cs):
        return 1
    return None if any(c.correct is None for c in cs) else 0


def pool_health(groups: Mapping[str, Sequence[Cand]], task: Task) -> Dict[str, Any]:
    allc = [c for cs in groups.values() for c in cs]
    parse = [c for c in allc if c.pred is not None]
    trunc = [c.truncated for c in allc if c.truncated is not None]
    decided = [float(c.correct) for c in allc if c.correct is not None]
    oracle = [v for v in (_oracle(cs) for cs in groups.values()) if v is not None]
    out = {
        "candidates": len(allc),
        "unparseable": len(allc) - len(parse),
        "truncated": sum(trunc) if trunc else None,
        "answer_share": [sum(1 for c in parse if c.pred == k) / len(parse) for k in range(task.K)]
        if parse else None,
        "pass1_mean": _mean(decided),
        "oracle": sum(oracle) / len(oracle) if oracle else None,
    }
    if task.kind == "survival":
        out["undetermined"] = len(allc) - len(decided)
        out["oracle_undetermined"] = len(groups) - len(oracle)
    return out


def patient_scores(cands: Sequence[Cand]) -> Optional[Dict[str, float]]:
    parse = [c for c in cands if c.pred is not None]
    if not parse:
        return None
    y = np.array([c.pred for c in parse], dtype=np.float64)
    w = _softmax(np.array([c.scores[AGGREGATION] for c in parse], dtype=np.float64))
    return {"yes_share": float(y.mean()), "prm_softmax_yes": float(w @ y)}


def patient_auc(groups: Mapping[str, Sequence[Cand]], labels: Mapping[str, int]) -> Dict[str, Any]:
    sc = {p: patient_scores(cs) for p, cs in sorted(groups.items())}
    kept = [p for p, s in sc.items() if s is not None]
    y = [labels[p] for p in kept]
    out: Dict[str, Any] = {"n": len(kept)}
    for k in ("yes_share", "prm_softmax_yes"):
        out[k] = auc([sc[p][k] for p in kept], y)
    out["scores"] = {p: sc[p] for p in kept}
    return out


def patient_risk_scores(cands: Sequence[Cand], task: Task) -> Optional[Dict[str, float]]:
    parse = [c for c in cands if c.pred is not None]
    if not parse:
        return None
    r = np.array([task.risk(c.pred) for c in parse], dtype=np.float64)
    w = _softmax(np.array([c.scores[AGGREGATION] for c in parse], dtype=np.float64))
    return {"mean_risk": float(r.mean()), "prm_expected_risk": float(w @ r)}


def patient_risk(groups: Mapping[str, Sequence[Cand]], targets: Mapping[str, SurvivalTarget], task: Task
                 ) -> Dict[str, Any]:
    sc = {p: patient_risk_scores(cs, task) for p, cs in sorted(groups.items())}
    kept = [p for p, s in sc.items() if s is not None]
    t = [targets[p].time for p in kept]
    e = [targets[p].event for p in kept]
    out: Dict[str, Any] = {"n": len(kept)}
    for k in ("mean_risk", "prm_expected_risk"):
        out[k] = c_index(t, e, [sc[p][k] for p in kept])
    out["scores"] = {p: sc[p] for p in kept}
    return out


def trajectory_ranking(groups: Mapping[str, Sequence[Cand]], task: Task) -> Dict[str, Any]:
    A, ok, ans = [], [], []
    pair_auc, pair_n = [], []
    undetermined = 0
    for p, cs in groups.items():
        parse = [c for c in cs if c.pred is not None]
        ranked = [c for c in parse if c.correct is not None]
        undetermined += len(parse) - len(ranked)
        a = [c.scores[AGGREGATION] for c in ranked]
        o = [int(c.correct) for c in ranked]
        A += a
        ok += o
        ans += [int(c.pred) for c in ranked]
        v = auc(a, o)
        if v is not None:
            n1 = sum(o)
            pair_auc.append(v)
            pair_n.append(n1 * (len(o) - n1))
    A_, ok_, ans_ = np.array(A, dtype=np.float64), np.array(ok, dtype=np.int64), np.array(ans, dtype=np.int64)
    within = {int(k): auc(A_[ans_ == k], ok_[ans_ == k]) for k in range(task.K)}
    defined = [v for v in within.values() if v is not None]
    binary = task.kind == "binary"
    pairs = int(sum(pair_n))
    out = {
        "n": len(A),
        "pooled_auc": auc(A_, ok_),
        "auc_within": within,
        "within_mean": sum(defined) / len(defined) if defined else None,
        "auc_within_yes": within.get(1) if binary else None,
        "auc_within_no": within.get(0) if binary else None,
        "polarity_auc": auc(A_, (ans_ == 0).astype(np.int64)) if binary else None,
        "pairs": pairs,
        "pairwise_micro": float(np.dot(pair_auc, pair_n) / pairs) if pairs else None,
        "pairwise_macro": _mean(pair_auc),
        "patients_with_both": len(pair_auc),
    }
    if not binary:
        out["undetermined"] = undetermined
    return out


def evaluate_pool(groups: Mapping[str, Sequence[Cand]], targets: Mapping[str, Any], task: Task
                  ) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pool": pool_health(groups, task),
        "strategies": {name: evaluate_strategy(groups, targets, task, fn) for name, fn in STRATEGIES.items()},
    }
    if task.kind == "binary":
        out["patient_auc"] = patient_auc(groups, targets)
    else:
        out["patient_risk"] = patient_risk(groups, targets, task)
    out["ranking"] = trajectory_ranking(groups, task)
    return out


def as_survival_target(t: Any, K: int, who: Any = "") -> SurvivalTarget:
    if not isinstance(t, SurvivalTarget):
        raise ValueError(f"{who}: a survival target is SurvivalTarget(time, event, bin), got {t!r}")
    if not np.isfinite(t.time) or t.time < 0 or t.event not in (0, 1) or not 0 <= int(t.bin) < K:
        raise ValueError(f"{who}: invalid survival target {t} for {K} stages")
    return t


def evaluate(scored: Sequence[Mapping[str, Any]], targets: Mapping[str, Any], task: Task,
             n_list: Sequence[int] = N_LIST) -> Dict[str, Any]:
    if any(int(n) != n or n < 1 for n in n_list) or len(set(n_list)) != len(n_list):
        raise ValueError(f"n_list must be distinct positive integers, got {list(n_list)}")
    binary = task.kind == "binary"
    if binary:
        bad = sorted(str(p) for p, y in targets.items() if y not in (0, 1))
        if bad:
            raise ValueError(f"binary targets must be 0/1; bad for {bad}")
        targets = {str(p): int(y) for p, y in targets.items()}
    else:
        targets = {str(p): as_survival_target(t, task.K, p) for p, t in targets.items()}
    groups, dropped = group_candidates(scored, targets, task)
    if not groups:
        raise ValueError("no scored candidate belongs to a patient with a target")
    tg = {p: targets[p] for p in groups}
    available = min(len(cs) for cs in groups.values())
    meta: Dict[str, Any] = {
        "task": task.kind, "answer_labels": list(task.labels),
        "n_list": [int(n) for n in n_list], "aggregation": AGGREGATION, "kappa": KAPPA,
        "patients": len(groups),
    }
    if binary:
        meta.update({"n_pos": sum(tg.values()), "n_neg": len(tg) - sum(tg.values()),
                     "prevalence": sum(tg.values()) / len(tg)})
    else:
        n_ev = sum(t.event for t in tg.values())
        meta.update({"n_events": n_ev, "n_censored": len(tg) - n_ev,
                     "n_undetermined": sum(1 for t in tg.values() if not task.determined(t)),
                     "per_bin": [sum(1 for t in tg.values() if t.bin == k) for k in range(task.K)]})
    meta.update({
        "min_candidates": available,
        "unlabelled_candidates": dropped,
        "patients_without_candidates": sorted(set(targets) - set(groups)),
        "evaluated_n": [], "skipped_n": {},
    })
    by_n: Dict[str, Any] = {}
    for n in n_list:
        if n > available:
            meta["skipped_n"][str(n)] = f"a patient has only {available} candidates"
            continue
        sub = {p: cs[:n] for p, cs in groups.items()}
        by_n[str(n)] = evaluate_pool(sub, tg, task)
        meta["evaluated_n"].append(int(n))
    out: Dict[str, Any] = {"meta": meta}
    if binary:
        out["labels"] = tg
    else:
        out["targets"] = {p: {"time": t.time, "event": t.event, "bin": t.bin} for p, t in tg.items()}
    out["by_n"] = by_n
    return out
