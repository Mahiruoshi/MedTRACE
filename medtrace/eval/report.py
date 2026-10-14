from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from ..data.tasks import Task


def _f(x: Optional[float], nd: int = 3) -> str:
    return "--" if x is None else f"{x:.{nd}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    w = [max([len(str(h))] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(w[0]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(cells)).rstrip()

    return "\n".join([line(headers), line(["-" * x for x in w])] + [line(r) for r in rows])


def _within(r: Mapping[str, Any], k: int) -> Optional[float]:
    w = r.get("auc_within") or {}
    return w.get(k, w.get(str(k)))


def _preamble(m: Mapping[str, Any]) -> List[str]:
    out = []
    if m["skipped_n"]:
        out.append("skipped N: " + "; ".join(f"N={k}: {v}" for k, v in m["skipped_n"].items()))
    if m["patients_without_candidates"] or m["unlabelled_candidates"]:
        out.append(f"WARNING: {len(m['patients_without_candidates'])} labelled patients without candidates, "
                   f"{m['unlabelled_candidates']} candidates without a label")
    return out


def _pool_line(p: Mapping[str, Any], answers: str) -> str:
    trunc = p["truncated"] if p["truncated"] is not None else "n/a"
    undet = (f", undetermined {p['undetermined']} ({p['oracle_undetermined']} patients' oracle)"
             if "undetermined" in p else "")
    return (f"pool: {p['candidates']} candidates, unparseable {p['unparseable']}, truncated {trunc}{undet} | "
            f"{answers} | pass@1 mean {_f(p['pass1_mean'])} | oracle {_f(p['oracle'])}")


def _pairs(r: Mapping[str, Any]) -> List[str]:
    return [_f(r["pairwise_micro"]), _f(r["pairwise_macro"]), f"{r['pairs']}/{r['patients_with_both']}pt"]


_PANEL = ("acc", "bal_acc", "sensitivity", "specificity", "precision", "f1", "mcc", "picks_pos")


def render_summary(metrics: Mapping[str, Any], task: Task) -> str:
    kind = metrics["meta"].get("task", task.kind)
    if kind != task.kind:
        raise ValueError(f"metrics of a {kind!r} task rendered with a {task.kind!r} task")
    return _render_binary(metrics, task) if task.kind == "binary" else _render_survival(metrics, task)


def _render_binary(metrics: Mapping[str, Any], task: Task) -> str:
    m = metrics["meta"]
    neg, pos = task.labels[0], task.labels[1]
    out = [f"patients {m['patients']} ({m['n_pos']}+/{m['n_neg']}-), prevalence {m['prevalence']:.3f} | "
           f"A = {m['aggregation']} (eq:agg_def)"]
    out += _preamble(m)
    for n in m["evaluated_n"]:
        b = metrics["by_n"][str(n)]
        p = b["pool"]
        share = p["answer_share"]
        out += ["", f"=== N={n} ===",
                _pool_line(p, f"{pos}-share {_f(share[1] if share else None)} vs prevalence {_f(m['prevalence'])}")]
        rows = [[name] + [_f(s[k]) for k in _PANEL] + [str(s["unresolved"])]
                for name, s in b["strategies"].items()]
        out.append(_table(["strategy", "acc", "bal_acc", "sens", "spec", "prec", "f1", "mcc", "picks+", "unres"],
                          rows))
        pa = b["patient_auc"]
        out.append(f"patient AUC (n={pa['n']}): {pos}-share {_f(pa['yes_share'])} | PRM-softmax-{pos} "
                   f"{_f(pa['prm_softmax_yes'])}")
        r = b["ranking"]
        out.append("trajectory ranking of A (correct vs incorrect):")
        out.append(_table(["pooled", f"in {pos}", f"in {neg}", f"polarity({neg})", "pair micro", "pair macro",
                           "pairs"],
                          [[_f(r["pooled_auc"]), _f(r["auc_within_yes"]), _f(r["auc_within_no"]),
                            _f(r["polarity_auc"])] + _pairs(r)]))
    return "\n".join(out)


def _render_survival(metrics: Mapping[str, Any], task: Task) -> str:
    m = metrics["meta"]
    labels = list(task.labels)
    out = [f"patients {m['patients']} ({m['n_events']} events/{m['n_censored']} censored), stages "
           f"{' / '.join(labels)} ({'/'.join(str(c) for c in m['per_bin'])} patients) | "
           f"A = {m['aggregation']} (eq:agg_def)",
           f"{m['n_undetermined']} patients censored before the last stage: their answers at or after the "
           f"censoring stage are undetermined, left out of acc, oracle, pass@1 mean and the rankings "
           f"(the C-indices use every patient)"]
    out += _preamble(m)
    for n in m["evaluated_n"]:
        b = metrics["by_n"][str(n)]
        p = b["pool"]
        share = p["answer_share"]
        answers = ", ".join(f"{lab} {_f(x)}" for lab, x in zip(labels, share)) if share else "--"
        out += ["", f"=== N={n} ===", _pool_line(p, f"answers {answers}")]
        out.append(_table(["strategy", "C-index", "acc", "unres", "undet"],
                          [[name, _f(s["c_index"]), _f(s["acc"]), str(s["unresolved"]), str(s["undetermined"])]
                           for name, s in b["strategies"].items()]))
        pr = b["patient_risk"]
        out.append(f"patient risk C-index (n={pr['n']}): mean risk {_f(pr['mean_risk'])} | PRM-expected risk "
                   f"{_f(pr['prm_expected_risk'])}")
        r = b["ranking"]
        out.append("trajectory ranking of A (correct vs incorrect):")
        out.append(_table(["pooled"] + [f"in {lab}" for lab in labels]
                          + ["within mean", "pair micro", "pair macro", "pairs", "undet"],
                          [[_f(r["pooled_auc"])] + [_f(_within(r, k)) for k in range(task.K)]
                           + [_f(r["within_mean"])] + _pairs(r) + [str(r["undetermined"])]]))
    return "\n".join(out)
