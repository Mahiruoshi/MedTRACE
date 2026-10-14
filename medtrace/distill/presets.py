from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Dict, Optional, Union

from ..config import deep_merge
from ..registry import SCORERS, SELECTORS

VARIANT_KEYS = ("name", "base", "scorer", "selector", "note")


def _v(chain: str, note: str, scorer: str = "probe_cosine", selector: str = "topk") -> Dict[str, Any]:
    return {"scorer": {"name": scorer}, "chain": chain, "selector": {"name": selector}, "note": note}


PRESETS: Dict[str, Dict[str, Any]] = {
    "ot_embed": _v("rnd>ot", "OT-Embed: s = cos(q, T_OT(Pi_rnd(F))), D_text = instr U pca"),
    "ada": _v("identity", "training-time keyframe sampling: dual-encoder relevance cos(g^V(I_t), g^T(Q_bio)) "
              "+ adaptive keyframe sampling", scorer="clip_query", selector="ada"),
}


def _component(value: Union[str, Mapping, None], kind: str) -> Dict[str, Any]:
    if isinstance(value, str):
        return {"name": value}
    if isinstance(value, Mapping) and "name" in value:
        return dict(value)
    raise ValueError(f"variant {kind} must be a name or a mapping with 'name', got {value!r}")


def _override(value: Union[str, Mapping, None], base: Dict[str, Any], kind: str, variant: str) -> Dict[str, Any]:
    if isinstance(value, str):
        value = {"name": value}
    if not isinstance(value, Mapping):
        raise ValueError(f"variant {variant!r}: {kind} must be a mapping of parameters, got {value!r}")
    value = dict(value)
    given = value.pop("name", base["name"])
    if str(given).lower() != base["name"]:
        raise ValueError(f"variant {variant!r}: {kind} {given!r} is not the base preset's {kind} "
                         f"{base['name']!r}; an inline variant only overrides its parameters")
    return deep_merge(base, value)


def resolve_variant(spec: Union[str, Mapping], name: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(spec, str):
        key = spec.lower()
        if key not in PRESETS:
            raise KeyError(f"unknown variant {spec!r}; presets: {', '.join(sorted(PRESETS))} "
                           f"(or define one inline on a base preset)")
        out = copy.deepcopy(PRESETS[key])
        out["name"] = name or key
    else:
        d = dict(spec)
        unknown = sorted(set(d) - set(VARIANT_KEYS))
        if unknown:
            raise KeyError(f"unknown variant keys {unknown}; allowed: {list(VARIANT_KEYS)}")
        base = d.pop("base", None)
        if not base:
            raise ValueError(f"an inline variant needs a 'base' preset ({', '.join(sorted(PRESETS))}), "
                             f"got {spec!r}")
        out = resolve_variant(base)
        label = name or d.get("name") or base
        for k in ("scorer", "selector"):
            if k in d:
                out[k] = _override(d.pop(k), out[k], k, label)
        if "note" in d:
            out["note"] = d["note"]
        out["name"] = label
    out["scorer"] = _component(out["scorer"], "scorer")
    out["selector"] = _component(out["selector"], "selector")
    out.setdefault("note", "")
    SCORERS.resolve(out["scorer"]["name"])
    SELECTORS.resolve(out["selector"]["name"])
    return out
