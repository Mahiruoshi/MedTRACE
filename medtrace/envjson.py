from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .seeds import STAGES

ENV_VAR = "MEDTRACE_ENV"

SCHEMA: Dict[str, Dict[str, type]] = {
    "seeds": {"inference": list, "split": int, "pool": list, "train": int, "stage_overrides": dict},
    "prompts": {"instruction": str, "biomarker_query": str, "question": str, "system": str, "user": str,
                "frame_line": str, "step_names": list, "answer_prefix": str, "answer_labels": list,
                "answer_aliases": dict},
}
INT_LISTS = ("seeds.inference", "seeds.pool")
STR_LISTS = ("prompts.step_names", "prompts.answer_labels")


class EnvNotFilled(RuntimeError):
    pass


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _read(source: str) -> Any:
    s = source.strip()
    if s.startswith(("{", "[")):
        return json.loads(s)
    p = Path(s).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"{ENV_VAR} names {s!r}, which is neither inline JSON nor a file")
    with open(p, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _check(env: Any) -> None:
    if not isinstance(env, dict):
        raise ValueError("the environment JSON must be an object with sections 'seeds' and 'prompts'")
    for sec, fields in SCHEMA.items():
        if not isinstance(env.get(sec), dict):
            raise ValueError(f"environment JSON: section {sec!r} missing")
        for k, typ in fields.items():
            if k not in env[sec]:
                raise ValueError(f"environment JSON: field {sec}.{k} missing (keep it, empty if unused)")
            v = env[sec][k]
            if v is None:
                continue
            if not isinstance(v, typ) or (typ is int and isinstance(v, bool)):
                raise TypeError(f"environment JSON: {sec}.{k} must be {typ.__name__} or empty, got {v!r}")
    for f in INT_LISTS:
        sec, k = f.split(".")
        if any(not _is_int(x) for x in env[sec][k] or ()):
            raise TypeError(f"environment JSON: {f} must be a list of integers")
    for f in STR_LISTS:
        sec, k = f.split(".")
        if any(not isinstance(x, str) or not x.strip() for x in env[sec][k] or ()):
            raise TypeError(f"environment JSON: {f} must be a list of non-empty strings")
    for lab, words in (env["prompts"]["answer_aliases"] or {}).items():
        if not isinstance(words, list) or any(not isinstance(x, str) or not x.strip() for x in words):
            raise TypeError(f"environment JSON: prompts.answer_aliases.{lab} must be a list of non-empty strings")
    for stage, v in (env["seeds"]["stage_overrides"] or {}).items():
        if stage not in STAGES:
            raise ValueError(f"environment JSON: seeds.stage_overrides.{stage}: unknown seed stage "
                             f"(stages: {', '.join(STAGES)})")
        if not _is_int(v):
            raise TypeError(f"environment JSON: seeds.stage_overrides.{stage} must be an integer, got {v!r}")


def load_env(source: Optional[str] = None) -> Dict[str, Any]:
    src = source if source is not None else os.environ.get(ENV_VAR)
    if not src:
        raise EnvNotFilled(f"set {ENV_VAR} to the environment JSON (a path or inline JSON); "
                           "template: env/medtrace.env.json")
    env = _read(src)
    _check(env)
    return copy.deepcopy(env)


def is_empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, (list, dict)) and len(v) == 0)


def require(env: Mapping[str, Any], *fields: str) -> None:
    empty = []
    for f in fields:
        sec, _, k = f.partition(".")
        if is_empty(env.get(sec, {}).get(k)):
            empty.append(f)
    if empty:
        raise EnvNotFilled(f"environment JSON ({ENV_VAR}): fill {', '.join(empty)} "
                           "(template: env/medtrace.env.json)")


def get(env: Mapping[str, Any], field: str) -> Any:
    require(env, field)
    sec, _, k = field.partition(".")
    return env[sec][k]


def seed(env: Mapping[str, Any], field: str, index: Optional[int] = None) -> int:
    v = get(env, field)
    if isinstance(v, list):
        if index is None:
            raise ValueError(f"{field} is a list; pass an index")
        if not 0 <= index < len(v):
            raise IndexError(f"{field}[{index}] out of range (have {len(v)})")
        v = v[index]
    if not _is_int(v):
        raise TypeError(f"{field}: seeds are integers, got {v!r}")
    return int(v)


def stage_overrides(env: Mapping[str, Any]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in (env["seeds"].get("stage_overrides") or {}).items()}
