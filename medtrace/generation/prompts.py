from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..data.cohort import IMAGE_EXTS, frame_id
from ..data.tasks import Task
from .parsing import vocabulary_hits

PLACEHOLDERS: Tuple[str, ...] = ("question", "instruction", "modality", "steps", "answers", "answer_prefix",
                                 "pid", "age", "m", "t", "slice_lines")
FRAME_PLACEHOLDERS: Tuple[str, ...] = ("j", "i", "t")

PUBLIC_KEYS = frozenset({"pid", "age", "T", "split"})
_FORBIDDEN_KEYS = frozenset({"label", "survival_time", "event", "meta", "frames_dir", "frames",
                             "y", "target"})

Position = Tuple[int, int, int]

_FMT = string.Formatter()


class LeakError(ValueError):
    pass


class PromptError(ValueError):
    pass


def positions_for(selected: Sequence[int], T: int) -> List[Position]:
    return [(k + 1, int(i), int(T)) for k, i in enumerate(selected)]


def template_fields(template: str, name: str = "template",
                    allowed: Sequence[str] = PLACEHOLDERS) -> List[str]:
    try:
        parts = list(_FMT.parse(template))
    except ValueError as e:
        raise PromptError(f"prompts.{name}: {e} (write literal braces as {{{{ and }}}})") from None
    out: List[str] = []
    for _, field, spec, conversion in parts:
        if field is None:
            continue
        if field == "" or field.isdigit():
            raise PromptError(f"prompts.{name}: positional placeholder {{{field}}}; "
                              f"use one of {list(allowed)}")
        base = re.split(r"[.\[]", field, maxsplit=1)[0]
        if base not in allowed:
            raise PromptError(f"prompts.{name}: unknown placeholder {{{field}}}; "
                              f"allowed: {list(allowed)}")
        if field != base:
            raise PromptError(f"prompts.{name}: placeholder {{{field}}} uses attribute/index access; "
                              f"only plain names are allowed")
        if conversion is not None or spec:
            shown = field + (f"!{conversion}" if conversion is not None else "") + (f":{spec}" if spec else "")
            raise PromptError(f"prompts.{name}: placeholder {{{shown}}} has a conversion or format spec; "
                              f"only plain names are allowed, e.g. {{{field}}}")
        out.append(base)
    return out


def frame_line_fields(task: Task) -> List[str]:
    if not task.frame_line.strip():
        raise PromptError("prompts.frame_line is empty; {slice_lines} renders one frame line per shown frame "
                          "(e.g. \"[Slice {j}] position {i}/{t}\")")
    if "\n" in task.frame_line or "\r" in task.frame_line:
        raise PromptError("prompts.frame_line must be a single line ({slice_lines} joins the frame lines "
                          "with newlines)")
    return template_fields(task.frame_line, "frame_line", FRAME_PLACEHOLDERS)


def check_templates(task: Task) -> Tuple[List[str], List[str]]:
    system = template_fields(task.system, "system")
    user = template_fields(task.user, "user")
    if user.count("slice_lines") != 1:
        raise PromptError(f"prompts.user must place {{slice_lines}} (the frame lines of the shown frames) "
                          f"exactly once, it places it {user.count('slice_lines')} times")
    if "slice_lines" in system:
        raise PromptError("prompts.system must not place {slice_lines}: the frame lines belong in the user "
                          "turn, after the images")
    frame_line_fields(task)
    return system, user


ENV_FIELDS: Dict[str, str] = {"question": "prompts.question", "instruction": "prompts.instruction",
                              "steps": "prompts.step_names", "answers": "prompts.answer_labels",
                              "answer_prefix": "prompts.answer_prefix", "slice_lines": "prompts.frame_line"}


def required_prompt_fields(task: Task) -> Tuple[str, ...]:
    system, user = check_templates(task)
    used = set(system) | set(user)
    return tuple(ENV_FIELDS[k] for k in PLACEHOLDERS if k in used and k in ENV_FIELDS)


def render(template: str, fields: Mapping[str, Any], name: str = "template") -> str:
    template_fields(template, name)
    missing = sorted(set(PLACEHOLDERS) - set(fields))
    if missing:
        raise PromptError(f"prompt fields missing {missing}")
    return template.format(**{k: fields[k] for k in PLACEHOLDERS})


def steps_block(task: Task) -> str:
    return "\n".join(f"Step {k} - {name}" for k, name in enumerate(task.step_names, start=1))


def frame_lines(task: Task, positions: Sequence[Position]) -> List[str]:
    frame_line_fields(task)
    return [task.frame_line.format(j=rank, i=idx + 1, t=T) for rank, idx, T in positions]


def prompt_fields(pub: Mapping, task: Task, modality: str, positions: Sequence[Position]) -> Dict[str, Any]:
    pid, age, T = _public(pub)
    _check_positions(positions, T)
    return {"question": task.question, "instruction": task.instruction, "modality": str(modality or ""),
            "steps": steps_block(task), "answers": " / ".join(task.labels),
            "answer_prefix": task.answer_prefix, "pid": pid, "age": age, "m": len(positions), "t": T,
            "slice_lines": "\n".join(frame_lines(task, positions))}


def _public(pub: Mapping) -> Tuple[str, str, int]:
    bad = sorted(k for k in pub if k in _FORBIDDEN_KEYS or k not in PUBLIC_KEYS)
    if bad:
        raise LeakError(f"pub carries non-public keys {bad}; pass Patient.public()")
    for k in ("pid", "age", "T"):
        if k not in pub:
            raise KeyError(f"pub is missing {k!r}")
    return str(pub["pid"]), str(pub["age"]), int(pub["T"])


def _check_positions(positions: Sequence[Position], T: int) -> None:
    if not positions:
        raise ValueError("no slices selected")
    seen = set()
    for k, (rank, idx, t) in enumerate(positions):
        if rank != k + 1:
            raise ValueError(f"ranks must be 1..M' in fed order; got {rank} at position {k}")
        if t != T:
            raise ValueError(f"slice {rank}: T={t} but the patient has T={T}")
        if not 0 <= idx < T:
            raise ValueError(f"slice {rank}: index {idx} outside [0, {T})")
        if idx in seen:
            raise ValueError(f"slice index {idx} selected twice")
        seen.add(idx)


_EXT_RE = re.compile("|".join(re.escape(e) for e in IMAGE_EXTS), re.IGNORECASE)
_TASK_TEXTS = ("question", "instruction", "frame_line")
_LINE_END = re.compile(r"[ \t]*(?:\r?\n|\Z)")


def check_task_texts(task: Task, names: Sequence[str] = _TASK_TEXTS) -> None:
    for name in names:
        if name not in _TASK_TEXTS:
            raise ValueError(f"check_task_texts checks prompts.{' / prompts.'.join(_TASK_TEXTS)}, not {name!r}")
        hits = vocabulary_hits(getattr(task, name), task)
        if hits:
            raise LeakError(f"prompts.{name} names answer word(s) {hits}; keep the answer vocabulary out "
                            f"of it (the templates can list the options with {{answers}})")


def assert_label_free(text: str, task: Task) -> None:
    check_task_texts(task)
    if _EXT_RE.search(text):
        raise LeakError("prompt contains an image file extension (a slice file name?)")
    if "\\" in text:
        raise LeakError("prompt contains a backslash (a file path?)")


def frame_line_pattern(task: Task) -> "re.Pattern[str]":
    frame_line_fields(task)
    parts = []
    for literal, field, _, _ in _FMT.parse(task.frame_line):
        parts.append(re.escape(literal))
        if field is not None:
            parts.append(r"\d+")
    return re.compile("".join(parts))


def check_frame_lines(system: str, user: str, lines: Sequence[str], task: Task) -> Dict[str, int]:
    rx = frame_line_pattern(task)
    hint = "(prompts.frame_line must be distinctive enough that no other prompt text looks like a frame line)"
    m = rx.search(system)
    if m:
        raise LeakError(f"the system turn contains frame-line text {m.group(0)[:80]!r} {hint}")
    block = "\n".join(lines)
    if not lines or not block:
        raise LeakError("no frame lines generated (prompts.user must place {slice_lines})")
    n = user.count(block)
    if n != 1:
        raise LeakError(f"the user turn holds the generated frame lines {n} times, expected exactly once")
    at = user.index(block)
    before, after = user[:at], user[at + len(block):]
    if not _LINE_END.match(after):
        raise LeakError(f"text follows the last frame line on its line: {after.splitlines()[0][:80]!r}")
    for part in (before, after):
        m = rx.search(part)
        if m:
            raise LeakError(f"the user turn contains frame-line text outside the generated lines "
                            f"{m.group(0)[:80]!r} {hint}")
    return {"frame_lines": len(lines)}


def prompt_record(pub: Mapping, task: Task, modality: str, selected: Sequence[int]) -> Dict:
    check_templates(task)
    pid, _, T = _public(pub)
    positions = positions_for(selected, T)
    fields = prompt_fields(pub, task, modality, positions)
    system = render(task.system, fields, "system")
    user = render(task.user, fields, "user")
    assert_label_free(system, task)
    assert_label_free(user, task)
    check_frame_lines(system, user, frame_lines(task, positions), task)
    return {"system": system, "question": user,
            "frame_ids": [frame_id(pid, int(i)) for i in selected]}
