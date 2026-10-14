from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..data.tasks import ANSWER_SEPARATORS, Task

_STEP_RE = re.compile(
    r"^\s*Step\s*(\d+)\s*[-:.–]\s*(.*)$", re.IGNORECASE | re.MULTILINE)

_MD_CHARS = r"[*_`\[\]()#>]"

_WORD_SEP = r"(?:[^\w\r\n]|_)+"

_LEAD = re.compile(r"[\W\d_]*")
_VERDICT_END = r"\s*(?:[-–:,.：，。]|$)"

_JOIN = r"(?:\s*(?:/|\bor\b|\bto\b)\s*|[-–])"

PAD_MARK = "(not produced)"


class _Hedge:
    def __repr__(self) -> str:
        return "<hedge>"


_HEDGE = _Hedge()
_Verdict = Union[str, _Hedge, None]


@dataclass(frozen=True)
class _Grammar:
    labels: Tuple[str, ...]
    groups: Dict[str, Tuple[str, int]]
    item: "re.Pattern[str]"
    bare_list: "re.Pattern[str]"
    bracket_list: "re.Pattern[str]"
    decisive: "re.Pattern[str]"
    hedge_right: Optional["re.Pattern[str]"]
    hedge_left: Optional["re.Pattern[str]"]
    answer_line: "re.Pattern[str]"
    answer_is: "re.Pattern[str]"

    def index(self, m: "re.Match[str]") -> int:
        return self.groups[m.lastgroup][1]


_GRAMMARS: Dict[tuple, _Grammar] = {}
_VOCABS: Dict[tuple, Tuple["re.Pattern[str]", Dict[str, Tuple[str, int]]]] = {}


def _phrase_re(phrase: str) -> str:
    return r"\b" + _WORD_SEP.join(re.escape(w) for w in phrase.split()) + r"\b"


def _entries(task: Task, phrases: Sequence[str]) -> List[Tuple[str, int]]:
    out = []
    for p in phrases:
        i = task.parse_label(p)
        if i is None:
            raise ValueError(f"answer phrase {p!r} does not map back to a label of the task")
        out.append((p, int(i)))
    return out


def _named(prefix: str, entries: Sequence[Tuple[str, int]], groups: Dict[str, Tuple[str, int]]) -> str:
    alts = []
    for j, (p, i) in enumerate(entries):
        groups[f"{prefix}{j}"] = (p, i)
        alts.append(f"(?P<{prefix}{j}>{_phrase_re(p)})")
    return "(?:" + "|".join(alts) + ")"


def _stages(task: Task) -> List[Tuple[str, int]]:
    by_label = {i: p for p, i in _entries(task, task.label_words())}
    words = [by_label[i].split() for i in range(task.K)]
    pre = 0
    while all(len(w) > pre for w in words) and len({w[pre] for w in words}) == 1:
        pre += 1
    suf = 0
    while all(len(w) > pre + suf for w in words) and len({w[len(w) - 1 - suf] for w in words}) == 1:
        suf += 1
    full = [" ".join(w) for w in words]
    cores = [" ".join(w[pre:len(w) - suf]) for w in words]
    out = [(p, i) for i, p in enumerate(full)]
    out += [(c, i) for i, c in enumerate(cores) if c and c not in full and cores.count(c) == 1]
    return sorted(out, key=lambda e: (-len(e[0]), e[0]))


def _prefix_parts(prefix: str) -> Tuple[str, str]:
    p = (prefix or "").strip()
    m = re.match(r"^(.*?\w)\W*$", p, re.DOTALL)
    if not m:
        raise ValueError(f"task.answer_prefix {prefix!r} must contain a word (a word and a separator, "
                         f"such as a colon); fill prompts.answer_prefix in the environment JSON")
    stem = m.group(1)
    return stem, "".join(c for c in p[m.end(1):] if not c.isspace())


def _build(task: Task) -> _Grammar:
    groups: Dict[str, Tuple[str, int]] = {}
    item = _named("l", _entries(task, task.label_words()), groups)
    words = _named("a", _entries(task, task.vocabulary()), groups)
    plain = "(?:" + "|".join(_phrase_re(p) for p in task.label_words()) + ")"
    sep = r"\s*(?:/|\bor\b)\s*" if task.K == 2 else r"\s*(?:/|,\s*(?:or\b)?|\bor\b)\s*"
    lst = plain + f"(?:{sep}{plain}){{{task.K - 1}}}"
    stem, own = _prefix_parts(task.answer_prefix)
    stem_re = r"\s+".join(re.escape(w) for w in stem.split())
    sep_cls = "".join(re.escape(c) for c in sorted(set("".join(ANSWER_SEPARATORS)) | set(own)))
    flags = re.IGNORECASE
    hedge_right = hedge_left = None
    if task.K > 2:
        stage = _named("h", _stages(task), groups)
        hedge_right = re.compile(_JOIN + stage, flags)
        hedge_left = re.compile(stage + _JOIN + r"\Z", flags)
    return _Grammar(
        labels=tuple(task.labels), groups=groups,
        item=re.compile(item, flags),
        bare_list=re.compile(lst, flags),
        bracket_list=re.compile(r"[\[\(<]\s*" + lst + r"\s*[\]\)>]", flags),
        decisive=re.compile(words + _VERDICT_END, flags),
        hedge_right=hedge_right, hedge_left=hedge_left,
        answer_line=re.compile(
            rf"^[ \t>*_#]*(?:Step\s*\d+\s*[-:.–]\s*)?\*{{0,2}}{stem_re}\*{{0,2}}\s*[{sep_cls}]\s*(.*)$",
            flags | re.MULTILINE),
        answer_is=re.compile(rf"(?:the\s+)?{stem_re}\s+is\s+(.+)", flags))


def _answer_words_key(task: Task) -> tuple:
    return tuple(task.labels), tuple(sorted((k, tuple(v)) for k, v in task.aliases.items()))


def _grammar(task: Task) -> _Grammar:
    key = (*_answer_words_key(task), task.answer_prefix)
    g = _GRAMMARS.get(key)
    if g is None:
        g = _GRAMMARS[key] = _build(task)
    return g


def _vocab(task: Task) -> Tuple["re.Pattern[str]", Dict[str, Tuple[str, int]]]:
    key = _answer_words_key(task)
    v = _VOCABS.get(key)
    if v is None:
        groups: Dict[str, Tuple[str, int]] = {}
        rx = re.compile(_named("a", _entries(task, task.vocabulary()), groups), re.IGNORECASE)
        v = _VOCABS[key] = (rx, groups)
    return v


def _covers_all(text: str, g: _Grammar) -> bool:
    return {g.index(m) for m in g.item.finditer(text)} == set(range(len(g.labels)))


def _strip_list(tail: str, rx: "re.Pattern[str]", g: _Grammar) -> str:
    out, i, last = [], 0, 0
    while True:
        m = rx.search(tail, i)
        if m is None:
            break
        if _covers_all(m.group(0), g):
            out += [tail[last:m.start()], " "]
            last = i = m.end()
        else:
            i = m.start() + 1
    out.append(tail[last:])
    return "".join(out)


def _strip_template(tail: str, g: _Grammar) -> str:
    return _strip_list(_strip_list(tail, g.bracket_list, g), g.bare_list, g)


def _hedged(text: str, start: int, end: int, idx: int, g: _Grammar) -> bool:
    if g.hedge_right is None:
        return False
    m = g.hedge_right.match(text, end)
    if m is not None and g.index(m) != idx:
        return True
    m = g.hedge_left.search(text, 0, start)
    return m is not None and g.index(m) != idx


def _first_label(tail: str, g: _Grammar) -> _Verdict:
    m = g.item.search(tail)
    if m is None:
        return None
    i = g.index(m)
    return _HEDGE if _hedged(tail, m.start(), m.end(), i, g) else g.labels[i]


def _decisive(line: str, g: _Grammar) -> _Verdict:
    s = line[_LEAD.match(line).end():]
    m = g.decisive.match(s)
    if m is None:
        return None
    i = g.index(m)
    return _HEDGE if _hedged(s, m.start(), m.end(m.lastgroup), i, g) else g.labels[i]


def _answer(v: _Verdict) -> Optional[str]:
    return None if v is _HEDGE else v


def vocabulary_hits(text: str, task: Task) -> List[str]:
    rx, groups = _vocab(task)
    return sorted({groups[m.lastgroup][0] for m in rx.finditer(text or "")})


def has_answer_line(text: str, task: Task) -> bool:
    return _grammar(task).answer_line.search(text) is not None


def extract_answer(text: str, task: Task) -> Optional[str]:
    g = _grammar(task)

    verdicts: List[_Verdict] = []
    for h in g.answer_line.finditer(text):
        v = _first_label(_strip_template(h.group(1), g), g)
        if v is None:
            nxt = text[h.end():].lstrip("\n").split("\n", 1)[0]
            nxt = _strip_template(re.sub(_MD_CHARS, "", nxt).strip(), g)
            v = _decisive(nxt, g)
        if v is not None:
            verdicts.append(v)
    if verdicts:
        return _answer(verdicts[-1])

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    last = _strip_template(lines[-1], g)
    m = g.answer_is.search(last)
    if m:
        return _answer(_first_label(m.group(1), g))
    return _answer(_decisive(last, g))


def parse_steps(text: str, task: Task) -> List[str]:
    if not task.step_names:
        raise ValueError("task.step_names is empty; fill prompts.step_names (the reasoning steps "
                         "before the answer line) in the environment JSON")
    n_steps = task.n_steps
    ans_hits = list(_grammar(task).answer_line.finditer(text))
    if ans_hits:
        cut = ans_hits[-1].start()
        body, answer = text[:cut], text[cut:].strip()
        nl = answer.find("\n\n")
        if nl > 0:
            answer = answer[:nl].strip()
    else:
        body, answer = text, ""

    hits = [m for m in _STEP_RE.finditer(body) if 1 <= int(m.group(1)) <= n_steps]
    steps: List[str] = []
    if hits:
        for i, m in enumerate(hits):
            end = hits[i + 1].start() if i + 1 < len(hits) else len(body)
            steps.append(body[m.start():end].strip())
    else:
        steps = [c.strip() for c in re.split(r"\n\s*\n", body) if c.strip()]

    steps = [s for s in steps if s]
    limit = n_steps - 1 if answer else n_steps
    if len(steps) > limit:
        steps = steps[: limit - 1] + ["\n\n".join(steps[limit - 1:])]
    names = task.step_names
    while len(steps) < limit:
        steps.append(f"Step {len(steps) + 1} - "
                     f"{names[min(len(steps), len(names) - 1)]}: "
                     f"{PAD_MARK}")
    if answer:
        steps.append(answer)
    return steps or [text.strip()]


def n_padded(steps: List[str]) -> int:
    return sum(1 for s in steps if PAD_MARK in s)


@dataclass
class Parsed:
    steps: List[str]
    answer: Optional[str]
    pred: Optional[int]
    n_padded: int
    answer_line: bool


def parse(text: str, task: Task) -> Parsed:
    steps = parse_steps(text, task)
    ans = extract_answer(text, task)
    return Parsed(steps=steps, answer=ans, pred=task.parse_label(ans), n_padded=n_padded(steps),
                  answer_line=has_answer_line(text, task))
