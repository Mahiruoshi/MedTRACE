from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

POS_TOKEN = "+"
NEG_TOKEN = "-"
IMAGE_MARKER = "<image>"
IMG_START, IMG_END, IMG_CONTEXT = "<img>", "</img>", "<IMG_CONTEXT>"
USER, ASSISTANT = 0, 1


@dataclass(frozen=True)
class ConvTemplate:
    name: str
    system_template: str
    system_message: str
    roles: Tuple[str, str]
    sep: str

    def render(self, system: Optional[str], messages: Sequence[Tuple[int, Optional[str]]]) -> str:
        out = self.system_template.format(system_message=system or self.system_message) + self.sep
        for role, msg in messages:
            out += self.roles[role] + msg + self.sep if msg else self.roles[role]
        return out


INTERNVL2_5 = ConvTemplate(
    name="internvl2_5",
    system_template="<|im_start|>system\n{system_message}",
    system_message="",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    sep="<|im_end|>\n",
)


@dataclass
class PRMPrompt:
    text: str
    image_paths: List[str]
    blocks: List[int]
    max_tiles: int
    slot_steps: List[int] = field(default_factory=list)

    @property
    def n_slots(self) -> int:
        return len(self.slot_steps)


def insert_image_markers(prompt: str, n: int) -> str:
    if n <= 0:
        return prompt
    anchor = INTERNVL2_5.roles[USER]
    i = prompt.find(anchor)
    tags = IMAGE_MARKER * n
    return prompt + tags if i == -1 else prompt[:i + len(anchor)] + tags + prompt[i + len(anchor):]


def expand_image_tokens(text: str, tiles_per_block: Sequence[int], num_image_token: int) -> str:
    if text.count(IMAGE_MARKER) != len(tiles_per_block):
        raise RuntimeError(f"{text.count(IMAGE_MARKER)} image markers for {len(tiles_per_block)} blocks: "
                           f"vision tokens and pixel batch would be misaligned")
    for n in tiles_per_block:
        text = text.replace(IMAGE_MARKER, IMG_START + IMG_CONTEXT * (num_image_token * int(n)) + IMG_END, 1)
    return text


@dataclass(frozen=True)
class SlotMatcher:
    header: Tuple[int, ...]
    pos: Tuple[int, ...]
    neg: Tuple[int, ...]

    @classmethod
    def from_encoder(cls, encode: Callable[[str], List[int]],
                     template: ConvTemplate = INTERNVL2_5) -> "SlotMatcher":
        head = tuple(encode(template.roles[ASSISTANT]))
        pos = tuple(encode(template.roles[ASSISTANT] + POS_TOKEN + template.sep))
        neg = tuple(encode(template.roles[ASSISTANT] + NEG_TOKEN + template.sep))
        for run in (pos, neg):
            if run[:len(head)] != head or len(run) <= len(head):
                raise RuntimeError("tokenizer merges the assistant header with the marker; "
                                   "slot positions would be off by one")
        if pos[len(head)] == neg[len(head)]:
            raise RuntimeError("'+' and '-' map to the same token id")
        return cls(head, pos, neg)

    @property
    def pos_id(self) -> int:
        return self.pos[len(self.header)]

    @property
    def neg_id(self) -> int:
        return self.neg[len(self.header)]

    def find(self, ids: Sequence[int]) -> List[int]:
        ids = list(ids)
        first, out = self.header[0], []
        for i, t in enumerate(ids):
            if t != first:
                continue
            for run in (self.pos, self.neg):
                if ids[i:i + len(run)] == list(run):
                    out.append(i + len(self.header) - 1)
                    break
        return out


class PRMLayout:
    name = "base"
    template = INTERNVL2_5
    default_max_tiles = 1
    default_max_images = 8

    def __init__(self, max_tiles: Optional[int] = None, max_images: Optional[int] = None):
        self.max_tiles = int(max_tiles if max_tiles is not None else self.default_max_tiles)
        self.max_images = int(max_images if max_images is not None else self.default_max_images)
        if self.max_tiles < 1 or self.max_images < 0:
            raise ValueError(f"max_tiles >= 1 and max_images >= 0 required ({self.max_tiles}, {self.max_images})")

    def system_for(self, system: str) -> str:
        raise NotImplementedError

    def messages(self, question: str, steps: Sequence[str], fills: Sequence[str]):
        raise NotImplementedError

    def blocks_for(self, n_images: int) -> List[int]:
        raise NotImplementedError

    def build(self, system: str, question: str, steps: Sequence[str], image_paths: Sequence[str],
              fills: Sequence[str]) -> PRMPrompt:
        steps = list(steps)
        fills = list(fills)
        if not steps:
            raise ValueError("no steps to score")
        if len(fills) != len(steps):
            raise ValueError(f"{len(fills)} fills for {len(steps)} steps")
        for i, (s, f) in enumerate(zip(steps, fills)):
            if not isinstance(s, str) or not s.strip():
                raise ValueError(f"step {i} is empty; pad missing steps with visible text")
            if f not in (POS_TOKEN, NEG_TOKEN):
                raise ValueError(f"fill {f!r} after step {i} must be '+' or '-' (layout {self.name!r} "
                                 f"scores every step; a value read from data would condition the "
                                 f"score on the label)")
        paths = list(image_paths)[:self.max_images]
        blocks = self.blocks_for(len(paths)) if paths else []
        text = self.template.render(self.system_for(system), self.messages(question, steps, fills))
        text = insert_image_markers(text, len(blocks))
        return PRMPrompt(text=text, image_paths=paths, blocks=blocks, max_tiles=self.max_tiles,
                         slot_steps=list(range(len(steps))))

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "template": self.template.name, "max_tiles": self.max_tiles,
                "max_images": self.max_images}


class TrainMatchedLayout(PRMLayout):
    name = "train_matched"
    default_max_tiles = 4
    default_max_images = 8

    def system_for(self, system: str) -> str:
        if not system:
            raise ValueError("train_matched needs the generator's system prompt (every training row has one)")
        return system

    def messages(self, question, steps, fills):
        msgs = [(USER, question)]
        for s, f in zip(steps, fills):
            msgs.append((ASSISTANT, s))
            msgs.append((ASSISTANT, f))
        return msgs

    def blocks_for(self, n_images: int) -> List[int]:
        return [n_images]
