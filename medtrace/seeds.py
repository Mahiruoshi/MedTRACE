from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict

STAGES = (
    "projection",
    "decode",
    "calib",
    "mc",
    "train",
)


def derive(master: int, stage: str) -> int:
    if stage not in STAGES:
        raise KeyError(f"unknown seed stage {stage!r}; declare it in medtrace.seeds.STAGES")
    h = hashlib.sha256(f"{master}:{stage}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def sub_seed(stage_seed: int, *keys) -> int:
    payload = ":".join([str(stage_seed), *[str(k) for k in keys]])
    h = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


@dataclass
class SeedRegistry:
    master: int
    overrides: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.master is None or isinstance(self.master, bool) or not isinstance(self.master, int):
            raise ValueError(f"master seed must be an int from the environment JSON, got {self.master!r}")

    def __getitem__(self, stage: str) -> int:
        if stage in self.overrides:
            return int(self.overrides[stage])
        return derive(self.master, stage)

    def as_dict(self) -> Dict[str, int]:
        return {"master": self.master, **{s: self[s] for s in STAGES},
                "overrides": dict(self.overrides)}
