from __future__ import annotations

from typing import Callable, Dict, List


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable] = {}

    def register(self, name: str):
        def deco(obj):
            key = name.lower()
            if key in self._items:
                raise KeyError(f"{self.kind} {name!r} registered twice")
            self._items[key] = obj
            return obj
        return deco

    def resolve(self, name: str) -> str:
        key = name.lower()
        if key not in self._items:
            raise KeyError(f"unknown {self.kind} {name!r}; known: {', '.join(self.names())}")
        return key

    def get(self, name: str, /) -> Callable:
        return self._items[self.resolve(name)]

    def build(self, name: str, /, *args, **kwargs):
        return self.get(name)(*args, **kwargs)

    def names(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        try:
            self.resolve(name)
            return True
        except KeyError:
            return False


MAPPINGS = Registry("mapping")
SCORERS = Registry("scorer")
SELECTORS = Registry("selector")
GENERATORS = Registry("generator")
AGGREGATORS = Registry("aggregator")
