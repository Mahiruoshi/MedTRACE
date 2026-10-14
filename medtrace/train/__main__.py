from __future__ import annotations

import importlib
import sys
from typing import Dict, Optional, Sequence

COMMANDS: Dict[str, str] = {
    "split": "medtrace.train.split",
    "mc": "medtrace.train.mc",
    "targets": "medtrace.train.targets",
    "fit": "medtrace.train.trainer",
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    rows = "\n".join(f"  {c:<{width}}  python -m {m}" for c, m in COMMANDS.items())
    return f"usage: python -m medtrace.train <command> [args ...]\n\ncommands:\n{rows}\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        sys.stdout.write(usage())
        return 0 if args else 2
    cmd, rest = args[0], args[1:]
    if cmd not in COMMANDS:
        sys.stderr.write(f"unknown command {cmd!r}\n\n{usage()}")
        return 2
    try:
        return int(importlib.import_module(COMMANDS[cmd]).main(list(rest)) or 0)
    except SystemExit as e:
        if e.code is None or isinstance(e.code, int):
            return int(e.code or 0)
        sys.stderr.write(f"{e.code}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
