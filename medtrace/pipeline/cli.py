from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from ..config import load_config

COMMANDS = ("distill", "generate", "score", "report", "all")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m medtrace", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--config", required=True, help="experiment YAML (with _base_ inheritance)")
    p.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[],
                   metavar="KEY=VALUE", help="dotted overrides, YAML-parsed")
    p.add_argument("--force", action="store_true",
                   help="recompute the stage(s), renaming a conflicting stage dir aside")
    p.add_argument("--gpu", type=int, default=None,
                   help="score: CUDA device index as seen by this process, i.e. within "
                        "CUDA_VISIBLE_DEVICES when set (default 0)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    from . import stages

    stages.load_plugins(cfg)
    if args.command == "distill":
        print(stages.stage_distill(cfg, force=args.force))
    elif args.command == "generate":
        print(stages.stage_generate(cfg, force=args.force))
    elif args.command == "score":
        print(stages.stage_score(cfg, gpu=args.gpu, force=args.force))
    elif args.command == "report":
        out = stages.stage_report(cfg, force=args.force)
        print(out.read_text(encoding="utf-8"))
    elif args.command == "all":
        out = stages.run_all(cfg, gpu=args.gpu, force=args.force)
        print(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
