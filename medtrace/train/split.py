from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..data.cohort import Manifest, manifest_sha256, natural_key
from ..data.tasks import Target, Task
from ..pipeline.rundir import read_json, write_json
from ..seeds import sub_seed

PARTS = ("fit", "meta")
ManifestLike = Union[str, os.PathLike, Manifest]


def strata_of(targets: Mapping[str, Target], task: Task) -> Dict[str, int]:
    return {str(pid): task.stratum(t) for pid, t in targets.items() if task.determined(t)}


def _strata(strata: Mapping[str, Any]) -> Dict[str, int]:
    ys: Dict[str, int] = {}
    for pid, s in strata.items():
        pid = str(pid)
        if pid in ys:
            raise ValueError(f"duplicate pid {pid!r}")
        if isinstance(s, bool) or not isinstance(s, int) or s < 0:
            raise ValueError(f"{pid}: stratum must be an int >= 0, got {s!r}")
        ys[pid] = int(s)
    return ys


def meta_counts(counts: Mapping[int, int], n_meta: int) -> Dict[int, int]:
    n = sum(int(c) for c in counts.values())
    if not 0 < n_meta < n:
        raise ValueError(f"n_meta={n_meta} must be in (0, {n})")
    take = {s: n_meta * int(c) // n for s, c in counts.items()}
    rem = {s: n_meta * int(c) % n for s, c in counts.items()}
    for s in sorted(counts, key=lambda s: (-rem[s], -s))[:n_meta - sum(take.values())]:
        take[s] += 1
    return take


def partition(strata: Mapping[str, int], n_meta: int, seed: int) -> Dict[str, List[str]]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int (seeds.split of the environment JSON), got {seed!r}")
    ys = _strata(strata)
    n = len(ys)
    if not 0 < n_meta < n:
        raise ValueError(f"n_meta={n_meta} must be in (0, {n}) (the determined train patients)")
    groups: Dict[int, List[str]] = {}
    for pid in sorted(ys):
        groups.setdefault(ys[pid], []).append(pid)
    take = meta_counts({s: len(v) for s, v in groups.items()}, n_meta)

    def rank(pid: str):
        return sub_seed(int(seed), pid), pid

    meta = [p for s in sorted(groups, reverse=True) for p in sorted(groups[s], key=rank)[:take[s]]]
    fit = sorted(set(ys) - set(meta), key=natural_key)
    return {"fit": fit, "meta": sorted(meta, key=natural_key)}


def save_partition(path: Union[str, os.PathLike], targets: Mapping[str, Target], task: Task,
                   manifest: ManifestLike, n_meta: int, seed: int) -> Dict[str, List[str]]:
    strata = strata_of(targets, task)
    parts = partition(strata, n_meta, seed)
    ys = _strata(strata)
    blob = {"seed": int(seed), "n_meta": int(n_meta),
            "manifest": None if isinstance(manifest, Mapping) else Path(manifest).name,
            "manifest_sha256": manifest_sha256(manifest),
            "undetermined": len(targets) - len(strata),
            "counts": {k: {"n": len(parts[k]),
                           "strata": {str(s): c for s, c in sorted(Counter(ys[p] for p in parts[k]).items())}}
                       for k in PARTS},
            "fit": parts["fit"], "meta": parts["meta"]}
    write_json(path, blob)
    return parts


def load_partition(path: Union[str, os.PathLike], manifest: ManifestLike) -> Dict[str, List[str]]:
    blob = read_json(path)
    want = manifest_sha256(manifest)
    if not isinstance(blob, Mapping) or blob.get("manifest_sha256") != want:
        got = blob.get("manifest_sha256") if isinstance(blob, Mapping) else None
        raise ValueError(f"{path}: partition drawn from manifest sha256 {got}, the current manifest is "
                         f"{want}; re-partition with medtrace.train.split")
    missing = [k for k in PARTS if not isinstance(blob.get(k), list)]
    if missing:
        raise KeyError(f"{path}: no {missing} pid list(s); write it with medtrace.train.split")
    parts = {k: [str(p) for p in blob[k]] for k in PARTS}
    _check_parts(parts)
    return parts


def _check_parts(parts: Mapping[str, Sequence[str]]) -> None:
    fit, meta = set(parts["fit"]), set(parts["meta"])
    if len(fit) != len(parts["fit"]) or len(meta) != len(parts["meta"]):
        raise ValueError("a part lists a pid twice")
    if fit & meta:
        raise ValueError(f"fit and meta overlap: {sorted(fit & meta)[:5]}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="python -m medtrace.train.split",
                                 description="Write the fit/meta partition of the train split "
                                             "(seed: seeds.split of the environment JSON, $MEDTRACE_ENV).")
    ap.add_argument("--out", required=True, help="partition JSON to write")
    ap.add_argument("--manifest", required=True, help="dataset manifest (README.md, 'Datasets')")
    ap.add_argument("--image-root", default=None,
                    help="base of relative frames_dir, relative to the working directory "
                         "(default: MEDTRACE_IMAGE_ROOT, else the manifest's)")
    ap.add_argument("--n-meta", type=int, required=True,
                    help="patients in the held-out meta part (of the determined train patients)")
    a = ap.parse_args(argv)
    from .. import envjson
    from ..data.cohort import load_patients, load_targets, task_type
    from ..data.tasks import load_task
    from ..paths import get_paths

    env = envjson.load_env()
    task = load_task(env, task_type(a.manifest))
    seed = envjson.seed(env, "seeds.split")
    patients = load_patients(a.manifest, "train", get_paths({"image_root": a.image_root}).image_root)
    targets = load_targets(a.manifest, "train", task, pids=[p.pid for p in patients])
    save_partition(a.out, targets, task, a.manifest, a.n_meta, seed)
    blob = read_json(a.out)
    print(json.dumps({**blob["counts"], "undetermined": blob["undetermined"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
