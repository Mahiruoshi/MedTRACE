from __future__ import annotations

import sys
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from ..features.store import FeatureStore
from ..mappings.base import Frames
from ..seeds import SeedRegistry

SHARE_WARN = 0.5


def _pid(p: Union[str, Any]) -> str:
    return p if isinstance(p, str) else str(p.pid)


def build_calibration(store: FeatureStore, patients: Sequence[Union[str, Any]], *, seeds: SeedRegistry,
                      n_patients: int = 16, max_frames: Optional[int] = None) -> Frames:
    pool = sorted({_pid(p) for p in patients})
    if not pool:
        raise ValueError("empty calibration pool")
    if int(n_patients) < 1:
        raise ValueError(f"calib.n_patients must be >= 1, got {n_patients}")
    seed = seeds["calib"]
    order = np.random.default_rng(seed).permutation(len(pool))
    taken: List[Frames] = []
    pids: List[str] = []
    skipped_long: List[str] = []
    for i in order:
        if len(pids) >= n_patients:
            break
        pid = pool[int(i)]
        if not store.has(pid):
            raise FileNotFoundError(f"calibration patient {pid} has no cached features in {store.root}; "
                                    f"extract the train split first")
        fr = store.load(pid)
        if max_frames is not None and fr.T > max_frames:
            skipped_long.append(pid)
            continue
        taken.append(fr)
        pids.append(pid)
    if not taken:
        raise ValueError(f"no eligible calibration patient: every series is longer than calib.max_frames="
                         f"{max_frames}")
    calib = Frames.concat(taken)
    counts = [f.T for f in taken]
    share = max(counts) / calib.T
    if share > SHARE_WARN:
        print(f"[calib] warning: patient {pids[counts.index(max(counts))]} supplies {max(counts)} of the "
              f"{calib.T} calibration frames ({share:.0%}); T_OT's moments and the PCA text pool are fitted "
              f"mostly on one series (calib.max_frames={max_frames}; a cap skips longer series).",
              file=sys.stderr, flush=True)
    calib.meta["calib"] = {"pids": pids, "n_patients": len(pids),
                           "n_frames": calib.T, "frames_per_patient": counts,
                           "largest_patient_share": float(share),
                           "requested": int(n_patients),
                           "max_frames": None if max_frames is None else int(max_frames),
                           "skipped_long": skipped_long,
                           "seed_stage": "calib", "seed": int(seed), "pool_size": len(pool)}
    return calib
