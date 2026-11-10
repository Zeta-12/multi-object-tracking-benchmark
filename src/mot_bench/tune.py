"""Grid search of a tracker's parameters on one split (train_half), scored with TrackEval."""

from __future__ import annotations

import csv
import itertools
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mot_bench import evaluate, mot, run

# min_score filters the detections before the tracker, the other keys are tracker parameters
DETECTION_KEYS = ("min_score",)


def grid(space: dict[str, list]) -> list[dict]:
    """Every combination of the values in `space`, last key varying fastest."""
    keys = list(space)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*space.values())]


def split_point(point: dict) -> tuple[dict, float | None]:
    params = {k: v for k, v in point.items() if k not in DETECTION_KEYS}
    return params, point.get("min_score")


def score(seqs: list[mot.Sequence], tracker: str, point: dict, out_dir: Path, benchmark: str) -> dict:
    params, min_score = split_point(point)
    run.run(seqs, tracker, params, out_dir, min_score)
    gt_dir = seqs[0].dir.parent
    metrics = evaluate.evaluate(gt_dir, out_dir, {s.name: s.info.length for s in seqs}, benchmark)["COMBINED"]
    shutil.rmtree(out_dir)
    return {**point, **metrics}


def search(
    seqs: list[mot.Sequence],
    tracker: str,
    space: dict[str, list],
    work_dir: Path,
    benchmark: str,
    jobs: int = 1,
) -> list[dict]:
    """One row per grid point: the point and its metrics over all `seqs`."""
    points = grid(space)
    dirs = [Path(work_dir) / f"trial-{i:03d}" for i in range(len(points))]
    if jobs == 1:
        return [score(seqs, tracker, p, d, benchmark) for p, d in zip(points, dirs, strict=True)]
    with ProcessPoolExecutor(jobs) as pool:
        futures = [pool.submit(score, seqs, tracker, p, d, benchmark) for p, d in zip(points, dirs, strict=True)]
        return [f.result() for f in futures]


def best(rows: list[dict], metric: str = "HOTA") -> dict:
    # first point wins a tie, so put the reference values first in the grid
    return max(rows, key=lambda r: r[metric])


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
