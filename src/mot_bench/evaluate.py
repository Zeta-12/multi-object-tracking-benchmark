"""HOTA, CLEAR (MOTA) and Identity (IDF1) metrics with TrackEval, the reference implementation."""

from __future__ import annotations

import contextlib
import functools
import io
from pathlib import Path

import numpy as np

COLUMNS = ["HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1", "IDSW", "FP", "FN", "Frag"]


@functools.cache
def _trackeval():
    # the last TrackEval commit (2022) still uses aliases removed in numpy 1.24
    for name, alias in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, name):
            setattr(np, name, alias)
    # it prints a warning about an optional dependency on import
    with contextlib.redirect_stdout(io.StringIO()):
        import trackeval
    return trackeval


def _summary(res: dict) -> dict[str, float]:
    hota, clear, ident = res["HOTA"], res["CLEAR"], res["Identity"]
    return {
        # HOTA fields are arrays over the 19 IoU thresholds 0.05..0.95
        "HOTA": float(np.mean(hota["HOTA"])),
        "DetA": float(np.mean(hota["DetA"])),
        "AssA": float(np.mean(hota["AssA"])),
        "LocA": float(np.mean(hota["LocA"])),
        "MOTA": float(clear["MOTA"]),
        "IDF1": float(ident["IDF1"]),
        "IDSW": int(clear["IDSW"]),
        "FP": int(clear["CLR_FP"]),
        "FN": int(clear["CLR_FN"]),
        "Frag": int(clear["Frag"]),
    }


def evaluate(gt_dir: Path, results_dir: Path, seq_lengths: dict[str, int], benchmark: str) -> dict[str, dict]:
    """Metrics per sequence and for all of them ("COMBINED").

    gt_dir/<seq>/gt/gt.txt is the ground truth, results_dir/data/<seq>.txt the tracker output.
    `benchmark` (MOT17 or MOT20) sets the distractor classes.
    """
    trackeval = _trackeval()
    results_dir = Path(results_dir)
    dataset_config = {
        "GT_FOLDER": str(gt_dir),
        "TRACKERS_FOLDER": str(results_dir.parent),
        "TRACKERS_TO_EVAL": [results_dir.name],
        "BENCHMARK": benchmark,
        "SKIP_SPLIT_FOL": True,
        "SEQ_INFO": dict(seq_lengths),
        "PRINT_CONFIG": False,
    }
    eval_config = {
        "USE_PARALLEL": False,
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    }
    metrics = [
        trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False}),
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        res, _ = trackeval.Evaluator(eval_config).evaluate([dataset], metrics)
    per_seq = res["MotChallenge2DBox"][results_dir.name]
    return {("COMBINED" if seq == "COMBINED_SEQ" else seq): _summary(r["pedestrian"]) for seq, r in per_seq.items()}


def table(metrics: dict[str, dict], columns: list[str] = COLUMNS) -> str:
    rows = [["sequence", *columns]]
    for name, m in metrics.items():
        rows.append([name] + [f"{100 * m[c]:.1f}" if isinstance(m[c], float) else str(m[c]) for c in columns])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(v.rjust(w) for v, w in zip(r, widths, strict=True)) for r in rows)
