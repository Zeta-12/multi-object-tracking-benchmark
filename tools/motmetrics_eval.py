"""MOTA, IDF1 and ID switches as tools/track.py of ByteTrack computes them, to compare with its README.

usage: python tools/motmetrics_eval.py <split dir> <results dir>

Needs motmetrics and lap. The ground truth keeps the boxes with the "considered" flag, the results are rounded
to 0.1 px as in write_results of the reference, and a pair is a match from an IoU of 0.5. Unlike TrackEval,
the tracker boxes matched to distractors or static people count as false positives, so MOTA comes out lower.
"""

import sys
import tempfile
from pathlib import Path

import motmetrics as mm
import numpy as np

split_dir, res_dir = Path(sys.argv[1]), Path(sys.argv[2])
mm.lap.default_solver = "lap"

accs, names = [], []
with tempfile.TemporaryDirectory() as tmp:
    for res in sorted(res_dir.glob("*.txt")):
        rows = np.loadtxt(res, delimiter=",", ndmin=2)
        rounded = Path(tmp) / res.name
        with open(rounded, "w") as f:
            for r in rows:
                x, y, w, h = (round(v, 1) for v in r[2:6])
                f.write(f"{int(r[0])},{int(r[1])},{x},{y},{w},{h},-1,-1,-1,-1\n")
        gt = mm.io.loadtxt(split_dir / res.stem / "gt" / "gt.txt", fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(rounded, fmt="mot15-2D", min_confidence=-1)
        accs.append(mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5))
        names.append(res.stem)

summary = mm.metrics.create().compute_many(
    accs, names=names, metrics=["mota", "idf1", "num_switches"], generate_overall=True
)
o = summary.loc["OVERALL"]
print(f"MOTA {100 * o['mota']:.1f}  IDF1 {100 * o['idf1']:.1f}  IDs {int(o['num_switches'])}")
