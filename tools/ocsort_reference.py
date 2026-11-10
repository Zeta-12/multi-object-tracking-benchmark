"""Run the official OC-SORT tracker on MOT17 public detections, to check mot_bench's OC-SORT against it.

usage: python tools/ocsort_reference.py <OC_SORT clone> <split dir> <out dir>

Needs numpy, scipy, lap and filterpy, not torch. Same inputs as `mot track` (det.txt sorted by frame, one
update per frame of seqinfo.ini, empty frames included), same settings and output filter as
evaluate_ocsort in yolox/evaluators/mot_evaluator.py, same file format as mot_bench.
"""

import configparser
import sys
from pathlib import Path

import numpy as np

clone, split_dir, out_dir = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(clone))

from trackers.ocsort_tracker.ocsort import OCSort  # noqa: E402

out_dir.mkdir(parents=True, exist_ok=True)
for seq_dir in sorted(p for p in split_dir.iterdir() if (p / "det" / "det.txt").is_file()):
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    length = int(ini["Sequence"]["seqLength"])

    det = np.loadtxt(seq_dir / "det" / "det.txt", delimiter=",", ndmin=2)[:, :7]
    det = det[np.argsort(det[:, 0], kind="stable")]
    boxes = np.concatenate([det[:, 2:4], det[:, 2:4] + det[:, 4:6], det[:, 6:7]], axis=1)

    # default arguments of utils/args.py
    tracker = OCSort(det_thresh=0.6, iou_threshold=0.3, asso_func="iou", delta_t=3, inertia=0.2)
    with open(out_dir / f"{seq_dir.name}.txt", "w") as f:
        for frame in range(1, length + 1):
            # img_info = img_size, so the boxes aren't rescaled
            for t in tracker.update(boxes[det[:, 0] == frame].copy(), (1, 1), (1, 1)):
                x, y, w, h = t[0], t[1], t[2] - t[0], t[3] - t[1]
                if w * h > 100 and not w / h > 1.6:
                    f.write(f"{frame},{int(t[4])},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
    print(seq_dir.name)
