"""Run the official BYTETracker on MOT17 detections, to check mot_bench's ByteTrack against it.

usage: python tools/bytetrack_reference.py <ByteTrack clone> <split dir> <out dir> [dets]

dets is the detection file, det/<dets>.txt (default det, the public detections).

Needs numpy, scipy and lap, not torch: the tracker gets (n, 5) boxes and scores, no detector runs. Same
inputs as `mot track` (det.txt sorted by frame, one update per frame of seqinfo.ini, empty frames
included), same output filter as yolox/evaluators/mot_evaluator.py, same file format as mot_bench.
"""

import configparser
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

clone, split_dir, out_dir = (Path(a) for a in sys.argv[1:4])
dets = sys.argv[4] if len(sys.argv) > 4 else "det"

# the code still uses np.float, removed in numpy 1.24
np.float = float

# imported by byte_tracker.py and matching.py, unused with (n, 5) inputs
for name in ("torch", "torch.nn", "torch.nn.functional", "cv2"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["torch"].nn = sys.modules["torch.nn"]
sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]


def bbox_overlaps(boxes, query_boxes):
    # cython_bbox 0.1.5 (MIT, from Fast R-CNN) needs a C compiler. Same operations, vectorised.
    b = boxes[:, None, :]
    q = query_boxes[None, :, :]
    box_area = (q[..., 2] - q[..., 0] + 1) * (q[..., 3] - q[..., 1] + 1)
    iw = np.minimum(b[..., 2], q[..., 2]) - np.maximum(b[..., 0], q[..., 0]) + 1
    ih = np.minimum(b[..., 3], q[..., 3]) - np.maximum(b[..., 1], q[..., 1]) + 1
    ua = (b[..., 2] - b[..., 0] + 1) * (b[..., 3] - b[..., 1] + 1) + box_area - iw * ih
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((iw > 0) & (ih > 0), iw * ih / ua, 0.0)


sys.modules["cython_bbox"] = types.ModuleType("cython_bbox")
sys.modules["cython_bbox"].bbox_overlaps = bbox_overlaps

# yolox/__init__.py imports the detector, only the tracker package is needed
for name, path in (("yolox", clone / "yolox"), ("yolox.tracker", clone / "yolox" / "tracker")):
    sys.modules[name] = types.ModuleType(name)
    sys.modules[name].__path__ = [str(path)]

from yolox.tracker.byte_tracker import BYTETracker  # noqa: E402


def track_buffer(name: str) -> int:
    # set by video in mot_evaluator.py
    if name[:8] in ("MOT17-05", "MOT17-06"):
        return 14
    if name[:8] in ("MOT17-13", "MOT17-14"):
        return 25
    return 30


out_dir.mkdir(parents=True, exist_ok=True)
for seq_dir in sorted(p for p in split_dir.iterdir() if (p / "det" / f"{dets}.txt").is_file()):
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    length = int(ini["Sequence"]["seqLength"])

    det = np.loadtxt(seq_dir / "det" / f"{dets}.txt", delimiter=",", ndmin=2)[:, :7]
    det = det[np.argsort(det[:, 0], kind="stable")]
    boxes = np.concatenate([det[:, 2:4], det[:, 2:4] + det[:, 4:6], det[:, 6:7]], axis=1)

    args = SimpleNamespace(
        track_thresh=0.6, track_buffer=track_buffer(seq_dir.name), match_thresh=0.9, mot20=False, min_box_area=100
    )
    tracker = BYTETracker(args)
    with open(out_dir / f"{seq_dir.name}.txt", "w") as f:
        for frame in range(1, length + 1):
            # img_info = img_size, so the boxes aren't rescaled
            for t in tracker.update(boxes[det[:, 0] == frame].copy(), (1, 1), (1, 1)):
                x, y, w, h = t.tlwh
                if w * h > args.min_box_area and not w / h > 1.6:
                    f.write(f"{frame},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{t.score:.4f},-1,-1,-1\n")
    print(seq_dir.name)
