"""Run the official BoT-SORT tracker on MOT17 detections, to check mot_bench's BoT-SORT against it.

usage: python tools/botsort_reference.py <BoT-SORT clone> <split dir> <out dir> <dets> [noreid]

Needs numpy (< 2, the code uses np.float), scipy and lap, not torch or opencv. The tracker gets the boxes and
scores of det/<dets>.txt, and the re-ID model is replaced by a lookup in det/<dets>.npy (from
tools/reid_features.py, the same values as the model in a BoT-SORT run). The camera motion is read by the
GMC class of the reference from its own files: tracker/GMC_files/MOT17_ablation for val_half,
tracker/GMC_files/MOTChallenge for the full videos. Settings and output filter of tools/track.py with
--default-parameters, same file format as mot_bench.

The reference script only calls the tracker on frames with a detection over 0.09. Every frame of the YOLOX
detections has one, so the tracker is called on every frame here, as in mot_bench.
"""

import configparser
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

clone, split_dir, out_dir = (Path(a).resolve() for a in sys.argv[1:4])
dets = sys.argv[4]
reid = sys.argv[5:] != ["noreid"]

np.float = float

for name in ("cv2", "matplotlib", "matplotlib.pyplot", "fast_reid", "fast_reid.fast_reid_interfece"):
    sys.modules[name] = types.ModuleType(name)


def bbox_overlaps(boxes, query_boxes):
    # cython_bbox 0.1.5, as in tools/bytetrack_reference.py
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


class Lookup:
    """Stands for FastReIDInterface: returns the features of the high detections of the current frame."""

    def __init__(self, *args, **kwargs):
        self.feats = None

    def inference(self, image, detections):
        if len(detections) != len(self.feats):
            raise RuntimeError(f"{len(detections)} detections but {len(self.feats)} features")
        return self.feats


sys.modules["fast_reid.fast_reid_interfece"].FastReIDInterface = Lookup
sys.path.insert(0, str(clone))

from tracker.bot_sort import BoTSORT  # noqa: E402


def track_buffer(name: str) -> int:
    # set by video in tools/track.py
    if name[:8] in ("MOT17-05", "MOT17-06"):
        return 14
    if name[:8] in ("MOT17-13", "MOT17-14"):
        return 25
    return 30


out_dir.mkdir(parents=True, exist_ok=True)
seqs = sorted(p for p in split_dir.iterdir() if (p / "det" / f"{dets}.txt").is_file())
# the GMC class opens its files relative to the clone
os.chdir(clone)
for seq_dir in seqs:
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    length = int(ini["Sequence"]["seqLength"])

    det = np.loadtxt(seq_dir / "det" / f"{dets}.txt", delimiter=",", ndmin=2)[:, :7]
    order = np.argsort(det[:, 0], kind="stable")
    det = det[order]
    boxes = np.concatenate([det[:, 2:4], det[:, 2:4] + det[:, 4:6], det[:, 6:7]], axis=1)
    feats = np.load(seq_dir / "det" / f"{dets}.npy")[order].astype(float) if reid else None

    args = SimpleNamespace(
        track_high_thresh=0.6,
        track_low_thresh=0.1,
        new_track_thresh=0.7,
        track_buffer=track_buffer(seq_dir.name),
        match_thresh=0.8,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=reid,
        fast_reid_config=None,
        fast_reid_weights=None,
        device="cpu",
        cmc_method="file",
        name=seq_dir.name,
        ablation=split_dir.name == "val_half",
        mot20=False,
    )
    tracker = BoTSORT(args, frame_rate=30)
    with open(out_dir / f"{seq_dir.name}.txt", "w") as f:
        for frame in range(1, length + 1):
            rows = det[:, 0] == frame
            frame_boxes = boxes[rows].copy()
            if reid:
                s = frame_boxes[:, 4]
                tracker.encoder.feats = feats[rows][s > 0.1][s[s > 0.1] > 0.6].copy()
            for t in tracker.update(frame_boxes, None):
                x, y, w, h = t.tlwh
                if w * h > 10 and not w / h > 1.6:
                    f.write(f"{frame},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{t.score:.4f},-1,-1,-1\n")
    print(seq_dir.name)
