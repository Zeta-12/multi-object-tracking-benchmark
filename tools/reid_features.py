"""Re-ID features of the BoT-SORT model for the detections of MOT sequences, as BoT-SORT computes them.

usage: python tools/reid_features.py <BoT-SORT clone> <weights> <split dir> <dets> <out dir>

The model is the FastReID SBS-S50 that the BoT-SORT authors trained on the first half of the MOT17 training
videos (mot17_sbs_S50.pth from the BoT-SORT README). Same steps as fast_reid/fast_reid_interfece.py: the box
is cut to integers and clipped to the image, the crop goes from BGR to RGB and is resized to 128 x 384, the
model runs in fp16 on batches of 8 and the feature is L2 normalised in fp16.

BoT-SORT only computes the features of the detections over its high threshold (0.6), in batches of 8 in the
order of the frame. These come first here, in the same batches, so they get the same values as in a BoT-SORT
run. The other detections over 0.1 follow. The scores are compared in float32, as in the tracker.

Writes <out dir>/<seq>.npy, float16 (n, 2048), one row per line of det/<dets>.txt, and zeros for the boxes
at 0.1 or under. Needs a CUDA GPU, torch, opencv, yacs, termcolor and pyyaml.
"""

import configparser
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

clone, weights, split_dir = (Path(a) for a in sys.argv[1:4])
dets, out_dir = sys.argv[4], Path(sys.argv[5])
sys.path.insert(0, str(clone))

from fast_reid.fastreid.config import get_cfg  # noqa: E402
from fast_reid.fastreid.modeling.meta_arch import build_model  # noqa: E402

HIGH, LOW, BATCH = np.float32(0.6), np.float32(0.1), 8

cfg = get_cfg()
cfg.merge_from_file(str(clone / "fast_reid" / "configs" / "MOT17" / "sbs_S50.yml"))
cfg.MODEL.BACKBONE.PRETRAIN = False
cfg.freeze()
model = build_model(cfg)
state = torch.load(weights, map_location="cpu", weights_only=True)["model"]
# the classifier of the training identities isn't used for the features, and the checkpointer of FastReID
# also skips the weights whose shape doesn't match
shapes = {k: v.shape for k, v in model.state_dict().items()}
skipped = [k for k, v in state.items() if k in shapes and v.shape != shapes[k]]
missing, unexpected = model.load_state_dict({k: v for k, v in state.items() if k not in skipped}, strict=False)
if set(missing) - set(skipped):
    raise SystemExit(f"missing weights: {missing}")
print("skipped:", skipped, "unused:", unexpected, flush=True)
model = model.eval().cuda().half()
size = tuple(cfg.INPUT.SIZE_TEST[::-1])


def crop(img: np.ndarray, box: np.ndarray) -> torch.Tensor | None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.astype(np.int_)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = cv2.resize(np.ascontiguousarray(img[y1:y2, x1:x2, ::-1]), size, interpolation=cv2.INTER_LINEAR)
    return torch.as_tensor(patch.astype("float32").transpose(2, 0, 1)).cuda().half()


out_dir.mkdir(parents=True, exist_ok=True)
for seq_dir in sorted(p for p in split_dir.iterdir() if (p / "det" / f"{dets}.txt").is_file()):
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    im_dir, im_ext = ini["Sequence"]["imDir"], ini["Sequence"]["imExt"]
    if not (seq_dir / im_dir).is_dir():
        continue

    det = np.loadtxt(seq_dir / "det" / f"{dets}.txt", delimiter=",", ndmin=2)
    boxes = np.concatenate([det[:, 2:4], det[:, 2:4] + det[:, 4:6]], axis=1)
    scores = det[:, 6].astype(np.float32)
    feats = np.zeros((len(det), 2048), np.float16)
    empty = nans = 0
    start = time.perf_counter()
    for frame in np.unique(det[:, 0]).astype(int):
        rows = np.flatnonzero(det[:, 0] == frame)
        high, low = rows[scores[rows] > HIGH], rows[(scores[rows] > LOW) & ~(scores[rows] > HIGH)]
        if not len(high) + len(low):
            continue
        img = cv2.imread(str(seq_dir / im_dir / f"{frame:06d}{im_ext}"))
        for group in (high, low):
            for b in range(0, len(group), BATCH):
                idx, patches = [], []
                for i in group[b : b + BATCH]:
                    patch = crop(img, boxes[i])
                    if patch is None:
                        empty += 1
                    else:
                        idx.append(i)
                        patches.append(patch)
                if not patches:
                    continue
                with torch.no_grad():
                    pred = model(torch.stack(patches))
                pred[torch.isinf(pred)] = 1.0
                f = F.normalize(pred).cpu().numpy()
                nans += int(np.isnan(f).any(axis=1).sum())
                feats[idx] = f
    seconds = time.perf_counter() - start
    np.save(out_dir / f"{seq_dir.name}.npy", feats)
    done = int((feats != 0).any(axis=1).sum())
    print(f"{seq_dir.name}: {done} features, {empty} empty crops, {nans} nan, {done / seconds:.0f} boxes/s", flush=True)
