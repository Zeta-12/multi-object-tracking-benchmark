"""Run the YOLOX detector of the ByteTrack ablation on MOT sequences and write MOTChallenge det files.

usage: python tools/yolox_detections.py <ByteTrack clone> <checkpoint> <split dir> <out dir> [fp32]

Same settings as tools/track.py of ByteTrack with the ablation model (exps/example/mot/yolox_x_ablation.py):
input 800 x 1440, fp16, conv and batch norm fused, confidence 0.01, NMS 0.7. Each box is written with the
score the tracker gets, objectness x class confidence, in the coordinates of the original image. The boxes
are divided by the resize ratio in float32, as in BYTETracker.update, and written without rounding. fp32 runs
the model in float32 instead, to see how much the tracking metrics move with the detector numerics alone.

Needs a CUDA GPU, torch, torchvision, opencv, loguru, thop, tabulate and pycocotools (imported by the exp).
The checkpoint is bytetrack_ablation.pth.tar from the ByteTrack README. The out files are named after the
sequences, ready for `mot add-dets`.
"""

import configparser
import sys
import time
from pathlib import Path

import cv2
import torch

clone, ckpt, split_dir, out_dir = (Path(a) for a in sys.argv[1:5])
fp16 = sys.argv[5:] != ["fp32"]
sys.path.insert(0, str(clone))

from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.exp import get_exp  # noqa: E402
from yolox.utils import fuse_model, postprocess  # noqa: E402

CONF, NMS = 0.01, 0.7

exp = get_exp(str(clone / "exps" / "example" / "mot" / "yolox_x_ablation.py"), None)
model = exp.get_model().cuda().eval()
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model"])
model = fuse_model(model)
if fp16:
    model = model.half()
transform = ValTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

out_dir.mkdir(parents=True, exist_ok=True)
for seq_dir in sorted(p for p in split_dir.iterdir() if (p / "seqinfo.ini").is_file()):
    ini = configparser.ConfigParser()
    ini.read(seq_dir / "seqinfo.ini")
    seq = ini["Sequence"]
    length, im_dir, im_ext = int(seq["seqLength"]), seq["imDir"], seq["imExt"]
    # only the FRCNN copy of each MOT17 video keeps the frames
    if not (seq_dir / im_dir).is_dir():
        continue

    lines, start = [], time.perf_counter()
    for frame in range(1, length + 1):
        img = cv2.imread(str(seq_dir / im_dir / f"{frame:06d}{im_ext}"))
        h, w = img.shape[:2]
        x, _ = transform(img, None, exp.test_size)
        x = torch.from_numpy(x)[None].cuda()
        with torch.no_grad():
            out = postprocess(model(x.half() if fp16 else x), exp.num_classes, CONF, NMS)[0]
        if out is None:
            continue
        out = out.cpu().numpy()
        scores = out[:, 4] * out[:, 5]
        boxes = out[:, :4]
        boxes /= min(exp.test_size[0] / float(h), exp.test_size[1] / float(w))
        for (x1, y1, x2, y2), s in zip(boxes.tolist(), scores.tolist(), strict=True):
            # float32 values, written in full so that x1 + w gives x2 back
            lines.append(f"{frame},-1,{x1!r},{y1!r},{x2 - x1!r},{y2 - y1!r},{s!r},-1,-1,-1\n")
    seconds = time.perf_counter() - start
    (out_dir / f"{seq_dir.name}.txt").write_text("".join(lines))
    print(f"{seq_dir.name}: {length} frames, {len(lines)} boxes, {length / seconds:.1f} fps", flush=True)
