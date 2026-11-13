# Multi-Object Tracking Benchmark

[![CI](https://github.com/Zeta-12/multi-object-tracking-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Zeta-12/multi-object-tracking-benchmark/actions/workflows/ci.yml)

Pedestrian trackers of the SORT family (SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT) written from scratch with
NumPy, and compared on MOT17 and MOT20 with [TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA, MOTA,
IDF1). All trackers get the same detections, so the differences come from the association step only, and each
implementation has to match the numbers published by its authors on the same split.

Work in progress: the data, the evaluation, SORT, ByteTrack, OC-SORT and BoT-SORT are in place. DeepSORT comes
next.

## Setup

```
uv sync
uv run mot download                     # MOT17 annotations and public detections, 10 MB
uv run mot info --split val_half
uv run mot track --tracker sort --detector FRCNN    # runs on val_half and prints the metrics
uv run mot track --tracker bytetrack --detector SDP --param track_buffer=60
uv run mot summary                      # every run of val_half
```

`mot download --dataset MOT20` gets the MOT20 annotations (19 MB). `mot download --images` adds the MOT17 frames,
needed to run a detector, re-ID or camera motion compensation: a 5.9 GB archive, 2 GB on disk once extracted (see
below). Archives are checked against their SHA-256 and downloads resume after a dropped connection.

## Evaluation split

The ground truth of the MOT17 and MOT20 test sets is only on the MOTChallenge server, which limits the number of
submissions. As in the ByteTrack, OC-SORT and BoT-SORT papers, the results here are on the second half of each
training sequence (`val_half`), and the first half (`train_half`) is left for tuning. With n frames, `train_half` is
frames 1 to n/2+1 and `val_half` the rest, renumbered from 1. This is the cut of ByteTrack's
`convert_mot17_to_coco.py`, and published val_half numbers are only comparable with it.

MOT17 ships every video three times, with the public detections of DPM, Faster R-CNN and SDP. The ground
truth and the frames are the same in the three copies, so only the FRCNN copy keeps the images.

Only pedestrians with the "considered" flag count in the metrics. Static people, people on a vehicle, reflections
and distractors are only used to drop the tracker boxes matched to them.

The metrics come from TrackEval, pinned to its last commit (12c8791, 2022). That commit still uses numpy aliases
removed in numpy 1.24, `evaluate.py` puts them back before importing it.

## Tuning

`mot tune` tries a grid of settings on `train_half`, keeps the one with the best HOTA and runs it once on
`val_half`. `min_score` drops the detections under a score before the tracker, the other keys are tracker
parameters:

```
uv run mot tune --detector SDP --grid min_score=null,0.5,0.7,0.9 --grid max_age=1,10,30 --grid min_hits=3,1 --jobs 6
```

The score of every grid point is written to `runs/MOT17-train_half/<name>/grid.csv`. Both halves come from the
same videos, so tuned settings are a bit optimistic, as in the papers that use this split.

## Trackers

All trackers take the detections of one frame and return the tracked boxes with their ids. They share the Kalman
filter and its box models (`kalman.py`), the IoU (`boxes.py`) and the assignment (`matching.py`).

SORT (Bewley et al., 2016) keeps one constant-velocity Kalman filter per track, on the box centre, area and aspect
ratio. Each frame, the predicted boxes are matched to the detections with the Hungarian algorithm on the IoU, and
pairs under 0.3 are dropped. There is no appearance model. A few details of the reference code change the numbers
and are kept here:

- with min_hits=3, a new track is only reported from its 4th consecutive detection, except during the first 3
  frames of a video;
- a track is only reported on the frames where it got a detection, and it is deleted after more than max_age=1
  frames without one;
- when every detection has at most one track above the IoU threshold and the other way round, these pairs are
  taken directly. This can differ from what the Hungarian step would return.

The SORT here gives the same output as the reference implementation ([abewley/sort](https://github.com/abewley/sort)
at 2236dff, with filterpy 1.4.5 and scipy for the assignment) on the 21 MOT17 training sequences: the same 193,561
boxes to the 0.01 px of the result files, and the same tracks up to the id numbers. To check it again:

```
# in a clone of abewley/sort (its imports also need matplotlib and scikit-image)
python sort.py --seq_path <this repo>/data/MOT17 --phase train
# here
uv run mot track --split train --name sort-all
uv run mot compare runs/MOT17-train/sort-all/data <abewley/sort clone>/output
```

ByteTrack (Zhang et al., 2022) also uses the low score detections. The detections over track_thresh (0.6) are
matched first, to every track including the lost ones, on 1 - IoU x score. The ones between 0.1 and 0.6 are then
matched by IoU to the tracks still free: they're often people half hidden, so they keep a track going but never
start one. Lost tracks are kept 30 frames. The Kalman state is the centre, the w/h ratio and the height, with a
noise proportional to the height (from DeepSORT). Details of the reference code kept here:

- the assignment is `lap.lapjv` with a cost limit, which isn't the Hungarian step followed by a threshold. A pair
  over the limit costs the limit, so one very good pair can win over two medium ones (`matching.capped`). It
  changes 37 of the 22,325 assignments made on the MOT17 training sequences;
- the IoU counts boxes one pixel wider and taller (`cython_bbox`);
- a new track needs a score of 0.7. It's reported from its second detection and isn't predicted before it;
- a detection with a score of exactly 0.6 is in neither group;
- a lost track is removed one frame late: it can still come back with its id on the next frame, but then the
  next time it's lost it's dropped at once;
- the output skips boxes under 100 px2 or more than 1.6 times wider than tall. This filter is in the
  evaluation script of the reference, not in the tracker;
- that script also sets the lost track buffer to 14 and 25 frames for the MOT17 videos at 14 and 25 fps. Here
  the buffer is scaled by the frame rate in seqinfo.ini, which gives the same values.

On the 21 MOT17 training sequences, with the public detections, the output is the same as the reference
([ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) at d1bf019, lap 0.5.14): the same 174,657 boxes and the
same tracks. `tools/bytetrack_reference.py` runs the reference tracker on the detections, without the detector:

```
# in a venv with numpy, scipy and lap
python tools/bytetrack_reference.py <ByteTrack clone> data/MOT17/train <out>
# here
uv run mot track --tracker bytetrack --split train --name bytetrack-all
uv run mot compare runs/MOT17-train/bytetrack-all/data <out>
```

OC-SORT (Cao et al., 2023) keeps the Kalman filter of SORT and adds three steps that trust the detections more than
the filter:

- ORU: when a lost track gets a detection again, its filter goes back to its state on the first missed frame and
  runs the gap again on a straight line between the two detections (`kalman.OruBox`);
- OCM: the cost adds inertia x score x (pi/2 - angle) / pi, where angle is between the motion of the track and
  the direction from its detection of 3 frames ago to each new detection;
- OCR: the tracks and detections left over are matched again on the IoU with the last detection of the track
  instead of its prediction.

Lost tracks are kept 30 frames, only the detections over 0.6 are used, and the reported box is the detection
itself, not the Kalman state. Details of the reference code kept here:

- after a gap the new detection is applied twice, once at the end of the virtual line and once for real;
- the measurement adds 1e-6 to the height in the w/h ratio;
- a new track is reported from its 4th detection, as in SORT;
- "no detection yet" is a box of -1 tested with sum() < 0, and the direction term is only used when the old
  detection has a score >= 0;
- the output skips boxes under 100 px2 or more than 1.6 times wider than tall, as in the evaluation script.

On the 21 MOT17 training sequences the output is the same as the reference
([noahcao/OC_SORT](https://github.com/noahcao/OC_SORT) at 2d34c67, the 2022 code of the paper, run with lap): the
same 156,787 boxes and the same tracks. The 2026 refactor of that repo (8462e7e) gives the same files, and so does
the `use_byte` option (169,118 boxes).

```
# in a venv with numpy, scipy, lap and filterpy
python tools/ocsort_reference.py <OC_SORT clone> data/MOT17/train <out>
# here
uv run mot track --tracker ocsort --split train --name ocsort-all
uv run mot compare runs/MOT17-train/ocsort-all/data <out>
```

BoT-SORT (Aharon et al., 2022) is ByteTrack with three changes:

- the Kalman state has the width and the height instead of the w/h ratio and the height (`kalman.XywhBox`);
- after the prediction the tracks are moved by the camera motion between the two frames (CMC), an affine map
  given with each frame;
- with re-ID features, the first association costs the smaller of the IoU cost of ByteTrack and half the cosine
  distance between the appearance of the track and the detection. The appearance only counts under 0.25 and when
  1 - IoU is under 0.5. The appearance of a track is a moving average of the features of its high score
  detections (0.9 old, 0.1 new).

Details of the reference code kept here:

- the first association matches up to a cost of 0.8 (0.9 in ByteTrack);
- a lost track drops its width and height velocities;
- the tracks born on the previous frame aren't predicted, but they're moved by the camera motion;
- every track of the tracked list is reported, also a track on the frame it's born;
- the output skips boxes under 10 px2 or more than 1.6 times wider than tall.

The appearance can lower the cost of a pair, never raise it. And a high score pair that passes the overlap test
already has an IoU cost of at most 1 - 0.5 x 0.6 = 0.7, under the 0.8 limit: the re-ID never adds a match the IoU
would refuse, it only changes which of the overlapping detections a track takes.

I wrote the trackers with the help of an AI assistant. Each one is checked against its reference implementation,
as above, before its results are reported here.

## YOLOX detections

The ByteTrack and OC-SORT papers report their MOT17 `val_half` numbers on the boxes of a YOLOX-X trained by the
ByteTrack authors on CrowdHuman and on `train_half` (`bytetrack_ablation.pth.tar`, from the ByteTrack README).
`tools/yolox_detections.py` runs it on the MOT17 frames with the settings of ByteTrack's `tools/track.py`: input
800 x 1440, fp16, confidence 0.01, NMS 0.7, each box with the score objectness x class. It needs a CUDA GPU and
the dependencies of the ByteTrack code. It ran at 9 fps on an RTX 3060 Ti, 10 minutes for the 7 videos.
`mot add-dets` copies the files into the dataset as `det/<name>.txt`, cuts the halves again, and `--dets`
picks them:

```
# in a venv with torch and the ByteTrack requirements, after mot download --images
python tools/yolox_detections.py <ByteTrack clone> bytetrack_ablation.pth.tar data/MOT17/train <out>
# here
uv run mot add-dets <out> --name yolox_ablation
uv run mot track --tracker bytetrack --dets yolox_ablation
```

The trackers still match the reference code on these boxes. OC-SORT gives the same 43,292 boxes on `val_half`
(94,543 on the full videos). ByteTrack gives the same 48,163 boxes but one, which has a height of 317.92 here and
317.93 in the reference, on the same track: the two Kalman filters do the same computation with different numpy
calls, and the last bits can tip a rounding to 0.01 px. The reference tools take the name of the detection file
as a 4th argument:

```
python tools/bytetrack_reference.py <ByteTrack clone> data/MOT17/val_half <out> yolox_ablation
uv run mot compare runs/MOT17-val_half/bytetrack-yolox_ablation/data <out>
```

## Re-ID features and camera motion

BoT-SORT needs two more inputs. The re-ID features come from the FastReID SBS-S50 that the BoT-SORT authors
trained on the first half of each MOT17 training video (`mot17_sbs_S50.pth`, from the BoT-SORT README).
`tools/reid_features.py` computes them on a GPU for every YOLOX box over 0.1, with the steps of the BoT-SORT
encoder: integer box clipped to the image, crop resized to 128 x 384, fp16, batches of 8, L2 norm. The boxes
over 0.6 come first, in the batches of a BoT-SORT run, and their features are the same as the ones of the BoT-SORT
encoder (checked on 884 boxes of MOT17-09, no difference). The 131,100 boxes took 6 minutes on the RTX 3060 Ti.
They're saved as `det/<name>.npy`, one row per line of `det/<name>.txt`, and cut into halves with it.

The camera motion is the one BoT-SORT ships and reads by default (`--cmc-method file` in its `tools/track.py`):
`tracker/GMC_files/MOTChallenge`, one affine map per frame of each video. Its `MOT17_ablation` files are the
`val_half` lines of these, so the full files cut like the rest give the same input. `mot add-cmc` copies them to
`cmc/<name>.txt` and `--cmc` picks them:

```
# in a venv with torch, opencv, yacs, termcolor and pyyaml
python tools/reid_features.py <BoT-SORT clone> mot17_sbs_S50.pth data/MOT17/train yolox_ablation <out>
# here
uv run mot add-features <out> --dets yolox_ablation
uv run mot add-cmc <BoT-SORT clone>/tracker/GMC_files/MOTChallenge --name botsort
uv run mot track --tracker botsort --dets yolox_ablation --cmc botsort
```

With these, BoT-SORT gives the same output as the reference
([NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT) at 2519854): the same 48,656 boxes and the same
tracks on `val_half`, 104,759 on the full videos, and 48,694 and 104,863 without re-ID (`reid=false`).
`tools/botsort_reference.py` runs the reference tracker with a lookup in these features in place of the re-ID
model, and its own reader of the camera motion files:

```
python tools/botsort_reference.py <BoT-SORT clone> data/MOT17/val_half <out> yolox_ablation
uv run mot compare runs/MOT17-val_half/botsort-yolox_ablation/data <out>
```

## Results

MOT17 `val_half`, public detections. "reference" is the setting of the reference code, "tuned" the best HOTA of
a grid on `train_half` (`mot summary` prints these numbers from the runs):

| Tracker | Settings | Detections | HOTA | DetA | AssA | MOTA | IDF1 | ID switches |
|---|---|---|---|---|---|---|---|---|
| SORT | reference | DPM | 29.3 | 26.9 | 32.5 | 24.5 | 33.8 | 459 |
| SORT | tuned | DPM | 31.9 | 28.1 | 36.6 | 26.5 | 39.7 | 637 |
| ByteTrack | reference | DPM | 30.1 | 21.4 | 42.4 | 26.4 | 36.9 | 116 |
| ByteTrack | tuned | DPM | 33.0 | 25.9 | 42.3 | 30.3 | 41.2 | 139 |
| OC-SORT | reference | DPM | 25.3 | 16.7 | 38.6 | 20.7 | 30.7 | 105 |
| OC-SORT | tuned | DPM | 31.1 | 27.7 | 35.3 | 26.3 | 39.1 | 563 |
| SORT | reference | FRCNN | 48.7 | 42.7 | 55.8 | 46.7 | 54.6 | 252 |
| SORT | tuned | FRCNN | 51.4 | 43.3 | 61.4 | 47.4 | 58.6 | 371 |
| ByteTrack | reference | FRCNN | 49.4 | 42.5 | 57.7 | 47.5 | 56.2 | 198 |
| ByteTrack | tuned | FRCNN | 50.1 | 42.5 | 59.2 | 47.6 | 57.1 | 178 |
| OC-SORT | reference | FRCNN | 49.8 | 41.9 | 59.4 | 46.0 | 56.2 | 154 |
| OC-SORT | tuned | FRCNN | 50.5 | 43.1 | 59.3 | 47.4 | 56.5 | 241 |
| SORT | reference | SDP | 53.9 | 52.6 | 55.4 | 60.0 | 62.6 | 367 |
| SORT | tuned | SDP | 56.9 | 54.3 | 59.7 | 62.4 | 67.5 | 471 |
| ByteTrack | reference | SDP | 56.4 | 54.1 | 59.0 | 62.9 | 67.4 | 282 |
| ByteTrack | tuned | SDP | 56.3 | 54.4 | 58.3 | 63.3 | 67.5 | 342 |
| OC-SORT | reference | SDP | 53.6 | 50.0 | 57.5 | 58.9 | 66.0 | 284 |
| OC-SORT | tuned | SDP | 54.8 | 52.9 | 56.9 | 62.3 | 66.9 | 491 |

Tuned settings:

| | DPM | FRCNN | SDP |
|---|---|---|---|
| SORT: min_score, max_age, min_hits | -0.25, 120, 1 | 0.6, 60, 0 | 0.6, 60, 1 |
| ByteTrack: track_thresh, track_buffer, match_thresh | 0.25, 30, 0.95 (low_thresh -0.5) | 0.7, 120, 0.95 | 0.4, 60, 0.8 |
| OC-SORT: det_thresh, max_age, min_hits, iou_threshold, inertia | -0.25, 30, 1, 0.2, 0 | 0.8, 120, 0, 0.2, 0.2 | 0.6, 60, 1, 0.2, 0.1 |

1. **With the public detections, the three trackers are within about 2 HOTA points of each other once tuned**,
   while the detector moves HOTA by 25 points. A tuned SORT is first with FRCNN (51.4, OC-SORT 50.5, ByteTrack
   50.1) and SDP (56.9, ByteTrack 56.3, OC-SORT 54.8), ByteTrack with DPM (33.0, SORT 31.9, OC-SORT 31.1). Most of the gap at the
   reference settings comes from the lost tracks: SORT drops a track after one missed frame, the two others keep
   it 30 frames. ByteTrack still switches ids less often (178 vs 371 for SORT with FRCNN).
2. **ByteTrack's second association needs low score boxes.** Turning it off (`low_thresh=0.6`) costs 0.2 HOTA
   with FRCNN and SDP, and 2.0 with DPM. Only 4.4% of the FRCNN detections and 2.8% of the SDP ones are between
   0.1 and 0.6, against 16.6% for DPM. 70% of the FRCNN scores are exactly 1.0 and SDP has none under 0.4. On
   the YOLOX boxes, kept down to a score of 0.01, it's worth 1.2 HOTA (see below).
3. **ORU, OCM and OCR don't help on these detections.** Turned off one at a time, each moves HOTA by at most 0.4
   with FRCNN and SDP, in both directions (table below). Pedestrians in MOT17 mostly walk in straight lines.
   The YOLOX boxes don't change this.
4. **Reporting the detection instead of the Kalman box costs OC-SORT 1.7 HOTA with SDP** (53.6 vs 55.3, LocA
   85.9 vs 87.7). The SDP boxes are the noisiest and the filter smooths them. With FRCNN the detection is a bit
   better (49.8 vs 49.6).
5. The tops of the grids are flat: on `train_half` the five best points are within 0.3 HOTA, so the pick
   between them is mostly noise. Tuning ByteTrack with SDP gives nothing on `val_half` (56.3 vs 56.4).

OC-SORT at the reference settings with parts changed, HOTA on MOT17 `val_half`:

| | DPM | FRCNN | SDP | YOLOX |
|---|---|---|---|---|
| OC-SORT | 25.3 | 49.8 | 53.6 | 66.4 |
| without ORU (`oru=false`) | 25.3 | 49.6 | 53.6 | 66.4 |
| without OCM (`inertia=0`) | 26.0 | 49.7 | 53.4 | 66.3 |
| without OCR (`ocr=false`) | 25.1 | 50.2 | 53.8 | 66.4 |
| without the three | 26.0 | 50.0 | 53.9 | 66.4 |
| Kalman box reported (`kalman_box=true`) | 25.8 | 49.6 | 55.3 | 66.4 |

The DPM scores aren't calibrated (-0.5 to 4.8). SORT keeps all of them at the reference settings, negative ones
included. The thresholds of ByteTrack and OC-SORT are made for scores between 0 and 1: at 0.6, OC-SORT keeps only
a third of the DPM boxes. The tuned runs lower them, to 0.25 for ByteTrack (low_thresh -0.5) and -0.25 for
OC-SORT.

MOT20 `val_half`, public detections:

| Tracker | Settings | HOTA | DetA | AssA | MOTA | IDF1 | ID switches |
|---|---|---|---|---|---|---|---|
| SORT | reference | 32.4 | 39.3 | 26.8 | 43.6 | 34.3 | 7,049 |
| SORT | tuned: max_age=60, min_hits=0 | 43.6 | 43.9 | 43.6 | 50.0 | 52.0 | 3,391 |

In the crowds of MOT20 people are hidden all the time, and keeping the lost tracks gives 11 points of HOTA. The
MOT20 detection scores are rounded to 0 or 1 (99% are 0 in MOT20-05), so there is no score threshold to tune, and
ByteTrack and OC-SORT, which split or filter the detections by score, can't use them. The YOLOX model of
ByteTrack for MOT20 was trained on all the MOT20 training videos, so it can't be scored on `val_half`.

MOT17 `val_half`, YOLOX detections of the ByteTrack ablation model, against the published numbers:

| Tracker | Metrics | Here | Published |
|---|---|---|---|
| ByteTrack | MOTA / IDF1 / ID switches (motmetrics) | 76.4 / 79.1 / 171 | 76.6 / 79.3 / 159 (ByteTrack README) |
| OC-SORT | HOTA / AssA / IDF1 / MOTA (TrackEval) | 66.4 / 69.1 / 78.1 / 74.5 | 66.5 / 68.9 / 77.7 / 74.9 (OC-SORT model zoo) |

ByteTrack computes its numbers with motmetrics in `tools/track.py`, and `tools/motmetrics_eval.py` does the same.
TrackEval gives a MOTA of 77.8 to the same files, as it first drops the tracker boxes matched to distractors and
static people. The reference trackers give the same output as here on these boxes, so the gaps come from the
detections, computed on other hardware and library versions than the authors'. Running the detector in fp32
instead of fp16, on the same GPU (`fp32` as 5th argument of `tools/yolox_detections.py`), moves the numbers by as
much: ByteTrack gets 76.6 / 79.2 / 167 and OC-SORT 66.2 / 68.7 / 77.7 / 74.5.

The same detections for every tracker, TrackEval:

| Tracker | Settings | HOTA | DetA | AssA | MOTA | IDF1 | ID switches |
|---|---|---|---|---|---|---|---|
| SORT | reference, min_score=0.6 | 62.9 | 64.3 | 62.0 | 74.3 | 70.8 | 365 |
| SORT | min_score=0.6, max_age=30 | 66.4 | 64.2 | 69.0 | 74.7 | 77.7 | 232 |
| ByteTrack | reference | 67.7 | 66.6 | 69.4 | 77.8 | 79.7 | 176 |
| ByteTrack | without the second association (`low_thresh=0.6`) | 66.5 | 65.7 | 67.8 | 76.8 | 77.5 | 200 |
| OC-SORT | reference | 66.4 | 64.2 | 69.1 | 74.5 | 78.1 | 211 |
| OC-SORT | `use_byte=true` | 66.7 | 65.5 | 68.5 | 75.2 | 77.8 | 266 |
| BoT-SORT | reference, with re-ID | 68.8 | 67.3 | 71.0 | 78.4 | 81.6 | 158 |
| BoT-SORT | without re-ID (`reid=false`) | 69.1 | 67.2 | 71.6 | 78.4 | 81.6 | 142 |
| BoT-SORT | without CMC (`cmc=false`) | 67.7 | 67.0 | 68.9 | 77.9 | 79.1 | 304 |
| BoT-SORT | without CMC and re-ID | 68.0 | 66.9 | 69.7 | 77.8 | 79.5 | 341 |

SORT has no score threshold of its own, so it gets the boxes from 0.6, the high threshold of the others. Nothing
is tuned on these detections: the detector was trained on `train_half`, and its boxes there are better than on
new frames. The same goes for the re-ID model, trained on the first half of the same videos: it has seen the
people of `val_half`, on other frames.

BoT-SORT against Table 1 of its paper (HOTA / MOTA / IDF1, TrackEval):

| Paper row | Paper | Here | Run here |
|---|---|---|---|
| ByteTrack, matching up to 0.8 | 67.88 / 77.66 / 79.77 | 67.7 / 77.5 / 79.4 | `bytetrack`, `match_thresh=0.8` |
| + Kalman state with the width | 68.12 / 77.67 / 79.89 | 68.0 / 77.8 / 79.5 | `botsort`, `cmc=false`, `reid=false` |
| BoT-SORT | 69.11 / 78.39 / 81.53 | 69.1 / 78.4 / 81.6 | `botsort`, `reid=false` |
| BoT-SORT-ReID | 69.17 / 78.46 / 82.07 | 68.8 / 78.4 / 81.6 | `botsort` |

The second row of the paper only changes the Kalman state. The run here is BoT-SORT without CMC and re-ID, which
also has the other details of its code (output of the new tracks, 10 px2 filter). With the fp32 detections,
BoT-SORT without re-ID gets 69.3 / 78.5 / 82.2: the gap to the paper with re-ID (0.4 HOTA, 0.5 IDF1) is within
what the detector numerics move.

6. **ByteTrack's second association is worth 1.2 HOTA on these boxes** (67.7 vs 66.5, IDF1 79.7 vs 77.5). 16.4%
   of the boxes are between 0.1 and 0.6, against 4.4% of the FRCNN ones. Without it, ByteTrack is level with
   OC-SORT.
7. **ORU, OCM and OCR together give nothing on MOT17, even on these boxes.** OC-SORT without the three scores 66.4,
   like the full tracker, and so does a SORT that keeps lost tracks 30 frames. The fp32 detections move OC-SORT
   by 0.2, so the differences of 0.1 in the ablation table are noise. In Table 5 of the OC-SORT paper the same
   three parts take HOTA from 64.9 to 66.5 on this split. The full tracker matches the 66.5, but the paper
   doesn't describe its baseline in enough detail to run it again. Its large gain is on DanceTrack (47.8 to
   52.1), which isn't covered here.
8. As with the public detections, most of SORT's gap comes from the lost tracks: keeping them 30 frames instead
   of 1 gives 3.5 HOTA and takes the ID switches from 365 to 232.
9. **The camera motion compensation is what makes BoT-SORT first, and only on the videos where the camera
   moves.** In the GMC files the camera of MOT17-05, -10, -11 and -13 moves 3 to 12 px per frame (median), the
   one of MOT17-02, -04 and -09 under 0.2 px. On the first four, CMC adds 1.3 to 6.6 HOTA (without re-ID), and
   the ID switches of MOT17-13 go from 147 to 3. On the static three nothing changes. Over the split it's +1.1
   HOTA and +2.1 IDF1, because the static MOT17-04 holds 45% of the ground truth boxes.
10. **The re-ID gives nothing measurable here**: 68.8 HOTA with it, 69.1 without (69.17 and 69.11 in the paper).
    By video it helps on MOT17-02, -10 and -11 (up to +2.6 HOTA) and costs on MOT17-04 and -09 (-1.5 and -3.8).
    As shown above, it can only change which overlapping detection a track takes.

HOTA by video, BoT-SORT on the YOLOX boxes:

| | 02 | 04 | 05 | 09 | 10 | 11 | 13 |
|---|---|---|---|---|---|---|---|
| camera shift, px per frame (median) | 0.16 | 0.06 | 4.63 | 0.07 | 2.95 | 4.34 | 11.59 |
| without CMC and re-ID | 47.8 | 79.4 | 63.6 | 68.1 | 52.0 | 66.2 | 63.8 |
| with CMC | 47.7 | 79.4 | 64.9 | 68.1 | 56.8 | 67.9 | 70.4 |
| with CMC and re-ID | 49.7 | 77.9 | 65.2 | 64.3 | 59.4 | 69.5 | 70.2 |

## Layout

```
src/mot_bench/
  data.py         MOTChallenge download (resume, SHA-256) and extraction, added detections, features, CMC
  mot.py          sequences, seqinfo.ini, gt / det / result / cmc files
  split.py        train_half / val_half
  boxes.py        box formats, IoU
  kalman.py       Kalman filter and box state models
  matching.py     Hungarian assignment
  trackers/       SORT, ByteTrack, OC-SORT, BoT-SORT
  tune.py         grid search on train_half
  run.py          runs a tracker over a split, writes the result files
  evaluate.py     TrackEval (HOTA, CLEAR, Identity)
  compare.py      checks two sets of results against each other
  cli.py          `mot` command
tools/
  bytetrack_reference.py   runs the official ByteTrack tracker on the same detections
  ocsort_reference.py      same for OC-SORT
  botsort_reference.py     same for BoT-SORT
  yolox_detections.py      YOLOX boxes of the ByteTrack ablation model, on a GPU
  motmetrics_eval.py       MOTA / IDF1 / ID switches computed as in ByteTrack
  reid_features.py         re-ID features of the BoT-SORT model, on a GPU
```

## Data and licences

- [MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT20/): CC BY-NC-SA 3.0.
  Nothing from the datasets is stored in this repository.
- YOLOX model: `bytetrack_ablation.pth.tar` from [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) (MIT),
  trained on MOT17 and [CrowdHuman](https://www.crowdhuman.org/), which is for non-commercial research only.
  Neither the weights nor the detections are stored here.
- Re-ID model: `mot17_sbs_S50.pth` from [NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT) (MIT,
  with [FastReID](https://github.com/JDAI-CV/fast-reid), Apache-2.0), trained on crops of MOT17 people. The
  camera motion files come from the same repository. The weights, the features and these files aren't stored
  here.
- Code: Apache-2.0.

## Author

**Julien Beloin**, Machine Learning & Computer Vision Engineer
([LinkedIn](https://www.linkedin.com/in/j-beloin), [GitHub](https://github.com/Zeta-12))
