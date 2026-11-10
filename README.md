# Multi-Object Tracking Benchmark

[![CI](https://github.com/Zeta-12/multi-object-tracking-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Zeta-12/multi-object-tracking-benchmark/actions/workflows/ci.yml)

Pedestrian trackers of the SORT family (SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT) written from scratch with
NumPy, and compared on MOT17 and MOT20 with [TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA, MOTA,
IDF1). All trackers get the same detections, so the differences come from the association step only, and each
implementation has to match the numbers published by its authors on the same split.

Work in progress: the data, the evaluation, SORT and ByteTrack are in place. OC-SORT, DeepSORT and BoT-SORT come
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

I wrote the trackers with the help of an AI assistant. Each one is checked against its reference implementation,
as above, before its results are reported here.

## Results

MOT17 `val_half`, public detections. "reference" is the setting of the reference code, "tuned" the best HOTA of
a grid on `train_half` (`mot summary` prints these numbers from the runs):

| Tracker | Settings | Detections | HOTA | DetA | AssA | MOTA | IDF1 | ID switches |
|---|---|---|---|---|---|---|---|---|
| SORT | reference | DPM | 29.3 | 26.9 | 32.5 | 24.5 | 33.8 | 459 |
| SORT | tuned | DPM | 31.9 | 28.1 | 36.6 | 26.5 | 39.7 | 637 |
| ByteTrack | reference | DPM | 30.1 | 21.4 | 42.4 | 26.4 | 36.9 | 116 |
| ByteTrack | tuned | DPM | 33.0 | 25.9 | 42.3 | 30.3 | 41.2 | 139 |
| SORT | reference | FRCNN | 48.7 | 42.7 | 55.8 | 46.7 | 54.6 | 252 |
| SORT | tuned | FRCNN | 51.4 | 43.3 | 61.4 | 47.4 | 58.6 | 371 |
| ByteTrack | reference | FRCNN | 49.4 | 42.5 | 57.7 | 47.5 | 56.2 | 198 |
| ByteTrack | tuned | FRCNN | 50.1 | 42.5 | 59.2 | 47.6 | 57.1 | 178 |
| SORT | reference | SDP | 53.9 | 52.6 | 55.4 | 60.0 | 62.6 | 367 |
| SORT | tuned | SDP | 56.9 | 54.3 | 59.7 | 62.4 | 67.5 | 471 |
| ByteTrack | reference | SDP | 56.4 | 54.1 | 59.0 | 62.9 | 67.4 | 282 |
| ByteTrack | tuned | SDP | 56.3 | 54.4 | 58.3 | 63.3 | 67.5 | 342 |

Tuned settings:

| | DPM | FRCNN | SDP |
|---|---|---|---|
| SORT: min_score, max_age, min_hits | -0.25, 120, 1 | 0.6, 60, 0 | 0.6, 60, 1 |
| ByteTrack: track_thresh, track_buffer, match_thresh | 0.25, 30, 0.95 (low_thresh -0.5) | 0.7, 120, 0.95 | 0.4, 60, 0.8 |

1. **With the public detections, a tuned SORT is level with ByteTrack** (HOTA 51.4 vs 50.1 with FRCNN, 56.9
   vs 56.3 with SDP, 31.9 vs 33.0 with DPM). Most of the gap at the reference settings comes from the lost
   tracks: SORT drops a track after one missed frame, ByteTrack keeps it 30 frames. ByteTrack still switches
   ids less often (178 vs 371 with FRCNN).
2. **ByteTrack's second association needs low score boxes.** Turning it off (`low_thresh=0.6`) costs 0.2 HOTA
   with FRCNN and SDP, and 2.0 with DPM. Only 4.4% of the FRCNN detections and 2.8% of the SDP ones are between
   0.1 and 0.6, against 16.6% for DPM. 70% of the FRCNN scores are exactly 1.0 and SDP has none under 0.4. The
   ByteTrack paper gets its gains on YOLOX boxes kept down to a score of 0.1, which come next here.
3. The tops of the grids are flat: on `train_half` the five best points are within 0.3 HOTA, so the pick
   between them is mostly noise. Tuning ByteTrack with SDP gives nothing on `val_half` (56.3 vs 56.4).

The DPM scores aren't calibrated (-0.5 to 4.8). SORT keeps all of them at the reference settings, negative ones
included, and ByteTrack's thresholds are made for scores between 0 and 1: its tuned run lowers them.

MOT20 `val_half`, public detections:

| Tracker | Settings | HOTA | DetA | AssA | MOTA | IDF1 | ID switches |
|---|---|---|---|---|---|---|---|
| SORT | reference | 32.4 | 39.3 | 26.8 | 43.6 | 34.3 | 7,049 |
| SORT | tuned: max_age=60, min_hits=0 | 43.6 | 43.9 | 43.6 | 50.0 | 52.0 | 3,391 |

In the crowds of MOT20 people are hidden all the time, and keeping the lost tracks gives 11 points of HOTA. The
MOT20 detection scores are rounded to 0 or 1 (99% are 0 in MOT20-05), so there is no score threshold to tune, and
ByteTrack, which splits the detections by score, can't use them. It will run on the YOLOX detections.

## Layout

```
src/mot_bench/
  data.py         MOTChallenge download (resume, SHA-256) and extraction
  mot.py          sequences, seqinfo.ini, gt / det / result files
  split.py        train_half / val_half
  boxes.py        box formats, IoU
  kalman.py       Kalman filter and box state models
  matching.py     Hungarian assignment
  trackers/       SORT, ByteTrack
  tune.py         grid search on train_half
  run.py          runs a tracker over a split, writes the result files
  evaluate.py     TrackEval (HOTA, CLEAR, Identity)
  compare.py      checks two sets of results against each other
  cli.py          `mot` command
tools/
  bytetrack_reference.py   runs the official ByteTrack tracker on the same detections
```

## Data and licences

- [MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT20/): CC BY-NC-SA 3.0.
  Nothing from the datasets is stored in this repository.
- Code: Apache-2.0.

## Author

**Julien Beloin**, Machine Learning & Computer Vision Engineer
([LinkedIn](https://www.linkedin.com/in/j-beloin), [GitHub](https://github.com/Zeta-12))
