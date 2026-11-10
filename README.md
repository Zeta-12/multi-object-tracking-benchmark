# Multi-Object Tracking Benchmark

[![CI](https://github.com/Zeta-12/multi-object-tracking-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Zeta-12/multi-object-tracking-benchmark/actions/workflows/ci.yml)

Pedestrian trackers of the SORT family (SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT) written from scratch with
NumPy, and compared on MOT17 and MOT20 with [TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA, MOTA,
IDF1). All trackers get the same detections, so the differences come from the association step only, and each
implementation has to match the numbers published by its authors on the same split.

Work in progress: the data, the evaluation and SORT are in place. ByteTrack, OC-SORT, DeepSORT and BoT-SORT come
next.

## Setup

```
uv sync
uv run mot download                     # MOT17 annotations and public detections, 10 MB
uv run mot info --split val_half
uv run mot track --tracker sort --detector FRCNN    # runs on val_half and prints the metrics
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

## Trackers

All trackers take the detections of one frame and return the tracked boxes with their ids. They share the Kalman
filter (`kalman.py`), the IoU (`boxes.py`) and the assignment (`matching.py`).

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

I wrote the trackers with the help of an AI assistant. Each one is checked against its reference implementation,
as above, before its results are reported here.

## Results

MOT17 `val_half`, public detections, reference settings (every detection is used, as in the SORT code):

| Tracker | Detections | HOTA | DetA | AssA | MOTA | IDF1 | ID switches | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| SORT | DPM | 29.3 | 26.9 | 32.5 | 24.5 | 33.8 | 459 | 5,933 | 34,291 |
| SORT | FRCNN | 48.7 | 42.7 | 55.8 | 46.7 | 54.6 | 252 | 631 | 27,831 |
| SORT | SDP | 53.9 | 52.6 | 55.4 | 60.0 | 62.6 | 367 | 351 | 20,861 |

The DPM scores aren't calibrated (-0.5 to 4.8) and SORT keeps all of them, negative ones included, hence the
false positives. The thresholds will be tuned on `train_half`.

## Layout

```
src/mot_bench/
  data.py         MOTChallenge download (resume, SHA-256) and extraction
  mot.py          sequences, seqinfo.ini, gt / det / result files
  split.py        train_half / val_half
  boxes.py        box formats, IoU
  kalman.py       Kalman filter and box state models
  matching.py     Hungarian assignment
  trackers/       SORT
  run.py          runs a tracker over a split, writes the result files
  evaluate.py     TrackEval (HOTA, CLEAR, Identity)
  compare.py      checks two sets of results against each other
  cli.py          `mot` command
```

## Data and licences

- [MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT20/): CC BY-NC-SA 3.0.
  Nothing from the datasets is stored in this repository.
- Code: Apache-2.0.

## Author

**Julien Beloin**, Machine Learning & Computer Vision Engineer
([LinkedIn](https://www.linkedin.com/in/j-beloin), [GitHub](https://github.com/Zeta-12))
