# Multi-Object Tracking Benchmark

[![CI](https://github.com/Zeta-12/multi-object-tracking-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Zeta-12/multi-object-tracking-benchmark/actions/workflows/ci.yml)

Pedestrian trackers of the SORT family (SORT, DeepSORT, ByteTrack, OC-SORT, BoT-SORT) written from scratch with
NumPy, and compared on MOT17 and MOT20 with [TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA, MOTA,
IDF1). All trackers get the same detections, so the differences come from the association step only, and each
implementation has to match the numbers published by its authors on the same split.

Work in progress: data download and the evaluation split are in place, the trackers come next.

## Setup

```
uv sync
uv run mot download                     # MOT17 annotations and public detections, 10 MB
uv run mot info --split val_half
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

## Layout

```
src/mot_bench/
  data.py         MOTChallenge download (resume, SHA-256) and extraction
  mot.py          sequences, seqinfo.ini, gt / det / result files
  split.py        train_half / val_half
  cli.py          `mot` command
```

## Data and licences

- [MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT20/): CC BY-NC-SA 3.0.
  Nothing from the datasets is stored in this repository.
- Code: Apache-2.0.

## Author

**Julien Beloin**, Machine Learning & Computer Vision Engineer
([LinkedIn](https://www.linkedin.com/in/j-beloin), [GitHub](https://github.com/Zeta-12))
