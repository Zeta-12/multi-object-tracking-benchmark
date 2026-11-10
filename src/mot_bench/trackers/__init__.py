"""Trackers. All take the detections of one frame as (n, 5) tlbr + score and return (m, 6) tlbr, id, score."""

from __future__ import annotations

import inspect

from mot_bench.trackers.bytetrack import ByteTrack
from mot_bench.trackers.sort import Sort

TRACKERS = {"sort": Sort, "bytetrack": ByteTrack}


def make(name: str, params: dict | None = None, frame_rate: float | None = None):
    """frame_rate goes to the trackers that take one, unless params sets it."""
    if name not in TRACKERS:
        raise KeyError(f"unknown tracker {name!r}, available: {', '.join(TRACKERS)}")
    cls = TRACKERS[name]
    params = dict(params or {})
    if frame_rate is not None and "frame_rate" in inspect.signature(cls).parameters:
        params.setdefault("frame_rate", frame_rate)
    return cls(**params)
