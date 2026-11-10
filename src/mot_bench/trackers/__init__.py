"""Trackers. All take the detections of one frame as (n, 5) tlbr + score and return (m, 6) tlbr, id, score."""

from __future__ import annotations

from mot_bench.trackers.sort import Sort

TRACKERS = {"sort": Sort}


def make(name: str, **params):
    if name not in TRACKERS:
        raise KeyError(f"unknown tracker {name!r}, available: {', '.join(TRACKERS)}")
    return TRACKERS[name](**params)
