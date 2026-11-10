"""Run a tracker over the sequences of a split and write MOTChallenge result files."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mot_bench import mot, trackers
from mot_bench.boxes import tlbr_to_tlwh, tlwh_to_tlbr

log = logging.getLogger(__name__)


@dataclass
class SeqRun:
    name: str
    frames: int
    detections: int
    seconds: float

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else float("inf")


def detections_by_frame(det: np.ndarray, length: int, min_score: float | None = None) -> list[np.ndarray]:
    """Detections of frames 1..length as (n, 5) tlbr + score."""
    if min_score is not None:
        det = det[det[:, 6] >= min_score]
    boxes = np.concatenate([tlwh_to_tlbr(det[:, 2:6]), det[:, 6:7]], axis=1)
    # det is sorted by frame
    bounds = np.searchsorted(det[:, 0], np.arange(1, length + 2))
    return [boxes[bounds[f] : bounds[f + 1]] for f in range(length)]


def track_sequence(tracker, seq: mot.Sequence, min_score: float | None = None) -> tuple[np.ndarray, SeqRun]:
    """Result rows (frame, id, x, y, w, h, score) and the timing of the tracker alone."""
    frames = detections_by_frame(seq.det(), seq.info.length, min_score)
    rows = []
    start = time.perf_counter()
    for f, dets in enumerate(frames, start=1):
        out = tracker.update(dets)
        if len(out):
            rows.append(np.column_stack([np.full(len(out), f), out[:, 4], tlbr_to_tlwh(out[:, :4]), out[:, 5]]))
    seconds = time.perf_counter() - start
    result = np.concatenate(rows) if rows else np.zeros((0, 7))
    return result, SeqRun(seq.name, seq.info.length, sum(len(d) for d in frames), seconds)


def run(
    seqs: list[mot.Sequence],
    tracker: str,
    params: dict,
    out_dir: Path,
    min_score: float | None = None,
) -> list[SeqRun]:
    """Results go to out_dir/data/<seq>.txt, the TrackEval layout. A new tracker is built for every sequence."""
    runs = []
    for seq in seqs:
        rows, info = track_sequence(trackers.make(tracker, **params), seq, min_score)
        mot.write_results(out_dir / "data" / f"{seq.name}.txt", rows)
        log.info("%s: %d boxes, %.0f fps", seq.name, len(rows), info.fps)
        runs.append(info)
    config = {"tracker": tracker, "params": params, "min_score": min_score}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return runs
