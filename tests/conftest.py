from pathlib import Path

import numpy as np
import pytest

from mot_bench.mot import SeqInfo, write_seqinfo


def write_sequence(root: Path, name: str, length: int = 10, n_objects: int = 3, seed: int = 0) -> Path:
    """A small sequence in MOTChallenge layout: objects moving right, one box per object and frame."""
    rng = np.random.default_rng(seed)
    seq_dir = root / name
    write_seqinfo(SeqInfo(name, length, 30.0, 1920, 1080), seq_dir / "seqinfo.ini")
    gt, det = [], []
    for k in range(n_objects):
        x0, y0 = rng.uniform(0, 1500), rng.uniform(0, 800)
        for f in range(1, length + 1):
            x = x0 + 5 * f
            # the last object is a static person (class 7), not evaluated
            cls = 7 if k == n_objects - 1 else 1
            gt.append(f"{f},{k + 1},{x:.0f},{y0:.0f},50,120,1,{cls},1")
            det.append(f"{f},-1,{x + 1:.1f},{y0 - 1:.1f},50.5,119.2,{rng.uniform(0.3, 1):.3f}")
    # det.txt of MOT17 FRCNN isn't sorted by frame
    det = det[::-1]
    (seq_dir / "gt").mkdir()
    (seq_dir / "det").mkdir()
    (seq_dir / "gt" / "gt.txt").write_text("\n".join(gt) + "\n")
    (seq_dir / "det" / "det.txt").write_text("\n".join(det) + "\n")
    return seq_dir


@pytest.fixture
def mot17(tmp_path):
    """<tmp>/MOT17/train with two copies of one video and a second video."""
    train = tmp_path / "MOT17" / "train"
    write_sequence(train, "MOT17-02-FRCNN", length=11)
    write_sequence(train, "MOT17-02-SDP", length=11, seed=1)
    write_sequence(train, "MOT17-04-FRCNN", length=8, n_objects=2, seed=2)
    return tmp_path / "MOT17"
