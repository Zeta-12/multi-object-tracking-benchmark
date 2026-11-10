"""MOTChallenge sequences and text files.

One line per box, comma separated, frames start at 1, boxes are (left, top, width, height) in pixels:

    gt.txt       frame, id, x, y, w, h, flag, class, visibility   (flag 0 = not evaluated)
    det.txt      frame, -1, x, y, w, h, score[, -1, -1, -1]
    results      frame, id, x, y, w, h, score, -1, -1, -1
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PEDESTRIAN = 1


@dataclass(frozen=True)
class SeqInfo:
    name: str
    length: int
    fps: float
    width: int
    height: int
    im_dir: str = "img1"
    im_ext: str = ".jpg"
    # frames removed before frame 1, only set on the half splits
    frame_offset: int = 0


def read_seqinfo(path: Path) -> SeqInfo:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    if not parser.read(path):
        raise FileNotFoundError(path)
    s = parser["Sequence"]
    return SeqInfo(
        name=s["name"],
        length=int(s["seqLength"]),
        fps=float(s["frameRate"]),
        width=int(s["imWidth"]),
        height=int(s["imHeight"]),
        im_dir=s.get("imDir", "img1"),
        im_ext=s.get("imExt", ".jpg"),
        frame_offset=int(s.get("frameOffset", 0)),
    )


def write_seqinfo(info: SeqInfo, path: Path) -> None:
    fps = int(info.fps) if float(info.fps).is_integer() else info.fps
    lines = [
        "[Sequence]",
        f"name={info.name}",
        f"imDir={info.im_dir}",
        f"frameRate={fps}",
        f"seqLength={info.length}",
        f"imWidth={info.width}",
        f"imHeight={info.height}",
        f"imExt={info.im_ext}",
    ]
    if info.frame_offset:
        lines.append(f"frameOffset={info.frame_offset}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_txt(path: Path, ncols: int) -> np.ndarray:
    """First `ncols` columns as float64, sorted by frame (the FRCNN detections of MOT17 aren't)."""
    if Path(path).stat().st_size == 0:
        return np.zeros((0, ncols))
    rows = np.loadtxt(path, delimiter=",", ndmin=2)[:, :ncols]
    return rows[np.argsort(rows[:, 0], kind="stable")]


def read_gt(path: Path) -> np.ndarray:
    return read_txt(path, 9)


def read_det(path: Path) -> np.ndarray:
    return read_txt(path, 7)


def write_results(path: Path, rows: np.ndarray) -> None:
    """rows: frame, id, x, y, w, h, score."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for frame, tid, x, y, w, h, score in rows:
            f.write(f"{int(frame)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1\n")


def evaluated(gt: np.ndarray) -> np.ndarray:
    """Boxes that count in the metrics: pedestrians with flag 1."""
    return gt[(gt[:, 6] == 1) & (gt[:, 7] == PEDESTRIAN)]


@dataclass(frozen=True)
class Sequence:
    dir: Path
    info: SeqInfo
    img_dir: Path

    @property
    def name(self) -> str:
        return self.info.name

    def gt(self) -> np.ndarray:
        return read_gt(self.dir / "gt" / "gt.txt")

    def det(self) -> np.ndarray:
        return read_det(self.dir / "det" / "det.txt")

    def image(self, frame: int) -> Path:
        """Image of `frame`, counted in this sequence (so frame 1 of val_half is the first val image)."""
        return self.img_dir / f"{frame + self.info.frame_offset:06d}{self.info.im_ext}"


def video_name(name: str) -> str:
    """The three MOT17 copies of a video (DPM, FRCNN, SDP) share the same frames, only FRCNN keeps them."""
    for suffix in ("-DPM", "-SDP"):
        if name.endswith(suffix):
            return name[: -len(suffix)] + "-FRCNN"
    return name


def sequences(data_dir: Path, dataset: str, split: str, detector: str | None = None) -> list[Sequence]:
    """Sequences of <data_dir>/<dataset>/<split>. `detector` keeps one MOT17 copy (DPM, FRCNN or SDP)."""
    split_dir = Path(data_dir) / dataset / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"{split_dir} not found, run `mot download --dataset {dataset}` first")
    source = Path(data_dir) / dataset / ("train" if split.endswith("_half") else split)
    out = []
    for seq_dir in sorted(p for p in split_dir.iterdir() if (p / "seqinfo.ini").is_file()):
        if detector and not seq_dir.name.endswith(f"-{detector}"):
            continue
        info = read_seqinfo(seq_dir / "seqinfo.ini")
        out.append(Sequence(seq_dir, info, source / video_name(info.name) / info.im_dir))
    return out
