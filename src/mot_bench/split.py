"""train_half / val_half splits of the training sequences.

The test ground truth of MOT17 and MOT20 is only on the evaluation server, so ablations are reported on
the second half of each training video (ByteTrack, OC-SORT and BoT-SORT all do). The frame ranges are
the ones of ByteTrack's tools/convert_mot17_to_coco.py: with n frames, train_half is 1..n//2+1 and
val_half is n//2+2..n, renumbered from 1. Published val_half numbers are only comparable with the same cut.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mot_bench.mot import read_seqinfo, write_seqinfo


def half_ranges(n: int) -> dict[str, tuple[int, int]]:
    """First and last frame (inclusive, 1-based) of each half."""
    return {"train_half": (1, n // 2 + 1), "val_half": (n // 2 + 2, n)}


def cut_lines(lines: list[str], first: int, last: int) -> list[str]:
    """Keep the lines of frames first..last and renumber them from 1. The rest of each line is untouched."""
    out = []
    for line in lines:
        if not line.strip():
            continue
        frame, rest = line.split(",", 1)
        f = int(float(frame))
        if first <= f <= last:
            out.append(f"{f - first + 1},{rest}")
    return out


def _cut_file(src: Path, dst: Path, first: int, last: int) -> int:
    lines = cut_lines(src.read_text().splitlines(), first, last)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(line + "\n" for line in lines))
    return len(lines)


def make_halves(dataset_dir: Path) -> list[Path]:
    """Write <dataset_dir>/{train_half,val_half}/<seq>/ (seqinfo, gt, det) from <dataset_dir>/train."""
    written = []
    for seq_dir in sorted(p for p in (Path(dataset_dir) / "train").iterdir() if (p / "seqinfo.ini").is_file()):
        info = read_seqinfo(seq_dir / "seqinfo.ini")
        for split, (first, last) in half_ranges(info.length).items():
            out = Path(dataset_dir) / split / info.name
            half = replace(info, length=last - first + 1, frame_offset=first - 1)
            write_seqinfo(half, out / "seqinfo.ini")
            for sub in ("gt/gt.txt", "det/det.txt"):
                if (seq_dir / sub).is_file():
                    _cut_file(seq_dir / sub, out / sub, first, last)
            written.append(out)
    return written
