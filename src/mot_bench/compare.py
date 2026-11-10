"""Compare two folders of MOTChallenge results, e.g. a tracker here and the reference implementation.

Two outputs agree when every frame has the same boxes (to the 0.01 px of the files) and the track ids
are the same up to a relabelling, i.e. the ids of the boxes in common map one to one.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def _boxes(path: Path) -> dict[tuple, int]:
    rows = np.loadtxt(path, delimiter=",", ndmin=2) if path.stat().st_size else np.zeros((0, 6))
    return {(int(r[0]), *np.round(r[2:6], 2).tolist()): int(r[1]) for r in rows}


def _conflicts(pairs: list[tuple[int, int]]) -> int:
    forward, backward = defaultdict(set), defaultdict(set)
    for a, b in pairs:
        forward[a].add(b)
        backward[b].add(a)
    return sum(len(v) > 1 for v in forward.values()) + sum(len(v) > 1 for v in backward.values())


def compare_files(a: Path, b: Path) -> dict[str, int]:
    ka, kb = _boxes(a), _boxes(b)
    common = ka.keys() & kb.keys()
    return {
        "boxes_a": len(ka),
        "boxes_b": len(kb),
        "only_a": len(ka) - len(common),
        "only_b": len(kb) - len(common),
        "id_conflicts": _conflicts([(ka[k], kb[k]) for k in common]),
    }


def compare(dir_a: Path, dir_b: Path) -> dict[str, dict[str, int]]:
    """Per sequence, for the files of dir_b."""
    out = {}
    for b in sorted(Path(dir_b).glob("*.txt")):
        a = Path(dir_a) / b.name
        if not a.is_file():
            raise FileNotFoundError(a)
        out[b.stem] = compare_files(a, b)
    return out


def same(results: dict[str, dict[str, int]]) -> bool:
    return all(r["only_a"] == r["only_b"] == r["id_conflicts"] == 0 for r in results.values())
