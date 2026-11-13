"""MOTChallenge downloads.

Archives are kept in <data_dir>/downloads and extracted to <data_dir>/<dataset>/{train,test}/<seq>/.
The "labels" archives hold seqinfo.ini, gt.txt and det.txt only, which is all the trackers need when
they run on the public detections. The images are only needed to run a detector, re-ID or camera
motion compensation.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import requests

from mot_bench import mot, split

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Archive:
    url: str
    sha256: str
    size: int


ARCHIVES = {
    ("MOT17", "labels"): Archive(
        "https://motchallenge.net/data/MOT17Labels.zip",
        "0aa79322e91583369f42f17c4d79a0b145380d8732487bba59272048dc82b2b9",
        10_107_022,
    ),
    ("MOT17", "images"): Archive(
        "https://motchallenge.net/data/MOT17.zip",
        "7b486bb2af5cfedbfabc020511ff933a3edbf12182ce4985eec1a210f20f9e06",
        5_860_214_001,
    ),
    ("MOT20", "labels"): Archive(
        "https://motchallenge.net/data/MOT20Labels.zip",
        "95decf629c332f80ace18024bf876996e6be89908ec593f05cda000270d6d9ad",
        19_246_343,
    ),
}

CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def fetch(archive: Archive, dest: Path, retries: int = 5) -> Path:
    """Download with resume (the image archives are 5-6 GB), then check the SHA-256."""
    if dest.exists() and dest.stat().st_size == archive.size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        done = part.stat().st_size if part.exists() else 0
        if done >= archive.size:
            break
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(archive.url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                if done and r.status_code != 206:
                    done = 0  # range ignored, start over
                log.info("%s: %.0f / %.0f MB", dest.name, done / 1e6, archive.size / 1e6)
                with part.open("ab" if done else "wb") as f:
                    for block in r.iter_content(CHUNK):
                        f.write(block)
            break
        except (requests.ConnectionError, requests.Timeout) as err:
            if attempt == retries:
                raise
            log.warning("%s, retrying in %d s", err, 10 * attempt)
            time.sleep(10 * attempt)
    digest = sha256(part)
    if digest != archive.sha256:
        part.unlink()
        raise RuntimeError(f"{dest.name}: SHA-256 {digest} does not match {archive.sha256}, file removed")
    part.replace(dest)
    return dest


def member_path(name: str) -> PurePosixPath | None:
    """Path inside the archive from its train/ or test/ folder (MOT20Labels.zip has an extra top folder)."""
    parts = PurePosixPath(name).parts
    if ".." in parts:
        raise ValueError(f"unsafe path in archive: {name}")
    for i, part in enumerate(parts):
        if part in ("train", "test"):
            return PurePosixPath(*parts[i:])
    return None


def _duplicate_frames(rel: PurePosixPath) -> bool:
    # train/MOT17-02-SDP/img1/000001.jpg, same file as in MOT17-02-FRCNN
    return len(rel.parts) > 2 and rel.parts[2] == "img1" and mot.video_name(rel.parts[1]) != rel.parts[1]


def extract(archive: Path, out_dir: Path) -> int:
    """Extract into out_dir/{train,test}/... and skip files already there. Returns the number of files written."""
    written = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            rel = member_path(info.filename)
            if info.is_dir() or rel is None or _duplicate_frames(rel):
                continue
            target = out_dir / rel
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as dst:
                while block := src.read(CHUNK):
                    dst.write(block)
            written += 1
    return written


def download(data_dir: Path, dataset: str, images: bool = False) -> Path:
    kinds = ["labels", "images"] if images else ["labels"]
    out = Path(data_dir) / dataset
    for kind in kinds:
        if (dataset, kind) not in ARCHIVES:
            raise KeyError(f"no {kind} archive registered for {dataset}")
        archive = ARCHIVES[dataset, kind]
        path = fetch(archive, Path(data_dir) / "downloads" / archive.url.rsplit("/", 1)[-1])
        n = extract(path, out)
        log.info("%s: %d files extracted to %s", path.name, n, out)
    halves = split.make_halves(out)
    log.info("%d half sequences written", len(halves))
    return out


def add_detections(data_dir: Path, dataset: str, src_dir: Path, name: str) -> list[str]:
    """Copy <src_dir>/<seq>.txt to det/<name>.txt of each training sequence, then cut the halves again."""
    if name == "det":
        raise ValueError("det is the name of the public detections")
    train = Path(data_dir) / dataset / "train"
    added = []
    for src in sorted(Path(src_dir).glob("*.txt")):
        seq_dir = train / src.stem
        if not (seq_dir / "seqinfo.ini").is_file():
            raise FileNotFoundError(f"no sequence {src.stem} in {train}")
        shutil.copyfile(src, seq_dir / "det" / f"{name}.txt")
        added.append(src.stem)
    split.make_halves(Path(data_dir) / dataset)
    return added
