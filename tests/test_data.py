import hashlib
import zipfile

import pytest
import requests

from mot_bench import data


def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return path


def test_extract_strips_the_top_folder_and_skips_copied_frames(tmp_path):
    archive = make_zip(
        tmp_path / "a.zip",
        {
            "MOT17/train/MOT17-02-FRCNN/img1/000001.jpg": b"jpg",
            "MOT17/train/MOT17-02-SDP/img1/000001.jpg": b"jpg",
            "MOT17/train/MOT17-02-SDP/det/det.txt": b"1,-1,0,0,1,1,0.5\n",
            "MOT17/readme.txt": b"",
        },
    )
    out = tmp_path / "MOT17"
    assert data.extract(archive, out) == 2
    assert (out / "train" / "MOT17-02-FRCNN" / "img1" / "000001.jpg").is_file()
    assert (out / "train" / "MOT17-02-SDP" / "det" / "det.txt").is_file()
    assert not (out / "train" / "MOT17-02-SDP" / "img1").exists()
    assert data.extract(archive, out) == 0


def test_unsafe_member_is_refused():
    with pytest.raises(ValueError):
        data.member_path("train/../../evil.txt")


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.body


def test_fetch_resumes_after_a_dropped_connection(tmp_path, monkeypatch):
    body = b"0123456789" * 100
    archive = data.Archive("https://example.org/x.zip", hashlib.sha256(body).hexdigest(), len(body))
    calls = []

    def get(url, headers, stream, timeout):
        calls.append(headers)
        if len(calls) == 1:
            (tmp_path / "x.zip.part").write_bytes(body[:400])
            raise requests.ConnectionError("reset")
        start = int(headers["Range"].split("=")[1].rstrip("-"))
        return FakeResponse(body[start:], status=206)

    monkeypatch.setattr(data.requests, "get", get)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    path = data.fetch(archive, tmp_path / "x.zip")
    assert path.read_bytes() == body
    assert calls == [{}, {"Range": "bytes=400-"}]
    assert not (tmp_path / "x.zip.part").exists()


def test_fetch_rejects_a_bad_checksum(tmp_path, monkeypatch):
    archive = data.Archive("https://example.org/x.zip", "0" * 64, 3)
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: FakeResponse(b"abc"))
    with pytest.raises(RuntimeError, match="SHA-256"):
        data.fetch(archive, tmp_path / "x.zip")
    assert not (tmp_path / "x.zip.part").exists()


def test_download_builds_the_halves(tmp_path, monkeypatch):
    seqinfo = "[Sequence]\nname=MOT20-01\nimDir=img1\nframeRate=25\nseqLength=9\nimWidth=1920\nimHeight=1080\n"
    archive = make_zip(
        tmp_path / "MOT20Labels.zip",
        {
            "MOT20Labels/train/MOT20-01/seqinfo.ini": seqinfo,
            "MOT20Labels/train/MOT20-01/gt/gt.txt": "".join(f"{f},1,0,0,10,10,1,1,1\n" for f in range(1, 10)),
            "MOT20Labels/train/MOT20-01/det/det.txt": "".join(f"{f},-1,0,0,10,10,0.9\n" for f in range(1, 10)),
        },
    )
    monkeypatch.setattr(data, "fetch", lambda a, dest: archive)
    out = data.download(tmp_path / "data", "MOT20")
    assert (out / "val_half" / "MOT20-01" / "gt" / "gt.txt").read_text().splitlines()[0] == "1,1,0,0,10,10,1,1,1"
    assert len((out / "val_half" / "MOT20-01" / "det" / "det.txt").read_text().splitlines()) == 4
