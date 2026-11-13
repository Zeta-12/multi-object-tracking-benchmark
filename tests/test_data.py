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


def test_add_detections_and_cut_them(mot17, tmp_path):
    from mot_bench import mot

    src = tmp_path / "yolox"
    src.mkdir()
    # frames 1 to 11, full precision values
    (src / "MOT17-02-FRCNN.txt").write_text(
        "".join(f"{f},-1,10.25,20.125,40.000000001,90.5,0.0123456789,-1,-1,-1\n" for f in range(1, 12))
    )
    assert data.add_detections(mot17.parent, "MOT17", src, "yolox") == ["MOT17-02-FRCNN"]

    seqs = mot.sequences(mot17.parent, "MOT17", "val_half", dets="yolox")
    assert [s.name for s in seqs] == ["MOT17-02-FRCNN"]
    det = seqs[0].det()
    # 11 frames: val_half is frames 7 to 11, renumbered from 1, nothing rounded
    assert det[:, 0].tolist() == [1, 2, 3, 4, 5]
    assert det[0, 4] == 40.000000001 and det[0, 6] == 0.0123456789
    # the public detections are still there
    assert len(mot.sequences(mot17.parent, "MOT17", "val_half")) == 3

    with pytest.raises(ValueError):
        data.add_detections(mot17.parent, "MOT17", src, "det")
    (src / "MOT17-99-FRCNN.txt").write_text("")
    with pytest.raises(FileNotFoundError):
        data.add_detections(mot17.parent, "MOT17", src, "yolox")


def test_add_features_follow_their_detections(mot17, tmp_path):
    import numpy as np

    from mot_bench import mot

    seq = mot17 / "train" / "MOT17-02-FRCNN"
    # det.txt of the fixture isn't sorted by frame: one feature per line, in the order of the file
    frames = np.loadtxt(seq / "det" / "det.txt", delimiter=",", usecols=0)
    src = tmp_path / "reid"
    src.mkdir()
    np.save(src / "MOT17-02-FRCNN.npy", np.column_stack([frames, np.arange(len(frames))]).astype(np.float16))
    assert data.add_features(mot17.parent, "MOT17", src, "det") == ["MOT17-02-FRCNN"]

    for split in ("train", "val_half", "train_half"):
        s = mot.sequences(mot17.parent, "MOT17", split, detector="FRCNN")[0]
        det, feats = s.det(), s.features()
        # each row still carries the frame of its detection, renumbered with the half
        offset = s.info.frame_offset
        np.testing.assert_array_equal(feats[:, 0] - offset, det[:, 0])
    assert mot.sequences(mot17.parent, "MOT17", "val_half", detector="SDP")[0].features() is None

    np.save(src / "MOT17-02-FRCNN.npy", np.zeros((3, 2), np.float16))
    with pytest.raises(ValueError):
        data.add_features(mot17.parent, "MOT17", src, "det")


def test_add_cmc_from_botsort_files(mot17, tmp_path):
    import numpy as np

    from mot_bench import mot

    src = tmp_path / "GMC_files"
    src.mkdir()
    # index from 0, tab separated, a trailing tab, as in BoT-SORT
    lines = [f"{i}\t1.000001\t0.000000\t{i}.5\t0.000000\t1.000001\t-{i}\t\n" for i in range(11)]
    (src / "GMC-MOT17-02.txt").write_text("".join(lines))
    added = data.add_cmc(mot17.parent, "MOT17", src, "botsort")
    assert added == ["MOT17-02-FRCNN", "MOT17-02-SDP"]

    first = (mot17 / "train" / "MOT17-02-SDP" / "cmc" / "botsort.txt").read_text().splitlines()[0]
    assert first == "1,1.000001,0.000000,0.5,0.000000,1.000001,-0"
    seqs = mot.sequences(mot17.parent, "MOT17", "val_half", cmc="botsort")
    assert [s.name for s in seqs] == ["MOT17-02-FRCNN", "MOT17-02-SDP"]
    warps = seqs[0].warps()
    # 11 frames: val_half starts at frame 7, index 6 of the file
    assert warps.shape == (5, 2, 3)
    np.testing.assert_array_equal(warps[0], [[1.000001, 0.0, 6.5], [0.0, 1.000001, -6.0]])
    # no cmc asked: None, a video without a file: no motion
    assert mot.sequences(mot17.parent, "MOT17", "val_half")[0].warps() is None
    np.testing.assert_array_equal(mot.read_cmc(seqs[0].dir / "cmc" / "botsort.txt", 7)[6], np.eye(2, 3))
