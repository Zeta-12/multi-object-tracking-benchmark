import numpy as np
import pytest

from mot_bench import mot, split


@pytest.mark.parametrize("n", [600, 525, 1050, 429, 8, 11])
def test_halves_match_bytetrack(n):
    r = split.half_ranges(n)
    # convert_mot17_to_coco.py, 0-based image indices
    train_idx, val_idx = [0, n // 2], [n // 2 + 1, n - 1]
    assert r["train_half"] == (train_idx[0] + 1, train_idx[1] + 1)
    assert r["val_half"] == (val_idx[0] + 1, val_idx[1] + 1)
    frames = list(range(r["train_half"][0], r["train_half"][1] + 1))
    frames += list(range(r["val_half"][0], r["val_half"][1] + 1))
    assert frames == list(range(1, n + 1))


def test_cut_lines_renumbers_and_keeps_the_rest():
    lines = ["5,2,1.5,2,3,4,1,1,0.75", "6,2,1.5,2,3,4,1,1,0.5", "9,1,0,0,1,1,0,7,1"]
    assert split.cut_lines(lines, 6, 9) == ["1,2,1.5,2,3,4,1,1,0.5", "4,1,0,0,1,1,0,7,1"]


def test_make_halves(mot17):
    split.make_halves(mot17)
    full = mot.read_gt(mot17 / "train" / "MOT17-02-FRCNN" / "gt" / "gt.txt")
    val = mot.sequences(mot17.parent, "MOT17", "val_half", detector="FRCNN")[0]
    train = mot.sequences(mot17.parent, "MOT17", "train_half", detector="FRCNN")[0]
    # 11 frames: train 1..6, val 7..11
    assert (train.info.length, val.info.length) == (6, 5)
    assert (train.info.frame_offset, val.info.frame_offset) == (0, 6)
    assert len(train.gt()) + len(val.gt()) == len(full)

    # frame 1 of val_half is frame 7 of the video, with the same boxes
    v = val.gt()
    np.testing.assert_array_equal(v[v[:, 0] == 1][:, 1:], full[full[:, 0] == 7][:, 1:])
    assert val.image(1) == mot17 / "train" / "MOT17-02-FRCNN" / "img1" / "000007.jpg"
    assert val.det()[:, 0].max() == 5
