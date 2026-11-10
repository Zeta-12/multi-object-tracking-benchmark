import numpy as np

from mot_bench import mot
from mot_bench.mot import SeqInfo


def test_seqinfo_round_trip(tmp_path):
    info = SeqInfo("MOT20-01", 214, 25.0, 1920, 1080, frame_offset=215)
    mot.write_seqinfo(info, tmp_path / "seqinfo.ini")
    assert mot.read_seqinfo(tmp_path / "seqinfo.ini") == info
    assert "frameRate=25\n" in (tmp_path / "seqinfo.ini").read_text()


def test_det_is_sorted_by_frame_and_cut_to_7_columns(tmp_path):
    p = tmp_path / "det.txt"
    p.write_text("3,-1,1,2,3,4,0.5,-1,-1,-1\n1,-1,5,6,7,8,0.9,-1,-1,-1\n2,-1,1,1,1,1,0.1,-1,-1,-1\n")
    det = mot.read_det(p)
    assert det.shape == (3, 7)
    assert det[:, 0].tolist() == [1, 2, 3]
    assert det[0, 6] == 0.9


def test_empty_file(tmp_path):
    p = tmp_path / "det.txt"
    p.write_text("")
    assert mot.read_det(p).shape == (0, 7)


def test_results_format(tmp_path):
    rows = np.array([[1, 3, 10.123, 20, 30, 60, 0.87654]])
    mot.write_results(tmp_path / "seq.txt", rows)
    assert (tmp_path / "seq.txt").read_text() == "1,3,10.12,20.00,30.00,60.00,0.8765,-1,-1,-1\n"


def test_evaluated_drops_static_people(mot17):
    gt = mot.read_gt(mot17 / "train" / "MOT17-04-FRCNN" / "gt" / "gt.txt")
    assert len(gt) == 16
    assert len(mot.evaluated(gt)) == 8


def test_copies_share_the_frcnn_frames(mot17):
    seqs = mot.sequences(mot17.parent, "MOT17", "train")
    assert [s.name for s in seqs] == ["MOT17-02-FRCNN", "MOT17-02-SDP", "MOT17-04-FRCNN"]
    assert seqs[1].img_dir == mot17 / "train" / "MOT17-02-FRCNN" / "img1"
    assert [s.name for s in mot.sequences(mot17.parent, "MOT17", "train", detector="SDP")] == ["MOT17-02-SDP"]
    assert mot.video_name("MOT20-05") == "MOT20-05"
