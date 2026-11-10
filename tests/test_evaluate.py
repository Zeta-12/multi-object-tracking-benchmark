import numpy as np
import pytest

from mot_bench import evaluate, mot, run


def gt_as_results(seq, out_dir):
    gt = seq.gt()
    mot.write_results(out_dir / "data" / f"{seq.name}.txt", np.column_stack([gt[:, :6], np.ones(len(gt))]))


@pytest.fixture
def seq(mot17):
    return mot.sequences(mot17.parent, "MOT17", "train", detector="FRCNN")[0]


def test_ground_truth_scores_100(seq, tmp_path):
    out = tmp_path / "runs" / "gt"
    gt_as_results(seq, out)
    m = evaluate.evaluate(seq.dir.parent, out, {seq.name: seq.info.length}, "MOT17")
    c = m["COMBINED"]
    assert c["HOTA"] == pytest.approx(1.0)
    assert c["MOTA"] == pytest.approx(1.0)
    assert c["IDF1"] == pytest.approx(1.0)
    # the static person (class 7) is in the output too and isn't counted as a false positive
    assert c["FP"] == 0 and c["FN"] == 0


def test_id_swap_hurts_association_only(seq, tmp_path):
    out = tmp_path / "runs" / "swap"
    gt = seq.gt()
    rows = np.column_stack([gt[:, :6], np.ones(len(gt))])
    late = rows[:, 0] >= 6
    one, two = late & (rows[:, 1] == 1), late & (rows[:, 1] == 2)
    rows[one, 1], rows[two, 1] = 2, 1
    mot.write_results(out / "data" / f"{seq.name}.txt", rows)

    c = evaluate.evaluate(seq.dir.parent, out, {seq.name: seq.info.length}, "MOT17")["COMBINED"]
    assert c["IDSW"] == 2
    assert c["DetA"] == pytest.approx(1.0)
    assert c["AssA"] < 0.7
    assert c["IDF1"] < 0.6


def test_sort_on_clean_detections(seq, tmp_path):
    out = tmp_path / "runs" / "sort"
    run.run([seq], "sort", {}, out)
    c = evaluate.evaluate(seq.dir.parent, out, {seq.name: seq.info.length}, "MOT17")["COMBINED"]
    assert c["MOTA"] == pytest.approx(1.0)
    assert c["IDF1"] == pytest.approx(1.0)
    assert (out / "config.json").is_file()
