import numpy as np

from mot_bench import compare, mot, run


def test_detections_by_frame():
    det = np.array(
        [
            [1, -1, 0, 0, 10, 20, 0.9],
            [1, -1, 5, 5, 10, 20, 0.2],
            [3, -1, 1, 2, 3, 4, 0.8],
        ]
    )
    frames = run.detections_by_frame(det, 4)
    assert [len(f) for f in frames] == [2, 0, 1, 0]
    assert frames[0][0].tolist() == [0, 0, 10, 20, 0.9]
    assert [len(f) for f in run.detections_by_frame(det, 4, min_score=0.5)] == [1, 0, 1, 0]


def test_results_match_themselves_up_to_relabelling(mot17, tmp_path):
    seqs = mot.sequences(mot17.parent, "MOT17", "train")
    run.run(seqs, "sort", {}, tmp_path / "a")
    for seq in seqs:
        rows = np.loadtxt(tmp_path / "a" / "data" / f"{seq.name}.txt", delimiter=",", ndmin=2)
        rows[:, 1] += 100
        mot.write_results(tmp_path / "b" / f"{seq.name}.txt", rows[:, :7])
    results = compare.compare(tmp_path / "a" / "data", tmp_path / "b")
    assert compare.same(results)

    # two tracks exchanging their ids halfway is a difference
    path = tmp_path / "b" / "MOT17-02-FRCNN.txt"
    rows = np.loadtxt(path, delimiter=",")
    late = rows[:, 0] > 5
    one, two = late & (rows[:, 1] == 101), late & (rows[:, 1] == 102)
    rows[one, 1], rows[two, 1] = 102, 101
    mot.write_results(path, rows[:, :7])
    results = compare.compare(tmp_path / "a" / "data", tmp_path / "b")
    assert results["MOT17-02-FRCNN"]["id_conflicts"] > 0
    assert not compare.same(results)
