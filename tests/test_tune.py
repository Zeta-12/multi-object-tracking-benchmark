import csv
import json

from mot_bench import cli, mot, split, tune


def test_grid_varies_the_last_key_fastest():
    assert tune.grid({"min_score": [None, 0.5], "max_age": [1, 3]}) == [
        {"min_score": None, "max_age": 1},
        {"min_score": None, "max_age": 3},
        {"min_score": 0.5, "max_age": 1},
        {"min_score": 0.5, "max_age": 3},
    ]


def test_min_score_is_not_a_tracker_parameter():
    assert tune.split_point({"min_score": 0.5, "max_age": 3}) == ({"max_age": 3}, 0.5)
    assert tune.split_point({"max_age": 3}) == ({"max_age": 3}, None)


def test_best_keeps_the_first_of_a_tie():
    rows = [{"a": 1, "HOTA": 0.5}, {"a": 2, "HOTA": 0.6}, {"a": 3, "HOTA": 0.6}]
    assert tune.best(rows)["a"] == 2


def test_parse_grid():
    space = cli.parse_grid(["min_score=null,-0.25,1", "max_age=1,3"])
    assert space == {"min_score": [None, -0.25, 1], "max_age": [1, 3]}


def test_search_scores_every_point(mot17, tmp_path):
    seqs = mot.sequences(mot17.parent, "MOT17", "train", detector="FRCNN")
    rows = tune.search(seqs, "sort", {"min_score": [None, 2.0], "max_age": [1, 3]}, tmp_path / "work", "MOT17")
    assert [(r["min_score"], r["max_age"]) for r in rows] == [(None, 1), (None, 3), (2.0, 1), (2.0, 3)]
    assert all(r["HOTA"] > 0.9 for r in rows[:2])
    # scores are under 1 in the fixture, nothing is left to track
    assert all(r["DetA"] == 0 for r in rows[2:])
    assert list((tmp_path / "work").iterdir()) == []


def test_tune_command(mot17, tmp_path):
    split.make_halves(mot17)
    out = tmp_path / "runs"
    args = ["--data-dir", str(mot17.parent), "tune", "--detector", "FRCNN", "--out", str(out)]
    cli.main([*args, "--grid", "min_score=2,null"])

    with open(out / "MOT17-train_half" / "sort-FRCNN-tuned" / "grid.csv") as f:
        assert len(list(csv.DictReader(f))) == 2
    val = out / "MOT17-val_half" / "sort-FRCNN-tuned"
    assert json.loads((val / "tuning.json").read_text())["best"] == {"min_score": None}
    assert json.loads((val / "metrics.json").read_text())["metrics"]["COMBINED"]["HOTA"] > 0.9
    assert "sort-FRCNN-tuned" in cli.summary_table(out, "MOT17", "val_half")
