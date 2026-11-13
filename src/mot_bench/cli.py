"""`mot` command line."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

DATASETS = ("MOT17", "MOT20")
SPLITS = ("train", "test", "train_half", "val_half")
DETS_HELP = "Detection file: det for the public ones (det/det.txt), or a name given to add-dets."
CMC_HELP = "Camera motion file, a name given to add-cmc (for BoT-SORT)."


def info_table(data_dir: Path, dataset: str, split: str, dets: str = "det") -> str:
    from mot_bench import mot

    rows = [("sequence", "frames", "fps", "size", "gt boxes", "det boxes")]
    for seq in mot.sequences(data_dir, dataset, split, dets=dets):
        gt = seq.dir / "gt" / "gt.txt"
        n_gt = len(mot.evaluated(mot.read_gt(gt))) if gt.is_file() else "-"
        i = seq.info
        rows.append((seq.name, i.length, f"{i.fps:g}", f"{i.width}x{i.height}", n_gt, len(seq.det())))
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    return "\n".join("  ".join(str(v).rjust(w) for v, w in zip(r, widths, strict=True)) for r in rows)


def parse_params(items: list[str]) -> dict:
    params = {}
    for item in items:
        key, _, value = item.partition("=")
        params[key] = parse_value(value)
    return params


def parse_value(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_grid(items: list[str]) -> dict[str, list]:
    """["min_score=0,0.5", "max_age=1,3"] -> {"min_score": [0, 0.5], "max_age": [1, 3]}"""
    space = {}
    for item in items:
        key, _, values = item.partition("=")
        space[key] = [parse_value(v) for v in values.split(",")]
    return space


def load_sequences(args, split: str) -> list:
    from mot_bench import mot

    seqs = mot.sequences(args.data_dir, args.dataset, split, args.detector, args.dets, args.cmc)
    if not seqs:
        raise SystemExit(
            f"no sequence in {args.data_dir}/{args.dataset}/{split} for detector {args.detector}, dets {args.dets}"
            f" and cmc {args.cmc}"
        )
    return seqs


def run_name(args, suffix: str = "") -> str:
    source = (args.detector or "all") if args.dets == "det" else args.dets
    return args.name or f"{args.tracker}-{source}{suffix}"


def evaluate_run(args, split: str, seqs: list, runs: list, out_dir: Path) -> dict:
    """Scores out_dir against the ground truth of the split and writes metrics.json."""
    from mot_bench import evaluate

    split_dir = Path(args.data_dir) / args.dataset / split
    metrics = evaluate.evaluate(split_dir, out_dir, {s.name: s.info.length for s in seqs}, args.dataset)
    report = {"metrics": metrics, "timing": {r.name: {"frames": r.frames, "seconds": r.seconds} for r in runs}}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return metrics


def track(args) -> None:
    from mot_bench import evaluate, run

    seqs = load_sequences(args, args.split)
    name = run_name(args)
    out_dir = args.out / f"{args.dataset}-{args.split}" / name
    runs = run.run(seqs, args.tracker, parse_params(args.param), out_dir, args.min_score)
    fps = sum(r.frames for r in runs) / sum(r.seconds for r in runs)
    print(f"{name}: {sum(r.frames for r in runs)} frames, tracker alone at {fps:.0f} fps")
    if args.split == "test":
        return
    print(evaluate.table(evaluate_run(args, args.split, seqs, runs, out_dir)))


def tune(args) -> None:
    from mot_bench import evaluate, run
    from mot_bench import tune as grid_search

    space = parse_grid(args.grid)
    if not space:
        raise SystemExit("nothing to tune, pass at least one --grid key=v1,v2,...")
    name = run_name(args, "-tuned")
    train_dir = args.out / f"{args.dataset}-train_half" / name
    rows = grid_search.search(
        load_sequences(args, "train_half"), args.tracker, space, train_dir, args.dataset, args.jobs
    )
    grid_search.write_csv(rows, train_dir / "grid.csv")

    columns = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]
    top = sorted(rows, key=lambda r: r[args.metric], reverse=True)[:5]
    label = {" ".join(f"{k}={r[k]}" for k in space): r for r in top}
    print(f"train_half, best {len(top)} of {len(rows)} by {args.metric}:")
    print(evaluate.table(label, columns, first="point"))

    point = {k: v for k, v in grid_search.best(rows, args.metric).items() if k in space}
    params, min_score = grid_search.split_point(point)
    seqs = load_sequences(args, "val_half")
    out_dir = args.out / f"{args.dataset}-val_half" / name
    runs = run.run(seqs, args.tracker, params, out_dir, min_score)
    tuning = {"tuned_on": "train_half", "metric": args.metric, "grid": space, "best": point}
    (out_dir / "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n")
    print(f"val_half with {point}:")
    print(evaluate.table(evaluate_run(args, "val_half", seqs, runs, out_dir)))


def summary_table(runs_dir: Path, dataset: str, split: str) -> str:
    """COMBINED metrics of every run in runs/<dataset>-<split>."""
    from mot_bench import evaluate

    split_dir = Path(runs_dir) / f"{dataset}-{split}"
    rows = {
        p.parent.name: json.loads(p.read_text())["metrics"]["COMBINED"]
        for p in sorted(split_dir.glob("*/metrics.json"))
    }
    if not rows:
        raise SystemExit(f"no evaluated run in {split_dir}")
    return evaluate.table(rows, ["HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1", "IDSW", "FP", "FN"], first="run")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mot", description="Multi-object tracking on MOT17 / MOT20.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download a dataset and build the train/val halves.")
    p_dl.add_argument("--dataset", choices=DATASETS, default="MOT17")
    p_dl.add_argument("--images", action="store_true", help="Also get the frames (MOT17: 5.9 GB).")

    p_info = sub.add_parser("info", help="List the sequences of a split.")
    p_info.add_argument("--dataset", choices=DATASETS, default="MOT17")
    p_info.add_argument("--split", choices=SPLITS, default="val_half")
    p_info.add_argument("--dets", default="det", help=DETS_HELP)

    p_add = sub.add_parser("add-dets", help="Add private detections to the training sequences.")
    p_add.add_argument("src", type=Path, help="folder with one MOTChallenge det file per sequence, <seq>.txt")
    p_add.add_argument("--name", required=True, help="saved as det/<name>.txt, used with --dets <name>")
    p_add.add_argument("--dataset", choices=DATASETS, default="MOT17")

    p_feat = sub.add_parser("add-features", help="Add the re-ID features of a detection file.")
    p_feat.add_argument("src", type=Path, help="folder with <seq>.npy, one feature per line of det/<dets>.txt")
    p_feat.add_argument("--dets", required=True, help="the detection file they belong to, saved as det/<dets>.npy")
    p_feat.add_argument("--dataset", choices=DATASETS, default="MOT17")

    p_cmc = sub.add_parser("add-cmc", help="Add camera motion files to the training sequences.")
    p_cmc.add_argument("src", type=Path, help="BoT-SORT tracker/GMC_files/MOTChallenge folder (GMC-<video>.txt)")
    p_cmc.add_argument("--name", required=True, help="saved as cmc/<name>.txt, used with --cmc <name>")
    p_cmc.add_argument("--dataset", choices=DATASETS, default="MOT17")

    p_track = sub.add_parser("track", help="Run a tracker on a split and evaluate it.")
    p_track.add_argument("--tracker", default="sort")
    p_track.add_argument("--dataset", choices=DATASETS, default="MOT17")
    p_track.add_argument("--split", choices=SPLITS, default="val_half")
    p_track.add_argument("--detector", help="MOT17 public detections to use: DPM, FRCNN or SDP (default: all three).")
    p_track.add_argument("--dets", default="det", help=DETS_HELP)
    p_track.add_argument("--cmc", help=CMC_HELP)
    p_track.add_argument("--min-score", type=float, help="Drop the detections under this score.")
    p_track.add_argument("--param", action="append", default=[], help="Tracker parameter, e.g. max_age=30.")
    p_track.add_argument("--name", help="Output name (default: <tracker>-<detector>).")
    p_track.add_argument("--out", type=Path, default=Path("runs"))

    p_tune = sub.add_parser("tune", help="Grid search on train_half, then run the best point on val_half.")
    p_tune.add_argument("--tracker", default="sort")
    p_tune.add_argument("--dataset", choices=DATASETS, default="MOT17")
    p_tune.add_argument("--detector", help="MOT17 public detections: DPM, FRCNN or SDP (default: all three).")
    p_tune.add_argument("--dets", default="det", help=DETS_HELP)
    p_tune.add_argument("--cmc", help=CMC_HELP)
    p_tune.add_argument(
        "--grid", action="append", default=[], help="Values to try, e.g. min_score=0,0.5 or max_age=1,3,5."
    )
    p_tune.add_argument("--metric", default="HOTA", choices=["HOTA", "MOTA", "IDF1"])
    p_tune.add_argument("--jobs", type=int, default=1, help="Grid points run in parallel.")
    p_tune.add_argument("--name", help="Output name (default: <tracker>-<detector>-tuned).")
    p_tune.add_argument("--out", type=Path, default=Path("runs"))

    p_sum = sub.add_parser("summary", help="Metrics of every run of a split.")
    p_sum.add_argument("--dataset", choices=DATASETS, default="MOT17")
    p_sum.add_argument("--split", choices=SPLITS, default="val_half")
    p_sum.add_argument("--out", type=Path, default=Path("runs"))

    p_cmp = sub.add_parser("compare", help="Check that two folders of results have the same tracks.")
    p_cmp.add_argument("ours", type=Path, help="e.g. runs/MOT17-train/sort-all/data")
    p_cmp.add_argument("reference", type=Path, help="output folder of the reference implementation")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.command == "download":
        from mot_bench import data

        print(data.download(args.data_dir, args.dataset, images=args.images))
    elif args.command == "info":
        print(info_table(args.data_dir, args.dataset, args.split, args.dets))
    elif args.command == "add-dets":
        from mot_bench import data

        added = data.add_detections(args.data_dir, args.dataset, args.src, args.name)
        print(f"det/{args.name}.txt added to {len(added)} sequences: {' '.join(added)}")
    elif args.command == "add-features":
        from mot_bench import data

        added = data.add_features(args.data_dir, args.dataset, args.src, args.dets)
        print(f"det/{args.dets}.npy added to {len(added)} sequences: {' '.join(added)}")
    elif args.command == "add-cmc":
        from mot_bench import data

        added = data.add_cmc(args.data_dir, args.dataset, args.src, args.name)
        print(f"cmc/{args.name}.txt added to {len(added)} sequences")
    elif args.command == "track":
        track(args)
    elif args.command == "tune":
        tune(args)
    elif args.command == "summary":
        print(summary_table(args.out, args.dataset, args.split))
    elif args.command == "compare":
        from mot_bench import compare

        results = compare.compare(args.ours, args.reference)
        for seq, r in results.items():
            print(seq, " ".join(f"{k}={v}" for k, v in r.items()))
        print("same tracks" if compare.same(results) else "the outputs differ")
