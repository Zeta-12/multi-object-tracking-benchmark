"""`mot` command line."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

DATASETS = ("MOT17", "MOT20")
SPLITS = ("train", "test", "train_half", "val_half")


def info_table(data_dir: Path, dataset: str, split: str) -> str:
    from mot_bench import mot

    rows = [("sequence", "frames", "fps", "size", "gt boxes", "det boxes")]
    for seq in mot.sequences(data_dir, dataset, split):
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
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value
    return params


def track(args) -> None:
    from mot_bench import evaluate, mot, run

    seqs = mot.sequences(args.data_dir, args.dataset, args.split, args.detector)
    if not seqs:
        raise SystemExit(f"no sequence in {args.data_dir}/{args.dataset}/{args.split} for detector {args.detector}")
    name = args.name or f"{args.tracker}-{args.detector or 'all'}"
    out_dir = args.out / f"{args.dataset}-{args.split}" / name
    runs = run.run(seqs, args.tracker, parse_params(args.param), out_dir, args.min_score)
    fps = sum(r.frames for r in runs) / sum(r.seconds for r in runs)
    print(f"{name}: {sum(r.frames for r in runs)} frames, tracker alone at {fps:.0f} fps")
    if args.split == "test":
        return
    split_dir = Path(args.data_dir) / args.dataset / args.split
    metrics = evaluate.evaluate(split_dir, out_dir, {s.name: s.info.length for s in seqs}, args.dataset)
    report = {"metrics": metrics, "timing": {r.name: {"frames": r.frames, "seconds": r.seconds} for r in runs}}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(evaluate.table(metrics))


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

    p_track = sub.add_parser("track", help="Run a tracker on a split and evaluate it.")
    p_track.add_argument("--tracker", default="sort")
    p_track.add_argument("--dataset", choices=DATASETS, default="MOT17")
    p_track.add_argument("--split", choices=SPLITS, default="val_half")
    p_track.add_argument("--detector", help="MOT17 public detections to use: DPM, FRCNN or SDP (default: all three).")
    p_track.add_argument("--min-score", type=float, help="Drop the detections under this score.")
    p_track.add_argument("--param", action="append", default=[], help="Tracker parameter, e.g. max_age=30.")
    p_track.add_argument("--name", help="Output name (default: <tracker>-<detector>).")
    p_track.add_argument("--out", type=Path, default=Path("runs"))

    p_cmp = sub.add_parser("compare", help="Check that two folders of results have the same tracks.")
    p_cmp.add_argument("ours", type=Path, help="e.g. runs/MOT17-train/sort-all/data")
    p_cmp.add_argument("reference", type=Path, help="output folder of the reference implementation")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.command == "download":
        from mot_bench import data

        print(data.download(args.data_dir, args.dataset, images=args.images))
    elif args.command == "info":
        print(info_table(args.data_dir, args.dataset, args.split))
    elif args.command == "track":
        track(args)
    elif args.command == "compare":
        from mot_bench import compare

        results = compare.compare(args.ours, args.reference)
        for seq, r in results.items():
            print(seq, " ".join(f"{k}={v}" for k, v in r.items()))
        print("same tracks" if compare.same(results) else "the outputs differ")
