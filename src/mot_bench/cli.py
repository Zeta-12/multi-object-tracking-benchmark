"""`mot` command line."""

from __future__ import annotations

import argparse
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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.command == "download":
        from mot_bench import data

        print(data.download(args.data_dir, args.dataset, images=args.images))
    elif args.command == "info":
        print(info_table(args.data_dir, args.dataset, args.split))
