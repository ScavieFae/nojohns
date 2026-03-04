#!/usr/bin/env python3
"""Parse .slp replay files into parquet format for world model training.

This is a convenience wrapper around slippi_db's parsing pipeline.
For most users, `prepare_dataset.py parse` is the better entry point.

Usage:
    python -m worldmodel.scripts.parse_replays --slp game.slp --output parsed/
    python -m worldmodel.scripts.parse_replays --root dataset/ --threads 4
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_single(slp_path: str, output_dir: str):
    """Parse a single .slp file."""
    try:
        from slippi_db.parse_peppi import get_slp
    except ImportError:
        logging.error("slippi_db not available. Install slippi-ai first.")
        sys.exit(1)

    import pyarrow.parquet as pq

    logging.info("Parsing %s", slp_path)
    game = get_slp(slp_path)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / Path(slp_path).stem
    # Write as parquet (uncompressed — prepare_dataset handles compression)
    pq.write_table(game, str(out_path))
    logging.info("Written to %s", out_path)


def parse_batch(root: str, threads: int):
    """Parse all .slp archives in a dataset directory."""
    try:
        from slippi_db.parse_local import run_parsing
    except ImportError:
        logging.error("slippi_db not available. Install slippi-ai first.")
        sys.exit(1)

    run_parsing(root=root, num_threads=threads)


def main():
    parser = argparse.ArgumentParser(description="Parse .slp replays to parquet")
    parser.add_argument("--slp", help="Single .slp file to parse")
    parser.add_argument("--output", default="parsed/", help="Output directory for single file")
    parser.add_argument("--root", help="Dataset root for batch parsing")
    parser.add_argument("--threads", type=int, default=1, help="Threads for batch parsing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.slp:
        parse_single(args.slp, args.output)
    elif args.root:
        parse_batch(args.root, args.threads)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
