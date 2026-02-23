#!/usr/bin/env python3
"""Prepare a dataset for world model training.

Wraps slippi_db's parsing pipeline:
    .slp files (raw archives) → parsed parquet → ready for training

Usage:
    # Parse .slp archives into parquet format:
    python -m worldmodel.scripts.prepare_dataset parse --root /path/to/dataset

    # Verify a parsed dataset is loadable:
    python -m worldmodel.scripts.prepare_dataset verify --dataset /path/to/dataset

    # Show stats for a parsed dataset:
    python -m worldmodel.scripts.prepare_dataset stats --dataset /path/to/dataset
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worldmodel.data.parse import load_game, load_games_from_dir


def cmd_parse(args):
    """Run slippi_db parsing on raw .slp archives."""
    try:
        from slippi_db.parse_local import run_parsing
    except ImportError:
        logging.error(
            "slippi_db not importable. Make sure slippi-ai is installed:\n"
            "  pip install -e docs/phillip-research/slippi-ai"
        )
        sys.exit(1)

    logging.info("Parsing replays from %s", args.root)
    run_parsing(
        root=args.root,
        num_threads=args.threads,
        reprocess=args.reprocess,
    )
    logging.info("Parsing complete.")


def cmd_verify(args):
    """Verify a parsed dataset loads correctly."""
    dataset_dir = Path(args.dataset)
    meta_path = dataset_dir / "meta.json"

    if not meta_path.exists():
        logging.error("No meta.json found at %s", meta_path)
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    training_entries = [e for e in meta if e.get("is_training")]
    logging.info("Found %d training games in meta.json", len(training_entries))

    # Try loading first game
    if training_entries:
        entry = training_entries[0]
        game_path = dataset_dir / "games" / entry["slp_md5"]
        try:
            game = load_game(game_path, compression=entry.get("compression", "zlib"))
            logging.info(
                "First game: %d frames, stage=%d, p0.x range=[%.1f, %.1f]",
                game.num_frames,
                game.stage,
                game.p0.x.min(),
                game.p0.x.max(),
            )
            logging.info("Dataset verified OK.")
        except Exception as e:
            logging.error("Failed to load first game: %s", e)
            sys.exit(1)


def cmd_stats(args):
    """Show stats for a parsed dataset."""
    dataset_dir = Path(args.dataset)
    meta_path = dataset_dir / "meta.json"

    with open(meta_path) as f:
        meta = json.load(f)

    training = [e for e in meta if e.get("is_training")]
    stages = Counter(e.get("stage") for e in training)
    chars = Counter()
    total_frames = 0

    for e in training:
        total_frames += e.get("lastFrame", 0)
        for p in e.get("players", []):
            chars[p.get("character")] += 1

    print(f"Total games: {len(meta)}")
    print(f"Training games: {len(training)}")
    print(f"Total frames: {total_frames:,} ({total_frames / 60:.0f}s = {total_frames / 3600:.1f}min)")
    print(f"\nStages: {dict(stages.most_common(10))}")
    print(f"Characters: {dict(chars.most_common(10))}")

    # Rejection reasons
    not_training = [e for e in meta if not e.get("is_training")]
    if not_training:
        reasons = Counter(e.get("not_training_reason", "unknown") for e in not_training)
        print(f"\nRejected: {len(not_training)}")
        for reason, count in reasons.most_common(5):
            print(f"  {reason}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Melee dataset for world model training")
    subparsers = parser.add_subparsers(dest="command")

    # parse subcommand
    p_parse = subparsers.add_parser("parse", help="Parse .slp archives")
    p_parse.add_argument("--root", required=True, help="Dataset root (contains Raw/ directory)")
    p_parse.add_argument("--threads", type=int, default=1, help="Parsing threads")
    p_parse.add_argument("--reprocess", action="store_true", help="Reprocess already-parsed files")
    p_parse.set_defaults(func=cmd_parse)

    # verify subcommand
    p_verify = subparsers.add_parser("verify", help="Verify a parsed dataset")
    p_verify.add_argument("--dataset", required=True, help="Parsed dataset directory")
    p_verify.set_defaults(func=cmd_verify)

    # stats subcommand
    p_stats = subparsers.add_parser("stats", help="Show dataset statistics")
    p_stats.add_argument("--dataset", required=True, help="Parsed dataset directory")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
