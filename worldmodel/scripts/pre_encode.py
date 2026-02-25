"""Pre-encode parsed games into a single .pt file for fast cloud loading.

Eliminates the per-file I/O bottleneck: instead of loading thousands of small
files from a network volume, upload one tensor file and torch.load() it.

Usage:
    .venv/bin/python worldmodel/scripts/pre_encode.py \
        --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
        --config worldmodel/experiments/mamba2-medium-gpu.yaml \
        --max-games 2000 \
        --output worldmodel/data/encoded-2k.pt
"""

import argparse
import time
import sys
import yaml

import torch

sys.path.insert(0, ".")
from worldmodel.data.parse import load_games_from_dir
from worldmodel.data.dataset import MeleeDataset
from worldmodel.model.encoding import EncodingConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-games", type=int, default=2000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    enc_cfg_dict = cfg.get("encoding", {})
    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None})

    print(f"Loading {args.max_games} games from {args.dataset}...")
    t0 = time.time()
    games = load_games_from_dir(args.dataset, max_games=args.max_games)
    print(f"Loaded {len(games)} games in {time.time() - t0:.1f}s")

    print("Encoding...")
    t1 = time.time()
    dataset = MeleeDataset(games, enc_cfg)
    print(f"Encoded in {time.time() - t1:.1f}s")
    print(f"  Floats: {dataset.floats.shape}")
    print(f"  Ints: {dataset.ints.shape}")
    print(f"  Total frames: {dataset.total_frames:,}")

    payload = {
        "floats": dataset.floats,
        "ints": dataset.ints,
        "game_offsets": torch.tensor(dataset.game_offsets),
        "game_lengths": dataset.game_lengths,
        "num_games": dataset.num_games,
        "encoding_config": enc_cfg_dict,
    }

    torch.save(payload, args.output)
    import os
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
