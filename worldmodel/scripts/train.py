#!/usr/bin/env python3
"""Train a Melee world model.

Usage:
    python -m worldmodel.scripts.train --dataset /path/to/parsed/dataset
    python -m worldmodel.scripts.train --config worldmodel/configs/fox_ditto_fd.yaml --dataset /path
    python -m worldmodel.scripts.train --dataset /path --epochs 10 --batch-size 128
"""

import argparse
import logging
import platform
import subprocess
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worldmodel.data.dataset import MeleeDataset
from worldmodel.data.parse import load_games_from_dir
from worldmodel.model.encoding import EncodingConfig
from worldmodel.model.mlp import FrameStackMLP
from worldmodel.training.metrics import LossWeights
from worldmodel.training.trainer import Trainer

try:
    import wandb
except ImportError:
    wandb = None


def load_config(config_path: str | None) -> dict:
    """Load YAML config, or return defaults."""
    if config_path:
        with open(config_path) as f:
            return yaml.safe_load(f)
    # Default config
    return {
        "encoding": {},
        "model": {"context_len": 10, "hidden_dim": 512, "trunk_dim": 256, "dropout": 0.1},
        "training": {
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 256,
            "num_epochs": 50,
            "train_split": 0.9,
        },
        "loss_weights": {"continuous": 1.0, "binary": 0.5, "action": 2.0, "jumps": 0.5},
        "save_dir": "worldmodel/checkpoints",
    }


def main():
    parser = argparse.ArgumentParser(description="Train Melee world model")
    parser.add_argument("--dataset", required=True, help="Path to parsed dataset directory")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", default=None, help="Override device (cpu/mps/cuda)")
    parser.add_argument("--save-dir", default=None, help="Override checkpoint save dir")
    parser.add_argument("--max-games", type=int, default=None, help="Max games to load")
    parser.add_argument("--stage", type=int, default=None, help="Filter by stage ID")
    parser.add_argument("--character", type=int, default=None, help="Filter by character ID")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--run-name", default=None, help="Name for this run (used in wandb + save dir)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    cfg = load_config(args.config)

    # CLI overrides
    data_cfg = cfg.get("data", {})
    if args.max_games is not None:
        data_cfg["max_games"] = args.max_games
    if args.stage is not None:
        data_cfg["stage_filter"] = args.stage
    if args.character is not None:
        data_cfg["character_filter"] = args.character

    train_cfg = cfg.get("training", {})
    if args.epochs is not None:
        train_cfg["num_epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.device is not None:
        train_cfg["device"] = args.device

    save_dir = args.save_dir or cfg.get("save_dir", "worldmodel/checkpoints")

    # Build encoding config
    enc_cfg_dict = cfg.get("encoding", {})
    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None})

    # Load data
    logging.info("Loading games from %s", args.dataset)
    games = load_games_from_dir(
        args.dataset,
        max_games=data_cfg.get("max_games"),
        stage_filter=data_cfg.get("stage_filter"),
        character_filter=data_cfg.get("character_filter"),
    )

    if not games:
        logging.error("No games loaded! Check dataset path and filters.")
        sys.exit(1)

    logging.info("Loaded %d games", len(games))

    # Build datasets
    dataset = MeleeDataset(games, enc_cfg)
    context_len = cfg.get("model", {}).get("context_len", 10)
    train_split = train_cfg.get("train_split", 0.9)
    train_ds = dataset.get_frame_dataset(context_len=context_len, train=True, train_split=train_split)
    val_ds = dataset.get_frame_dataset(context_len=context_len, train=False, train_split=train_split)

    logging.info("Train: %d examples, Val: %d examples", len(train_ds), len(val_ds))

    # Build model
    model_cfg = cfg.get("model", {})
    model = FrameStackMLP(
        cfg=enc_cfg,
        context_len=model_cfg.get("context_len", 10),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        trunk_dim=model_cfg.get("trunk_dim", 256),
        dropout=model_cfg.get("dropout", 0.1),
    )

    param_count = sum(p.numel() for p in model.parameters())
    logging.info("Model parameters: %s", f"{param_count:,}")

    # Build trainer
    loss_cfg = cfg.get("loss_weights", {})
    loss_weights = LossWeights(**loss_cfg) if loss_cfg else None

    # Collect full run config for tracking
    run_config = {
        "model": model_cfg,
        "training": train_cfg,
        "loss_weights": loss_cfg,
        "encoding": enc_cfg_dict,
        "data": {
            "dataset_path": args.dataset,
            "max_games": data_cfg.get("max_games"),
            "num_games_loaded": len(games),
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "total_frames": dataset.total_frames,
        },
        "model_params": param_count,
        "machine": platform.node(),
        "device": train_cfg.get("device", "auto"),
    }

    # Git commit hash for reproducibility
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        run_config["git_commit"] = git_hash
    except Exception:
        pass

    # Init wandb
    if wandb and not args.no_wandb:
        wandb.init(
            project="melee-worldmodel",
            name=args.run_name,
            config=run_config,
        )
        logging.info("Wandb run: %s", wandb.run.url)
    elif not wandb:
        logging.info("wandb not installed — logging to file only")

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=enc_cfg,
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        batch_size=train_cfg.get("batch_size", 256),
        num_epochs=train_cfg.get("num_epochs", 50),
        loss_weights=loss_weights,
        save_dir=save_dir,
        device=train_cfg.get("device"),
        resume_from=args.resume,
    )

    # Train
    history = trainer.train()

    # Summary
    if history:
        final = history[-1]
        logging.info("--- Training complete ---")
        logging.info("Final loss: %.4f", final.get("loss/total", 0))
        logging.info("Action accuracy: %.3f", final.get("metric/p0_action_acc", 0))
        logging.info("Position MAE: %.2f game units", final.get("metric/position_mae", 0))
        if "metric/action_change_acc" in final:
            logging.info("Action-change accuracy: %.3f", final["metric/action_change_acc"])

    if wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
