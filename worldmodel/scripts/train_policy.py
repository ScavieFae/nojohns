#!/usr/bin/env python3
"""Train an imitation learning policy on Melee replays.

The policy learns to predict controller inputs from game state —
essentially learning "what would a human press in this situation?"

Usage:
    # Quick test (5 games):
    python -m worldmodel.scripts.train_policy \
        --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
        --max-games 5 --epochs 5 --no-wandb

    # Real training (2K games, overnight on ScavieFae):
    python -m worldmodel.scripts.train_policy \
        --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
        --max-games 2000 --epochs 10 --batch-size 512 --device mps \
        --save-dir ~/claude-projects/nojohns-training/checkpoints/policy-v1 \
        --run-name "policy-imitation-2k" -v
"""

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worldmodel.data.dataset import MeleeDataset
from worldmodel.data.parse import load_games_from_dir
from worldmodel.data.policy_dataset import PolicyFrameDataset
from worldmodel.model.encoding import EncodingConfig
from worldmodel.model.policy_mlp import PolicyMLP
from worldmodel.training.policy_trainer import PolicyLossWeights, PolicyTrainer

try:
    import wandb
except ImportError:
    wandb = None


def main():
    parser = argparse.ArgumentParser(description="Train imitation learning policy")
    parser.add_argument("--dataset", required=True, help="Path to parsed dataset directory")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", default=None, help="Override device (cpu/mps/cuda)")
    parser.add_argument("--save-dir", default=None, help="Override checkpoint save dir")
    parser.add_argument("--max-games", type=int, default=None, help="Max games to load")
    parser.add_argument("--predict-player", type=int, default=0, choices=[0, 1],
                        help="Which player's controller to predict (default: 0)")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--run-name", default=None, help="Name for this run (wandb + save dir)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Defaults
    model_cfg = {"context_len": 10, "hidden_dim": 512, "trunk_dim": 256, "dropout": 0.1}
    train_cfg = {
        "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 256,
        "num_epochs": 50, "train_split": 0.9,
    }
    loss_cfg = {"analog": 1.0, "button": 1.0}

    # Load YAML config if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model_cfg.update(cfg.get("model", {}))
        train_cfg.update(cfg.get("training", {}))
        loss_cfg.update(cfg.get("loss_weights", {}))

    # CLI overrides
    if args.epochs is not None:
        train_cfg["num_epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.device is not None:
        train_cfg["device"] = args.device

    save_dir = args.save_dir or "worldmodel/checkpoints/policy"

    enc_cfg = EncodingConfig()

    # Load data
    logging.info("Loading games from %s", args.dataset)
    games = load_games_from_dir(
        args.dataset,
        max_games=args.max_games,
    )
    if not games:
        logging.error("No games loaded! Check dataset path.")
        sys.exit(1)

    logging.info("Loaded %d games", len(games))

    game_md5s = [g.meta["slp_md5"] for g in games if g.meta]
    data_fingerprint = hashlib.sha256("|".join(sorted(game_md5s)).encode()).hexdigest()[:12]
    logging.info("Data fingerprint: %s (%d games)", data_fingerprint, len(game_md5s))

    # Encode
    dataset = MeleeDataset(games, enc_cfg)

    context_len = model_cfg.get("context_len", 10)
    train_split = train_cfg.get("train_split", 0.9)
    split_idx = max(1, int(dataset.num_games * train_split))

    train_ds = PolicyFrameDataset(
        dataset, range(0, split_idx),
        context_len=context_len,
        predict_player=args.predict_player,
    )
    val_ds = PolicyFrameDataset(
        dataset, range(split_idx, dataset.num_games),
        context_len=context_len,
        predict_player=args.predict_player,
    )

    logging.info("Train: %d examples, Val: %d examples", len(train_ds), len(val_ds))

    # Build model
    model = PolicyMLP(
        cfg=enc_cfg,
        context_len=model_cfg.get("context_len", 10),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        trunk_dim=model_cfg.get("trunk_dim", 256),
        dropout=model_cfg.get("dropout", 0.1),
    )

    param_count = sum(p.numel() for p in model.parameters())
    logging.info("Policy model parameters: %s", f"{param_count:,}")

    # Run config for tracking
    run_config = {
        "model": {**model_cfg, "type": "policy_mlp"},
        "training": train_cfg,
        "loss_weights": loss_cfg,
        "data": {
            "dataset_path": args.dataset,
            "max_games": args.max_games,
            "num_games_loaded": len(game_md5s),
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "total_frames": dataset.total_frames,
            "fingerprint": data_fingerprint,
            "predict_player": args.predict_player,
        },
        "model_params": param_count,
        "machine": platform.node(),
        "device": train_cfg.get("device", "auto"),
    }

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
            tags=["policy", "imitation"],
        )
        logging.info("Wandb run: %s", wandb.run.url)
    elif not wandb:
        logging.info("wandb not installed — logging to file only")

    loss_weights = PolicyLossWeights(**loss_cfg) if loss_cfg else None

    trainer = PolicyTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
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
        logging.info("--- Policy training complete ---")
        logging.info("Final loss: %.4f", final.get("loss/total", 0))
        logging.info("Stick MAE: %.4f", final.get("metric/stick_mae", 0))
        logging.info("Button accuracy: %.3f", final.get("metric/button_acc", 0))
        if "metric/button_pressed_acc" in final:
            logging.info("Button-pressed accuracy: %.3f", final["metric/button_pressed_acc"])

    # Save manifest
    manifest_dir = Path(save_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": run_config,
        "data_fingerprint": data_fingerprint,
        "game_md5s": game_md5s,
        "results": history[-1] if history else {},
        "all_epochs": history,
    }

    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    logging.info("Saved run manifest: %s", manifest_path)

    if wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
