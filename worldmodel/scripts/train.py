#!/usr/bin/env python3
"""Train a Melee world model.

Usage:
    python -m worldmodel.scripts.train --dataset /path/to/parsed/dataset
    python -m worldmodel.scripts.train --config worldmodel/configs/fox_ditto_fd.yaml --dataset /path
    python -m worldmodel.scripts.train --dataset /path --epochs 10 --batch-size 128
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worldmodel.data.dataset import MeleeDataset, StreamingMeleeDataset
from worldmodel.data.parse import load_games_from_dir
from worldmodel.model.encoding import EncodingConfig
from worldmodel.model.mlp import FrameStackMLP
from worldmodel.model.mamba2 import FrameStackMamba2
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
    parser.add_argument("--streaming", action="store_true", help="Stream from disk (for datasets too large for RAM)")
    parser.add_argument("--buffer-size", type=int, default=1000, help="Games per streaming buffer (default 1000)")
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

    # Derive experiment name from config filename (e.g. "exp-1a-state-age-embed" from path)
    experiment_name = None
    if args.config:
        experiment_name = Path(args.config).stem  # strip directory + .yaml

    base_save_dir = args.save_dir or cfg.get("save_dir", "worldmodel/checkpoints")
    # Experiment name drives save subdirectory
    if experiment_name and not args.save_dir:
        save_dir = str(Path(base_save_dir) / experiment_name)
    else:
        save_dir = base_save_dir

    # Default run-name to experiment name
    if args.run_name is None and experiment_name:
        args.run_name = experiment_name

    if experiment_name:
        logging.info("Experiment: %s", experiment_name)

    # Build encoding config
    enc_cfg_dict = cfg.get("encoding", {})
    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None})

    # Load data
    logging.info("Loading games from %s", args.dataset)
    context_len = cfg.get("model", {}).get("context_len", 10)
    train_split = train_cfg.get("train_split", 0.9)
    dataset_dir = Path(args.dataset)

    if args.streaming:
        # Streaming mode: load meta.json for game entries, don't load all games into RAM
        import json as json_mod
        meta_path = dataset_dir / "meta.json"
        with open(meta_path) as f:
            all_entries = json_mod.load(f)

        # Filter to training-eligible games
        game_entries = []
        for entry in all_entries:
            if not entry.get("is_training", False):
                continue
            if data_cfg.get("stage_filter") is not None and entry.get("stage") != data_cfg["stage_filter"]:
                continue
            if data_cfg.get("character_filter") is not None:
                players = entry.get("players", [])
                if not all(p.get("character") == data_cfg["character_filter"] for p in players):
                    continue
            game_path = dataset_dir / "games" / entry["slp_md5"]
            if game_path.exists():
                game_entries.append(entry)
            if data_cfg.get("max_games") and len(game_entries) >= data_cfg["max_games"]:
                break

        if not game_entries:
            logging.error("No games found! Check dataset path and filters.")
            sys.exit(1)

        logging.info("Found %d games for streaming", len(game_entries))

        game_md5s = [e["slp_md5"] for e in game_entries]
        data_fingerprint = hashlib.sha256("|".join(sorted(game_md5s)).encode()).hexdigest()[:12]
        logging.info("Data fingerprint: %s (%d games)", data_fingerprint, len(game_md5s))

        train_ds = StreamingMeleeDataset(
            game_entries, dataset_dir, enc_cfg,
            context_len=context_len,
            buffer_size=args.buffer_size,
            train=True, train_split=train_split,
        )
        # Val set: load into memory (only 10% of games, fits in RAM)
        val_entries = game_entries[int(len(game_entries) * train_split):]
        val_games = []
        for entry in val_entries:
            from worldmodel.data.parse import load_game
            try:
                g = load_game(dataset_dir / "games" / entry["slp_md5"], entry.get("compression", "zlib"))
                g.meta = entry
                val_games.append(g)
            except Exception:
                continue
        if val_games:
            val_dataset = MeleeDataset(val_games, enc_cfg)
            val_ds = val_dataset.get_frame_dataset(context_len=context_len, train=True, train_split=1.0)
            total_frames = len(game_entries) * 4500  # approximate
        else:
            val_ds = None
            total_frames = len(game_entries) * 4500

        logging.info("Streaming train: ~%d games, Val: %d games (%d examples)",
                     len(game_entries) - len(val_entries), len(val_games),
                     len(val_ds) if val_ds else 0)

    else:
        # In-memory mode (original behavior)
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

        game_md5s = [g.meta["slp_md5"] for g in games if g.meta]
        data_fingerprint = hashlib.sha256("|".join(sorted(game_md5s)).encode()).hexdigest()[:12]
        logging.info("Data fingerprint: %s (%d games)", data_fingerprint, len(game_md5s))

        dataset = MeleeDataset(games, enc_cfg)
        train_ds = dataset.get_frame_dataset(context_len=context_len, train=True, train_split=train_split)
        val_ds = dataset.get_frame_dataset(context_len=context_len, train=False, train_split=train_split)
        total_frames = dataset.total_frames

        logging.info("Train: %d examples, Val: %d examples", len(train_ds), len(val_ds) if val_ds else 0)

    # Build model
    model_cfg = cfg.get("model", {})
    arch = model_cfg.get("arch", "mlp")

    if arch == "mamba2":
        model = FrameStackMamba2(
            cfg=enc_cfg,
            context_len=model_cfg.get("context_len", 10),
            d_model=model_cfg.get("d_model", 256),
            d_state=model_cfg.get("d_state", 64),
            n_layers=model_cfg.get("n_layers", 2),
            headdim=model_cfg.get("headdim", 64),
            dropout=model_cfg.get("dropout", 0.1),
            chunk_size=model_cfg.get("chunk_size"),
        )
    else:
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
            "num_games_loaded": len(game_md5s),
            "train_examples": len(train_ds),
            "val_examples": len(val_ds) if val_ds else 0,
            "total_frames": total_frames,
            "fingerprint": data_fingerprint,
            "streaming": args.streaming,
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

    # Save manifest alongside checkpoints
    manifest_dir = Path(save_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": run_config,
        "data_fingerprint": data_fingerprint,
        "game_md5s": game_md5s,
        "results": history[-1] if history else {},
        "all_epochs": history,
    }
    manifest_path = manifest_dir / "manifest.json"

    def _json_default(obj):
        """Handle numpy types in JSON serialization."""
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    logging.info("Saved run manifest: %s", manifest_path)

    if wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
