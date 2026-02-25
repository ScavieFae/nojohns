"""Training loop for the world model."""

import dataclasses
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from torch.utils.data import IterableDataset

from worldmodel.data.dataset import MeleeFrameDataset
from worldmodel.model.encoding import EncodingConfig
from worldmodel.training.metrics import (
    ActionBreakdown,
    EpochMetrics,
    LossWeights,
    MetricsTracker,
    action_name,
)

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_dataset: MeleeFrameDataset,
        val_dataset: Optional[MeleeFrameDataset],
        cfg: EncodingConfig,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        num_epochs: int = 50,
        loss_weights: Optional[LossWeights] = None,
        save_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
        resume_from: Optional[str | Path] = None,
        rollout_every_n: int = 5,
        rollout_games: int = 3,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        logger.info("Using device: %s", self.device)

        self.model = model.to(self.device)
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir) if save_dir else None
        self.start_epoch = 0

        # DataLoader — no custom collate, default stacking works with tuple returns
        # IterableDataset handles its own shuffling; map-style Dataset uses shuffle=True
        is_iterable = isinstance(train_dataset, IterableDataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=not is_iterable,
            num_workers=0,
            drop_last=True,
            pin_memory=(device != "cpu"),
        )
        self.val_loader = None
        if val_dataset and len(val_dataset) > 0:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
                pin_memory=(device != "cpu"),
            )

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        self.metrics = MetricsTracker(cfg, loss_weights)
        self.history: list[dict] = []

        # Shape preflight: pull one batch and verify dims match config
        self._verify_shapes(train_dataset)

        # Rollout validation setup
        self.rollout_every_n = rollout_every_n
        self._val_games: list[tuple[torch.Tensor, torch.Tensor]] = []

        if val_dataset and rollout_every_n > 0 and hasattr(val_dataset, 'game_range') and hasattr(val_dataset, 'data'):
            offsets = val_dataset.data.game_offsets
            games_to_use = list(val_dataset.game_range)[:rollout_games]
            for gi in games_to_use:
                start = offsets[gi]
                end = offsets[gi + 1]
                if end - start > model.context_len + 50:
                    self._val_games.append((
                        val_dataset.data.floats[start:end].clone(),
                        val_dataset.data.ints[start:end].clone(),
                    ))
            if self._val_games:
                logger.info("Rollout validation: %d games, every %d epochs", len(self._val_games), rollout_every_n)

        # Resume from checkpoint
        if resume_from:
            self._load_checkpoint(resume_from)

    def _verify_shapes(self, dataset) -> None:
        """Preflight check: verify first sample's tensor shapes match config.

        Catches config/data mismatches immediately instead of mid-epoch.
        """
        try:
            sample = dataset[0] if hasattr(dataset, '__getitem__') else next(iter(dataset))
        except Exception:
            logger.warning("Shape preflight: couldn't read sample, skipping check")
            return

        float_ctx, int_ctx, next_ctrl, float_tgt, int_tgt = sample
        cfg = self.cfg
        K = self.model.context_len
        expected_float = cfg.float_per_player * 2
        expected_int = cfg.int_per_frame
        expected_ctrl = cfg.ctrl_conditioning_dim

        errors = []
        if float_ctx.shape != (K, expected_float):
            errors.append(f"float_ctx: got {tuple(float_ctx.shape)}, expected ({K}, {expected_float})")
        if int_ctx.shape != (K, expected_int):
            errors.append(f"int_ctx: got {tuple(int_ctx.shape)}, expected ({K}, {expected_int})")
        if next_ctrl.shape != (expected_ctrl,):
            errors.append(f"next_ctrl: got {tuple(next_ctrl.shape)}, expected ({expected_ctrl},)")
        if float_tgt.shape != (30,):
            errors.append(f"float_tgt: got {tuple(float_tgt.shape)}, expected (30,)")
        if int_tgt.shape != (cfg.target_int_dim,):
            errors.append(f"int_tgt: got {tuple(int_tgt.shape)}, expected ({cfg.target_int_dim},)")

        if errors:
            msg = "Shape preflight FAILED — data/config mismatch:\n  " + "\n  ".join(errors)
            msg += f"\n  Config: state_age_as_embed={cfg.state_age_as_embed}, press_events={cfg.press_events}"
            raise ValueError(msg)

        logger.info(
            "Shape preflight OK: float_ctx=(%d,%d) int_ctx=(%d,%d) next_ctrl=(%d,)",
            K, expected_float, K, expected_int, expected_ctrl,
        )

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        epoch_metrics = EpochMetrics()

        for float_ctx, int_ctx, next_ctrl, float_tgt, int_tgt in self.train_loader:
            float_ctx = float_ctx.to(self.device)
            int_ctx = int_ctx.to(self.device)
            next_ctrl = next_ctrl.to(self.device)
            float_tgt = float_tgt.to(self.device)
            int_tgt = int_tgt.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(float_ctx, int_ctx, next_ctrl)
            loss, batch_metrics = self.metrics.compute_loss(predictions, float_tgt, int_tgt, int_ctx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_metrics.update(batch_metrics)

        return epoch_metrics.averaged()

    @torch.no_grad()
    def _val_epoch(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        epoch_metrics = EpochMetrics()
        action_tracker = ActionBreakdown(self.cfg.action_vocab)

        for float_ctx, int_ctx, next_ctrl, float_tgt, int_tgt in self.val_loader:
            float_ctx = float_ctx.to(self.device)
            int_ctx = int_ctx.to(self.device)
            next_ctrl = next_ctrl.to(self.device)
            float_tgt = float_tgt.to(self.device)
            int_tgt = int_tgt.to(self.device)

            predictions = self.model(float_ctx, int_ctx, next_ctrl)
            _, batch_metrics = self.metrics.compute_loss(predictions, float_tgt, int_tgt, int_ctx)
            epoch_metrics.update(batch_metrics)

            # Per-action tracking (both players — p1_action is at index 6)
            action_tracker.update(predictions["p0_action_logits"], int_tgt[:, 0])
            action_tracker.update(predictions["p1_action_logits"], int_tgt[:, 6])

        result = {f"val_{k}": v for k, v in epoch_metrics.averaged().items()}
        result.update(action_tracker.summary_metrics())

        # Log worst actions to text
        worst = action_tracker.worst_actions(5)
        if worst:
            parts = [f"{action_name(aid)}({acc:.1%}, n={n})" for aid, acc, n in worst]
            logger.info("Worst actions: %s", ", ".join(parts))

        return result

    @torch.no_grad()
    def _rollout_validation(self) -> dict[str, float]:
        """Run autoregressive rollouts on validation games and measure drift."""
        from worldmodel.scripts.rollout import rollout, decode_frame

        horizons = [10, 50, 100, 200, 300]
        pos_errors: dict[int, list[float]] = {h: [] for h in horizons}
        action_matches: dict[int, list[float]] = {h: [] for h in horizons}
        stocks_matches: dict[int, list[float]] = {h: [] for h in horizons}
        alive: dict[int, list[float]] = {h: [] for h in horizons}

        K = self.model.context_len

        for float_data, int_data in self._val_games:
            pred_frames = rollout(
                self.model, float_data, int_data, self.cfg,
                max_frames=max(horizons),
                device=str(self.device),
            )

            for h in horizons:
                frame_idx = K + h
                if frame_idx >= len(pred_frames) or frame_idx >= float_data.shape[0]:
                    continue

                pred = pred_frames[frame_idx]
                actual = decode_frame(float_data[frame_idx], int_data[frame_idx], self.cfg)

                # Position MAE (game units, both players)
                p0_err = ((pred["p0"]["x"] - actual["p0"]["x"])**2 +
                          (pred["p0"]["y"] - actual["p0"]["y"])**2) ** 0.5
                p1_err = ((pred["p1"]["x"] - actual["p1"]["x"])**2 +
                          (pred["p1"]["y"] - actual["p1"]["y"])**2) ** 0.5
                pos_errors[h].append((p0_err + p1_err) / 2)

                # Action accuracy
                p0_act_match = 1.0 if pred["p0"]["action"] == actual["p0"]["action"] else 0.0
                p1_act_match = 1.0 if pred["p1"]["action"] == actual["p1"]["action"] else 0.0
                action_matches[h].append((p0_act_match + p1_act_match) / 2)

                # Stocks match
                p0_stk = 1.0 if abs(pred["p0"]["stocks"] - actual["p0"]["stocks"]) < 0.5 else 0.0
                p1_stk = 1.0 if abs(pred["p1"]["stocks"] - actual["p1"]["stocks"]) < 0.5 else 0.0
                stocks_matches[h].append((p0_stk + p1_stk) / 2)

                # Alive check: positions within stage bounds
                positions_ok = all(
                    abs(pred[p]["x"]) < 300 and abs(pred[p]["y"]) < 300
                    for p in ["p0", "p1"]
                )
                alive[h].append(1.0 if positions_ok else 0.0)

        metrics = {}
        for h in horizons:
            if pos_errors[h]:
                metrics[f"rollout/pos_mae_{h}f"] = sum(pos_errors[h]) / len(pos_errors[h])
                metrics[f"rollout/action_acc_{h}f"] = sum(action_matches[h]) / len(action_matches[h])
                metrics[f"rollout/stocks_match_{h}f"] = sum(stocks_matches[h]) / len(stocks_matches[h])
                metrics[f"rollout/alive_{h}f"] = sum(alive[h]) / len(alive[h])

        return metrics

    def train(self) -> list[dict]:
        logger.info(
            "Starting training: epochs %d→%d, %d train examples, batch_size=%d",
            self.start_epoch + 1, self.num_epochs,
            len(self.train_loader.dataset), self.batch_size,
        )

        best_val_loss = float("inf")

        for epoch in range(self.start_epoch, self.num_epochs):
            t0 = time.time()
            train_metrics = self._train_epoch()
            val_metrics = self._val_epoch()
            self.scheduler.step()

            # Rollout validation every N epochs
            rollout_metrics = {}
            if self._val_games and self.rollout_every_n > 0 and (epoch + 1) % self.rollout_every_n == 0:
                rollout_metrics = self._rollout_validation()
                logger.info(
                    "Rollout @ epoch %d: pos_mae_50f=%.1f action_acc_50f=%.3f",
                    epoch + 1,
                    rollout_metrics.get("rollout/pos_mae_50f", 0),
                    rollout_metrics.get("rollout/action_acc_50f", 0),
                )

            elapsed = time.time() - t0
            combined = {**train_metrics, **val_metrics, **rollout_metrics, "epoch": epoch, "time": elapsed}
            self.history.append(combined)

            loss_str = f"loss={train_metrics['loss/total']:.4f}"
            acc_str = f"action_acc={train_metrics['metric/p0_action_acc']:.3f}"
            pos_str = f"pos_mae={train_metrics['metric/position_mae']:.2f}"

            val_str = ""
            if val_metrics:
                val_str = f" | val_loss={val_metrics['val_loss/total']:.4f}"
                val_str += f" val_acc={val_metrics['val_metric/p0_action_acc']:.3f}"

            change_str = ""
            if "metric/action_change_acc" in train_metrics:
                change_str = f" change_acc={train_metrics['metric/action_change_acc']:.3f}"

            logger.info(
                "Epoch %d/%d [%.1fs]: %s %s %s%s%s",
                epoch + 1, self.num_epochs, elapsed,
                loss_str, acc_str, pos_str, change_str, val_str,
            )

            # Log to wandb
            if wandb and wandb.run:
                wandb.log({**combined, "lr": self.scheduler.get_last_lr()[0]})

            if self.save_dir and val_metrics:
                val_loss = val_metrics.get("val_loss/total", float("inf"))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best.pt", epoch, val_loss)

        if self.save_dir:
            self._save_checkpoint("final.pt", self.num_epochs - 1)

        return self.history

    def _load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing:
            logger.warning("Checkpoint missing %d keys (new heads will train from scratch): %s", len(missing), ", ".join(missing))
        if unexpected:
            logger.warning("Checkpoint has %d unexpected keys: %s", len(unexpected), ", ".join(unexpected))
        if missing or unexpected:
            logger.warning("Model architecture changed — reinitializing optimizer (LR schedule restarts)")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        # Advance scheduler to match
        for _ in range(self.start_epoch):
            self.scheduler.step()
        logger.info(
            "Resumed from %s (epoch %d, val_loss=%.4f)",
            path, checkpoint["epoch"] + 1, checkpoint.get("val_loss", 0),
        )

    def _save_checkpoint(self, name: str, epoch: int, val_loss: float = 0.0) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "encoding_config": dataclasses.asdict(self.cfg),
                "context_len": self.model.context_len,
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)
