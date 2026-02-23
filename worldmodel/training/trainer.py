"""Training loop for the world model."""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from worldmodel.data.dataset import MeleeFrameDataset
from worldmodel.model.encoding import EncodingConfig
from worldmodel.model.mlp import FrameStackMLP
from worldmodel.training.metrics import EpochMetrics, LossWeights, MetricsTracker

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        model: FrameStackMLP,
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

        # DataLoader — no custom collate, default stacking works with tuple returns
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
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

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        epoch_metrics = EpochMetrics()

        for float_ctx, int_ctx, float_tgt, int_tgt in self.train_loader:
            float_ctx = float_ctx.to(self.device)
            int_ctx = int_ctx.to(self.device)
            float_tgt = float_tgt.to(self.device)
            int_tgt = int_tgt.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(float_ctx, int_ctx)
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

        for float_ctx, int_ctx, float_tgt, int_tgt in self.val_loader:
            float_ctx = float_ctx.to(self.device)
            int_ctx = int_ctx.to(self.device)
            float_tgt = float_tgt.to(self.device)
            int_tgt = int_tgt.to(self.device)

            predictions = self.model(float_ctx, int_ctx)
            _, batch_metrics = self.metrics.compute_loss(predictions, float_tgt, int_tgt, int_ctx)
            epoch_metrics.update(batch_metrics)

        return {f"val_{k}": v for k, v in epoch_metrics.averaged().items()}

    def train(self) -> list[dict]:
        logger.info(
            "Starting training: %d epochs, %d train examples, batch_size=%d",
            self.num_epochs, len(self.train_loader.dataset), self.batch_size,
        )

        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            t0 = time.time()
            train_metrics = self._train_epoch()
            val_metrics = self._val_epoch()
            self.scheduler.step()

            elapsed = time.time() - t0
            combined = {**train_metrics, **val_metrics, "epoch": epoch, "time": elapsed}
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

            if self.save_dir and val_metrics:
                val_loss = val_metrics.get("val_loss/total", float("inf"))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best.pt", epoch, val_loss)

        if self.save_dir:
            self._save_checkpoint("final.pt", self.num_epochs - 1)

        return self.history

    def _save_checkpoint(self, name: str, epoch: int, val_loss: float = 0.0) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "action_vocab": self.cfg.action_vocab,
                    "action_embed_dim": self.cfg.action_embed_dim,
                    "jumps_vocab": self.cfg.jumps_vocab,
                    "jumps_embed_dim": self.cfg.jumps_embed_dim,
                    "context_len": self.model.context_len,
                },
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)
