"""Training loop and metrics for imitation learning policy."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from worldmodel.data.policy_dataset import ANALOG_DIM, BUTTON_DIM
from worldmodel.model.policy_mlp import PolicyMLP

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)

BUTTON_NAMES = ["A", "B", "X", "Y", "Z", "L", "R", "D_UP"]


@dataclass
class PolicyLossWeights:
    analog: float = 1.0
    button: float = 1.0


@dataclass
class PolicyBatchMetrics:
    total_loss: float = 0.0
    analog_loss: float = 0.0
    button_loss: float = 0.0
    analog_mae: float = 0.0
    stick_mae: float = 0.0  # main + c stick only (not trigger)
    button_acc: float = 0.0
    button_pressed_acc: float = 0.0  # accuracy on frames where a button IS pressed
    num_button_pressed: int = 0


@dataclass
class PolicyEpochMetrics:
    total_loss: float = 0.0
    analog_loss: float = 0.0
    button_loss: float = 0.0
    analog_mae: float = 0.0
    stick_mae: float = 0.0
    button_acc: float = 0.0
    button_pressed_acc: float = 0.0
    num_button_pressed: int = 0
    num_batches: int = 0

    def update(self, bm: PolicyBatchMetrics) -> None:
        self.total_loss += bm.total_loss
        self.analog_loss += bm.analog_loss
        self.button_loss += bm.button_loss
        self.analog_mae += bm.analog_mae
        self.stick_mae += bm.stick_mae
        self.button_acc += bm.button_acc
        self.button_pressed_acc += bm.button_pressed_acc
        self.num_button_pressed += bm.num_button_pressed
        self.num_batches += 1

    def averaged(self) -> dict[str, float]:
        n = max(self.num_batches, 1)
        result = {
            "loss/total": self.total_loss / n,
            "loss/analog": self.analog_loss / n,
            "loss/button": self.button_loss / n,
            "metric/analog_mae": self.analog_mae / n,
            "metric/stick_mae": self.stick_mae / n,
            "metric/button_acc": self.button_acc / n,
        }
        if self.num_button_pressed > 0:
            result["metric/button_pressed_acc"] = (
                self.button_pressed_acc / self.num_button_pressed
            )
        return result


class PolicyMetrics:
    """Compute policy losses and track metrics."""

    def __init__(self, weights: PolicyLossWeights | None = None):
        self.weights = weights or PolicyLossWeights()

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        ctrl_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, PolicyBatchMetrics]:
        metrics = PolicyBatchMetrics()

        # Split target: analog (first 5) and buttons (last 8)
        analog_tgt = ctrl_tgt[:, :ANALOG_DIM]  # (B, 5)
        button_tgt = ctrl_tgt[:, ANALOG_DIM:]  # (B, 8)

        # Analog loss: MSE on [0, 1] values
        analog_pred = predictions["analog_pred"]
        analog_loss = F.mse_loss(analog_pred, analog_tgt)
        metrics.analog_loss = analog_loss.item()

        with torch.no_grad():
            metrics.analog_mae = (analog_pred - analog_tgt).abs().mean().item()
            # Stick MAE: just the 4 stick axes (not trigger)
            metrics.stick_mae = (analog_pred[:, :4] - analog_tgt[:, :4]).abs().mean().item()

        # Button loss: BCE with logits
        button_logits = predictions["button_logits"]
        button_loss = F.binary_cross_entropy_with_logits(button_logits, button_tgt)
        metrics.button_loss = button_loss.item()

        with torch.no_grad():
            button_pred = (button_logits > 0).float()
            metrics.button_acc = (button_pred == button_tgt).float().mean().item()

            # Accuracy specifically on frames where any button is pressed
            any_pressed = button_tgt.sum(dim=1) > 0  # (B,)
            n_pressed = any_pressed.sum().item()
            metrics.num_button_pressed = int(n_pressed)
            if n_pressed > 0:
                pressed_correct = (button_pred[any_pressed] == button_tgt[any_pressed])
                metrics.button_pressed_acc = pressed_correct.float().mean().item() * n_pressed

        total = self.weights.analog * analog_loss + self.weights.button * button_loss
        metrics.total_loss = total.item()

        return total, metrics


class PolicyTrainer:
    """Training loop for imitation learning policy."""

    def __init__(
        self,
        model: PolicyMLP,
        train_dataset,
        val_dataset=None,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        num_epochs: int = 50,
        loss_weights: Optional[PolicyLossWeights] = None,
        save_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
        resume_from: Optional[str | Path] = None,
        epoch_callback: Optional[Callable] = None,
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
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir) if save_dir else None
        self.start_epoch = 0

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
        self.metrics = PolicyMetrics(loss_weights)
        self.epoch_callback = epoch_callback
        self.history: list[dict] = []

        if resume_from:
            self._load_checkpoint(resume_from)

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        epoch_metrics = PolicyEpochMetrics()

        for float_ctx, int_ctx, ctrl_tgt in self.train_loader:
            float_ctx = float_ctx.to(self.device)
            int_ctx = int_ctx.to(self.device)
            ctrl_tgt = ctrl_tgt.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(float_ctx, int_ctx)
            loss, batch_metrics = self.metrics.compute_loss(predictions, ctrl_tgt)
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
        epoch_metrics = PolicyEpochMetrics()

        for float_ctx, int_ctx, ctrl_tgt in self.val_loader:
            float_ctx = float_ctx.to(self.device)
            int_ctx = int_ctx.to(self.device)
            ctrl_tgt = ctrl_tgt.to(self.device)

            predictions = self.model(float_ctx, int_ctx)
            _, batch_metrics = self.metrics.compute_loss(predictions, ctrl_tgt)
            epoch_metrics.update(batch_metrics)

        return {f"val_{k}": v for k, v in epoch_metrics.averaged().items()}

    def train(self) -> list[dict]:
        logger.info(
            "Starting policy training: epochs %d→%d, %d train examples, batch_size=%d",
            self.start_epoch + 1, self.num_epochs,
            len(self.train_loader.dataset), self.batch_size,
        )

        best_val_loss = float("inf")

        for epoch in range(self.start_epoch, self.num_epochs):
            t0 = time.time()
            train_metrics = self._train_epoch()
            val_metrics = self._val_epoch()
            self.scheduler.step()

            elapsed = time.time() - t0
            combined = {**train_metrics, **val_metrics, "epoch": epoch, "time": elapsed}
            self.history.append(combined)

            loss_str = f"loss={train_metrics['loss/total']:.4f}"
            stick_str = f"stick_mae={train_metrics['metric/stick_mae']:.4f}"
            btn_str = f"btn_acc={train_metrics['metric/button_acc']:.3f}"

            val_str = ""
            if val_metrics:
                val_str = f" | val_loss={val_metrics['val_loss/total']:.4f}"
                val_str += f" val_btn={val_metrics['val_metric/button_acc']:.3f}"

            pressed_str = ""
            if "metric/button_pressed_acc" in train_metrics:
                pressed_str = f" pressed_acc={train_metrics['metric/button_pressed_acc']:.3f}"

            logger.info(
                "Epoch %d/%d [%.1fs]: %s %s %s%s%s",
                epoch + 1, self.num_epochs, elapsed,
                loss_str, stick_str, btn_str, pressed_str, val_str,
            )

            if wandb and wandb.run:
                wandb.log({**combined, "lr": self.scheduler.get_last_lr()[0]})

            if self.save_dir:
                self._save_checkpoint("latest.pt", epoch)
                if val_metrics:
                    val_loss = val_metrics.get("val_loss/total", float("inf"))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self._save_checkpoint("best.pt", epoch, val_loss)

            if self.epoch_callback:
                self.epoch_callback()

        return self.history

    def _load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
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
                "config": {
                    "context_len": self.model.context_len,
                    "model_type": "policy_mlp",
                },
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)
