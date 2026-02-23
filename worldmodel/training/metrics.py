"""Loss computation and metric tracking for world model training."""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from worldmodel.model.encoding import EncodingConfig

logger = logging.getLogger(__name__)


@dataclass
class LossWeights:
    continuous: float = 1.0
    binary: float = 0.5
    action: float = 2.0
    jumps: float = 0.5


@dataclass
class EpochMetrics:
    total_loss: float = 0.0
    continuous_loss: float = 0.0
    binary_loss: float = 0.0
    action_loss: float = 0.0
    jumps_loss: float = 0.0
    position_mae: float = 0.0
    damage_mae: float = 0.0
    p0_action_acc: float = 0.0
    p1_action_acc: float = 0.0
    p0_action_change_acc: float = 0.0
    binary_acc: float = 0.0
    num_batches: int = 0
    num_action_changes: int = 0

    def update(self, bm: "BatchMetrics") -> None:
        self.total_loss += bm.total_loss
        self.continuous_loss += bm.continuous_loss
        self.binary_loss += bm.binary_loss
        self.action_loss += bm.action_loss
        self.jumps_loss += bm.jumps_loss
        self.position_mae += bm.position_mae
        self.damage_mae += bm.damage_mae
        self.p0_action_acc += bm.p0_action_acc
        self.p1_action_acc += bm.p1_action_acc
        self.p0_action_change_acc += bm.p0_action_change_correct
        self.binary_acc += bm.binary_acc
        self.num_batches += 1
        self.num_action_changes += bm.num_action_changes

    def averaged(self) -> dict[str, float]:
        n = max(self.num_batches, 1)
        result = {
            "loss/total": self.total_loss / n,
            "loss/continuous": self.continuous_loss / n,
            "loss/binary": self.binary_loss / n,
            "loss/action": self.action_loss / n,
            "loss/jumps": self.jumps_loss / n,
            "metric/position_mae": self.position_mae / n,
            "metric/damage_mae": self.damage_mae / n,
            "metric/p0_action_acc": self.p0_action_acc / n,
            "metric/p1_action_acc": self.p1_action_acc / n,
            "metric/binary_acc": self.binary_acc / n,
        }
        if self.num_action_changes > 0:
            result["metric/action_change_acc"] = self.p0_action_change_acc / self.num_action_changes
        return result


@dataclass
class BatchMetrics:
    total_loss: float = 0.0
    continuous_loss: float = 0.0
    binary_loss: float = 0.0
    action_loss: float = 0.0
    jumps_loss: float = 0.0
    position_mae: float = 0.0
    damage_mae: float = 0.0
    p0_action_acc: float = 0.0
    p1_action_acc: float = 0.0
    p0_action_change_correct: float = 0.0
    num_action_changes: int = 0
    binary_acc: float = 0.0


class MetricsTracker:
    """Computes losses and tracks metrics.

    Target layout:
        float_tgt: (B, 14) = [p0_cont_delta(4), p1_cont_delta(4), p0_binary(3), p1_binary(3)]
        int_tgt:   (B, 4)  = [p0_action, p0_jumps, p1_action, p1_jumps]
    """

    def __init__(self, cfg: EncodingConfig, weights: LossWeights | None = None):
        self.cfg = cfg
        self.weights = weights or LossWeights()

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        float_tgt: torch.Tensor,
        int_tgt: torch.Tensor,
        int_ctx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, BatchMetrics]:
        metrics = BatchMetrics()

        # Unpack targets
        cont_delta_true = float_tgt[:, :8]  # (B, 8)
        binary_true = float_tgt[:, 8:]  # (B, 6)
        p0_action_true = int_tgt[:, 0]  # (B,)
        p0_jumps_true = int_tgt[:, 1]
        p1_action_true = int_tgt[:, 2]
        p1_jumps_true = int_tgt[:, 3]

        # Continuous delta loss (MSE)
        cont_loss = F.mse_loss(predictions["continuous_delta"], cont_delta_true)
        metrics.continuous_loss = cont_loss.item()

        with torch.no_grad():
            pred_d = predictions["continuous_delta"].detach()
            # Position MAE (un-normalized): indices 1,2 (p0 x,y) and 5,6 (p1 x,y)
            pos_pred = torch.cat([pred_d[:, 1:3], pred_d[:, 5:7]], dim=1)
            pos_true = torch.cat([cont_delta_true[:, 1:3], cont_delta_true[:, 5:7]], dim=1)
            metrics.position_mae = ((pos_pred - pos_true).abs() / self.cfg.xy_scale).mean().item()
            # Damage MAE
            dmg_pred = torch.stack([pred_d[:, 0], pred_d[:, 4]], dim=1)
            dmg_true = torch.stack([cont_delta_true[:, 0], cont_delta_true[:, 4]], dim=1)
            metrics.damage_mae = ((dmg_pred - dmg_true).abs() / self.cfg.percent_scale).mean().item()

        # Binary loss (BCE with logits)
        bin_loss = F.binary_cross_entropy_with_logits(predictions["binary_logits"], binary_true)
        metrics.binary_loss = bin_loss.item()
        with torch.no_grad():
            metrics.binary_acc = ((predictions["binary_logits"] > 0).float() == binary_true).float().mean().item()

        # Action loss (CE per player)
        p0_act_loss = F.cross_entropy(predictions["p0_action_logits"], p0_action_true)
        p1_act_loss = F.cross_entropy(predictions["p1_action_logits"], p1_action_true)
        action_loss = (p0_act_loss + p1_act_loss) / 2
        metrics.action_loss = action_loss.item()

        with torch.no_grad():
            p0_pred = predictions["p0_action_logits"].argmax(dim=-1)
            p1_pred = predictions["p1_action_logits"].argmax(dim=-1)
            metrics.p0_action_acc = (p0_pred == p0_action_true).float().mean().item()
            metrics.p1_action_acc = (p1_pred == p1_action_true).float().mean().item()

            if int_ctx is not None:
                p0_curr_action = int_ctx[:, -1, 0]
                changed = p0_curr_action != p0_action_true
                n_changed = changed.sum().item()
                metrics.num_action_changes = int(n_changed)
                if n_changed > 0:
                    metrics.p0_action_change_correct = (p0_pred[changed] == p0_action_true[changed]).sum().item()

        # Jumps loss
        p0_j_loss = F.cross_entropy(predictions["p0_jumps_logits"], p0_jumps_true)
        p1_j_loss = F.cross_entropy(predictions["p1_jumps_logits"], p1_jumps_true)
        jumps_loss = (p0_j_loss + p1_j_loss) / 2
        metrics.jumps_loss = jumps_loss.item()

        total = (
            self.weights.continuous * cont_loss
            + self.weights.binary * bin_loss
            + self.weights.action * action_loss
            + self.weights.jumps * jumps_loss
        )
        metrics.total_loss = total.item()

        return total, metrics
