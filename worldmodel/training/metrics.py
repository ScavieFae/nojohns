"""Loss computation and metric tracking for world model training."""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from worldmodel.model.encoding import EncodingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action categories (by libmelee Action enum value)
# ---------------------------------------------------------------------------
ACTION_CATEGORIES: dict[str, set[int]] = {
    "idle": {14},
    "movement": set(range(15, 24)) | set(range(39, 42)) | {244},
    "aerial": set(range(24, 38)) | {42, 43},
    "ground_attack": set(range(44, 65)),
    "aerial_attack": set(range(65, 75)),
    "damage": {38} | set(range(75, 92)),
    "shield_dodge": {178, 179, 180, 181, 182, 233, 234, 235, 236},
    "grab": set(range(212, 233)) | set(range(239, 244)),
    "edge": set(range(245, 264)),
    "special": set(range(342, 398)),
}

_ACTION_TO_CATEGORY: dict[int, str] = {}
for _cat, _ids in ACTION_CATEGORIES.items():
    for _aid in _ids:
        _ACTION_TO_CATEGORY[_aid] = _cat

# Action names (best-effort from libmelee)
try:
    from melee import Action as _MeleeAction
    _ACTION_NAMES: dict[int, str] = {a.value: a.name for a in _MeleeAction}
except ImportError:
    _ACTION_NAMES = {}


def action_name(idx: int) -> str:
    return _ACTION_NAMES.get(idx, f"action_{idx}")


class ActionBreakdown:
    """Per-action accuracy tracker for validation epochs."""

    def __init__(self, num_actions: int = 400):
        self.correct = torch.zeros(num_actions, dtype=torch.long)
        self.total = torch.zeros(num_actions, dtype=torch.long)

    def update(self, pred_logits: torch.Tensor, true_labels: torch.Tensor) -> None:
        """Accumulate hits/misses per action ID. Handles both players."""
        pred = pred_logits.argmax(dim=-1).cpu()
        true = true_labels.cpu()
        correct_mask = pred == true
        for aid in true.unique():
            aid_int = aid.item()
            if aid_int >= len(self.total):
                continue
            mask = true == aid
            self.total[aid_int] += mask.sum()
            self.correct[aid_int] += correct_mask[mask].sum()

    def category_accuracies(self) -> dict[str, float]:
        """Accuracy per gameplay category."""
        result = {}
        for cat, ids in ACTION_CATEGORIES.items():
            ids_list = [i for i in ids if i < len(self.total) and self.total[i] > 0]
            if not ids_list:
                continue
            cat_correct = sum(self.correct[i].item() for i in ids_list)
            cat_total = sum(self.total[i].item() for i in ids_list)
            result[cat] = cat_correct / cat_total
        return result

    def worst_actions(self, n: int = 5) -> list[tuple[int, float, int]]:
        """Top N actions by total error count. Returns (action_id, accuracy, total_count)."""
        active = (self.total > 0).nonzero(as_tuple=True)[0]
        if len(active) == 0:
            return []
        errors = self.total[active] - self.correct[active]
        sorted_idx = errors.argsort(descending=True)[:n]
        result = []
        for idx in sorted_idx:
            aid = active[idx].item()
            acc = self.correct[aid].item() / self.total[aid].item()
            result.append((aid, acc, self.total[aid].item()))
        return result

    def summary_metrics(self, prefix: str = "val_action") -> dict[str, float]:
        """Metrics dict for wandb logging."""
        metrics: dict[str, float] = {}
        for cat, acc in self.category_accuracies().items():
            metrics[f"{prefix}/cat_{cat}"] = acc
        # "other" category — everything not in a named category
        categorized: set[int] = set()
        for ids in ACTION_CATEGORIES.values():
            categorized |= ids
        other_correct = other_total = 0
        for i in range(len(self.total)):
            if i not in categorized and self.total[i] > 0:
                other_correct += self.correct[i].item()
                other_total += self.total[i].item()
        if other_total > 0:
            metrics[f"{prefix}/cat_other"] = other_correct / other_total
        metrics[f"{prefix}/num_active_actions"] = float((self.total > 0).sum().item())
        return metrics


@dataclass
class LossWeights:
    continuous: float = 1.0
    velocity: float = 0.5
    dynamics: float = 0.5
    binary: float = 0.5
    action: float = 2.0
    jumps: float = 0.5
    l_cancel: float = 0.3
    hurtbox: float = 0.3
    ground: float = 0.3
    last_attack: float = 0.3
    action_change_weight: float = 1.0  # E010b: multiplier for action-change frames (1.0 = off)


@dataclass
class EpochMetrics:
    total_loss: float = 0.0
    continuous_loss: float = 0.0
    velocity_loss: float = 0.0
    dynamics_loss: float = 0.0
    binary_loss: float = 0.0
    action_loss: float = 0.0
    jumps_loss: float = 0.0
    l_cancel_loss: float = 0.0
    hurtbox_loss: float = 0.0
    ground_loss: float = 0.0
    last_attack_loss: float = 0.0
    position_mae: float = 0.0
    velocity_mae: float = 0.0
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
        self.velocity_loss += bm.velocity_loss
        self.dynamics_loss += bm.dynamics_loss
        self.binary_loss += bm.binary_loss
        self.action_loss += bm.action_loss
        self.jumps_loss += bm.jumps_loss
        self.l_cancel_loss += bm.l_cancel_loss
        self.hurtbox_loss += bm.hurtbox_loss
        self.ground_loss += bm.ground_loss
        self.last_attack_loss += bm.last_attack_loss
        self.position_mae += bm.position_mae
        self.velocity_mae += bm.velocity_mae
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
            "loss/velocity": self.velocity_loss / n,
            "loss/dynamics": self.dynamics_loss / n,
            "loss/binary": self.binary_loss / n,
            "loss/action": self.action_loss / n,
            "loss/jumps": self.jumps_loss / n,
            "loss/l_cancel": self.l_cancel_loss / n,
            "loss/hurtbox": self.hurtbox_loss / n,
            "loss/ground": self.ground_loss / n,
            "loss/last_attack": self.last_attack_loss / n,
            "metric/position_mae": self.position_mae / n,
            "metric/velocity_mae": self.velocity_mae / n,
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
    velocity_loss: float = 0.0
    dynamics_loss: float = 0.0
    binary_loss: float = 0.0
    action_loss: float = 0.0
    jumps_loss: float = 0.0
    l_cancel_loss: float = 0.0
    hurtbox_loss: float = 0.0
    ground_loss: float = 0.0
    last_attack_loss: float = 0.0
    position_mae: float = 0.0
    velocity_mae: float = 0.0
    damage_mae: float = 0.0
    p0_action_acc: float = 0.0
    p1_action_acc: float = 0.0
    p0_action_change_correct: float = 0.0
    num_action_changes: int = 0
    binary_acc: float = 0.0


class MetricsTracker:
    """Computes losses and tracks metrics.

    Target layout:
        float_tgt: (B, 30) = [p0_cont_delta(4), p1_cont_delta(4), p0_vel_delta(5), p1_vel_delta(5),
                               p0_binary(3), p1_binary(3), p0_dynamics(3), p1_dynamics(3)]
        int_tgt:   (B, 12) = [p0_action, p0_jumps, p0_l_cancel, p0_hurtbox, p0_ground, p0_last_attack,
                               p1_action, p1_jumps, p1_l_cancel, p1_hurtbox, p1_ground, p1_last_attack]
    """

    def __init__(self, cfg: EncodingConfig, weights: LossWeights | None = None):
        self.cfg = cfg
        self.weights = weights or LossWeights()
        # Config-driven target layout offsets
        self._cont_end = 8
        self._vel_end = 18
        self._bin_end = 18 + cfg.predicted_binary_dim
        self._dyn_end = self._bin_end + cfg.predicted_dynamics_dim

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        float_tgt: torch.Tensor,
        int_tgt: torch.Tensor,
        int_ctx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, BatchMetrics]:
        metrics = BatchMetrics()

        # E008c: multi-position — reshape (B, K, ...) to (B*K, ...) for uniform loss
        multi_pos = predictions.get("_multi_position", False)
        if multi_pos:
            B, K = float_tgt.shape[:2]
            float_tgt = float_tgt.reshape(B * K, -1)
            int_tgt = int_tgt.reshape(B * K, -1)
            predictions = {k: v.reshape(B * K, *v.shape[2:])
                           for k, v in predictions.items() if isinstance(v, torch.Tensor)}
            # For action-change: at position i, "previous action" is int_ctx[:, i, 0]
            if int_ctx is not None:
                int_ctx = int_ctx.reshape(B * K, 1, -1)  # (B*K, 1, I)

        # Unpack targets (config-driven slicing)
        cont_delta_true = float_tgt[:, :self._cont_end]       # (B, 8) position/shield deltas
        vel_delta_true = float_tgt[:, self._cont_end:self._vel_end]  # (B, 10) velocity deltas
        binary_true = float_tgt[:, self._vel_end:self._bin_end]      # (B, 6..86) binary flags
        dynamics_true = float_tgt[:, self._bin_end:self._dyn_end]    # (B, 6..8) hitlag/stocks/combo[/hitstun]

        # int_tgt layout: [p0: action, jumps, l_cancel, hurtbox, ground, last_attack,
        #                  p1: action, jumps, l_cancel, hurtbox, ground, last_attack]
        p0_action_true = int_tgt[:, 0]  # (B,)
        p0_jumps_true = int_tgt[:, 1]
        p0_l_cancel_true = int_tgt[:, 2]
        p0_hurtbox_true = int_tgt[:, 3]
        p0_ground_true = int_tgt[:, 4]
        p0_last_attack_true = int_tgt[:, 5]
        p1_action_true = int_tgt[:, 6]
        p1_jumps_true = int_tgt[:, 7]
        p1_l_cancel_true = int_tgt[:, 8]
        p1_hurtbox_true = int_tgt[:, 9]
        p1_ground_true = int_tgt[:, 10]
        p1_last_attack_true = int_tgt[:, 11]

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

        # Velocity delta loss (MSE)
        vel_loss = F.mse_loss(predictions["velocity_delta"], vel_delta_true)
        metrics.velocity_loss = vel_loss.item()

        with torch.no_grad():
            vel_pred = predictions["velocity_delta"].detach()
            metrics.velocity_mae = ((vel_pred - vel_delta_true).abs() / self.cfg.velocity_scale).mean().item()

        # Dynamics loss (MSE — absolute values for hitlag, stocks, combo)
        dyn_loss = F.mse_loss(predictions["dynamics_pred"], dynamics_true)
        metrics.dynamics_loss = dyn_loss.item()

        # Binary loss (BCE with logits)
        bin_loss = F.binary_cross_entropy_with_logits(predictions["binary_logits"], binary_true)
        metrics.binary_loss = bin_loss.item()
        with torch.no_grad():
            metrics.binary_acc = ((predictions["binary_logits"] > 0).float() == binary_true).float().mean().item()

        # Action loss (CE per player)
        # E010b: weight action-change frames higher to improve transition prediction
        w = self.weights.action_change_weight
        if int_ctx is not None and w > 1.0:
            # Detect frames where action changes from context's last frame
            prev_p0 = int_ctx[:, -1, 0]  # previous p0 action
            prev_p1 = int_ctx[:, -1, self.cfg.int_per_player]  # previous p1 action
            p0_changed = (prev_p0 != p0_action_true).float()
            p1_changed = (prev_p1 != p1_action_true).float()
            p0_sample_w = 1.0 + (w - 1.0) * p0_changed  # 1.0 for steady, w for change
            p1_sample_w = 1.0 + (w - 1.0) * p1_changed
            p0_act_loss = (F.cross_entropy(predictions["p0_action_logits"], p0_action_true, reduction='none') * p0_sample_w).mean()
            p1_act_loss = (F.cross_entropy(predictions["p1_action_logits"], p1_action_true, reduction='none') * p1_sample_w).mean()
        else:
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

        # Combat context losses (CE per player, averaged)
        lc_loss = (F.cross_entropy(predictions["p0_l_cancel_logits"], p0_l_cancel_true)
                   + F.cross_entropy(predictions["p1_l_cancel_logits"], p1_l_cancel_true)) / 2
        metrics.l_cancel_loss = lc_loss.item()

        hb_loss = (F.cross_entropy(predictions["p0_hurtbox_logits"], p0_hurtbox_true)
                   + F.cross_entropy(predictions["p1_hurtbox_logits"], p1_hurtbox_true)) / 2
        metrics.hurtbox_loss = hb_loss.item()

        gnd_loss = (F.cross_entropy(predictions["p0_ground_logits"], p0_ground_true)
                    + F.cross_entropy(predictions["p1_ground_logits"], p1_ground_true)) / 2
        metrics.ground_loss = gnd_loss.item()

        la_loss = (F.cross_entropy(predictions["p0_last_attack_logits"], p0_last_attack_true)
                   + F.cross_entropy(predictions["p1_last_attack_logits"], p1_last_attack_true)) / 2
        metrics.last_attack_loss = la_loss.item()

        total = (
            self.weights.continuous * cont_loss
            + self.weights.velocity * vel_loss
            + self.weights.dynamics * dyn_loss
            + self.weights.binary * bin_loss
            + self.weights.action * action_loss
            + self.weights.jumps * jumps_loss
            + self.weights.l_cancel * lc_loss
            + self.weights.hurtbox * hb_loss
            + self.weights.ground * gnd_loss
            + self.weights.last_attack * la_loss
        )
        metrics.total_loss = total.item()

        return total, metrics
