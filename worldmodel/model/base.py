"""World model protocol — interface that all architectures implement."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class WorldModel(Protocol):
    """Protocol for next-frame prediction models.

    All world models take encoded game state and predict next-frame changes.
    The prediction is split into separate heads for different field types.
    """

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict next-frame state from encoded input.

        Args:
            x: Encoded game state tensor. Shape depends on architecture:
               - MLP: (B, K * frame_dim) for K context frames
               - Mamba: (B, T, frame_dim) for T-length sequences

        Returns:
            Dict with prediction heads:
                continuous_delta: (B, 8) — Δ[percent, x, y, shield] per player
                binary_logits: (B, 6) — logits for [facing, invuln, ground] per player
                p0_action_logits: (B, num_actions) — next action distribution for P0
                p1_action_logits: (B, num_actions) — next action distribution for P1
                p0_jumps_logits: (B, num_jumps) — next jumps distribution for P0
                p1_jumps_logits: (B, num_jumps) — next jumps distribution for P1
        """
        ...
