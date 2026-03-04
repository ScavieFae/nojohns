"""Imitation learning policy: game state → controller output.

Same embedding structure and trunk as the world model (FrameStackMLP),
but with controller prediction heads instead of state prediction heads.

Input: K frames of full game state (state + controller history)
Output: predicted controller for the target player on the next frame
  - analog: (5,) — main_x, main_y, c_x, c_y, shoulder [0, 1]
  - button_logits: (8,) — A, B, X, Y, Z, L, R, D_UP (raw logits)
"""

import torch
import torch.nn as nn

from worldmodel.data.policy_dataset import ANALOG_DIM, BUTTON_DIM
from worldmodel.model.encoding import EncodingConfig


class PolicyMLP(nn.Module):
    """Frame-stacking MLP policy for imitation learning.

    Architecture mirrors the world model's trunk (same embeddings, same hidden dims)
    but replaces the multi-head state predictor with a controller predictor.
    """

    def __init__(
        self,
        cfg: EncodingConfig,
        context_len: int = 10,
        hidden_dim: int = 512,
        trunk_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cfg = cfg
        self.context_len = context_len

        # Same learned embeddings as world model
        self.action_embed = nn.Embedding(cfg.action_vocab, cfg.action_embed_dim)
        self.jumps_embed = nn.Embedding(cfg.jumps_vocab, cfg.jumps_embed_dim)
        self.character_embed = nn.Embedding(cfg.character_vocab, cfg.character_embed_dim)
        self.stage_embed = nn.Embedding(cfg.stage_vocab, cfg.stage_embed_dim)
        self.l_cancel_embed = nn.Embedding(cfg.l_cancel_vocab, cfg.l_cancel_embed_dim)
        self.hurtbox_embed = nn.Embedding(cfg.hurtbox_vocab, cfg.hurtbox_embed_dim)
        self.ground_embed = nn.Embedding(cfg.ground_vocab, cfg.ground_embed_dim)
        self.last_attack_embed = nn.Embedding(cfg.last_attack_vocab, cfg.last_attack_embed_dim)

        # Same per-frame dim as world model: 58 + 120 + 4 = 182
        float_per_frame = cfg.float_per_player * 2
        embed_per_frame = (
            2 * (cfg.action_embed_dim + cfg.jumps_embed_dim + cfg.character_embed_dim
                 + cfg.l_cancel_embed_dim + cfg.hurtbox_embed_dim
                 + cfg.ground_embed_dim + cfg.last_attack_embed_dim)
            + cfg.stage_embed_dim
        )
        self.frame_dim = float_per_frame + embed_per_frame
        input_dim = self.frame_dim * context_len

        # Shared trunk (same structure as world model)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, trunk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Controller prediction heads
        # Analog: main_stick x/y, c_stick x/y, shoulder — continuous [0, 1]
        self.analog_head = nn.Linear(trunk_dim, ANALOG_DIM)
        # Buttons: A, B, X, Y, Z, L, R, D_UP — binary logits
        self.button_head = nn.Linear(trunk_dim, BUTTON_DIM)

    def forward(
        self,
        float_ctx: torch.Tensor,
        int_ctx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            float_ctx: (B, K, 58) — context frames (full state + controller history)
            int_ctx:   (B, K, 15) — context categoricals

        Returns:
            Dict with:
                analog_pred: (B, 5) — predicted stick/trigger values (sigmoid, [0,1])
                button_logits: (B, 8) — predicted button logits (raw, for BCEWithLogitsLoss)
        """
        B, K, _ = float_ctx.shape

        # Embed categoricals (identical to world model)
        p0_action_emb = self.action_embed(int_ctx[:, :, 0])
        p0_jumps_emb = self.jumps_embed(int_ctx[:, :, 1])
        p0_char_emb = self.character_embed(int_ctx[:, :, 2])
        p0_lc_emb = self.l_cancel_embed(int_ctx[:, :, 3])
        p0_hb_emb = self.hurtbox_embed(int_ctx[:, :, 4])
        p0_gnd_emb = self.ground_embed(int_ctx[:, :, 5])
        p0_la_emb = self.last_attack_embed(int_ctx[:, :, 6])
        p1_action_emb = self.action_embed(int_ctx[:, :, 7])
        p1_jumps_emb = self.jumps_embed(int_ctx[:, :, 8])
        p1_char_emb = self.character_embed(int_ctx[:, :, 9])
        p1_lc_emb = self.l_cancel_embed(int_ctx[:, :, 10])
        p1_hb_emb = self.hurtbox_embed(int_ctx[:, :, 11])
        p1_gnd_emb = self.ground_embed(int_ctx[:, :, 12])
        p1_la_emb = self.last_attack_embed(int_ctx[:, :, 13])
        stage_emb = self.stage_embed(int_ctx[:, :, 14])

        frame_enc = torch.cat([
            float_ctx,
            p0_action_emb, p0_jumps_emb, p0_char_emb,
            p0_lc_emb, p0_hb_emb, p0_gnd_emb, p0_la_emb,
            p1_action_emb, p1_jumps_emb, p1_char_emb,
            p1_lc_emb, p1_hb_emb, p1_gnd_emb, p1_la_emb,
            stage_emb,
        ], dim=-1)

        x = frame_enc.reshape(B, -1)
        h = self.trunk(x)

        return {
            "analog_pred": torch.sigmoid(self.analog_head(h)),
            "button_logits": self.button_head(h),
        }
