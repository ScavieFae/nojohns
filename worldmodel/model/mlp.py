"""Phase 1: Frame-stacking MLP world model.

Concatenates K recent frames (via learned embeddings for categoricals)
and predicts next-frame state changes through separate heads.

v2.2: input-conditioned — receives next-frame controller input alongside context.
  float_ctx:  (B, K, 58) — context window [t-K, ..., t-1]
  int_ctx:    (B, K, 15) — context categoricals
  next_ctrl:  (B, 26)    — frame t's controller input [p0_ctrl(13), p1_ctrl(13)]
  frame_dim:  58 + 2×(32+4+8+2+2+4+8) + 4 = 182
  input_dim:  182 × K + 26 = 1846 (for K=10)
"""

import torch
import torch.nn as nn

from worldmodel.model.encoding import EncodingConfig

# Controller input dimensions: sticks(4) + shoulders(2) + buttons(7) = 13 per player × 2
CTRL_DIM = 26


class FrameStackMLP(nn.Module):
    """Input-conditioned MLP world model with frame stacking.

    Takes a context window of K frames plus the current frame's controller input,
    and predicts the current frame's state. This separates physics prediction
    (what happens given this input) from decision prediction (what will be pressed).

    Input format (v2.2):
        float_ctx:  (B, K, 58) — context frames
        int_ctx:    (B, K, 15) — context categoricals
        next_ctrl:  (B, 26)    — current frame's controller input
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

        # Learned embeddings for categoricals
        self.action_embed = nn.Embedding(cfg.action_vocab, cfg.action_embed_dim)
        self.jumps_embed = nn.Embedding(cfg.jumps_vocab, cfg.jumps_embed_dim)
        self.character_embed = nn.Embedding(cfg.character_vocab, cfg.character_embed_dim)
        self.stage_embed = nn.Embedding(cfg.stage_vocab, cfg.stage_embed_dim)
        # Combat context embeddings (v2.1)
        self.l_cancel_embed = nn.Embedding(cfg.l_cancel_vocab, cfg.l_cancel_embed_dim)
        self.hurtbox_embed = nn.Embedding(cfg.hurtbox_vocab, cfg.hurtbox_embed_dim)
        self.ground_embed = nn.Embedding(cfg.ground_vocab, cfg.ground_embed_dim)
        self.last_attack_embed = nn.Embedding(cfg.last_attack_vocab, cfg.last_attack_embed_dim)

        # Per-frame dim: floats + embeddings
        # = 58 + 2×(32+4+8+2+2+4+8) + 4 = 58 + 120 + 4 = 182
        float_per_frame = cfg.float_per_player * 2
        embed_per_frame = (
            2 * (cfg.action_embed_dim + cfg.jumps_embed_dim + cfg.character_embed_dim
                 + cfg.l_cancel_embed_dim + cfg.hurtbox_embed_dim
                 + cfg.ground_embed_dim + cfg.last_attack_embed_dim)
            + cfg.stage_embed_dim
        )
        self.frame_dim = float_per_frame + embed_per_frame
        # v2.2: context window + controller input for the frame we're predicting
        input_dim = self.frame_dim * context_len + CTRL_DIM

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, trunk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Prediction heads (unchanged — same targets as v1/v2)
        self.continuous_head = nn.Linear(trunk_dim, 8)
        self.binary_head = nn.Linear(trunk_dim, 6)
        self.p0_action_head = nn.Linear(trunk_dim, cfg.action_vocab)
        self.p1_action_head = nn.Linear(trunk_dim, cfg.action_vocab)
        self.p0_jumps_head = nn.Linear(trunk_dim, cfg.jumps_vocab)
        self.p1_jumps_head = nn.Linear(trunk_dim, cfg.jumps_vocab)

    def forward(
        self,
        float_ctx: torch.Tensor,
        int_ctx: torch.Tensor,
        next_ctrl: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            float_ctx:  (B, K, 58) — context frames
            int_ctx:    (B, K, 15) — context categoricals
            next_ctrl:  (B, 26)    — controller input for the frame being predicted

        Returns:
            Dict with prediction head outputs.
        """
        B, K, _ = float_ctx.shape

        # Embed categoricals: int_ctx columns are:
        # [p0_action(0), p0_jumps(1), p0_char(2), p0_l_cancel(3), p0_hurtbox(4),
        #  p0_ground(5), p0_last_attack(6),
        #  p1_action(7), p1_jumps(8), p1_char(9), p1_l_cancel(10), p1_hurtbox(11),
        #  p1_ground(12), p1_last_attack(13), stage(14)]
        p0_action_emb = self.action_embed(int_ctx[:, :, 0])          # (B, K, 32)
        p0_jumps_emb = self.jumps_embed(int_ctx[:, :, 1])            # (B, K, 4)
        p0_char_emb = self.character_embed(int_ctx[:, :, 2])         # (B, K, 8)
        p0_lc_emb = self.l_cancel_embed(int_ctx[:, :, 3])            # (B, K, 2)
        p0_hb_emb = self.hurtbox_embed(int_ctx[:, :, 4])             # (B, K, 2)
        p0_gnd_emb = self.ground_embed(int_ctx[:, :, 5])             # (B, K, 4)
        p0_la_emb = self.last_attack_embed(int_ctx[:, :, 6])         # (B, K, 8)
        p1_action_emb = self.action_embed(int_ctx[:, :, 7])          # (B, K, 32)
        p1_jumps_emb = self.jumps_embed(int_ctx[:, :, 8])            # (B, K, 4)
        p1_char_emb = self.character_embed(int_ctx[:, :, 9])         # (B, K, 8)
        p1_lc_emb = self.l_cancel_embed(int_ctx[:, :, 10])           # (B, K, 2)
        p1_hb_emb = self.hurtbox_embed(int_ctx[:, :, 11])            # (B, K, 2)
        p1_gnd_emb = self.ground_embed(int_ctx[:, :, 12])            # (B, K, 4)
        p1_la_emb = self.last_attack_embed(int_ctx[:, :, 13])        # (B, K, 8)
        stage_emb = self.stage_embed(int_ctx[:, :, 14])              # (B, K, 4)

        # Concat floats + embeddings per frame: (B, K, frame_dim)
        frame_enc = torch.cat([
            float_ctx,
            p0_action_emb, p0_jumps_emb, p0_char_emb,
            p0_lc_emb, p0_hb_emb, p0_gnd_emb, p0_la_emb,
            p1_action_emb, p1_jumps_emb, p1_char_emb,
            p1_lc_emb, p1_hb_emb, p1_gnd_emb, p1_la_emb,
            stage_emb,
        ], dim=-1)

        # Flatten context window + append controller input: (B, K*frame_dim + 26)
        x = torch.cat([frame_enc.reshape(B, -1), next_ctrl], dim=-1)

        # Trunk
        h = self.trunk(x)

        return {
            "continuous_delta": self.continuous_head(h),
            "binary_logits": self.binary_head(h),
            "p0_action_logits": self.p0_action_head(h),
            "p1_action_logits": self.p1_action_head(h),
            "p0_jumps_logits": self.p0_jumps_head(h),
            "p1_jumps_logits": self.p1_jumps_head(h),
        }
