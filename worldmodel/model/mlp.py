"""Phase 1: Frame-stacking MLP world model.

Concatenates K recent frames (via learned embeddings for categoricals)
and predicts next-frame state changes through separate heads.

v2.2+: input-conditioned — receives next-frame controller input alongside context.
  float_ctx:  (B, K, F) — context window [t-K, ..., t-1]
  int_ctx:    (B, K, I) — context categoricals
  next_ctrl:  (B, C)    — frame t's controller input (C = ctrl_conditioning_dim)

Dimensions are config-driven — experiment flags (state_age_as_embed, press_events)
change F, I, and C without any code modifications.
"""

import torch
import torch.nn as nn

from worldmodel.model.encoding import EncodingConfig


class FrameStackMLP(nn.Module):
    """Input-conditioned MLP world model with frame stacking.

    Takes a context window of K frames plus the current frame's controller input,
    and predicts the current frame's state. This separates physics prediction
    (what happens given this input) from decision prediction (what will be pressed).
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
        # Exp 1a: state_age as learned embedding
        if cfg.state_age_as_embed:
            self.state_age_embed = nn.Embedding(cfg.state_age_embed_vocab, cfg.state_age_embed_dim)

        # Per-frame dim: floats + embeddings (dynamic based on config flags)
        float_per_frame = cfg.float_per_player * 2
        base_embed_per_player = (
            cfg.action_embed_dim + cfg.jumps_embed_dim + cfg.character_embed_dim
            + cfg.l_cancel_embed_dim + cfg.hurtbox_embed_dim
            + cfg.ground_embed_dim + cfg.last_attack_embed_dim
        )
        if cfg.state_age_as_embed:
            base_embed_per_player += cfg.state_age_embed_dim
        embed_per_frame = 2 * base_embed_per_player + cfg.stage_embed_dim
        self.frame_dim = float_per_frame + embed_per_frame
        # Context window + controller conditioning input
        input_dim = self.frame_dim * context_len + cfg.ctrl_conditioning_dim

        # Int column indices (depend on int_per_player)
        ipp = cfg.int_per_player
        self._int_cols = {
            "p0_action": 0, "p0_jumps": 1, "p0_char": 2,
            "p0_l_cancel": 3, "p0_hurtbox": 4, "p0_ground": 5, "p0_last_attack": 6,
            "p1_action": ipp, "p1_jumps": ipp + 1, "p1_char": ipp + 2,
            "p1_l_cancel": ipp + 3, "p1_hurtbox": ipp + 4,
            "p1_ground": ipp + 5, "p1_last_attack": ipp + 6,
            "stage": ipp * 2,
        }
        if cfg.state_age_as_embed:
            self._int_cols["p0_state_age"] = 7  # after p0's 7 base categoricals
            self._int_cols["p1_state_age"] = ipp + 7

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, trunk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Prediction heads
        self.continuous_head = nn.Linear(trunk_dim, 8)
        self.binary_head = nn.Linear(trunk_dim, 6)
        self.velocity_head = nn.Linear(trunk_dim, cfg.predicted_velocity_dim)   # 10
        self.dynamics_head = nn.Linear(trunk_dim, cfg.predicted_dynamics_dim)   # 6
        self.p0_action_head = nn.Linear(trunk_dim, cfg.action_vocab)
        self.p1_action_head = nn.Linear(trunk_dim, cfg.action_vocab)
        self.p0_jumps_head = nn.Linear(trunk_dim, cfg.jumps_vocab)
        self.p1_jumps_head = nn.Linear(trunk_dim, cfg.jumps_vocab)
        # Combat context heads (predict per player)
        self.p0_l_cancel_head = nn.Linear(trunk_dim, cfg.l_cancel_vocab)
        self.p1_l_cancel_head = nn.Linear(trunk_dim, cfg.l_cancel_vocab)
        self.p0_hurtbox_head = nn.Linear(trunk_dim, cfg.hurtbox_vocab)
        self.p1_hurtbox_head = nn.Linear(trunk_dim, cfg.hurtbox_vocab)
        self.p0_ground_head = nn.Linear(trunk_dim, cfg.ground_vocab)
        self.p1_ground_head = nn.Linear(trunk_dim, cfg.ground_vocab)
        self.p0_last_attack_head = nn.Linear(trunk_dim, cfg.last_attack_vocab)
        self.p1_last_attack_head = nn.Linear(trunk_dim, cfg.last_attack_vocab)

    def forward(
        self,
        float_ctx: torch.Tensor,
        int_ctx: torch.Tensor,
        next_ctrl: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            float_ctx:  (B, K, F) — context frames
            int_ctx:    (B, K, I) — context categoricals
            next_ctrl:  (B, C)    — controller conditioning input

        Returns:
            Dict with prediction head outputs.
        """
        B, K, _ = float_ctx.shape
        c = self._int_cols

        # Embed categoricals using config-driven column indices
        p0_action_emb = self.action_embed(int_ctx[:, :, c["p0_action"]])
        p0_jumps_emb = self.jumps_embed(int_ctx[:, :, c["p0_jumps"]])
        p0_char_emb = self.character_embed(int_ctx[:, :, c["p0_char"]])
        p0_lc_emb = self.l_cancel_embed(int_ctx[:, :, c["p0_l_cancel"]])
        p0_hb_emb = self.hurtbox_embed(int_ctx[:, :, c["p0_hurtbox"]])
        p0_gnd_emb = self.ground_embed(int_ctx[:, :, c["p0_ground"]])
        p0_la_emb = self.last_attack_embed(int_ctx[:, :, c["p0_last_attack"]])

        p1_action_emb = self.action_embed(int_ctx[:, :, c["p1_action"]])
        p1_jumps_emb = self.jumps_embed(int_ctx[:, :, c["p1_jumps"]])
        p1_char_emb = self.character_embed(int_ctx[:, :, c["p1_char"]])
        p1_lc_emb = self.l_cancel_embed(int_ctx[:, :, c["p1_l_cancel"]])
        p1_hb_emb = self.hurtbox_embed(int_ctx[:, :, c["p1_hurtbox"]])
        p1_gnd_emb = self.ground_embed(int_ctx[:, :, c["p1_ground"]])
        p1_la_emb = self.last_attack_embed(int_ctx[:, :, c["p1_last_attack"]])
        stage_emb = self.stage_embed(int_ctx[:, :, c["stage"]])

        # Build per-frame encoding: floats + embeddings
        parts = [
            float_ctx,
            p0_action_emb, p0_jumps_emb, p0_char_emb,
            p0_lc_emb, p0_hb_emb, p0_gnd_emb, p0_la_emb,
        ]
        if self.cfg.state_age_as_embed:
            parts.append(self.state_age_embed(int_ctx[:, :, c["p0_state_age"]]))
        parts.extend([
            p1_action_emb, p1_jumps_emb, p1_char_emb,
            p1_lc_emb, p1_hb_emb, p1_gnd_emb, p1_la_emb,
        ])
        if self.cfg.state_age_as_embed:
            parts.append(self.state_age_embed(int_ctx[:, :, c["p1_state_age"]]))
        parts.append(stage_emb)

        frame_enc = torch.cat(parts, dim=-1)  # (B, K, frame_dim)

        # Flatten context window + append controller conditioning
        x = torch.cat([frame_enc.reshape(B, -1), next_ctrl], dim=-1)

        # Trunk
        h = self.trunk(x)

        return {
            "continuous_delta": self.continuous_head(h),
            "binary_logits": self.binary_head(h),
            "velocity_delta": self.velocity_head(h),
            "dynamics_pred": self.dynamics_head(h),
            "p0_action_logits": self.p0_action_head(h),
            "p1_action_logits": self.p1_action_head(h),
            "p0_jumps_logits": self.p0_jumps_head(h),
            "p1_jumps_logits": self.p1_jumps_head(h),
            "p0_l_cancel_logits": self.p0_l_cancel_head(h),
            "p1_l_cancel_logits": self.p1_l_cancel_head(h),
            "p0_hurtbox_logits": self.p0_hurtbox_head(h),
            "p1_hurtbox_logits": self.p1_hurtbox_head(h),
            "p0_ground_logits": self.p0_ground_head(h),
            "p1_ground_logits": self.p1_ground_head(h),
            "p0_last_attack_logits": self.p0_last_attack_head(h),
            "p1_last_attack_logits": self.p1_last_attack_head(h),
        }
