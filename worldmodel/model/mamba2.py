"""Mamba-2 world model: sequential state space model for frame prediction.

Replaces the MLP trunk with Mamba-2 blocks that process the context window
as a temporal sequence. Same embedding layers, same prediction heads, same
forward signature — the dataset and trainer don't know the difference.

Uses sequential scan (not SSD chunked algorithm) since K=10 is too short
for chunk-based matmul to pay off. If K grows past ~64, swap the scan loop
for SSD — the Mamba2Block interface doesn't change.

Pure PyTorch, MPS-compatible. No CUDA, no Triton.
Reference: tommyip/mamba2-minimal (MIT license).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from worldmodel.model.encoding import EncodingConfig


# ---------- Primitives ----------


def _silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU defined manually — torch's version had MPS issues historically."""
    return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with optional SiLU gating."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if z is not None:
            x = x * _silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ---------- Mamba-2 Block ----------


class Mamba2Block(nn.Module):
    """Single Mamba-2 layer using sequential scan.

    Architecture per timestep:
        1. in_proj → z (gate), xBC (conv input), dt (timestep per head)
        2. conv1d along sequence → SiLU activation
        3. Split xBC → x, B (input gate), C (output gate)
        4. Sequential SSM: h = decay·h + dt·outer(x, B), y = inner(h, C)
        5. Skip connection with D, gated RMSNorm with z, out_proj

    Args:
        d_model: Input/output dimension.
        d_state: SSM state dimension (N). 64 is Mamba-2's sweet spot.
        d_conv: Depthwise convolution kernel size. 4 captures ~67ms at 60fps.
        expand: Expansion factor for inner dimension (d_inner = expand * d_model).
        headdim: Dimension per SSM head (P). Multiple channels share SSM dynamics.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.headdim = headdim
        assert self.d_inner % headdim == 0, (
            f"d_inner ({self.d_inner}) must be divisible by headdim ({headdim})"
        )
        self.nheads = self.d_inner // headdim

        # Input projection: z (gate) + xBC (goes through conv) + dt (per-head timestep)
        d_in_proj = 2 * self.d_inner + 2 * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # Depthwise conv over sequence dimension (local pattern detection)
        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim,
            kernel_size=d_conv, groups=conv_dim,
            padding=d_conv - 1,
        )

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        self.A_log = nn.Parameter(torch.empty(self.nheads))
        self.D = nn.Parameter(torch.empty(self.nheads))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_params()

    def _init_params(self):
        nn.init.uniform_(self.dt_bias, -1.0, 1.0)
        # Spread decay rates across heads: head 0 decays slowest, last decays fastest
        with torch.no_grad():
            self.A_log.copy_(torch.log(torch.linspace(1, self.nheads, self.nheads)))
        nn.init.ones_(self.D)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Process a sequence via sequential SSM scan.

        Args:
            u: (batch, seqlen, d_model)

        Returns:
            (batch, seqlen, d_model)
        """
        batch, seqlen, _ = u.shape

        A = -torch.exp(self.A_log)  # (nheads,) — always negative for stability

        # Project input
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # 1D conv along sequence dimension, then SiLU
        xBC = _silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen, :]
        )

        # Split into SSM components
        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1,
        )
        x = x.view(batch, seqlen, self.nheads, self.headdim)

        # Sequential SSM scan (10 steps for K=10 — tight and simple)
        h = torch.zeros(
            batch, self.nheads, self.headdim, self.d_state,
            device=u.device, dtype=u.dtype,
        )
        ys = []
        for t in range(seqlen):
            # Discretize: decay = exp(A * dt)
            dA = torch.exp(A * dt[:, t, :])  # (batch, nheads)

            # State update: h = decay·h + dt·outer(x, B)
            h = h * dA.unsqueeze(-1).unsqueeze(-1) + torch.einsum(
                "bh, bn, bhp -> bhpn",
                dt[:, t, :],        # (batch, nheads)
                B[:, t, :],         # (batch, d_state)
                x[:, t, :, :],      # (batch, nheads, headdim)
            )

            # Read out: y = inner(h, C)
            y_t = torch.einsum("bhpn, bn -> bhp", h, C[:, t, :])
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (batch, seqlen, nheads, headdim)

        # Skip connection with D
        y = y + x * self.D.unsqueeze(-1)

        # Flatten heads
        y = y.view(batch, seqlen, self.d_inner)

        # Gated output: RMSNorm(y, gate=z) → out_proj
        y = self.norm(y, z)
        return self.out_proj(y)


# ---------- Full World Model ----------


class FrameStackMamba2(nn.Module):
    """Mamba-2 world model with input-conditioned prediction.

    Same interface as FrameStackMLP:
        forward(float_ctx, int_ctx, next_ctrl) → dict of prediction tensors

    Architecture:
        frame_enc (B,K,frame_dim) → project → Mamba2 layers → last timestep
        → add ctrl_proj(next_ctrl) → prediction heads

    The trunk changes from flatten+MLP to project+Mamba2+last_timestep.
    Everything else (embeddings, heads, data pipeline) is identical.
    """

    def __init__(
        self,
        cfg: EncodingConfig,
        context_len: int = 10,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 2,
        headdim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cfg = cfg
        self.context_len = context_len

        # --- Embeddings (identical to FrameStackMLP) ---
        self.action_embed = nn.Embedding(cfg.action_vocab, cfg.action_embed_dim)
        self.jumps_embed = nn.Embedding(cfg.jumps_vocab, cfg.jumps_embed_dim)
        self.character_embed = nn.Embedding(cfg.character_vocab, cfg.character_embed_dim)
        self.stage_embed = nn.Embedding(cfg.stage_vocab, cfg.stage_embed_dim)
        self.l_cancel_embed = nn.Embedding(cfg.l_cancel_vocab, cfg.l_cancel_embed_dim)
        self.hurtbox_embed = nn.Embedding(cfg.hurtbox_vocab, cfg.hurtbox_embed_dim)
        self.ground_embed = nn.Embedding(cfg.ground_vocab, cfg.ground_embed_dim)
        self.last_attack_embed = nn.Embedding(cfg.last_attack_vocab, cfg.last_attack_embed_dim)
        if cfg.state_age_as_embed:
            self.state_age_embed = nn.Embedding(
                cfg.state_age_embed_vocab, cfg.state_age_embed_dim,
            )

        # Per-frame dimension (same calculation as FrameStackMLP)
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

        # Int column indices (same as FrameStackMLP)
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
            self._int_cols["p0_state_age"] = 7
            self._int_cols["p1_state_age"] = ipp + 7

        # --- Mamba-2 backbone (replaces MLP trunk) ---
        self.frame_proj = nn.Linear(self.frame_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": Mamba2Block(d_model, d_state, headdim=headdim),
                "norm": RMSNorm(d_model),
            })
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # Controller conditioning (additive, not concatenative)
        self.ctrl_proj = nn.Linear(cfg.ctrl_conditioning_dim, d_model)

        # --- Prediction heads (identical to FrameStackMLP) ---
        self.continuous_head = nn.Linear(d_model, 8)
        self.binary_head = nn.Linear(d_model, 6)
        self.p0_action_head = nn.Linear(d_model, cfg.action_vocab)
        self.p1_action_head = nn.Linear(d_model, cfg.action_vocab)
        self.p0_jumps_head = nn.Linear(d_model, cfg.jumps_vocab)
        self.p1_jumps_head = nn.Linear(d_model, cfg.jumps_vocab)

    def _encode_frames(
        self, float_ctx: torch.Tensor, int_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Encode context frames into per-frame vectors.

        Identical logic to FrameStackMLP.forward() embedding section.

        Args:
            float_ctx: (B, K, F) continuous features
            int_ctx:   (B, K, I) categorical indices

        Returns:
            (B, K, frame_dim) encoded frames
        """
        c = self._int_cols

        # Player 0 embeddings
        p0_action_emb = self.action_embed(int_ctx[:, :, c["p0_action"]])
        p0_jumps_emb = self.jumps_embed(int_ctx[:, :, c["p0_jumps"]])
        p0_char_emb = self.character_embed(int_ctx[:, :, c["p0_char"]])
        p0_lc_emb = self.l_cancel_embed(int_ctx[:, :, c["p0_l_cancel"]])
        p0_hb_emb = self.hurtbox_embed(int_ctx[:, :, c["p0_hurtbox"]])
        p0_gnd_emb = self.ground_embed(int_ctx[:, :, c["p0_ground"]])
        p0_la_emb = self.last_attack_embed(int_ctx[:, :, c["p0_last_attack"]])

        # Player 1 embeddings
        p1_action_emb = self.action_embed(int_ctx[:, :, c["p1_action"]])
        p1_jumps_emb = self.jumps_embed(int_ctx[:, :, c["p1_jumps"]])
        p1_char_emb = self.character_embed(int_ctx[:, :, c["p1_char"]])
        p1_lc_emb = self.l_cancel_embed(int_ctx[:, :, c["p1_l_cancel"]])
        p1_hb_emb = self.hurtbox_embed(int_ctx[:, :, c["p1_hurtbox"]])
        p1_gnd_emb = self.ground_embed(int_ctx[:, :, c["p1_ground"]])
        p1_la_emb = self.last_attack_embed(int_ctx[:, :, c["p1_last_attack"]])

        stage_emb = self.stage_embed(int_ctx[:, :, c["stage"]])

        # Assemble per-frame vector
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

        return torch.cat(parts, dim=-1)  # (B, K, frame_dim)

    def forward(
        self,
        float_ctx: torch.Tensor,
        int_ctx: torch.Tensor,
        next_ctrl: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass — same signature as FrameStackMLP.

        Args:
            float_ctx:  (B, K, F) context frames
            int_ctx:    (B, K, I) context categoricals
            next_ctrl:  (B, C)   controller conditioning input

        Returns:
            Dict with prediction head outputs.
        """
        B, K, _ = float_ctx.shape

        # Encode frames (identical to MLP)
        frame_enc = self._encode_frames(float_ctx, int_ctx)  # (B, K, frame_dim)

        # Project to model dimension (replaces MLP's flatten)
        x = self.input_dropout(self.frame_proj(frame_enc))  # (B, K, d_model)

        # Mamba-2 layers with pre-norm residual connections
        for layer in self.layers:
            x = x + layer["mamba"](layer["norm"](x))

        # Take last timestep — SSM state carries all context from the sequence
        h = self.final_norm(x[:, -1, :])  # (B, d_model)

        # Condition on controller input (additive)
        h = h + self.ctrl_proj(next_ctrl)  # (B, d_model)

        # Prediction heads
        return {
            "continuous_delta": self.continuous_head(h),
            "binary_logits": self.binary_head(h),
            "p0_action_logits": self.p0_action_head(h),
            "p1_action_logits": self.p1_action_head(h),
            "p0_jumps_logits": self.p0_jumps_head(h),
            "p1_jumps_logits": self.p1_jumps_head(h),
        }
