"""Mamba-2 world model: sequential state space model for frame prediction.

Replaces the MLP trunk with Mamba-2 blocks that process the context window
as a temporal sequence. Same embedding layers, same prediction heads, same
forward signature — the dataset and trainer don't know the difference.

Two scan modes:
  - Sequential scan (chunk_size=None): simple for-loop, correct baseline.
  - SSD chunked algorithm (chunk_size=int): parallel matmul-based, faster
    on GPU/MPS for longer sequences. Requires seqlen % chunk_size == 0.

Pure PyTorch, MPS-compatible. No CUDA, no Triton, no einops.
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


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum in log-space for SSD decay computation.

    Given log-decay values x along the last dimension, computes the
    lower-triangular matrix of cumulative sums. Used to build the
    causal decay matrix within and between chunks.

    Args:
        x: (..., T) log-decay values (negative)

    Returns:
        (..., T, T) lower-triangular cumulative sums, upper triangle = -inf
    """
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)  # (..., T, T)
    # Zero out above the diagonal (we only accumulate past → present)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    # Cumulative sum along rows
    x_segsum = torch.cumsum(x, dim=-2)
    # Mask upper triangle to -inf (no future information)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


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
    """Single Mamba-2 layer with selectable scan mode.

    Architecture per timestep:
        1. in_proj → z (gate), xBC (conv input), dt (timestep per head)
        2. conv1d along sequence → SiLU activation
        3. Split xBC → x, B (input gate), C (output gate)
        4. SSM scan (sequential or SSD chunked)
        5. Skip connection with D, gated RMSNorm with z, out_proj

    Args:
        d_model: Input/output dimension.
        d_state: SSM state dimension (N). 64 is Mamba-2's sweet spot.
        d_conv: Depthwise convolution kernel size. 4 captures ~67ms at 60fps.
        expand: Expansion factor for inner dimension (d_inner = expand * d_model).
        headdim: Dimension per SSM head (P). Multiple channels share SSM dynamics.
        chunk_size: If set, use SSD chunked algorithm. None = sequential scan.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.chunk_size = chunk_size
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

    def _sequential_scan(
        self,
        x: torch.Tensor,   # (batch, seqlen, nheads, headdim)
        A: torch.Tensor,    # (nheads,) raw negative decay
        B: torch.Tensor,    # (batch, seqlen, d_state)
        C: torch.Tensor,    # (batch, seqlen, d_state)
        dt: torch.Tensor,   # (batch, seqlen, nheads)
    ) -> torch.Tensor:
        """Sequential SSM scan — simple for-loop, always correct."""
        batch, seqlen, nheads, headdim = x.shape

        h = torch.zeros(
            batch, nheads, headdim, self.d_state,
            device=x.device, dtype=x.dtype,
        )
        ys = []
        for t in range(seqlen):
            dA = torch.exp(A * dt[:, t, :])  # (batch, nheads)
            h = h * dA.unsqueeze(-1).unsqueeze(-1) + torch.einsum(
                "bh, bn, bhp -> bhpn",
                dt[:, t, :], B[:, t, :], x[:, t, :, :],
            )
            y_t = torch.einsum("bhpn, bn -> bhp", h, C[:, t, :])
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (batch, seqlen, nheads, headdim)

    def _ssd_scan(
        self,
        x: torch.Tensor,   # (batch, seqlen, nheads, headdim)
        A: torch.Tensor,    # (nheads,) raw negative decay
        B: torch.Tensor,    # (batch, seqlen, d_state)
        C: torch.Tensor,    # (batch, seqlen, d_state)
        dt: torch.Tensor,   # (batch, seqlen, nheads)
    ) -> torch.Tensor:
        """SSD chunked algorithm — parallel matmuls, faster for longer sequences.

        Splits the sequence into chunks, computes quadratic attention within
        each chunk, and propagates SSM states between chunks.
        """
        batch, seqlen, nheads, headdim = x.shape
        d_state = B.shape[-1]
        cs = self.chunk_size
        nchunks = seqlen // cs
        assert seqlen % cs == 0, f"seqlen ({seqlen}) must be divisible by chunk_size ({cs})"

        # Discretized log-decay: (batch, seqlen, nheads) — negative values
        A_disc = A * dt  # (nheads,) broadcast with (batch, seqlen, nheads)

        # Reshape into chunks: (batch, nchunks, chunk_size, ...)
        x = x.reshape(batch, nchunks, cs, nheads, headdim)
        A_disc = A_disc.reshape(batch, nchunks, cs, nheads)
        B = B.reshape(batch, nchunks, cs, d_state)
        C = C.reshape(batch, nchunks, cs, d_state)

        # Rearrange A for per-head cumsum: (batch, nchunks, nheads, chunk_size)
        A_ch = A_disc.permute(0, 1, 3, 2)
        A_cumsum = torch.cumsum(A_ch, dim=-1)

        # --- Step 1: Intra-chunk (diagonal blocks) ---
        # L = causal decay matrix within each chunk
        L = torch.exp(_segsum(A_ch))  # (batch, nchunks, nheads, cs, cs)
        # Y_diag: within-chunk output via quadratic attention
        Y_diag = torch.einsum("bcln, bcsn, bchls, bcshp -> bclhp", C, B, L, x)

        # --- Step 2: Compute states at chunk boundaries ---
        # decay_states[l] = how much position l's contribution decays to end of chunk
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        # (batch, nchunks, nheads, chunk_size)

        # Accumulate state within each chunk: sum of (decayed x ⊗ B)
        states = torch.einsum("bchl, bcln, bclhp -> bchpn", decay_states, B, x)
        # (batch, nchunks, nheads, headdim, d_state)

        # --- Step 3: Inter-chunk SSM recurrence ---
        # Propagate states forward across chunk boundaries
        initial_states = torch.zeros(
            batch, 1, nheads, headdim, d_state,
            device=x.device, dtype=x.dtype,
        )
        states = torch.cat([initial_states, states], dim=1)
        # (batch, nchunks+1, nheads, headdim, d_state)

        # Total decay per chunk: (batch, nheads, nchunks)
        total_decay = A_cumsum[:, :, :, -1].permute(0, 2, 1)
        # Pad with zero at start: (batch, nheads, nchunks+1)
        total_decay = F.pad(total_decay, (1, 0))

        # Inter-chunk decay matrix
        decay_chunk = torch.exp(_segsum(total_decay))
        # (batch, nheads, nchunks+1, nchunks+1)

        # Propagate: each chunk's state = sum of all prior states, decayed
        # states: (batch, nchunks+1, nheads, headdim, d_state) — reindex as (b,c,h,p,n)
        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        # Slice: each chunk c needs the accumulated state ENTERING it (from prior chunks)
        # new_states[0] = initial (for chunk 0), new_states[c] = state entering chunk c
        states = new_states[:, :-1]  # (batch, nchunks, nheads, headdim, d_state)

        # --- Step 4: State → output ---
        # Decay the inter-chunk state to each position within the chunk
        state_decay_out = torch.exp(A_cumsum)  # (batch, nchunks, nheads, chunk_size)

        Y_off = torch.einsum("bcln, bchpn, bchl -> bclhp", C, states, state_decay_out)

        # Combine intra-chunk and inter-chunk
        Y = Y_diag + Y_off  # (batch, nchunks, chunk_size, nheads, headdim)
        return Y.reshape(batch, seqlen, nheads, headdim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Process a sequence via SSM scan.

        Automatically selects sequential or SSD based on chunk_size config.

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

        # Select scan mode
        if self.chunk_size is not None and seqlen % self.chunk_size == 0:
            # SSD expects x pre-scaled by dt (absorbs the dt*B*x term)
            x_scaled = x * dt.unsqueeze(-1)  # (batch, seqlen, nheads, headdim)
            y = self._ssd_scan(x_scaled, A, B, C, dt)
        else:
            y = self._sequential_scan(x, A, B, C, dt)

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
        chunk_size: int | None = None,
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
                "mamba": Mamba2Block(d_model, d_state, headdim=headdim, chunk_size=chunk_size),
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
        self.velocity_head = nn.Linear(d_model, cfg.predicted_velocity_dim)   # 10
        self.dynamics_head = nn.Linear(d_model, cfg.predicted_dynamics_dim)   # 6
        self.p0_action_head = nn.Linear(d_model, cfg.action_vocab)
        self.p1_action_head = nn.Linear(d_model, cfg.action_vocab)
        self.p0_jumps_head = nn.Linear(d_model, cfg.jumps_vocab)
        self.p1_jumps_head = nn.Linear(d_model, cfg.jumps_vocab)
        # Combat context heads (predict per player)
        self.p0_l_cancel_head = nn.Linear(d_model, cfg.l_cancel_vocab)
        self.p1_l_cancel_head = nn.Linear(d_model, cfg.l_cancel_vocab)
        self.p0_hurtbox_head = nn.Linear(d_model, cfg.hurtbox_vocab)
        self.p1_hurtbox_head = nn.Linear(d_model, cfg.hurtbox_vocab)
        self.p0_ground_head = nn.Linear(d_model, cfg.ground_vocab)
        self.p1_ground_head = nn.Linear(d_model, cfg.ground_vocab)
        self.p0_last_attack_head = nn.Linear(d_model, cfg.last_attack_vocab)
        self.p1_last_attack_head = nn.Linear(d_model, cfg.last_attack_vocab)

    def _encode_frames(
        self, float_ctx: torch.Tensor, int_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Encode context frames into per-frame vectors.

        Identical logic to FrameStackMLP.forward() embedding section.
        """
        c = self._int_cols

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
        """Forward pass — same signature as FrameStackMLP."""
        B, K, _ = float_ctx.shape

        frame_enc = self._encode_frames(float_ctx, int_ctx)
        x = self.input_dropout(self.frame_proj(frame_enc))

        for layer in self.layers:
            x = x + layer["mamba"](layer["norm"](x))

        h = self.final_norm(x[:, -1, :])
        h = h + self.ctrl_proj(next_ctrl)

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
