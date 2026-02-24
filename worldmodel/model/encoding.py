"""State encoding: game frames → flat tensors for the world model.

Converts per-player per-frame data into a fixed-size tensor representation.
Continuous values are normalized, categoricals get learned embeddings.

Normalization scales match slippi-ai's embed.py:
  - xy: ×0.05    (positions range roughly -200 to 200)
  - percent: ×0.01  (damage 0-999)
  - shield: ×0.01   (0-60)
  - velocity: ×0.05  (same scale as position — units/frame)
  - state_age: ×0.01 (action frames, typically 0-100+)
  - hitlag: ×0.1     (typically 0-10 frames)
  - stocks: ×0.25    (0-4 stocks → 0-1 range)
  - controller: already [0, 1], no scaling needed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from worldmodel.data.parse import PlayerFrame


# --- Encoding config ---


@dataclass
class EncodingConfig:
    """Configuration for state encoding dimensions."""

    # Normalization scales
    xy_scale: float = 0.05
    percent_scale: float = 0.01
    shield_scale: float = 0.01
    velocity_scale: float = 0.05
    state_age_scale: float = 0.01
    hitlag_scale: float = 0.1
    stocks_scale: float = 0.25

    # Embedding sizes for categoricals
    action_vocab: int = 400  # action states 0..399
    action_embed_dim: int = 32
    jumps_vocab: int = 8  # jumps_left 0..7
    jumps_embed_dim: int = 4
    character_vocab: int = 33  # internal character IDs 0..32
    character_embed_dim: int = 8
    stage_vocab: int = 33  # stage IDs 0..32 (legal stages are a subset)
    stage_embed_dim: int = 4

    # Derived dimensions (per player)
    # core_continuous: percent, x, y, shield = 4
    # velocity: speed_air_x, speed_y, speed_ground_x, speed_attack_x, speed_attack_y = 5
    # dynamics: state_age, hitlag, stocks = 3
    # binary: facing, invulnerable, on_ground = 3
    # controller: main_x, main_y, c_x, c_y, shoulder, 8 buttons = 13
    # embeddings: action (32) + jumps (4) + character (8) = 44
    # Per player = 4 + 5 + 3 + 3 + 13 + 44 = 72
    # Stage embedding (shared): 4
    # Total per frame = 72 * 2 + 4 = 148

    @property
    def core_continuous_dim(self) -> int:
        return 4  # percent, x, y, shield

    @property
    def velocity_dim(self) -> int:
        return 5  # speed_air_x, speed_y, speed_ground_x, speed_attack_x, speed_attack_y

    @property
    def dynamics_dim(self) -> int:
        return 3  # state_age, hitlag, stocks

    @property
    def continuous_dim(self) -> int:
        return self.core_continuous_dim + self.velocity_dim + self.dynamics_dim  # 12

    @property
    def binary_dim(self) -> int:
        return 3  # facing, invulnerable, on_ground

    @property
    def controller_dim(self) -> int:
        return 13  # 2 sticks (4) + shoulder (1) + 8 buttons

    @property
    def float_per_player(self) -> int:
        return self.continuous_dim + self.binary_dim + self.controller_dim  # 28

    @property
    def embed_dim(self) -> int:
        return self.action_embed_dim + self.jumps_embed_dim + self.character_embed_dim

    @property
    def player_dim(self) -> int:
        return self.float_per_player + self.embed_dim  # 28 + 44 = 72

    @property
    def frame_dim(self) -> int:
        return self.player_dim * 2 + self.stage_embed_dim  # 72*2 + 4 = 148


# --- Tensor encoding (numpy → torch, no learned params) ---


def encode_player_frames(pf: PlayerFrame, cfg: EncodingConfig) -> dict[str, torch.Tensor]:
    """Convert a PlayerFrame's numpy arrays into normalized torch tensors.

    Returns dict with keys:
        continuous: (T, 12) — [percent, x, y, shield, 5×velocity, state_age, hitlag, stocks]
        binary: (T, 3) — [facing, invulnerable, on_ground]
        controller: (T, 13) — [main_x, main_y, c_x, c_y, shoulder, A..D_UP]
        action: (T,) — int64 action indices
        jumps_left: (T,) — int64 jumps indices
        character: (T,) — int64 character indices
    """
    T = len(pf.percent)

    continuous = np.stack(
        [
            # Core continuous (predict deltas for these)
            pf.percent * cfg.percent_scale,
            pf.x * cfg.xy_scale,
            pf.y * cfg.xy_scale,
            pf.shield_strength * cfg.shield_scale,
            # Velocity (input features)
            pf.speed_air_x * cfg.velocity_scale,
            pf.speed_y * cfg.velocity_scale,
            pf.speed_ground_x * cfg.velocity_scale,
            pf.speed_attack_x * cfg.velocity_scale,
            pf.speed_attack_y * cfg.velocity_scale,
            # Dynamics (input features)
            pf.state_age * cfg.state_age_scale,
            pf.hitlag * cfg.hitlag_scale,
            pf.stocks * cfg.stocks_scale,
        ],
        axis=1,
    )  # (T, 12)

    binary = np.stack(
        [pf.facing, pf.invulnerable, pf.on_ground],
        axis=1,
    )  # (T, 3)

    controller = np.concatenate(
        [
            pf.main_stick_x[:, None],
            pf.main_stick_y[:, None],
            pf.c_stick_x[:, None],
            pf.c_stick_y[:, None],
            pf.shoulder[:, None],
            pf.buttons,  # (T, 8)
        ],
        axis=1,
    )  # (T, 13)

    # Clamp categoricals to valid range
    action = np.clip(pf.action, 0, cfg.action_vocab - 1)
    jumps = np.clip(pf.jumps_left, 0, cfg.jumps_vocab - 1)
    character = np.clip(pf.character, 0, cfg.character_vocab - 1)

    return {
        "continuous": torch.from_numpy(continuous).float(),
        "binary": torch.from_numpy(binary).float(),
        "controller": torch.from_numpy(controller).float(),
        "action": torch.from_numpy(action).long(),
        "jumps_left": torch.from_numpy(jumps).long(),
        "character": torch.from_numpy(character).long(),
    }


# --- Embedding module (learned params, part of the model) ---


class StateEncoder(nn.Module):
    """Encodes per-frame game state into a flat vector via learned embeddings.

    Takes pre-encoded tensors (from encode_player_frames) and produces
    a single flat vector per frame by embedding categoricals and concatenating.
    """

    def __init__(self, cfg: EncodingConfig):
        super().__init__()
        self.cfg = cfg
        self.action_embed = nn.Embedding(cfg.action_vocab, cfg.action_embed_dim)
        self.jumps_embed = nn.Embedding(cfg.jumps_vocab, cfg.jumps_embed_dim)
        self.character_embed = nn.Embedding(cfg.character_vocab, cfg.character_embed_dim)
        self.stage_embed = nn.Embedding(cfg.stage_vocab, cfg.stage_embed_dim)

    def forward_player(
        self,
        continuous: torch.Tensor,  # (B, 12)
        binary: torch.Tensor,  # (B, 3)
        controller: torch.Tensor,  # (B, 13)
        action: torch.Tensor,  # (B,)
        jumps_left: torch.Tensor,  # (B,)
        character: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Encode one player's state. Returns (B, player_dim)."""
        action_emb = self.action_embed(action)  # (B, 32)
        jumps_emb = self.jumps_embed(jumps_left)  # (B, 4)
        char_emb = self.character_embed(character)  # (B, 8)
        return torch.cat([continuous, binary, controller, action_emb, jumps_emb, char_emb], dim=-1)

    def forward(
        self,
        p0_continuous: torch.Tensor,
        p0_binary: torch.Tensor,
        p0_controller: torch.Tensor,
        p0_action: torch.Tensor,
        p0_jumps: torch.Tensor,
        p0_character: torch.Tensor,
        p1_continuous: torch.Tensor,
        p1_binary: torch.Tensor,
        p1_controller: torch.Tensor,
        p1_action: torch.Tensor,
        p1_jumps: torch.Tensor,
        p1_character: torch.Tensor,
        stage: torch.Tensor,
    ) -> torch.Tensor:
        """Encode full frame state for both players. Returns (B, frame_dim)."""
        p0 = self.forward_player(p0_continuous, p0_binary, p0_controller, p0_action, p0_jumps, p0_character)
        p1 = self.forward_player(p1_continuous, p1_binary, p1_controller, p1_action, p1_jumps, p1_character)
        stage_emb = self.stage_embed(stage)  # (B, 4)
        return torch.cat([p0, p1, stage_emb], dim=-1)  # (B, frame_dim)
