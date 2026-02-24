"""PyTorch Dataset for Melee world model training.

Performance-critical: with 17M+ examples, __getitem__ must be pure tensor slicing.
Returns 3 tensors per sample (float context, int context, float targets) to avoid
dict overhead and custom collation.

v2 layout (encoding with velocity, dynamics, character, stage):
  Float per player: [cont(12), binary(3), controller(13)] = 28
  Float per frame: 28 * 2 = 56
  Int per frame: [p0_action, p0_jumps, p0_character, p1_action, p1_jumps, p1_character, stage] = 7
  Target float: [p0_cont_delta(4), p1_cont_delta(4), p0_binary(3), p1_binary(3)] = 14
  Target int: [p0_action, p0_jumps, p1_action, p1_jumps] = 4
"""

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from worldmodel.data.parse import ParsedGame
from worldmodel.model.encoding import EncodingConfig, encode_player_frames

logger = logging.getLogger(__name__)

# v2 layout: per player = continuous(12) + binary(3) + controller(13) = 28
# Per frame = 28 * 2 players = 56
FLOAT_PER_PLAYER = 28
FLOAT_PER_FRAME = FLOAT_PER_PLAYER * 2  # 56
# [p0_action, p0_jumps, p0_character, p1_action, p1_jumps, p1_character, stage] = 7
INT_PER_FRAME = 7
# Targets unchanged from v1
TARGET_FLOAT_DIM = 14
TARGET_INT_DIM = 4


class MeleeDataset:
    """Container holding all game data as contiguous tensors."""

    def __init__(self, games: list[ParsedGame], cfg: EncodingConfig):
        self.cfg = cfg
        self.num_games = len(games)

        # Build contiguous float and int arrays for all frames
        all_floats = []  # (T, FLOAT_PER_FRAME)
        all_ints = []  # (T, INT_PER_FRAME)
        game_lengths = []

        for game in games:
            p0 = encode_player_frames(game.p0, cfg)
            p1 = encode_player_frames(game.p1, cfg)
            T = game.num_frames

            # Float: [continuous(12), binary(3), controller(13)] per player = 28 each
            frame_float = torch.cat([
                p0["continuous"],  # (T, 12) — percent, x, y, shield, 5×vel, state_age, hitlag, stocks
                p0["binary"],  # (T, 3)
                p0["controller"],  # (T, 13)
                p1["continuous"],  # (T, 12)
                p1["binary"],  # (T, 3)
                p1["controller"],  # (T, 13)
            ], dim=1)  # (T, 56)
            all_floats.append(frame_float)

            # Int: action + jumps + character per player, plus stage
            stage_col = torch.full((T,), game.stage, dtype=torch.long)
            stage_col = torch.clamp(stage_col, 0, cfg.stage_vocab - 1)

            frame_int = torch.stack([
                p0["action"],  # (T,)
                p0["jumps_left"],
                p0["character"],
                p1["action"],
                p1["jumps_left"],
                p1["character"],
                stage_col,
            ], dim=1)  # (T, 7)
            all_ints.append(frame_int)

            game_lengths.append(T)

        self.floats = torch.cat(all_floats, dim=0)  # (total_frames, 56)
        self.ints = torch.cat(all_ints, dim=0)  # (total_frames, 7)
        self.game_offsets = np.cumsum([0] + game_lengths)
        self.total_frames = self.game_offsets[-1]
        self.game_lengths = game_lengths

        logger.info("Encoded %d games, %d total frames", self.num_games, self.total_frames)

    def get_frame_dataset(
        self,
        context_len: int = 10,
        train: bool = True,
        train_split: float = 0.9,
    ) -> "MeleeFrameDataset":
        split_idx = max(1, int(self.num_games * train_split))
        if train:
            game_range = range(0, split_idx)
        else:
            game_range = range(split_idx, self.num_games)
        return MeleeFrameDataset(self, game_range, context_len)


class MeleeFrameDataset(Dataset):
    """Frame-level dataset returning flat tensors.

    v2 layout:
        float_ctx: (K, 56) — [p0: cont(12)+bin(3)+ctrl(13), p1: same] per frame
        int_ctx:   (K, 7)  — [p0_action, p0_jumps, p0_char, p1_action, p1_jumps, p1_char, stage]
        float_tgt: (14,)   — [p0_cont_delta(4), p1_cont_delta(4), p0_binary(3), p1_binary(3)]
        int_tgt:   (4,)    — [p0_action, p0_jumps, p1_action, p1_jumps]
    """

    # v2 float layout offsets (per player block = 28 floats)
    # continuous: [0:12]  (percent, x, y, shield, 5×vel, state_age, hitlag, stocks)
    # binary:     [12:15] (facing, invuln, on_ground)
    # controller: [15:28] (sticks, shoulder, buttons)
    P0_CONT = slice(0, 4)      # core continuous: percent, x, y, shield (delta targets)
    P0_BIN = slice(12, 15)     # binary: facing, invuln, on_ground
    P1_CONT = slice(28, 32)    # p1 core continuous (28 = FLOAT_PER_PLAYER)
    P1_BIN = slice(40, 43)     # p1 binary (28 + 12 = 40)

    def __init__(self, data: MeleeDataset, game_range: range, context_len: int = 10):
        self.data = data
        self.context_len = context_len

        indices = []
        for gi in game_range:
            start = data.game_offsets[gi]
            end = data.game_offsets[gi + 1]
            for t in range(start + context_len, end - 1):
                indices.append(t)

        self.valid_indices = np.array(indices, dtype=np.int64)
        logger.info(
            "FrameDataset: %d examples from %d games (context=%d)",
            len(self.valid_indices), len(game_range), context_len,
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.valid_indices[idx])
        K = self.context_len

        float_ctx = self.data.floats[t - K:t]  # (K, 56)
        int_ctx = self.data.ints[t - K:t]  # (K, 7)

        # Targets: delta for continuous, raw for binary/categorical
        curr_float = self.data.floats[t]  # (56,)
        next_float = self.data.floats[t + 1]  # (56,)

        # Continuous delta: first 4 of each player's block (percent, x, y, shield)
        p0_cont_delta = next_float[self.P0_CONT] - curr_float[self.P0_CONT]
        p1_cont_delta = next_float[self.P1_CONT] - curr_float[self.P1_CONT]
        # Binary targets (next frame)
        p0_binary = next_float[self.P0_BIN]
        p1_binary = next_float[self.P1_BIN]

        float_tgt = torch.cat([p0_cont_delta, p1_cont_delta, p0_binary, p1_binary])  # (14,)

        # Int targets: next frame action/jumps (not character — that's constant)
        next_ints = self.data.ints[t + 1]
        int_tgt = torch.stack([
            next_ints[0],  # p0_action
            next_ints[1],  # p0_jumps
            next_ints[3],  # p1_action
            next_ints[4],  # p1_jumps
        ])  # (4,)

        return float_ctx, int_ctx, float_tgt, int_tgt
