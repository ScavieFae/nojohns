"""PyTorch Dataset for Melee world model training.

Performance-critical: with 17M+ examples, __getitem__ must be pure tensor slicing.
Returns 3 tensors per sample (float context, int context, float targets) to avoid
dict overhead and custom collation.
"""

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from worldmodel.data.parse import ParsedGame
from worldmodel.model.encoding import EncodingConfig, encode_player_frames

logger = logging.getLogger(__name__)

# Layout of the float context tensor per frame:
# [p0_cont(4), p0_bin(3), p0_ctrl(13), p1_cont(4), p1_bin(3), p1_ctrl(13)] = 40 floats
FLOAT_PER_FRAME = 40
# Layout of the int context tensor per frame:
# [p0_action, p0_jumps, p1_action, p1_jumps] = 4 ints
INT_PER_FRAME = 4
# Layout of targets:
# [cont_delta(8), binary(6)] = 14 floats + [p0_action, p1_action, p0_jumps, p1_jumps] = 4 ints
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

            # Float: concat all continuous data per frame
            frame_float = torch.cat([
                p0["continuous"],  # (T, 4)
                p0["binary"],  # (T, 3)
                p0["controller"],  # (T, 13)
                p1["continuous"],  # (T, 4)
                p1["binary"],  # (T, 3)
                p1["controller"],  # (T, 13)
            ], dim=1)  # (T, 40)
            all_floats.append(frame_float)

            # Int: action + jumps per player
            frame_int = torch.stack([
                p0["action"],  # (T,)
                p0["jumps_left"],
                p1["action"],
                p1["jumps_left"],
            ], dim=1)  # (T, 4)
            all_ints.append(frame_int)

            game_lengths.append(T)

        self.floats = torch.cat(all_floats, dim=0)  # (total_frames, 40)
        self.ints = torch.cat(all_ints, dim=0)  # (total_frames, 4)
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

    __getitem__ returns:
        float_ctx: (K, 40) — context window float features
        int_ctx:   (K, 4)  — context window categorical indices
        float_tgt: (14,)   — target float features [cont_delta(8), binary(6)]
        int_tgt:   (4,)    — target categorical indices
    """

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

        float_ctx = self.data.floats[t - K:t]  # (K, 40)
        int_ctx = self.data.ints[t - K:t]  # (K, 4)

        # Targets: delta for continuous, raw for binary/categorical
        # Float targets: [p0_cont_delta(4), p1_cont_delta(4), p0_binary(3), p1_binary(3)]
        curr_float = self.data.floats[t]  # (40,)
        next_float = self.data.floats[t + 1]  # (40,)

        # Continuous delta: indices 0:4 (p0) and 20:24 (p1)
        p0_cont_delta = next_float[0:4] - curr_float[0:4]
        p1_cont_delta = next_float[20:24] - curr_float[20:24]
        # Binary targets (next frame): indices 4:7 (p0) and 24:27 (p1)
        p0_binary = next_float[4:7]
        p1_binary = next_float[24:27]

        float_tgt = torch.cat([p0_cont_delta, p1_cont_delta, p0_binary, p1_binary])  # (14,)

        # Int targets: next frame action/jumps
        int_tgt = self.data.ints[t + 1]  # (4,) = [p0_action, p0_jumps, p1_action, p1_jumps]

        return float_ctx, int_ctx, float_tgt, int_tgt
