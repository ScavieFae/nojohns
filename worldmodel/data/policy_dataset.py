"""Imitation learning dataset: game state → controller output.

Given K frames of full game state (including previous controller inputs),
predict what player 0 (or player 1) will press on the next frame.

Returns 3 tensors per sample:
    float_ctx:  (K, 58) — full context frames [t-K, ..., t-1]
    int_ctx:    (K, 15) — context categoricals
    ctrl_tgt:   (13,)   — target player's controller on frame t
                          [main_x, main_y, c_x, c_y, shoulder, A, B, X, Y, Z, L, R, D_UP]
"""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from worldmodel.data.dataset import FLOAT_PER_PLAYER, MeleeDataset

logger = logging.getLogger(__name__)

# Controller layout within each player's float block (29 floats per player)
# Columns 16:29 = [main_x, main_y, c_x, c_y, shoulder, A, B, X, Y, Z, L, R, D_UP]
CTRL_OFFSET = 16
CTRL_DIM = 13
# Analog: first 5 (sticks + trigger), buttons: last 8
ANALOG_DIM = 5
BUTTON_DIM = 8


class PolicyFrameDataset(Dataset):
    """Imitation learning dataset — predict one player's controller from game state.

    Context window includes full frame data (state + controller) for previous K frames.
    This gives the model access to the player's recent input history, which helps
    predict input patterns (dash-dancing, L-cancel timing, etc).

    The target is the controller state for frame t — what the player actually pressed.
    """

    P0_CTRL = slice(CTRL_OFFSET, CTRL_OFFSET + CTRL_DIM)  # floats[16:29]
    P1_CTRL = slice(FLOAT_PER_PLAYER + CTRL_OFFSET,
                    FLOAT_PER_PLAYER + CTRL_OFFSET + CTRL_DIM)  # floats[45:58]

    def __init__(
        self,
        data: MeleeDataset,
        game_range: range,
        context_len: int = 10,
        predict_player: int = 0,
    ):
        self.data = data
        self.context_len = context_len
        self.ctrl_slice = self.P0_CTRL if predict_player == 0 else self.P1_CTRL

        indices = []
        for gi in game_range:
            start = data.game_offsets[gi]
            end = data.game_offsets[gi + 1]
            for t in range(start + context_len, end):
                indices.append(t)

        self.valid_indices = np.array(indices, dtype=np.int64)
        logger.info(
            "PolicyDataset: %d examples from %d games (context=%d, player=%d)",
            len(self.valid_indices), len(game_range), context_len, predict_player,
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.valid_indices[idx])
        K = self.context_len

        float_ctx = self.data.floats[t - K:t]  # (K, 58)
        int_ctx = self.data.ints[t - K:t]  # (K, 15)
        ctrl_tgt = self.data.floats[t][self.ctrl_slice]  # (13,)

        return float_ctx, int_ctx, ctrl_tgt
