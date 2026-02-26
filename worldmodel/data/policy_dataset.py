"""Imitation learning dataset: game state → controller output.

Given K frames of full game state (including previous controller inputs),
predict what player 0 (or player 1) will press on the next frame.

Returns 3 tensors per sample:
    float_ctx:  (K, float_per_player*2) — full context frames [t-K, ..., t-1]
    int_ctx:    (K, int_per_frame) — context categoricals
    ctrl_tgt:   (13,)   — target player's controller on frame t
                          [main_x, main_y, c_x, c_y, shoulder, A, B, X, Y, Z, L, R, D_UP]
"""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from worldmodel.data.dataset import MeleeDataset
from worldmodel.model.encoding import EncodingConfig

logger = logging.getLogger(__name__)

# Controller constants (same across all configs)
CTRL_DIM = 13  # main_x, main_y, c_x, c_y, shoulder, A, B, X, Y, Z, L, R, D_UP
ANALOG_DIM = 5
BUTTON_DIM = 8


class PolicyFrameDataset(Dataset):
    """Imitation learning dataset — predict one player's controller from game state.

    Config-driven: works with any EncodingConfig (including state_age_as_embed).
    """

    def __init__(
        self,
        data: MeleeDataset,
        game_range: range,
        context_len: int = 10,
        predict_player: int = 0,
        cfg: EncodingConfig | None = None,
    ):
        self.data = data
        self.context_len = context_len

        if cfg is None:
            cfg = EncodingConfig()

        # Controller offset: continuous + binary dims, then controller starts
        fp = cfg.float_per_player
        ctrl_offset = cfg.continuous_dim + cfg.binary_dim
        if predict_player == 0:
            self.ctrl_slice = slice(ctrl_offset, ctrl_offset + CTRL_DIM)
        else:
            self.ctrl_slice = slice(fp + ctrl_offset, fp + ctrl_offset + CTRL_DIM)

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

        float_ctx = self.data.floats[t - K:t]
        int_ctx = self.data.ints[t - K:t]
        ctrl_tgt = self.data.floats[t][self.ctrl_slice]

        return float_ctx, int_ctx, ctrl_tgt
