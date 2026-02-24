"""PyTorch Dataset for Melee world model training.

Performance-critical: with 17M+ examples, __getitem__ must be pure tensor slicing.
Returns 5 tensors per sample to avoid dict overhead and custom collation.

v2.2+ layout — input-conditioned world model:
  The model receives the current frame's controller input alongside the context
  window, and predicts the current frame's state. Dimensions are config-driven
  (EncodingConfig flags change them — see encoding.py).

  float_ctx:  (K, F) — [p0: cont+bin(3)+ctrl(13), p1: same] per frame
  int_ctx:    (K, I) — per-player categoricals [+ state_age when embedded] + stage
  next_ctrl:  (C,)   — controller input for frame t [+ press events when enabled]
  float_tgt:  (14,)  — [p0_cont_delta(4), p1_cont_delta(4), p0_binary(3), p1_binary(3)]
  int_tgt:    (4,)   — [p0_action, p0_jumps, p1_action, p1_jumps]

  Baseline: F=58, I=15, C=26. See EncodingConfig for experiment variants.
  Context = frames [t-K, ..., t-1]. Target = frame t's state given frame t's input.
"""

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from worldmodel.data.parse import ParsedGame, load_game
from worldmodel.model.encoding import EncodingConfig, encode_player_frames

logger = logging.getLogger(__name__)

# Backward-compat constants: match v2.2 baseline (default EncodingConfig).
# Used by policy_dataset.py and rollout.py — will migrate to config-driven later.
FLOAT_PER_PLAYER = 29
FLOAT_PER_FRAME = 58
CTRL_DIM = 26
INT_PER_PLAYER = 7
INT_PER_FRAME = 15
TARGET_FLOAT_DIM = 14
TARGET_INT_DIM = 4


def _encode_game(game: ParsedGame, cfg: EncodingConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a single game into float and int tensors.

    Returns:
        frame_float: (T, float_per_player*2) — continuous + binary + controller per player
        frame_int: (T, int_per_frame) — categoricals per player + stage
    """
    p0 = encode_player_frames(game.p0, cfg)
    p1 = encode_player_frames(game.p1, cfg)
    T = game.num_frames

    # Float: [continuous, binary(3), controller(13)] per player
    frame_float = torch.cat([
        p0["continuous"],  # (T, continuous_dim)
        p0["binary"],  # (T, 3)
        p0["controller"],  # (T, 13)
        p1["continuous"],  # (T, continuous_dim)
        p1["binary"],  # (T, 3)
        p1["controller"],  # (T, 13)
    ], dim=1)  # (T, float_per_player*2)

    # Int: categoricals per player + stage
    stage_col = torch.full((T,), game.stage, dtype=torch.long)
    stage_col = torch.clamp(stage_col, 0, cfg.stage_vocab - 1)

    int_cols = [
        p0["action"],
        p0["jumps_left"],
        p0["character"],
        p0["l_cancel"],
        p0["hurtbox_state"],
        p0["ground"],
        p0["last_attack_landed"],
    ]
    if cfg.state_age_as_embed:
        int_cols.append(p0["state_age_int"])
    int_cols.extend([
        p1["action"],
        p1["jumps_left"],
        p1["character"],
        p1["l_cancel"],
        p1["hurtbox_state"],
        p1["ground"],
        p1["last_attack_landed"],
    ])
    if cfg.state_age_as_embed:
        int_cols.append(p1["state_age_int"])
    int_cols.append(stage_col)

    frame_int = torch.stack(int_cols, dim=1)  # (T, int_per_frame)

    return frame_float, frame_int


class MeleeDataset:
    """Container holding all game data as contiguous tensors (in-memory)."""

    def __init__(self, games: list[ParsedGame], cfg: EncodingConfig):
        self.cfg = cfg
        self.num_games = len(games)

        all_floats = []
        all_ints = []
        game_lengths = []

        for game in games:
            frame_float, frame_int = _encode_game(game, cfg)
            all_floats.append(frame_float)
            all_ints.append(frame_int)
            game_lengths.append(game.num_frames)

        self.floats = torch.cat(all_floats, dim=0)  # (total_frames, 58)
        self.ints = torch.cat(all_ints, dim=0)  # (total_frames, 15)
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
    """Input-conditioned frame dataset.

    v2.2+: returns 5 tensors — context window + next-frame controller input + targets.
    The model receives the controller input for frame t and predicts frame t's state.

        float_ctx:  (K, F) — context frames [t-K, ..., t-1]   (F = float_per_player*2)
        int_ctx:    (K, I) — context categoricals              (I = int_per_frame)
        next_ctrl:  (C,)   — frame t's controller input        (C = ctrl_conditioning_dim)
        float_tgt:  (14,)  — [p0_cont_delta(4), p1_cont_delta(4), p0_binary(3), p1_binary(3)]
        int_tgt:    (4,)   — [p0_action, p0_jumps, p1_action, p1_jumps]

    Dimensions are config-driven (EncodingConfig flags change them).
    """

    def __init__(self, data: MeleeDataset, game_range: range, context_len: int = 10):
        self.data = data
        self.context_len = context_len
        cfg = data.cfg

        # Compute slice offsets from config (per-player float block)
        fp = cfg.float_per_player
        cd = cfg.continuous_dim
        bd = cfg.binary_dim
        ctrl_d = cfg.controller_dim

        bin_start = cd
        bin_end = bin_start + bd
        ctrl_start = bin_end
        ctrl_end = ctrl_start + ctrl_d

        self._p0_cont = slice(0, cfg.core_continuous_dim)  # delta targets (4)
        self._p0_bin = slice(bin_start, bin_end)
        self._p0_ctrl = slice(ctrl_start, ctrl_end)
        self._p1_cont = slice(fp, fp + cfg.core_continuous_dim)
        self._p1_bin = slice(fp + bin_start, fp + bin_end)
        self._p1_ctrl = slice(fp + ctrl_start, fp + ctrl_end)

        # Button slices for press events (buttons are last 8 of controller)
        # controller layout: main_x, main_y, c_x, c_y, shoulder, 8 buttons
        btn_offset = 5  # 4 sticks + 1 shoulder
        self._p0_buttons = slice(ctrl_start + btn_offset, ctrl_end)
        self._p1_buttons = slice(fp + ctrl_start + btn_offset, fp + ctrl_end)

        # Int column offsets
        ipp = cfg.int_per_player
        self._p1_int_offset = ipp  # where p1's int columns start

        self._press_events = cfg.press_events

        indices = []
        # press_events needs t-1 for prev buttons, so start at context_len (which already guarantees t-1 exists)
        for gi in game_range:
            start = data.game_offsets[gi]
            end = data.game_offsets[gi + 1]
            for t in range(start + context_len, end):
                indices.append(t)

        self.valid_indices = np.array(indices, dtype=np.int64)
        logger.info(
            "FrameDataset: %d examples from %d games (context=%d)",
            len(self.valid_indices), len(game_range), context_len,
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.valid_indices[idx])
        K = self.context_len

        float_ctx = self.data.floats[t - K:t]  # (K, F)
        int_ctx = self.data.ints[t - K:t]  # (K, I)

        # Controller input for frame t (what we're conditioning on)
        curr_float = self.data.floats[t]
        ctrl_parts = [
            curr_float[self._p0_ctrl],  # p0 controller (13)
            curr_float[self._p1_ctrl],  # p1 controller (13)
        ]

        if self._press_events:
            prev_float = self.data.floats[t - 1]
            # Press event = button newly pressed this frame (was <0.5, now >0.5)
            p0_curr_btn = curr_float[self._p0_buttons]
            p0_prev_btn = prev_float[self._p0_buttons]
            p1_curr_btn = curr_float[self._p1_buttons]
            p1_prev_btn = prev_float[self._p1_buttons]
            p0_press = ((p0_curr_btn > 0.5) & (p0_prev_btn < 0.5)).float()  # (8,)
            p1_press = ((p1_curr_btn > 0.5) & (p1_prev_btn < 0.5)).float()  # (8,)
            ctrl_parts.extend([p0_press, p1_press])

        next_ctrl = torch.cat(ctrl_parts)  # (26,) or (42,)

        # Targets: frame t's state
        if not self._press_events:
            prev_float = self.data.floats[t - 1]

        # Continuous delta: frame t minus frame t-1
        p0_cont_delta = curr_float[self._p0_cont] - prev_float[self._p0_cont]
        p1_cont_delta = curr_float[self._p1_cont] - prev_float[self._p1_cont]
        p0_binary = curr_float[self._p0_bin]
        p1_binary = curr_float[self._p1_bin]

        float_tgt = torch.cat([p0_cont_delta, p1_cont_delta, p0_binary, p1_binary])  # (14,)

        # Int targets: frame t's action/jumps
        curr_ints = self.data.ints[t]
        p1_off = self._p1_int_offset
        int_tgt = torch.stack([
            curr_ints[0],          # p0_action
            curr_ints[1],          # p0_jumps
            curr_ints[p1_off],     # p1_action
            curr_ints[p1_off + 1], # p1_jumps
        ])  # (4,)

        return float_ctx, int_ctx, next_ctrl, float_tgt, int_tgt


# --- Streaming dataset for large datasets ---


class StreamingMeleeDataset(IterableDataset):
    """Memory-efficient dataset that loads games from disk in chunks.

    Instead of holding all games in RAM, loads buffer_size games at a time,
    generates all valid frame indices, shuffles within the buffer, and yields.
    Games are shuffled each epoch for good cross-game diversity.

    Train/val split is by game (same as MeleeDataset).
    """

    def __init__(
        self,
        game_entries: list[dict],
        dataset_dir: Path,
        cfg: EncodingConfig,
        context_len: int = 10,
        buffer_size: int = 1000,
        train: bool = True,
        train_split: float = 0.9,
    ):
        self.cfg = cfg
        self.context_len = context_len
        self.buffer_size = buffer_size

        # Split by game
        split_idx = max(1, int(len(game_entries) * train_split))
        if train:
            self.entries = game_entries[:split_idx]
        else:
            self.entries = game_entries[split_idx:]

        self.dataset_dir = Path(dataset_dir)
        self._approx_frames = len(self.entries) * 4500  # rough estimate

    def __len__(self) -> int:
        """Approximate length for progress bars and logging."""
        return self._approx_frames

    def __iter__(self):
        # Shuffle game order each epoch
        entries = list(self.entries)
        random.shuffle(entries)

        # Process in chunks
        for chunk_start in range(0, len(entries), self.buffer_size):
            chunk = entries[chunk_start:chunk_start + self.buffer_size]

            # Load and encode chunk
            games = []
            for entry in chunk:
                compression = entry.get("compression", "zlib")
                game_path = self.dataset_dir / "games" / entry["slp_md5"]
                try:
                    games.append(load_game(game_path, compression=compression))
                except Exception as e:
                    logger.debug("Skipping %s: %s", entry["slp_md5"], e)
                    continue

            if not games:
                continue

            # Reuse MeleeDataset for encoding (it builds contiguous tensors)
            data = MeleeDataset(games, self.cfg)
            # Use all games in chunk (train/val split already done in __init__)
            frame_ds = data.get_frame_dataset(
                context_len=self.context_len, train=True, train_split=1.0,
            )

            # Shuffle indices within buffer for good batch diversity
            indices = list(range(len(frame_ds)))
            random.shuffle(indices)

            for idx in indices:
                yield frame_ds[idx]

            # Free memory before loading next chunk
            del data, frame_ds, games

        logger.debug("StreamingMeleeDataset: completed epoch")
