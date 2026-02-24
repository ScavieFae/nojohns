"""PyTorch Dataset for Melee world model training.

Performance-critical: with 17M+ examples, __getitem__ must be pure tensor slicing.
Returns 5 tensors per sample to avoid dict overhead and custom collation.

v2.2 layout — input-conditioned world model:
  The model receives the current frame's controller input alongside the context
  window, and predicts the current frame's state. This separates physics prediction
  (deterministic given inputs) from decision prediction (what will the player press).

  float_ctx:  (K, 58) — [p0: cont(13)+bin(3)+ctrl(13), p1: same] per frame
  int_ctx:    (K, 15) — per-player categoricals + stage
  next_ctrl:  (26,)   — controller input for frame t [p0_ctrl(13), p1_ctrl(13)]
  float_tgt:  (14,)   — [p0_cont_delta(4), p1_cont_delta(4), p0_binary(3), p1_binary(3)]
  int_tgt:    (4,)    — [p0_action, p0_jumps, p1_action, p1_jumps]

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

# v2.2 layout: per player = continuous(13) + binary(3) + controller(13) = 29
# Per frame = 29 * 2 players = 58
FLOAT_PER_PLAYER = 29
FLOAT_PER_FRAME = FLOAT_PER_PLAYER * 2  # 58
# Controller: sticks(4) + shoulders(2) + buttons(7) = 13 per player
CTRL_PER_PLAYER = 13
CTRL_DIM = CTRL_PER_PLAYER * 2  # 26 — the input-conditioning tensor
# [p0: action,jumps,char,l_cancel,hurtbox,ground,last_attack,
#  p1: action,jumps,char,l_cancel,hurtbox,ground,last_attack, stage] = 15
INT_PER_PLAYER = 7
INT_PER_FRAME = INT_PER_PLAYER * 2 + 1  # 15
# Targets unchanged
TARGET_FLOAT_DIM = 14
TARGET_INT_DIM = 4


def _encode_game(game: ParsedGame, cfg: EncodingConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a single game into float and int tensors.

    Returns:
        frame_float: (T, 58) — continuous + binary + controller per player
        frame_int: (T, 15) — categoricals per player + stage
    """
    p0 = encode_player_frames(game.p0, cfg)
    p1 = encode_player_frames(game.p1, cfg)
    T = game.num_frames

    # Float: [continuous(13), binary(3), controller(13)] per player = 29 each
    frame_float = torch.cat([
        p0["continuous"],  # (T, 13)
        p0["binary"],  # (T, 3)
        p0["controller"],  # (T, 13)
        p1["continuous"],  # (T, 13)
        p1["binary"],  # (T, 3)
        p1["controller"],  # (T, 13)
    ], dim=1)  # (T, 58)

    # Int: categoricals per player + stage
    stage_col = torch.full((T,), game.stage, dtype=torch.long)
    stage_col = torch.clamp(stage_col, 0, cfg.stage_vocab - 1)

    frame_int = torch.stack([
        p0["action"],
        p0["jumps_left"],
        p0["character"],
        p0["l_cancel"],
        p0["hurtbox_state"],
        p0["ground"],
        p0["last_attack_landed"],
        p1["action"],
        p1["jumps_left"],
        p1["character"],
        p1["l_cancel"],
        p1["hurtbox_state"],
        p1["ground"],
        p1["last_attack_landed"],
        stage_col,
    ], dim=1)  # (T, 15)

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

    v2.2: returns 5 tensors — context window + next-frame controller input + targets.
    The model receives the controller input for frame t and predicts frame t's state.
    This makes action prediction a physics problem (deterministic given inputs)
    rather than a decision-guessing problem.

        float_ctx:  (K, 58) — context frames [t-K, ..., t-1]
        int_ctx:    (K, 15) — context categoricals
        next_ctrl:  (26,)   — frame t's controller input [p0_ctrl(13), p1_ctrl(13)]
        float_tgt:  (14,)   — [p0_cont_delta(4), p1_cont_delta(4), p0_binary(3), p1_binary(3)]
        int_tgt:    (4,)    — [p0_action, p0_jumps, p1_action, p1_jumps]

    Targets are frame t's state. Delta is computed as frame t minus frame t-1.
    """

    # Float layout offsets (per player block = 29 floats)
    # continuous: [0:13]  (percent, x, y, shield, 5×vel, state_age, hitlag, stocks, combo_count)
    # binary:     [13:16] (facing, invuln, on_ground)
    # controller: [16:29] (sticks, shoulder, buttons)
    P0_CONT = slice(0, 4)      # core continuous: percent, x, y, shield (delta targets)
    P0_BIN = slice(13, 16)     # binary: facing, invuln, on_ground
    P0_CTRL = slice(16, 29)    # p0 controller (13 floats)
    P1_CONT = slice(29, 33)    # p1 core continuous (29 = FLOAT_PER_PLAYER)
    P1_BIN = slice(42, 45)     # p1 binary (29 + 13 = 42)
    P1_CTRL = slice(45, 58)    # p1 controller (13 floats)

    def __init__(self, data: MeleeDataset, game_range: range, context_len: int = 10):
        self.data = data
        self.context_len = context_len

        indices = []
        for gi in game_range:
            start = data.game_offsets[gi]
            end = data.game_offsets[gi + 1]
            # v2.2: predict frame t given t's controller, no need for t+1
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

        float_ctx = self.data.floats[t - K:t]  # (K, 58) — frames t-K to t-1
        int_ctx = self.data.ints[t - K:t]  # (K, 15)

        # Controller input for frame t (what we're conditioning on)
        curr_float = self.data.floats[t]  # (58,)
        next_ctrl = torch.cat([
            curr_float[self.P0_CTRL],  # p0 controller (13)
            curr_float[self.P1_CTRL],  # p1 controller (13)
        ])  # (26,)

        # Targets: frame t's state (the result of applying next_ctrl)
        prev_float = self.data.floats[t - 1]  # last context frame

        # Continuous delta: frame t minus frame t-1
        p0_cont_delta = curr_float[self.P0_CONT] - prev_float[self.P0_CONT]
        p1_cont_delta = curr_float[self.P1_CONT] - prev_float[self.P1_CONT]
        # Binary targets (frame t)
        p0_binary = curr_float[self.P0_BIN]
        p1_binary = curr_float[self.P1_BIN]

        float_tgt = torch.cat([p0_cont_delta, p1_cont_delta, p0_binary, p1_binary])  # (14,)

        # Int targets: frame t's action/jumps
        curr_ints = self.data.ints[t]
        int_tgt = torch.stack([
            curr_ints[0],  # p0_action
            curr_ints[1],  # p0_jumps
            curr_ints[INT_PER_PLAYER],  # p1_action (index 7)
            curr_ints[INT_PER_PLAYER + 1],  # p1_jumps (index 8)
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
