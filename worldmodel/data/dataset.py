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
  float_tgt:  (D,)   — [p0_cont_delta(4), p1_cont_delta(4), p0_vel_delta(5), p1_vel_delta(5),
                         p0_binary(B), p1_binary(B), p0_dynamics(Y), p1_dynamics(Y)]
                        D=30 baseline, D=112 with state_flags+hitstun. B=binary_dim, Y=dynamics per player.
  int_tgt:    (12,)  — [p0_action, p0_jumps, p0_l_cancel, p0_hurtbox, p0_ground, p0_last_attack,
                         p1_action, p1_jumps, p1_l_cancel, p1_hurtbox, p1_ground, p1_last_attack]

  Baseline: F=58, I=15, C=26. See EncodingConfig for experiment variants.
  Context = frames [t-K, ..., t-1]. Target = frame t+d's state given ctrl(t)..ctrl(t+d).
  When lookahead=0 (default), d=0 and this reduces to: predict frame t given ctrl(t).

  focal_offset=D (E008): context window extends D frames past the prediction target.
  All K frames have full state+ctrl — model sees the future trajectory and learns
  to predict the focal frame from both past and future context.
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
TARGET_FLOAT_DIM = 30
TARGET_INT_DIM = 12


def _encode_game(game: ParsedGame, cfg: EncodingConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a single game into float and int tensors.

    Returns:
        frame_float: (T, float_per_player*2) — continuous + binary + controller per player
        frame_int: (T, int_per_frame) — categoricals per player + stage
    """
    items = game.items if cfg.projectiles else None
    p0 = encode_player_frames(game.p0, cfg, items=items)
    p1 = encode_player_frames(game.p1, cfg, items=items)
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

    @classmethod
    def from_tensors(cls, floats, ints, game_offsets, game_lengths, num_games, cfg):
        """Reconstruct from pre-encoded tensors (e.g. loaded from a .pt file)."""
        ds = cls.__new__(cls)
        ds.cfg = cfg
        ds.floats = floats
        ds.ints = ints
        ds.game_offsets = game_offsets if isinstance(game_offsets, np.ndarray) else game_offsets.numpy()
        ds.game_lengths = game_lengths
        ds.num_games = num_games
        ds.total_frames = int(ds.game_offsets[-1])
        return ds

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
        float_tgt:  (30,)  — [p0_cont_delta(4), p1_cont_delta(4), p0_vel_delta(5), p1_vel_delta(5),
                              p0_binary(3), p1_binary(3), p0_dynamics(3), p1_dynamics(3)]
        int_tgt:    (12,)  — [p0: action, jumps, l_cancel, hurtbox, ground, last_attack,
                              p1: action, jumps, l_cancel, hurtbox, ground, last_attack]

    Dimensions are config-driven (EncodingConfig flags change them).
    """

    def __init__(self, data: MeleeDataset, game_range: range, context_len: int = 10):
        self.data = data
        self.game_range = game_range
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

        # Velocity slices (indices 4:9 within each player's continuous block)
        vel_start = cfg.core_continuous_dim           # 4
        vel_end = vel_start + cfg.velocity_dim        # 9
        self._p0_vel = slice(vel_start, vel_end)
        self._p1_vel = slice(fp + vel_start, fp + vel_end)

        # Dynamics indices: hitlag, stocks, combo_count [, hitstun]
        # Layout after velocity: [state_age (if not embed)], hitlag, stocks, combo_count [, hitstun]
        dyn_start = vel_end + (0 if cfg.state_age_as_embed else 1)  # skip state_age float
        self._p0_dyn = [dyn_start, dyn_start + 1, dyn_start + 2]  # hitlag, stocks, combo
        if cfg.hitstun:
            self._p0_dyn.append(dyn_start + 3)  # hitstun_remaining
        self._p1_dyn = [fp + i for i in self._p0_dyn]

        # Button slices for press events (buttons are last 8 of controller)
        # controller layout: main_x, main_y, c_x, c_y, shoulder, 8 buttons
        btn_offset = 5  # 4 sticks + 1 shoulder
        self._p0_buttons = slice(ctrl_start + btn_offset, ctrl_end)
        self._p1_buttons = slice(fp + ctrl_start + btn_offset, fp + ctrl_end)

        # Int column offsets
        ipp = cfg.int_per_player
        self._p1_int_offset = ipp  # where p1's int columns start

        self._press_events = cfg.press_events
        self._lookahead = cfg.lookahead
        self._focal_offset = cfg.focal_offset
        self._multi_position = cfg.multi_position

        if cfg.focal_offset > 0:
            assert cfg.focal_offset < context_len, (
                f"focal_offset ({cfg.focal_offset}) must be < context_len ({context_len})"
            )

        indices = []
        # press_events needs t-1 for prev buttons, so start at context_len (which already guarantees t-1 exists)
        # lookahead=d needs frames up to t+d, so end range shrinks by d
        # focal_offset=D needs D extra frames after t for context tail, so end shrinks by D too
        tail = max(cfg.lookahead, cfg.focal_offset)
        for gi in game_range:
            start = data.game_offsets[gi]
            end = data.game_offsets[gi + 1]
            for t in range(start + context_len, end - tail):
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
        d = self._lookahead  # 0 = predict frame t, 1 = predict frame t+1, etc.
        D = self._focal_offset  # 0 = predict end of window (default), >0 = predict inside window

        # E008: focal offset — context window extends D frames past the prediction target.
        # Context includes full state+ctrl for ALL K frames, including D future frames.
        # The model sees where the game actually goes and learns to predict the focal
        # frame (t) from both past context and future trajectory.
        #   D=0: context [t-K, ..., t-1], predict t  (baseline)
        #   D=3: context [t-K, ..., t-1, t, t+1, t+2], predict t  (3 future frames)
        ctx_end = t + D  # context extends D frames past t
        float_ctx = self.data.floats[ctx_end - K:ctx_end]  # (K, F)
        int_ctx = self.data.ints[ctx_end - K:ctx_end]  # (K, I)

        # Controller input: ctrl(t) through ctrl(t+d), concatenated
        ctrl_parts = []
        for offset in range(d + 1):
            frame_float = self.data.floats[t + offset]
            ctrl_parts.append(frame_float[self._p0_ctrl])  # p0 controller (13)
            ctrl_parts.append(frame_float[self._p1_ctrl])  # p1 controller (13)

            if self._press_events:
                prev_float = self.data.floats[t + offset - 1]
                p0_curr_btn = frame_float[self._p0_buttons]
                p0_prev_btn = prev_float[self._p0_buttons]
                p1_curr_btn = frame_float[self._p1_buttons]
                p1_prev_btn = prev_float[self._p1_buttons]
                p0_press = ((p0_curr_btn > 0.5) & (p0_prev_btn < 0.5)).float()
                p1_press = ((p1_curr_btn > 0.5) & (p1_prev_btn < 0.5)).float()
                ctrl_parts.extend([p0_press, p1_press])

        next_ctrl = torch.cat(ctrl_parts)  # (26*(1+d),) or (42*(1+d),)

        # E008c: multi-position — return K targets (one per context position)
        if self._multi_position:
            return self._get_multi_position(t, d, float_ctx, int_ctx, next_ctrl)

        # Target frame: t+d (same as baseline — focal_offset only changes context, not target)
        tgt_idx = t + d
        float_tgt, int_tgt = self._build_targets(tgt_idx)

        return float_ctx, int_ctx, next_ctrl, float_tgt, int_tgt

    def _build_targets(self, tgt_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build float and int targets for a single frame."""
        tgt_float = self.data.floats[tgt_idx]
        prev_float = self.data.floats[tgt_idx - 1]

        # Continuous delta: target minus previous
        p0_cont_delta = tgt_float[self._p0_cont] - prev_float[self._p0_cont]
        p1_cont_delta = tgt_float[self._p1_cont] - prev_float[self._p1_cont]

        # Velocity deltas
        p0_vel_delta = tgt_float[self._p0_vel] - prev_float[self._p0_vel]
        p1_vel_delta = tgt_float[self._p1_vel] - prev_float[self._p1_vel]

        p0_binary = tgt_float[self._p0_bin]
        p1_binary = tgt_float[self._p1_bin]

        # Dynamics absolute (hitlag, stocks, combo_count)
        p0_dyn = tgt_float[self._p0_dyn]
        p1_dyn = tgt_float[self._p1_dyn]

        float_tgt = torch.cat([
            p0_cont_delta, p1_cont_delta,     # (8)
            p0_vel_delta, p1_vel_delta,        # (10)
            p0_binary, p1_binary,              # (6)
            p0_dyn, p1_dyn,                    # (6)
        ])  # (D,)

        # Int targets: categoricals (6 per player)
        tgt_ints = self.data.ints[tgt_idx]
        p1_off = self._p1_int_offset
        int_tgt = torch.stack([
            tgt_ints[0],          # p0_action
            tgt_ints[1],          # p0_jumps
            tgt_ints[3],          # p0_l_cancel
            tgt_ints[4],          # p0_hurtbox
            tgt_ints[5],          # p0_ground
            tgt_ints[6],          # p0_last_attack
            tgt_ints[p1_off],     # p1_action
            tgt_ints[p1_off + 1], # p1_jumps
            tgt_ints[p1_off + 3], # p1_l_cancel
            tgt_ints[p1_off + 4], # p1_hurtbox
            tgt_ints[p1_off + 5], # p1_ground
            tgt_ints[p1_off + 6], # p1_last_attack
        ])  # (12,)

        return float_tgt, int_tgt

    def _get_multi_position(
        self,
        t: int,
        d: int,
        float_ctx: torch.Tensor,
        int_ctx: torch.Tensor,
        next_ctrl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """E008c: compute K targets — one for each context position.

        At position i, predict frame (ctx_start + i + 1). The previous frame
        for computing deltas is ctx_start + i = the context frame itself.
        """
        K = self.context_len
        D = self._focal_offset
        ctx_start = t + D - K  # first context frame index

        # Vectorized: targets are context frames shifted by 1
        prev_floats = self.data.floats[ctx_start:ctx_start + K]     # (K, F) — same as float_ctx
        tgt_floats = self.data.floats[ctx_start + 1:ctx_start + K + 1]  # (K, F)

        p0_cont_delta = tgt_floats[:, self._p0_cont] - prev_floats[:, self._p0_cont]  # (K, 4)
        p1_cont_delta = tgt_floats[:, self._p1_cont] - prev_floats[:, self._p1_cont]
        p0_vel_delta = tgt_floats[:, self._p0_vel] - prev_floats[:, self._p0_vel]      # (K, 5)
        p1_vel_delta = tgt_floats[:, self._p1_vel] - prev_floats[:, self._p1_vel]
        p0_binary = tgt_floats[:, self._p0_bin]                                         # (K, B)
        p1_binary = tgt_floats[:, self._p1_bin]
        p0_dyn = tgt_floats[:, self._p0_dyn]                                            # (K, Y)
        p1_dyn = tgt_floats[:, self._p1_dyn]

        float_tgt = torch.cat([
            p0_cont_delta, p1_cont_delta,
            p0_vel_delta, p1_vel_delta,
            p0_binary, p1_binary,
            p0_dyn, p1_dyn,
        ], dim=1)  # (K, D)

        # Int targets for all K positions
        tgt_ints = self.data.ints[ctx_start + 1:ctx_start + K + 1]  # (K, I)
        p1_off = self._p1_int_offset
        int_tgt = torch.stack([
            tgt_ints[:, 0], tgt_ints[:, 1], tgt_ints[:, 3],
            tgt_ints[:, 4], tgt_ints[:, 5], tgt_ints[:, 6],
            tgt_ints[:, p1_off], tgt_ints[:, p1_off + 1], tgt_ints[:, p1_off + 3],
            tgt_ints[:, p1_off + 4], tgt_ints[:, p1_off + 5], tgt_ints[:, p1_off + 6],
        ], dim=1)  # (K, 12)

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
