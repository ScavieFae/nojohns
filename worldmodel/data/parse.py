"""Thin wrapper around slippi_db parsing.

Handles the zlib-compressed parquet format that slippi_db produces,
and converts nested PyArrow structs into flat numpy arrays suitable
for world model training.
"""

import io
import json
import logging
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

BUTTON_NAMES = ["A", "B", "X", "Y", "Z", "L", "R", "D_UP"]


@dataclass
class PlayerFrame:
    """Flat numpy arrays for one player across all frames in a game."""

    percent: np.ndarray  # (T,) float32
    x: np.ndarray  # (T,) float32
    y: np.ndarray  # (T,) float32
    shield_strength: np.ndarray  # (T,) float32
    facing: np.ndarray  # (T,) bool → float32
    invulnerable: np.ndarray  # (T,) bool → float32
    on_ground: np.ndarray  # (T,) bool → float32
    action: np.ndarray  # (T,) int64 (categorical)
    jumps_left: np.ndarray  # (T,) int64 (categorical)
    character: np.ndarray  # (T,) int64 (categorical, constant per game)
    # Velocity (5 components)
    speed_air_x: np.ndarray  # (T,) float32
    speed_y: np.ndarray  # (T,) float32
    speed_ground_x: np.ndarray  # (T,) float32
    speed_attack_x: np.ndarray  # (T,) float32
    speed_attack_y: np.ndarray  # (T,) float32
    # Dynamics
    state_age: np.ndarray  # (T,) float32 — frame counter within current action
    hitlag: np.ndarray  # (T,) float32 — hitlag frames remaining
    stocks: np.ndarray  # (T,) float32 — stocks remaining
    # Combat context (v2.1 — in parquet, not yet wired into model)
    l_cancel: np.ndarray  # (T,) int64 — 0=N/A, 1=successful, 2=missed
    hurtbox_state: np.ndarray  # (T,) int64 — 0=vulnerable, 1=invulnerable, 2=intangible
    ground: np.ndarray  # (T,) int64 — ground surface ID (remapped: 65535→0 airborne sentinel)
    last_attack_landed: np.ndarray  # (T,) int64 — attack ID of last connected move
    combo_count: np.ndarray  # (T,) int64 — current combo counter
    # Controller
    main_stick_x: np.ndarray  # (T,) float32
    main_stick_y: np.ndarray  # (T,) float32
    c_stick_x: np.ndarray  # (T,) float32
    c_stick_y: np.ndarray  # (T,) float32
    shoulder: np.ndarray  # (T,) float32
    buttons: np.ndarray  # (T, 8) float32 — one per button


@dataclass
class ParsedGame:
    """A single parsed game with both players' frame data."""

    p0: PlayerFrame
    p1: PlayerFrame
    stage: int
    num_frames: int
    # Metadata (optional, for filtering)
    meta: Optional[dict] = None


def _safe_field(struct_array, name: str, num_frames: int, dtype=np.float32) -> np.ndarray:
    """Read a field from a PyArrow struct, returning zeros if the field doesn't exist."""
    try:
        return np.array(struct_array.field(name).to_pylist(), dtype=dtype)
    except (KeyError, Exception):
        return np.zeros(num_frames, dtype=dtype)


def _extract_player(struct_array, num_frames: int) -> PlayerFrame:
    """Extract flat numpy arrays from a PyArrow StructArray for one player."""
    # Continuous fields
    percent = np.array(struct_array.field("percent").to_pylist(), dtype=np.float32)
    x = np.array(struct_array.field("x").to_pylist(), dtype=np.float32)
    y = np.array(struct_array.field("y").to_pylist(), dtype=np.float32)
    shield = np.array(struct_array.field("shield_strength").to_pylist(), dtype=np.float32)

    # Boolean fields → float
    facing = np.array(struct_array.field("facing").to_pylist(), dtype=np.float32)
    invuln = np.array(struct_array.field("invulnerable").to_pylist(), dtype=np.float32)
    on_ground = np.array(struct_array.field("on_ground").to_pylist(), dtype=np.float32)

    # Categorical fields
    action = np.array(struct_array.field("action").to_pylist(), dtype=np.int64)
    jumps_left = np.array(struct_array.field("jumps_left").to_pylist(), dtype=np.int64)
    character = _safe_field(struct_array, "character", num_frames, dtype=np.int64)

    # Velocity (v2 fields — zeros if parsed with old schema)
    speed_air_x = _safe_field(struct_array, "speed_air_x", num_frames)
    speed_y = _safe_field(struct_array, "speed_y", num_frames)
    speed_ground_x = _safe_field(struct_array, "speed_ground_x", num_frames)
    speed_attack_x = _safe_field(struct_array, "speed_attack_x", num_frames)
    speed_attack_y = _safe_field(struct_array, "speed_attack_y", num_frames)

    # Dynamics (v2 fields — zeros if parsed with old schema)
    state_age = _safe_field(struct_array, "state_age", num_frames)
    hitlag = _safe_field(struct_array, "hitlag", num_frames)
    stocks = _safe_field(struct_array, "stocks", num_frames)

    # Combat context (v2.1 — zeros if parsed with old schema)
    l_cancel = _safe_field(struct_array, "l_cancel", num_frames, dtype=np.int64)
    hurtbox_state = _safe_field(struct_array, "hurtbox_state", num_frames, dtype=np.int64)
    ground_raw = _safe_field(struct_array, "ground", num_frames, dtype=np.int64)
    # Remap 65535 (airborne sentinel in peppi_py) → 0, shift real surfaces up by 1
    ground = np.where(ground_raw == 65535, 0, ground_raw + 1).astype(np.int64)
    last_attack_landed = _safe_field(struct_array, "last_attack_landed", num_frames, dtype=np.int64)
    combo_count = _safe_field(struct_array, "combo_count", num_frames, dtype=np.int64)

    # Controller
    ctrl = struct_array.field("controller")
    main = ctrl.field("main_stick")
    c = ctrl.field("c_stick")
    main_x = np.array(main.field("x").to_pylist(), dtype=np.float32)
    main_y = np.array(main.field("y").to_pylist(), dtype=np.float32)
    c_x = np.array(c.field("x").to_pylist(), dtype=np.float32)
    c_y = np.array(c.field("y").to_pylist(), dtype=np.float32)
    shoulder = np.array(ctrl.field("shoulder").to_pylist(), dtype=np.float32)

    # Buttons
    btn_struct = ctrl.field("buttons")
    button_arrays = []
    for name in BUTTON_NAMES:
        button_arrays.append(np.array(btn_struct.field(name).to_pylist(), dtype=np.float32))
    buttons = np.stack(button_arrays, axis=1)  # (T, 8)

    return PlayerFrame(
        percent=percent,
        x=x,
        y=y,
        shield_strength=shield,
        facing=facing,
        invulnerable=invuln,
        on_ground=on_ground,
        action=action,
        jumps_left=jumps_left,
        character=character,
        speed_air_x=speed_air_x,
        speed_y=speed_y,
        speed_ground_x=speed_ground_x,
        speed_attack_x=speed_attack_x,
        speed_attack_y=speed_attack_y,
        state_age=state_age,
        hitlag=hitlag,
        stocks=stocks,
        l_cancel=l_cancel,
        hurtbox_state=hurtbox_state,
        ground=ground,
        last_attack_landed=last_attack_landed,
        combo_count=combo_count,
        main_stick_x=main_x,
        main_stick_y=main_y,
        c_stick_x=c_x,
        c_stick_y=c_y,
        shoulder=shoulder,
        buttons=buttons,
    )


def load_game(path: str | Path, compression: str = "zlib") -> ParsedGame:
    """Load a single parsed game file (zlib-compressed parquet).

    Args:
        path: Path to the parsed game file (md5-named, no extension).
        compression: Compression type ('zlib' or 'none').

    Returns:
        ParsedGame with flat numpy arrays for both players.
    """
    path = Path(path)
    with open(path, "rb") as f:
        data = f.read()

    if compression == "zlib":
        data = zlib.decompress(data)

    table = pq.read_table(io.BytesIO(data))
    root = table.column("root").combine_chunks()
    num_frames = len(root)

    p0 = _extract_player(root.field("p0"), num_frames)
    p1 = _extract_player(root.field("p1"), num_frames)

    # Stage is constant across all frames — take first value
    stage = root.field("stage")[0].as_py()

    return ParsedGame(p0=p0, p1=p1, stage=stage, num_frames=num_frames)


def load_games_from_dir(
    dataset_dir: str | Path,
    max_games: Optional[int] = None,
    stage_filter: Optional[int] = None,
    character_filter: Optional[int] = None,
) -> list[ParsedGame]:
    """Load all parsed games from a slippi_db dataset directory.

    Expected layout:
        dataset_dir/
            meta.json       — array of metadata dicts
            games/          — md5-named parsed game files

    Args:
        dataset_dir: Root of the parsed dataset.
        max_games: Cap on number of games to load.
        stage_filter: Only load games on this stage (melee.Stage value).
        character_filter: Only load games where both players use this character.

    Returns:
        List of ParsedGame objects.
    """
    dataset_dir = Path(dataset_dir)
    meta_path = dataset_dir / "meta.json"

    with open(meta_path) as f:
        meta_list = json.load(f)

    games = []
    for entry in meta_list:
        if not entry.get("is_training", False):
            continue

        if stage_filter is not None and entry.get("stage") != stage_filter:
            continue

        if character_filter is not None:
            players = entry.get("players", [])
            if not all(p.get("character") == character_filter for p in players):
                continue

        compression = entry.get("compression", "zlib")
        game_path = dataset_dir / "games" / entry["slp_md5"]

        if not game_path.exists():
            logger.warning("Game file missing: %s", game_path)
            continue

        try:
            game = load_game(game_path, compression=compression)
            game.meta = entry
            games.append(game)
        except Exception as e:
            logger.warning("Failed to load %s: %s", game_path, e)
            continue

        if max_games and len(games) >= max_games:
            break

    logger.info("Loaded %d games from %s", len(games), dataset_dir)
    return games
