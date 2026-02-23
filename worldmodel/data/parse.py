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
