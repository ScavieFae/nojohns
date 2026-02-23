#!/usr/bin/env python3
"""Build a world model training dataset from raw .slp replay files.

Walks a directory tree, parses every .slp file with peppi_py, converts to
our ParsedGame format, and saves as zlib-compressed parquet + meta.json.

This bypasses slippi_db's batch pipeline (which expects a specific layout)
and works with any directory structure containing .slp files.

Usage:
    python -m worldmodel.scripts.build_dataset \
        --input /path/to/replays \
        --output /path/to/nojohns-training/data/parsed \
        --workers 4

    # Multiple input dirs:
    python -m worldmodel.scripts.build_dataset \
        --input /path/to/tournaments /path/to/phillip-matchups ~/Slippi \
        --output /path/to/parsed
"""

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)


def _pa_to_np(arr, dtype=np.float32) -> np.ndarray:
    """Convert peppi_py PyArrow array to numpy, handling Arrow scalar types."""
    return np.array(arr.to_pylist(), dtype=dtype)


def _to_libmelee_stick(arr) -> np.ndarray:
    """Convert peppi_py stick [-1, 1] → libmelee stick [0, 1]."""
    return _pa_to_np(arr) / 2.0 + 0.5


def _extract_buttons(button_bits_arr) -> dict[str, np.ndarray]:
    """Extract individual button bools from peppi_py physical button bitmask."""
    bits = _pa_to_np(button_bits_arr, dtype=np.uint16)
    return {
        "A": ((bits >> 8) & 1).astype(bool),
        "B": ((bits >> 9) & 1).astype(bool),
        "X": ((bits >> 10) & 1).astype(bool),
        "Y": ((bits >> 11) & 1).astype(bool),
        "Z": ((bits >> 4) & 1).astype(bool),
        "L": ((bits >> 6) & 1).astype(bool),
        "R": ((bits >> 5) & 1).astype(bool),
        "D_UP": ((bits >> 3) & 1).astype(bool),
    }


def _build_player_arrays(port_data, num_frames: int) -> dict:
    """Build numpy arrays for one player from peppi_py PortData."""
    pre = port_data.leader.pre
    post = port_data.leader.post

    # Controller inputs from pre-frame data
    main_x = _to_libmelee_stick(pre.joystick.x)
    main_y = _to_libmelee_stick(pre.joystick.y)
    c_x = _to_libmelee_stick(pre.cstick.x)
    c_y = _to_libmelee_stick(pre.cstick.y)
    # Shoulder: max of L and R physical triggers
    l_trigger = _pa_to_np(pre.triggers_physical.l)
    r_trigger = _pa_to_np(pre.triggers_physical.r)
    shoulder = np.maximum(l_trigger, r_trigger)
    buttons = _extract_buttons(pre.buttons_physical)

    # Post-frame player state
    percent = _pa_to_np(post.percent, dtype=np.uint16)
    x = _pa_to_np(post.position.x)
    y = _pa_to_np(post.position.y)
    action = _pa_to_np(post.state, dtype=np.uint16)
    # direction is float: 1.0 = right, -1.0 = left → bool: True = right
    facing = (_pa_to_np(post.direction) > 0).astype(bool)
    invulnerable = np.zeros(num_frames, dtype=bool)  # Not in peppi_py post
    character = _pa_to_np(post.character, dtype=np.uint8) if post.character is not None else np.zeros(num_frames, dtype=np.uint8)
    jumps_left = _pa_to_np(post.jumps, dtype=np.uint8) if post.jumps is not None else np.zeros(num_frames, dtype=np.uint8)
    shield = _pa_to_np(post.shield) if post.shield is not None else np.full(num_frames, 60.0, dtype=np.float32)
    # airborne is uint8: 0=grounded, nonzero=airborne
    on_ground = (_pa_to_np(post.airborne, dtype=np.uint8) == 0) if post.airborne is not None else np.ones(num_frames, dtype=bool)

    return {
        "percent": percent, "facing": facing, "x": x, "y": y,
        "action": action, "invulnerable": invulnerable,
        "character": character, "jumps_left": jumps_left,
        "shield_strength": shield, "on_ground": on_ground,
        "main_stick_x": main_x, "main_stick_y": main_y,
        "c_stick_x": c_x, "c_stick_y": c_y, "shoulder": shoulder,
        "buttons": buttons,
    }


def _build_parquet_table(game, p0_arrays: dict, p1_arrays: dict, num_frames: int):
    """Build a PyArrow table matching the slippi_db GAME_TYPE schema."""
    import pyarrow as pa

    def make_player_struct(p: dict) -> pa.StructArray:
        btn = pa.StructArray.from_arrays(
            [pa.array(p["buttons"][b]) for b in ["A", "B", "X", "Y", "Z", "L", "R", "D_UP"]],
            names=["A", "B", "X", "Y", "Z", "L", "R", "D_UP"],
        )
        ctrl = pa.StructArray.from_arrays(
            [
                pa.StructArray.from_arrays([pa.array(p["main_stick_x"]), pa.array(p["main_stick_y"])], names=["x", "y"]),
                pa.StructArray.from_arrays([pa.array(p["c_stick_x"]), pa.array(p["c_stick_y"])], names=["x", "y"]),
                pa.array(p["shoulder"]),
                btn,
            ],
            names=["main_stick", "c_stick", "shoulder", "buttons"],
        )
        # Nana stub (empty — we skip Ice Climbers games)
        nana = pa.StructArray.from_arrays(
            [
                pa.array(np.zeros(num_frames, dtype=bool)),
                pa.array(np.zeros(num_frames, dtype=np.uint16)),
                pa.array(np.zeros(num_frames, dtype=bool)),
                pa.array(np.zeros(num_frames, dtype=np.float32)),
                pa.array(np.zeros(num_frames, dtype=np.float32)),
                pa.array(np.zeros(num_frames, dtype=np.uint16)),
                pa.array(np.zeros(num_frames, dtype=bool)),
                pa.array(np.zeros(num_frames, dtype=np.uint8)),
                pa.array(np.zeros(num_frames, dtype=np.uint8)),
                pa.array(np.zeros(num_frames, dtype=np.float32)),
                pa.array(np.zeros(num_frames, dtype=bool)),
            ],
            names=["exists", "percent", "facing", "x", "y", "action", "invulnerable",
                   "character", "jumps_left", "shield_strength", "on_ground"],
        )
        return pa.StructArray.from_arrays(
            [
                pa.array(p["percent"]), pa.array(p["facing"]),
                pa.array(p["x"]), pa.array(p["y"]),
                pa.array(p["action"]), pa.array(p["invulnerable"]),
                pa.array(p["character"]), pa.array(p["jumps_left"]),
                pa.array(p["shield_strength"]), pa.array(p["on_ground"]),
                ctrl, nana,
            ],
            names=["percent", "facing", "x", "y", "action", "invulnerable",
                   "character", "jumps_left", "shield_strength", "on_ground",
                   "controller", "nana"],
        )

    p0_struct = make_player_struct(p0_arrays)
    p1_struct = make_player_struct(p1_arrays)

    # Stage (constant), Randall (zeros), FoD (zeros), Items (zeros)
    stage = pa.array(np.zeros(num_frames, dtype=np.uint8))  # Filled from metadata
    randall = pa.StructArray.from_arrays(
        [pa.array(np.zeros(num_frames, dtype=np.float32))] * 2, names=["x", "y"]
    )
    fod = pa.StructArray.from_arrays(
        [pa.array(np.zeros(num_frames, dtype=np.float32))] * 2, names=["left", "right"]
    )
    # Items: 15 empty slots
    empty_item = pa.StructArray.from_arrays(
        [
            pa.array(np.zeros(num_frames, dtype=bool)),
            pa.array(np.zeros(num_frames, dtype=np.uint16)),
            pa.array(np.zeros(num_frames, dtype=np.uint8)),
            pa.array(np.zeros(num_frames, dtype=np.float32)),
            pa.array(np.zeros(num_frames, dtype=np.float32)),
        ],
        names=["exists", "type", "state", "x", "y"],
    )
    items = pa.StructArray.from_arrays(
        [empty_item] * 15,
        names=[f"item_{i}" for i in range(15)],
    )

    root = pa.StructArray.from_arrays(
        [p0_struct, p1_struct, stage, randall, fod, items],
        names=["p0", "p1", "stage", "randall", "fod_platforms", "items"],
    )
    return pa.table({"root": root})


def parse_single_slp(slp_path: str) -> dict | None:
    """Parse one .slp file and return metadata + game data bytes.

    Returns None on failure. Runs in worker processes.
    """
    try:
        import peppi_py
        import pyarrow.parquet as pq

        # Parse
        peppi_game = peppi_py.read_slippi(slp_path)

        num_ports = len(peppi_game.frames.ports)
        if num_ports != 2:
            return None

        p0_port = peppi_game.frames.ports[0]
        num_frames = len(p0_port.leader.pre.position.x)

        if num_frames < 120:
            return None

        # Build numpy arrays for both players
        p0_arrays = _build_player_arrays(peppi_game.frames.ports[0], num_frames)
        p1_arrays = _build_player_arrays(peppi_game.frames.ports[1], num_frames)

        # Build parquet table
        table = _build_parquet_table(peppi_game, p0_arrays, p1_arrays, num_frames)

        # Write as zlib-compressed parquet
        buf = io.BytesIO()
        pq.write_table(table, buf)
        compressed = zlib.compress(buf.getvalue())

        # Metadata
        start = peppi_game.start
        players = []
        for i in range(num_ports):
            char_arr = peppi_game.frames.ports[i].leader.post.character
            char_val = int(char_arr.to_pylist()[0]) if char_arr is not None else -1
            players.append({"port": i, "character": char_val})

        stage_val = int(start.stage) if start.stage is not None else -1

        with open(slp_path, "rb") as f:
            slp_bytes = f.read()
        md5 = hashlib.md5(slp_bytes).hexdigest()

        return {
            "meta": {
                "name": os.path.basename(slp_path),
                "slp_md5": md5,
                "slp_size": len(slp_bytes),
                "lastFrame": num_frames,
                "num_players": num_ports,
                "players": players,
                "stage": stage_val,
                "valid": True,
                "is_training": True,
                "not_training_reason": "",
                "pq_size": len(compressed),
                "compression": "zlib",
                "source": str(slp_path),
            },
            "data": compressed,
            "md5": md5,
        }

    except Exception as e:
        return None


def find_slp_files(input_dirs: list[str]) -> list[str]:
    """Recursively find all .slp files in the given directories."""
    slp_files = []
    for d in input_dirs:
        d = os.path.expanduser(d)
        if os.path.isfile(d) and d.endswith(".slp"):
            slp_files.append(d)
        elif os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".slp"):
                        slp_files.append(os.path.join(root, f))
    return sorted(set(slp_files))


def main():
    parser = argparse.ArgumentParser(description="Build world model dataset from .slp files")
    parser.add_argument("--input", nargs="+", required=True, help="Input directories containing .slp files")
    parser.add_argument("--output", required=True, help="Output directory for parsed dataset")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Max files to process")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Find all .slp files
    slp_files = find_slp_files(args.input)
    if args.limit:
        slp_files = slp_files[: args.limit]

    logging.info("Found %d .slp files", len(slp_files))
    if not slp_files:
        logging.error("No .slp files found in %s", args.input)
        sys.exit(1)

    # Setup output
    output_dir = Path(args.output)
    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing meta.json to skip already-parsed files
    meta_path = output_dir / "meta.json"
    existing_meta = []
    existing_md5s = set()
    if meta_path.exists():
        with open(meta_path) as f:
            existing_meta = json.load(f)
        existing_md5s = {e["slp_md5"] for e in existing_meta}
        logging.info("Found %d already-parsed games, will skip duplicates", len(existing_md5s))

    # Also check existing game files on disk (handles interrupted runs without meta.json)
    for gf in games_dir.iterdir():
        existing_md5s.add(gf.name)

    # Filter out already-parsed files by checking if game file exists
    slp_to_parse = slp_files  # We'll check md5 after parsing

    # Parse in parallel with timeout
    meta_list = list(existing_meta)
    parsed = 0
    failed = 0
    skipped = 0
    SAVE_INTERVAL = 200  # Save meta.json every N new games

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(parse_single_slp, f): f for f in slp_to_parse}
        total = len(futures)

        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result(timeout=30)  # 30s timeout per file
            except Exception:
                failed += 1
                continue

            if result is None:
                failed += 1
            elif result["md5"] in existing_md5s:
                skipped += 1
            else:
                # Save game data
                game_path = games_dir / result["md5"]
                with open(game_path, "wb") as f:
                    f.write(result["data"])
                meta_list.append(result["meta"])
                existing_md5s.add(result["md5"])
                parsed += 1

            if (i + 1) % 50 == 0 or (i + 1) == total:
                logging.info(
                    "Progress: %d/%d (parsed=%d, skipped=%d, failed=%d)",
                    i + 1, total, parsed, skipped, failed,
                )

            # Incremental save
            if parsed > 0 and parsed % SAVE_INTERVAL == 0:
                with open(meta_path, "w") as f:
                    json.dump(meta_list, f)
                logging.info("Saved meta.json (%d games)", len(meta_list))

    # Final save
    with open(meta_path, "w") as f:
        json.dump(meta_list, f, indent=2)

    logging.info(
        "Done! %d new games parsed, %d skipped, %d failed. Total: %d games in %s",
        parsed, skipped, failed, len(meta_list), output_dir,
    )


if __name__ == "__main__":
    main()
