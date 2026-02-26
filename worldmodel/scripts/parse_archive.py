#!/usr/bin/env python3
"""Stream-parse .slp replays from zip/7z archives into our parquet format.

Extracts one replay at a time, parses with peppi_py, writes zlib-compressed
parquet to the output directory. Never holds more than one replay in memory,
so disk usage stays minimal regardless of archive size.

Output format matches our existing parsed-v2 layout:
    output_dir/
        meta.json       — array of metadata dicts
        games/          — md5-named zlib-compressed parquet files

Usage:
    # Parse a .zip archive:
    python -m worldmodel.scripts.parse_archive \
        --archive ~/downloads/ranked-anonymized-6-171694.zip \
        --output ~/data/parsed-v2 \
        --workers 4

    # Parse a .7z archive (requires py7zr):
    python -m worldmodel.scripts.parse_archive \
        --archive ~/downloads/ranked-anonymized-1-116248.7z \
        --output ~/data/parsed-v2
"""

import argparse
import gzip
import hashlib
import io
import json
import logging
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

from worldmodel.scripts.item_utils import extract_items, empty_items_pa

# Button bitmasks (matching slippi_db/libmelee convention)
BUTTON_MASKS = {
    "A": 0x0100,
    "B": 0x0200,
    "X": 0x0400,
    "Y": 0x0800,
    "Z": 0x0010,
    "L": 0x0040,
    "R": 0x0020,
    "D_UP": 0x0008,
}


def peppi_to_parquet(game, stage_id: int, slp_path: str | None = None) -> pa.Table:
    """Convert a peppi_py Game to our parquet schema.

    Produces the same column layout as slippi_db's output that our
    parse.py/load_game() expects: a single 'root' column containing
    a struct with p0, p1, stage, randall, fod_platforms, and items fields.
    """
    frames = game.frames
    num_frames = len(frames.id)

    players = {}
    for i, port in enumerate(frames.ports):
        post = port.leader.post
        pre = port.leader.pre
        vel = post.velocities

        # Sticks: peppi gives [-1, 1], our pipeline expects [0, 1] (libmelee)
        joy_x = (pre.joystick.x.to_numpy() / 2.0 + 0.5).astype(np.float32)
        joy_y = (pre.joystick.y.to_numpy() / 2.0 + 0.5).astype(np.float32)
        c_x = (pre.cstick.x.to_numpy() / 2.0 + 0.5).astype(np.float32)
        c_y = (pre.cstick.y.to_numpy() / 2.0 + 0.5).astype(np.float32)
        shoulder = pre.triggers.to_numpy().astype(np.float32)

        # Buttons: bitmask → individual bools
        button_bits = pre.buttons.to_numpy().astype(np.uint32)
        button_arrays = {}
        for btn_name, mask in BUTTON_MASKS.items():
            button_arrays[btn_name] = pa.array(
                (button_bits & mask).astype(bool)
            )

        main_stick = pa.StructArray.from_arrays(
            [pa.array(joy_x), pa.array(joy_y)], names=["x", "y"]
        )
        c_stick = pa.StructArray.from_arrays(
            [pa.array(c_x), pa.array(c_y)], names=["x", "y"]
        )
        controller = pa.StructArray.from_arrays(
            [main_stick, c_stick, pa.array(shoulder),
             pa.StructArray.from_arrays(
                 list(button_arrays.values()),
                 names=list(button_arrays.keys()),
             )],
            names=["main_stick", "c_stick", "shoulder", "buttons"],
        )

        # Hurtbox: handle potential None
        if post.hurtbox_state is not None:
            hurtbox = post.hurtbox_state.to_numpy().astype(np.int64)
        else:
            hurtbox = np.zeros(num_frames, dtype=np.int64)

        # L-cancel: handle potential None
        if post.l_cancel is not None:
            l_cancel = post.l_cancel.to_numpy().astype(np.int64)
        else:
            l_cancel = np.zeros(num_frames, dtype=np.int64)

        # Last attack landed: handle potential None
        if post.last_attack_landed is not None:
            last_attack = post.last_attack_landed.to_numpy().astype(np.int64)
        else:
            last_attack = np.zeros(num_frames, dtype=np.int64)

        player = pa.StructArray.from_arrays(
            [
                pa.array(post.percent.to_numpy().astype(np.float32)),
                pa.array(post.position.x.to_numpy().astype(np.float32)),
                pa.array(post.position.y.to_numpy().astype(np.float32)),
                pa.array(post.shield.to_numpy().astype(np.float32)),
                pa.array((post.direction.to_numpy() > 0).astype(np.float32)),
                pa.array(hurtbox.astype(np.float32) != 0),  # invulnerable
                pa.array(np.logical_not(post.airborne.to_numpy().astype(bool))),
                pa.array(post.state.to_numpy().astype(np.int64)),
                pa.array(post.jumps.to_numpy().astype(np.int64)),
                pa.array(post.character.to_numpy().astype(np.int64)),
                # Velocity
                pa.array(vel.self_x_air.to_numpy().astype(np.float32)),
                pa.array(vel.self_y.to_numpy().astype(np.float32)),
                pa.array(vel.self_x_ground.to_numpy().astype(np.float32)),
                pa.array(vel.knockback_x.to_numpy().astype(np.float32)),
                pa.array(vel.knockback_y.to_numpy().astype(np.float32)),
                # Dynamics
                pa.array(post.state_age.to_numpy().astype(np.float32)),
                pa.array(post.hitlag.to_numpy().astype(np.float32)),
                pa.array(post.stocks.to_numpy().astype(np.float32)),
                # Combat context
                pa.array(l_cancel),
                pa.array(hurtbox),
                pa.array(post.ground.to_numpy().astype(np.int64)),
                pa.array(last_attack),
                pa.array(post.combo_count.to_numpy().astype(np.int64)),
                # Controller
                controller,
            ],
            names=[
                "percent", "x", "y", "shield_strength",
                "facing", "invulnerable", "on_ground",
                "action", "jumps_left", "character",
                "speed_air_x", "speed_y", "speed_ground_x",
                "speed_attack_x", "speed_attack_y",
                "state_age", "hitlag", "stocks",
                "l_cancel", "hurtbox_state", "ground",
                "last_attack_landed", "combo_count",
                "controller",
            ],
        )
        players[f"p{i}"] = player

    stage = pa.array(np.full(num_frames, stage_id, dtype=np.int64))
    randall = pa.StructArray.from_arrays(
        [pa.array(np.zeros(num_frames, dtype=np.float32))] * 2, names=["x", "y"]
    )
    fod = pa.StructArray.from_arrays(
        [pa.array(np.zeros(num_frames, dtype=np.float32))] * 2, names=["left", "right"]
    )

    # Extract items from raw arrow data if path available
    items_pa = None
    if slp_path is not None:
        items_pa = extract_items(slp_path, num_frames)
    if items_pa is None:
        items_pa = empty_items_pa(num_frames)

    root = pa.StructArray.from_arrays(
        [players["p0"], players["p1"], stage, randall, fod, items_pa],
        names=["p0", "p1", "stage", "randall", "fod_platforms", "items"],
    )
    return pa.table({"root": root})


def parse_slp_bytes(slp_bytes: bytes) -> pa.Table | None:
    """Parse raw .slp bytes into a parquet table. Returns None on failure."""
    import peppi_py

    with tempfile.NamedTemporaryFile(suffix=".slp", delete=False) as tf:
        tf.write(slp_bytes)
        tf_path = tf.name

    try:
        game = peppi_py.read_slippi(tf_path)
        if len(game.frames.ports) != 2:
            return None

        stage_id = game.start.stage
        num_frames = len(game.frames.id)
        if num_frames < 60:  # skip very short games
            return None

        return peppi_to_parquet(game, stage_id, slp_path=tf_path)
    except Exception as e:
        logger.debug("Parse failed: %s", e)
        return None
    finally:
        Path(tf_path).unlink(missing_ok=True)


def save_parsed_game(table: pa.Table, output_dir: Path, slp_md5: str) -> int:
    """Write a parsed game as zlib-compressed parquet. Returns byte size."""
    buf = io.BytesIO()
    pq.write_table(table, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw)

    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    out_path = games_dir / slp_md5
    out_path.write_bytes(compressed)
    return len(compressed)


def iter_zip(archive_path: str):
    """Yield (filename, slp_bytes) from a .zip of .slp.gz files."""
    import zipfile

    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".slp.gz"):
                continue
            try:
                gz_data = zf.read(name)
                slp_data = gzip.decompress(gz_data)
                yield name, slp_data
            except Exception as e:
                logger.debug("Skip %s: %s", name, e)
                continue


def iter_7z(archive_path: str, chunk_size: int = 500):
    """Yield (filename, slp_bytes) from a .7z archive.

    Solid 7z archives are slow to random-access, so we extract in chunks
    of chunk_size files at a time. Each chunk decompresses a sequential
    block, keeping memory bounded while avoiding O(n²) re-decompression.
    """
    try:
        import py7zr
    except ImportError:
        logger.error("py7zr not installed. Run: pip install py7zr")
        sys.exit(1)

    with py7zr.SevenZipFile(archive_path, "r") as zf:
        all_names = [n for n in zf.getnames() if n.endswith(".slp.gz") or n.endswith(".slp")]

    # Process in chunks to avoid re-decompressing the whole archive per file
    for chunk_start in range(0, len(all_names), chunk_size):
        chunk_names = all_names[chunk_start:chunk_start + chunk_size]
        try:
            with py7zr.SevenZipFile(archive_path, "r") as zf:
                extracted = zf.read(chunk_names)
        except Exception as e:
            logger.warning("Failed to extract chunk at %d: %s", chunk_start, e)
            continue

        for name in chunk_names:
            if name not in extracted:
                continue
            try:
                data = extracted[name].read()
                if name.endswith(".gz"):
                    data = gzip.decompress(data)
                yield name, data
            except Exception as e:
                logger.debug("Skip %s: %s", name, e)
                continue

        del extracted  # free memory between chunks


def main():
    parser = argparse.ArgumentParser(description="Stream-parse replays from archives")
    parser.add_argument("--archive", required=True, help="Path to .zip or .7z archive")
    parser.add_argument("--output", required=True, help="Output directory (parsed-v2 format)")
    parser.add_argument("--limit", type=int, default=0, help="Max games to parse (0=all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip games already in output")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    archive_path = Path(args.archive).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing metadata if appending
    meta_path = output_dir / "meta.json"
    existing_meta = []
    existing_md5s = set()
    if meta_path.exists():
        with open(meta_path) as f:
            existing_meta = json.load(f)
        existing_md5s = {e["slp_md5"] for e in existing_meta}
        logger.info("Found %d existing games in %s", len(existing_md5s), output_dir)

    # Choose iterator based on extension
    ext = archive_path.suffix.lower()
    if ext == ".zip":
        replay_iter = iter_zip(str(archive_path))
    elif ext == ".7z":
        replay_iter = iter_7z(str(archive_path))
    else:
        logger.error("Unsupported archive format: %s", ext)
        sys.exit(1)

    new_meta = list(existing_meta)
    parsed = 0
    skipped = 0
    failed = 0
    total_bytes = 0
    t0 = time.time()

    for name, slp_bytes in replay_iter:
        # Use md5 of the .slp content as the game ID
        slp_md5 = hashlib.md5(slp_bytes).hexdigest()

        if args.skip_existing and slp_md5 in existing_md5s:
            skipped += 1
            continue

        table = parse_slp_bytes(slp_bytes)
        if table is None:
            failed += 1
            continue

        nbytes = save_parsed_game(table, output_dir, slp_md5)
        total_bytes += nbytes

        num_frames = table.num_rows
        new_meta.append({
            "slp_md5": slp_md5,
            "source_file": name,
            "num_frames": num_frames,
            "stage": table.column("root").combine_chunks().field("stage")[0].as_py(),
            "is_training": True,
            "compression": "zlib",
        })
        existing_md5s.add(slp_md5)
        parsed += 1

        if parsed % 100 == 0:
            elapsed = time.time() - t0
            rate = parsed / elapsed
            logger.info(
                "Parsed %d games (%.1f/s), %d failed, %d skipped, %.1f MB written",
                parsed, rate, failed, skipped, total_bytes / 1e6,
            )
            # Save metadata periodically (crash recovery)
            with open(meta_path, "w") as f:
                json.dump(new_meta, f)

        if args.limit and parsed >= args.limit:
            logger.info("Reached limit of %d games", args.limit)
            break

    # Final metadata save
    with open(meta_path, "w") as f:
        json.dump(new_meta, f)

    elapsed = time.time() - t0
    logger.info(
        "Done: %d parsed, %d failed, %d skipped in %.0fs (%.1f/s). %.1f MB total.",
        parsed, failed, skipped, elapsed, parsed / max(elapsed, 1), total_bytes / 1e6,
    )


if __name__ == "__main__":
    main()
