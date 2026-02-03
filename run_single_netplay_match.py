#!/usr/bin/env python3
"""
Helper script to run a single netplay match in isolation.

This gets spawned as a subprocess to ensure fresh libmelee state.
"""

import argparse
import logging
import sys
from datetime import datetime

from melee import Character

from nojohns.cli import load_fighter
from games.melee.netplay import NetplayConfig, NetplayRunner, NetplayDisconnectedError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run single netplay match")
    parser.add_argument("--opponent", required=True, help="Opponent connect code")
    parser.add_argument("--character", required=True, help="Character name (e.g. FOX, MARTH)")
    parser.add_argument("-d", "--dolphin", required=True, help="Path to Slippi Dolphin")
    parser.add_argument("-i", "--iso", required=True, help="Path to Melee ISO")
    parser.add_argument("--delay", type=int, default=6, help="Online delay (frames)")
    parser.add_argument("--throttle", type=int, default=3, help="AI input throttle (get new input every N frames)")
    parser.add_argument("--match-num", type=int, default=1, help="Match number for logging")

    args = parser.parse_args()

    # Convert character name to enum
    try:
        char = getattr(Character, args.character)
    except AttributeError:
        logger.error(f"Invalid character: {args.character}")
        sys.exit(1)

    # Load fighter
    fighter = load_fighter("random")
    if not fighter:
        logger.error("Failed to load random fighter")
        sys.exit(1)

    logger.info(f"MATCH {args.match_num}: {args.character}")

    # Configure and run
    config = NetplayConfig(
        dolphin_path=args.dolphin,
        iso_path=args.iso,
        opponent_code=args.opponent,
        character=char,
        online_delay=args.delay,
        input_throttle=args.throttle,
        fullscreen=False,
    )

    runner = NetplayRunner(config)

    start_time = datetime.now()
    outcome = "UNKNOWN"
    duration_seconds = 0.0

    try:
        result = runner.run_netplay(fighter, games=1)

        if result.games:
            game = result.games[0]
            duration_seconds = game.duration_frames / 60.0

        outcome = "COMPLETED"
        logger.info(f"✅ Match completed! Duration: {duration_seconds:.1f}s")

    except NetplayDisconnectedError as e:
        duration_real = (datetime.now() - start_time).total_seconds()
        outcome = "FREEZE/DISCONNECT"
        logger.warning(f"❌ Match failed: {e}")
        logger.info(f"Real duration before freeze: {duration_real:.1f}s")

    except Exception as e:
        outcome = "ERROR"
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)

    # Output result as JSON for parent process to parse
    import json
    result_data = {
        "outcome": outcome,
        "duration_seconds": duration_seconds,
        "character": args.character,
        "match_num": args.match_num,
    }
    print(f"RESULT:{json.dumps(result_data)}")

    # Exit code: 0 for success, 1 for failure
    sys.exit(0 if outcome == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
