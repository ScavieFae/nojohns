#!/usr/bin/env python3
"""
Netplay stability test - run multiple matches with different characters.

Both sides run this script independently with their own opponent code.
Logs results to a timestamped file for comparison afterwards.

Usage:
    python test_netplay_stability.py --opponent SCAV#382 --label scaviefae
    python test_netplay_stability.py --opponent SCAVIEFAE#123 --label scav
"""

import argparse
import logging
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from melee import Character

# Test configuration - ALL Melee characters!
CHARACTERS = [
    # Top tier
    Character.FOX,
    Character.FALCO,
    Character.MARTH,
    # Character.SHEIK,  # Removed - menu nav can't find it reliably
    Character.JIGGLYPUFF,
    Character.PEACH,
    Character.CPTFALCON,
    Character.PIKACHU,
    Character.SAMUS,
    Character.YLINK,
    # Character.ICECLIMBERS,  # libmelee calls them POPO, and they're weird in netplay
    # Mid tier
    Character.LINK,
    Character.LUIGI,
    Character.GANONDORF,
    Character.DOC,
    Character.MARIO,
    Character.YOSHI,
    Character.DK,
    # Low tier
    Character.ZELDA,
    Character.ROY,
    Character.MEWTWO,
    Character.NESS,
    Character.GAMEANDWATCH,
    Character.BOWSER,
    Character.KIRBY,
    Character.PICHU,
]

DELAY = 6
MATCHES_TO_RUN = 10
SUCCESS_THRESHOLD_SECONDS = 60  # Match is "successful" if it lasts 60+ seconds


def setup_logging(label: str) -> tuple[logging.Logger, Path]:
    """Set up dual logging to console and file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"netplay_test_{label}_{timestamp}.log")

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

    return logger, log_file


def main():
    parser = argparse.ArgumentParser(description="Test netplay stability across characters")
    parser.add_argument("--opponent", required=True, help="Opponent Slippi connect code")
    parser.add_argument("--label", required=True, help="Label for this side (for log file)")
    parser.add_argument(
        "-d",
        "--dolphin",
        required=True,
        help="Path to Slippi Dolphin",
    )
    parser.add_argument(
        "-i",
        "--iso",
        required=True,
        help="Path to Melee ISO",
    )
    parser.add_argument(
        "--dolphin-home",
        default=None,
        help="Dolphin home dir (with Slippi account config)",
    )

    args = parser.parse_args()

    logger, log_file = setup_logging(args.label)

    logger.info("=" * 80)
    logger.info("NETPLAY STABILITY TEST")
    logger.info(f"Label: {args.label}")
    logger.info(f"Opponent: {args.opponent}")
    logger.info(f"Delay: {DELAY} frames")
    logger.info(f"Characters: {len(CHARACTERS)} (cycling)")
    logger.info(f"Target: {MATCHES_TO_RUN} matches initiated")
    logger.info(f"Success threshold: {SUCCESS_THRESHOLD_SECONDS}s in-game")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    # Results tracking
    results = []

    for match_num in range(1, MATCHES_TO_RUN + 1):
        # Random character selection for variety
        char = random.choice(CHARACTERS)
        char_name = char.name

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"MATCH {match_num}/{MATCHES_TO_RUN}: {char_name} (random)")
        logger.info("=" * 80)

        start_real = datetime.now()
        outcome = "UNKNOWN"
        duration_seconds = 0.0
        error_msg = None

        # Run match in subprocess for fresh libmelee state
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "run_single_netplay_match.py",
                    "--opponent", args.opponent,
                    "--character", char_name,
                    "-d", args.dolphin,
                    "-i", args.iso,
                    "--delay", str(DELAY),
                    "--match-num", str(match_num),
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per match
            )

            # Parse result from subprocess output
            for line in result.stdout.split("\n"):
                if line.startswith("RESULT:"):
                    import json
                    result_data = json.loads(line.split("RESULT:", 1)[1])
                    outcome = result_data["outcome"]
                    duration_seconds = result_data["duration_seconds"]
                    break

            # Also show subprocess logs
            if result.stderr:
                logger.info("Match subprocess output:")
                for line in result.stderr.split("\n"):
                    if line.strip():
                        logger.info(f"  {line}")

        except subprocess.TimeoutExpired:
            outcome = "TIMEOUT"
            error_msg = "Match exceeded 10 minute timeout"
            logger.error(f"❌ Match timed out after 10 minutes")

        except Exception as e:
            outcome = "ERROR"
            error_msg = str(e)
            logger.error(f"❌ Subprocess error: {e}", exc_info=True)

        finally:
            end_real = datetime.now()
            elapsed_real = (end_real - start_real).total_seconds()

            # Ensure cleanup between matches
            try:
                subprocess.run(
                    ["killall", "Slippi Dolphin"],
                    capture_output=True,
                    timeout=5,
                )
                # Wait for Dolphin to fully terminate
                for _ in range(10):
                    check = subprocess.run(
                        ["pgrep", "-f", "Slippi Dolphin"],
                        capture_output=True,
                    )
                    if check.returncode != 0:
                        break
                    time.sleep(0.5)

                # Brief pause for cleanup
                time.sleep(5)
            except Exception:
                pass

        # Determine success
        success = duration_seconds >= SUCCESS_THRESHOLD_SECONDS

        # Log result
        result_entry = {
            "match_num": match_num,
            "character": char_name,
            "outcome": outcome,
            "duration_game_seconds": duration_seconds,
            "duration_real_seconds": elapsed_real,
            "success": success,
            "error": error_msg,
            "timestamp": start_real.isoformat(),
        }
        results.append(result_entry)

        logger.info(f"Result: {outcome} | Duration: {duration_seconds:.1f}s | Success: {success}")

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST COMPLETE - SUMMARY")
    logger.info("=" * 80)

    successes = sum(1 for r in results if r["success"])
    logger.info(f"Matches initiated: {len(results)}")
    logger.info(f"Successful (≥60s): {successes}/{len(results)}")
    logger.info("")
    logger.info("Per-character breakdown:")

    for char in CHARACTERS:
        char_results = [r for r in results if r["character"] == char.name]
        if char_results:
            char_successes = sum(1 for r in char_results if r["success"])
            avg_duration = sum(r["duration_game_seconds"] for r in char_results) / len(
                char_results
            )
            logger.info(
                f"  {char.name:15s}: {char_successes}/{len(char_results)} success, "
                f"avg {avg_duration:.1f}s"
            )

    logger.info("")
    logger.info("Detailed results:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        logger.info(
            f"  {status} Match {r['match_num']:2d} | {r['character']:15s} | "
            f"{r['outcome']:20s} | {r['duration_game_seconds']:5.1f}s"
        )

    logger.info("")
    logger.info(f"Full log saved to: {log_file}")


if __name__ == "__main__":
    main()
