#!/usr/bin/env python3
"""
Quick test to connect to Scav#382 for netplay.
"""

import logging
from pathlib import Path

from nojohns.fighter import RandomFighter
from games.melee.netplay import NetplayConfig, NetplayRunner
import melee

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Paths
DOLPHIN_PATH = Path.home() / 'Library/Application Support/Slippi Launcher/netplay'
ISO_PATH = Path('/Users/queenmab/claude-projects/games/melee/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso')

if __name__ == '__main__':
    print("🌐 Netplay Test: Connecting to Scav#382")
    print(f"Dolphin: {DOLPHIN_PATH}")
    print(f"ISO: {ISO_PATH}")

    # Create fighter
    fighter = RandomFighter()
    print(f"\nFighter: {fighter.metadata.display_name}")

    # Configure netplay
    config = NetplayConfig(
        dolphin_path=str(DOLPHIN_PATH),
        iso_path=str(ISO_PATH),
        opponent_code="SCAV#382",
        character=melee.Character.FOX,
        online_delay=6,
        input_throttle=3,  # Get new input every 3 frames
        fullscreen=False,
    )

    print(f"Opponent: {config.opponent_code}")
    print(f"Character: {config.character.name}")
    print(f"Online delay: {config.online_delay} frames")

    # Run!
    runner = NetplayRunner(config)

    try:
        print("\n" + "=" * 60)
        print("SEARCHING FOR MATCH...")
        print("=" * 60)
        print("\nDolphin should launch and search for Scav#382...")
        print("Press Ctrl+C to cancel\n")

        result = runner.run_netplay(fighter, games=1)

        print("\n" + "=" * 60)
        print("🏆 MATCH COMPLETE!")
        print("=" * 60)
        print(f"\nScore: {result.score}")
        print(f"Games played: {len(result.games)}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
