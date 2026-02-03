#!/usr/bin/env python3
"""
Run Phillip vs Random fighter.

Usage (from project root):
    .venv/bin/python scripts/run_phillip_vs_random.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path so nojohns/fighters/games are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

print("Importing modules...")
from fighters.phillip import PhillipFighter, PhillipConfig
from nojohns.fighter import RandomFighter
from games.melee import DolphinConfig, MatchSettings, MatchRunner
import melee

print("✅ Imports successful!")

# Configuration
MODEL_PATH = PROJECT_ROOT / 'fighters/phillip/models/all_d21_imitation_v3.pkl'
DOLPHIN_PATH = Path.home() / 'Library/Application Support/Slippi Launcher/netplay'
ISO_PATH = Path('/Users/queenmab/claude-projects/games/melee/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso')

print(f"\nConfiguration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Dolphin: {DOLPHIN_PATH}")
print(f"  ISO: {ISO_PATH}")

# Verify files exist
if not MODEL_PATH.exists():
    print(f"\n❌ Model not found at {MODEL_PATH}")
    print("Download from: https://dl.dropbox.com/scl/fi/bppnln3rfktxfdocottuw/all_d21_imitation_v3?rlkey=46yqbsp7vi5222x04qt4npbkq&st=6knz106y&dl=1")
    sys.exit(1)

dolphin_check = DOLPHIN_PATH / 'Slippi Dolphin.app'
if not dolphin_check.exists():
    print(f"\n❌ Dolphin not found at {dolphin_check}")
    sys.exit(1)

if not ISO_PATH.exists():
    print(f"\n❌ ISO not found at {ISO_PATH}")
    sys.exit(1)

print("\n✅ All files found!")

def main():
    # Create fighters
    print("\nCreating fighters...")

    try:
        phillip_config = PhillipConfig(
            model_path=MODEL_PATH,
            async_inference=True,
            use_gpu=False,
        )
        phillip = PhillipFighter(phillip_config)
        print(f"  ✅ Phillip: {phillip.metadata.name}")
        print(f"     {phillip.metadata.description}")
    except Exception as e:
        print(f"  ❌ Failed to create Phillip: {e}")
        import traceback
        traceback.print_exc()
        return 1

    random_fighter = RandomFighter()
    print(f"  ✅ Random: {random_fighter.metadata.name}")

    # Set up match
    print("\nSetting up match...")

    dolphin_config = DolphinConfig(
        dolphin_path=str(DOLPHIN_PATH),
        iso_path=str(ISO_PATH),
    )

    settings = MatchSettings(
        games=1,
        stocks=4,
        time_minutes=8,
        stage=melee.Stage.FINAL_DESTINATION,
        p1_character=melee.Character.FOX,
        p2_character=melee.Character.FOX,
    )

    print(f"  Stage: {settings.stage.name}")
    print(f"  Format: Best of {settings.games}, {settings.stocks} stocks")
    print(f"  P1 (Phillip): {settings.p1_character.name}")
    print(f"  P2 (Random): {settings.p2_character.name}")

    # Create runner
    runner = MatchRunner(dolphin_config)

    # Run the match!
    print("\n" + "=" * 60)
    print("STARTING MATCH: Phillip vs Random")
    print("=" * 60)
    print("\nDolphin should launch... Press Ctrl+C to stop\n")

    try:
        def on_game_end(game):
            print(f"\n🏁 Game complete!")
            print(f"   Winner: P{game.winner_port}")
            print(f"   Final stocks: P1={game.p1_stocks}, P2={game.p2_stocks}")

        result = runner.run_match(
            phillip,
            random_fighter,
            settings,
            on_game_end=on_game_end,
        )

        print("\n" + "=" * 60)
        print("🏆 MATCH COMPLETE!")
        print("=" * 60)
        print(f"\nWinner: P{result.winner_port}")
        print(f"Score: {result.score}")
        print(f"Total games: {len(result.games)}")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Match interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\n❌ Match failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
