#!/usr/bin/env python3
"""
nojohns/cli.py - Command line interface for No Johns

Usage:
    nojohns fight <fighter1> <fighter2> [options]
    nojohns list-fighters
    nojohns info <fighter>
"""

import argparse
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_fight(args):
    """Run a fight between two fighters."""
    from nojohns import (
        DolphinConfig,
        MatchSettings,
        MatchRunner,
        DoNothingFighter,
        RandomFighter,
    )
    from melee import Character, Stage
    
    # Load fighters
    fighter1 = load_fighter(args.fighter1)
    fighter2 = load_fighter(args.fighter2)
    
    if fighter1 is None or fighter2 is None:
        return 1
    
    # Set up Dolphin
    dolphin = DolphinConfig(
        dolphin_path=args.dolphin,
        iso_path=args.iso,
    )
    
    # Match settings
    settings = MatchSettings(
        games=args.games,
        stocks=args.stocks,
        time_minutes=args.time,
        stage=Stage[args.stage.upper()],
        p1_character=Character[args.p1_char.upper()],
        p2_character=Character[args.p2_char.upper()],
    )
    
    logger.info(f"🎮 No Johns - Fight!")
    logger.info(f"   P1: {fighter1.metadata.display_name} ({settings.p1_character.name})")
    logger.info(f"   P2: {fighter2.metadata.display_name} ({settings.p2_character.name})")
    logger.info(f"   Format: Bo{settings.games}, {settings.stocks} stock, {settings.stage.name}")
    logger.info("")
    
    # Run match
    runner = MatchRunner(dolphin)
    
    def on_game_end(game):
        logger.info(f"   Game over! P{game.winner_port} wins ({game.p1_stocks}-{game.p2_stocks})")
    
    try:
        result = runner.run_match(
            fighter1, fighter2, settings,
            on_game_end=on_game_end,
        )
        
        logger.info("")
        logger.info(f"🏆 Match Complete!")
        logger.info(f"   Winner: P{result.winner_port} ({result.score})")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nMatch cancelled.")
        return 130
    except Exception as e:
        logger.error(f"Match error: {e}")
        return 1


def cmd_list_fighters(args):
    """List available fighters."""
    fighters = get_available_fighters()
    
    print("\n📋 Available Fighters\n")
    print(f"{'Name':<15} {'Type':<12} {'Characters':<20} {'GPU'}")
    print("-" * 60)
    
    for f in fighters:
        chars = ", ".join(c.name for c in f.characters[:3])
        if len(f.characters) > 3:
            chars += "..."
        gpu = "Yes" if f.gpu_required else "No"
        print(f"{f.name:<15} {f.fighter_type:<12} {chars:<20} {gpu}")
    
    print()
    return 0


def cmd_info(args):
    """Show detailed info about a fighter."""
    fighter = load_fighter(args.fighter)
    if fighter is None:
        return 1
    
    meta = fighter.metadata
    
    print(f"\n🎮 {meta.display_name} v{meta.version}")
    print(f"   by {meta.author}")
    print()
    print(f"   Type: {meta.fighter_type}")
    print(f"   Characters: {', '.join(c.name for c in meta.characters)}")
    print(f"   GPU Required: {'Yes' if meta.gpu_required else 'No'}")
    print(f"   Min RAM: {meta.min_ram_gb}GB")
    print(f"   Frame Delay: {meta.avg_frame_delay}")
    print()
    print(f"   {meta.description}")
    
    if meta.repo_url:
        print(f"\n   Repo: {meta.repo_url}")
    
    print()
    return 0


def load_fighter(name: str):
    """Load a fighter by name."""
    from nojohns import DoNothingFighter, RandomFighter
    
    # Built-in fighters
    builtins = {
        "do-nothing": DoNothingFighter,
        "random": RandomFighter,
    }
    
    if name in builtins:
        return builtins[name]()
    
    # Try to load from fighters/ directory
    # TODO: Implement fighter registry
    
    logger.error(f"Unknown fighter: {name}")
    logger.error(f"Available: {', '.join(builtins.keys())}")
    return None


def get_available_fighters():
    """Get metadata for all available fighters."""
    from nojohns import DoNothingFighter, RandomFighter
    
    return [
        DoNothingFighter().metadata,
        RandomFighter().metadata,
    ]


def main():
    parser = argparse.ArgumentParser(
        prog="nojohns",
        description="Melee AI tournaments for Moltbots",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # fight command
    fight_parser = subparsers.add_parser("fight", help="Run a fight between two fighters")
    fight_parser.add_argument("fighter1", help="Fighter for P1")
    fight_parser.add_argument("fighter2", help="Fighter for P2")
    fight_parser.add_argument("--dolphin", "-d", required=True, help="Path to Slippi Dolphin")
    fight_parser.add_argument("--iso", "-i", required=True, help="Path to Melee ISO")
    fight_parser.add_argument("--games", "-g", type=int, default=1, help="Number of games (default: 1)")
    fight_parser.add_argument("--stocks", "-s", type=int, default=4, help="Stocks per game (default: 4)")
    fight_parser.add_argument("--time", "-t", type=int, default=8, help="Time limit in minutes (default: 8)")
    fight_parser.add_argument("--stage", default="FINAL_DESTINATION", help="Stage (default: FINAL_DESTINATION)")
    fight_parser.add_argument("--p1-char", default="FOX", help="P1 character (default: FOX)")
    fight_parser.add_argument("--p2-char", default="FOX", help="P2 character (default: FOX)")
    fight_parser.add_argument("--headless", action="store_true", help="Run without display (faster)")
    fight_parser.set_defaults(func=cmd_fight)
    
    # list-fighters command
    list_parser = subparsers.add_parser("list-fighters", help="List available fighters")
    list_parser.set_defaults(func=cmd_list_fighters)
    
    # info command
    info_parser = subparsers.add_parser("info", help="Show info about a fighter")
    info_parser.add_argument("fighter", help="Fighter name")
    info_parser.set_defaults(func=cmd_info)
    
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
