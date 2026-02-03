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
    from games.melee import (
        DolphinConfig,
        MatchSettings,
        MatchRunner,
    )
    from nojohns import DoNothingFighter, RandomFighter
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


def cmd_netplay(args):
    """Run a fighter over Slippi netplay against a remote opponent."""
    from games.melee import NetplayConfig, NetplayRunner
    from melee import Character, Stage

    fighter = load_fighter(args.fighter)
    if fighter is None:
        return 1

    config = NetplayConfig(
        dolphin_path=args.dolphin,
        iso_path=args.iso,
        opponent_code=args.code,
        character=Character[args.char.upper()],
        stage=Stage[args.stage.upper()],
        stocks=args.stocks,
        time_minutes=args.time,
        online_delay=args.delay,
        dolphin_home_path=args.dolphin_home,
    )

    logger.info(f"🎮 No Johns - Netplay!")
    logger.info(f"   Fighter: {fighter.metadata.display_name} ({config.character.name})")
    logger.info(f"   Opponent code: {config.opponent_code}")
    logger.info(f"   Format: Bo{args.games}, {config.stocks} stock, {config.stage.name}")
    logger.info("")

    runner = NetplayRunner(config)

    def on_game_end(game):
        logger.info(f"   Game over! P{game.winner_port} wins ({game.p1_stocks}-{game.p2_stocks})")

    try:
        result = runner.run_netplay(
            fighter,
            games=args.games,
            on_game_end=on_game_end,
        )

        logger.info("")
        logger.info(f"🏆 Netplay Complete!")
        logger.info(f"   Winner: P{result.winner_port} ({result.score})")

        return 0

    except KeyboardInterrupt:
        logger.info("\nNetplay cancelled.")
        return 130
    except Exception as e:
        logger.error(f"Netplay error: {e}")
        return 1


def cmd_netplay_test(args):
    """Run two fighters on two local Dolphins connected via Slippi."""
    from games.melee import netplay_test
    from melee import Character, Stage

    fighter1 = load_fighter(args.fighter1)
    fighter2 = load_fighter(args.fighter2)

    if fighter1 is None or fighter2 is None:
        return 1

    logger.info(f"🎮 No Johns - Netplay Test (two local Dolphins)")
    logger.info(f"   Side 1: {fighter1.metadata.display_name} (code: {args.code1})")
    logger.info(f"   Side 2: {fighter2.metadata.display_name} (code: {args.code2})")
    logger.info(f"   Format: Bo{args.games}")
    logger.info("")

    try:
        result1, result2 = netplay_test(
            fighter1=fighter1,
            fighter2=fighter2,
            dolphin_path=args.dolphin,
            iso_path=args.iso,
            code1=args.code1,
            code2=args.code2,
            home1=args.home1,
            home2=args.home2,
            games=args.games,
            character1=Character[args.p1_char.upper()],
            character2=Character[args.p2_char.upper()],
            stage=Stage[args.stage.upper()],
        )

        logger.info("")
        logger.info(f"🏆 Netplay Test Complete!")
        logger.info(f"   Side 1 sees: P{result1.winner_port} won ({result1.score})")
        logger.info(f"   Side 2 sees: P{result2.winner_port} won ({result2.score})")

        return 0

    except KeyboardInterrupt:
        logger.info("\nNetplay test cancelled.")
        return 130
    except Exception as e:
        logger.error(f"Netplay test error: {e}")
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

    # SmashBot — look for the clone next to the adapter
    if name == "smashbot":
        from fighters.smashbot import SmashBotFighter

        default_path = Path(__file__).resolve().parent.parent / "fighters" / "smashbot" / "SmashBot"
        if not default_path.is_dir():
            logger.error(f"SmashBot not found at {default_path}")
            logger.error("Clone it: git clone https://github.com/altf4/SmashBot fighters/smashbot/SmashBot")
            return None
        return SmashBotFighter(str(default_path))

    # TODO: Implement fighter registry for arbitrary fighters

    logger.error(f"Unknown fighter: {name}")
    logger.error(f"Available: {', '.join(list(builtins.keys()) + ['smashbot'])}")
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

    # netplay command
    netplay_parser = subparsers.add_parser("netplay", help="Run a fighter over Slippi netplay")
    netplay_parser.add_argument("fighter", help="Fighter to run")
    netplay_parser.add_argument("--code", "-c", required=True, help="Opponent's Slippi connect code (e.g. ABCD#123)")
    netplay_parser.add_argument("--dolphin", "-d", required=True, help="Path to Slippi Dolphin")
    netplay_parser.add_argument("--iso", "-i", required=True, help="Path to Melee ISO")
    netplay_parser.add_argument("--char", default="FOX", help="Character (default: FOX)")
    netplay_parser.add_argument("--stage", default="FINAL_DESTINATION", help="Stage (default: FINAL_DESTINATION)")
    netplay_parser.add_argument("--games", "-g", type=int, default=1, help="Number of games (default: 1)")
    netplay_parser.add_argument("--stocks", "-s", type=int, default=4, help="Stocks per game (default: 4)")
    netplay_parser.add_argument("--time", "-t", type=int, default=8, help="Time limit in minutes (default: 8)")
    netplay_parser.add_argument("--delay", type=int, default=2, help="Online input delay in frames (default: 2)")
    netplay_parser.add_argument("--dolphin-home", default=None, help="Dolphin home dir (for Slippi account)")
    netplay_parser.set_defaults(func=cmd_netplay)

    # netplay-test command
    nptest_parser = subparsers.add_parser("netplay-test", help="Test two fighters on two local Dolphins via Slippi")
    nptest_parser.add_argument("fighter1", help="Fighter for side 1")
    nptest_parser.add_argument("fighter2", help="Fighter for side 2")
    nptest_parser.add_argument("--code1", required=True, help="Slippi connect code for side 1")
    nptest_parser.add_argument("--code2", required=True, help="Slippi connect code for side 2")
    nptest_parser.add_argument("--home1", required=True, help="Dolphin home dir for side 1")
    nptest_parser.add_argument("--home2", required=True, help="Dolphin home dir for side 2")
    nptest_parser.add_argument("--dolphin", "-d", required=True, help="Path to Slippi Dolphin")
    nptest_parser.add_argument("--iso", "-i", required=True, help="Path to Melee ISO")
    nptest_parser.add_argument("--games", "-g", type=int, default=1, help="Number of games (default: 1)")
    nptest_parser.add_argument("--stage", default="FINAL_DESTINATION", help="Stage (default: FINAL_DESTINATION)")
    nptest_parser.add_argument("--p1-char", default="FOX", help="Side 1 character (default: FOX)")
    nptest_parser.add_argument("--p2-char", default="FOX", help="Side 2 character (default: FOX)")
    nptest_parser.set_defaults(func=cmd_netplay_test)

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
