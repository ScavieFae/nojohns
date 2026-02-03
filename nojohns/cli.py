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


def cmd_matchmake(args):
    """Join the arena queue, wait for a match, play, and report results."""
    import json
    import time
    import urllib.error
    import urllib.request

    from games.melee import NetplayConfig, NetplayRunner, NetplayDisconnectedError
    from melee import Character, Stage

    fighter = load_fighter(args.fighter)
    if fighter is None:
        return 1

    server = args.server.rstrip("/")

    # --- Helper: HTTP calls via stdlib ---
    def _post(path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{server}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def _get(path: str) -> dict:
        req = urllib.request.Request(f"{server}{path}")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def _delete(path: str) -> dict:
        req = urllib.request.Request(f"{server}{path}", method="DELETE")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    # --- Step 1: Join queue ---
    queue_id = None
    match_id = None
    try:
        try:
            result = _post("/queue/join", {
                "connect_code": args.code,
                "fighter_name": args.fighter,
            })
        except urllib.error.URLError as e:
            logger.error(f"Cannot reach arena server at {server}: {e}")
            return 1

        queue_id = result["queue_id"]
        status = result["status"]

        if status == "matched":
            match_id = result["match_id"]
            opponent_code = result["opponent_code"]
            logger.info(f"Matched! Opponent: {opponent_code}")
        else:
            position = result.get("position", "?")
            logger.info(f"Joined queue (position: {position})")
            logger.info("Waiting for opponent...")

            # --- Step 2: Poll for match ---
            poll_count = 0
            max_polls = 150  # 5 minutes at 2-second intervals
            while status == "waiting" and poll_count < max_polls:
                time.sleep(2)
                poll_count += 1
                try:
                    result = _get(f"/queue/{queue_id}")
                except urllib.error.URLError:
                    logger.warning("Lost connection to arena, retrying...")
                    continue

                status = result["status"]

                if status == "matched":
                    match_id = result["match_id"]
                    opponent_code = result["opponent_code"]
                    logger.info(f"Matched! Opponent: {opponent_code}")
                    break

            if status != "matched":
                logger.error("Queue timed out — no opponent found.")
                _delete(f"/queue/{queue_id}")
                return 1

        # --- Step 3: Run netplay ---
        logger.info("Launching netplay...")

        config = NetplayConfig(
            dolphin_path=args.dolphin,
            iso_path=args.iso,
            opponent_code=opponent_code,
            character=Character[args.char.upper()],
            stage=Stage[args.stage.upper()],
            stocks=args.stocks,
            time_minutes=args.time,
            online_delay=args.delay,
            input_throttle=args.throttle,
            dolphin_home_path=args.dolphin_home,
        )

        runner = NetplayRunner(config)
        start_time = time.time()
        outcome = "COMPLETED"

        try:
            match_result = runner.run_netplay(fighter, games=1)
            duration = time.time() - start_time
            logger.info(f"Match complete! Result: {outcome}")
        except NetplayDisconnectedError:
            duration = time.time() - start_time
            outcome = "DISCONNECT"
            logger.warning(f"Match ended with disconnect after {duration:.1f}s")
        except Exception as e:
            duration = time.time() - start_time
            outcome = "ERROR"
            logger.error(f"Match error: {e}")

        # --- Step 4: Report result ---
        try:
            _post(f"/matches/{match_id}/result", {
                "queue_id": queue_id,
                "outcome": outcome,
                "duration_seconds": round(duration, 1),
            })
            logger.info("Reported result to arena.")
        except urllib.error.URLError as e:
            logger.warning(f"Failed to report result: {e}")

        return 0

    except KeyboardInterrupt:
        logger.info("\nCancelled.")
        return 130
    finally:
        # Always clean up our queue entry so stale entries don't pile up
        if queue_id and match_id is None:
            try:
                _delete(f"/queue/{queue_id}")
            except Exception:
                pass


def cmd_arena(args):
    """Start the arena matchmaking server."""
    try:
        import uvicorn
    except ImportError:
        logger.error("Arena requires extra dependencies: pip install nojohns[arena]")
        return 1

    from arena.server import app

    # Set DB path on app state so lifespan picks it up
    app.state.db_path = args.db
    logger.info(f"Starting arena server on port {args.port} (db: {args.db})")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    return 0


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
    from nojohns.registry import list_fighters as registry_list

    fighters = registry_list()

    print("\n📋 Available Fighters\n")
    print(f"{'Name':<15} {'Type':<12} {'Characters':<20} {'GPU'}")
    print("-" * 60)

    for f in fighters:
        chars = ", ".join(f.characters[:3])
        if len(f.characters) > 3:
            chars += "..."
        gpu = "Yes" if f.hardware.get("gpu_required") else "No"
        print(f"{f.name:<15} {f.fighter_type:<12} {chars:<20} {gpu}")

    print()
    return 0


def cmd_info(args):
    """Show detailed info about a fighter."""
    from nojohns.registry import get_fighter_info, FighterNotFoundError

    info = get_fighter_info(args.fighter)
    if info is None:
        logger.error(f"Unknown fighter: {args.fighter}")
        return 1

    gpu = "Yes" if info.hardware.get("gpu_required") else "No"
    ram = info.hardware.get("min_ram_gb", "?")

    print(f"\n🎮 {info.display_name} v{info.version}")
    print(f"   by {info.author}")
    print()
    print(f"   Type: {info.fighter_type}")
    print(f"   Characters: {', '.join(info.characters)}")
    print(f"   GPU Required: {gpu}")
    print(f"   Min RAM: {ram}GB")
    print(f"   Frame Delay: {info.avg_frame_delay}")
    print()
    print(f"   {info.description}")

    if info.repo_url:
        print(f"\n   Repo: {info.repo_url}")

    print()
    return 0


def load_fighter(name: str):
    """Load a fighter by name via the registry."""
    from nojohns.registry import (
        load_fighter as registry_load,
        FighterNotFoundError,
        FighterLoadError,
    )

    try:
        return registry_load(name)
    except FighterNotFoundError as e:
        logger.error(str(e))
        return None
    except FighterLoadError as e:
        logger.error(str(e))
        return None


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

    # matchmake command
    mm_parser = subparsers.add_parser("matchmake", help="Join arena queue, get matched, play netplay")
    mm_parser.add_argument("fighter", help="Fighter to run")
    mm_parser.add_argument("--code", "-c", required=True, help="Your Slippi connect code (e.g. SCAV#382)")
    mm_parser.add_argument("--server", required=True, help="Arena server URL (e.g. http://localhost:8000)")
    mm_parser.add_argument("--dolphin", "-d", required=True, help="Path to Slippi Dolphin")
    mm_parser.add_argument("--iso", "-i", required=True, help="Path to Melee ISO")
    mm_parser.add_argument("--char", default="FOX", help="Character (default: FOX)")
    mm_parser.add_argument("--stage", default="FINAL_DESTINATION", help="Stage (default: FINAL_DESTINATION)")
    mm_parser.add_argument("--stocks", "-s", type=int, default=4, help="Stocks per game (default: 4)")
    mm_parser.add_argument("--time", "-t", type=int, default=8, help="Time limit in minutes (default: 8)")
    mm_parser.add_argument("--delay", type=int, default=6, help="Online input delay in frames (default: 6)")
    mm_parser.add_argument("--throttle", type=int, default=3, help="AI input throttle (default: 3)")
    mm_parser.add_argument("--dolphin-home", default=None, help="Dolphin home dir (for Slippi account)")
    mm_parser.set_defaults(func=cmd_matchmake)

    # arena command
    arena_parser = subparsers.add_parser("arena", help="Start the matchmaking server")
    arena_parser.add_argument("--port", "-p", type=int, default=8000, help="Server port (default: 8000)")
    arena_parser.add_argument("--db", default="arena.db", help="SQLite database path (default: arena.db)")
    arena_parser.set_defaults(func=cmd_arena)

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
