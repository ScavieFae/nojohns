#!/usr/bin/env python3
"""
nojohns/cli.py - Command line interface for No Johns

Usage:
    nojohns setup                  # Create ~/.nojohns/ config
    nojohns setup melee            # Configure Melee paths
    nojohns setup melee phillip    # Install Phillip fighter
    nojohns fight <f1> <f2>        # Run a local fight
    nojohns netplay <f> --code X   # Netplay against remote opponent
    nojohns matchmake <f>          # Arena matchmaking
    nojohns list-fighters
    nojohns info <fighter>
"""

import argparse
import logging
import sys
from pathlib import Path

from nojohns.config import (
    CONFIG_DIR,
    CONFIG_PATH,
    GameConfig,
    NojohnsConfig,
    load_config,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Arg Resolution
# ============================================================================

def _resolve_args(args, game_cfg: GameConfig | None, nj_cfg: NojohnsConfig):
    """Merge CLI args with config. CLI wins. Mutates args in place."""
    # dolphin: CLI > config
    if getattr(args, "dolphin", None) is None and game_cfg and game_cfg.dolphin_path:
        args.dolphin = game_cfg.dolphin_path
    # iso: CLI > config
    if getattr(args, "iso", None) is None and game_cfg and game_cfg.iso_path:
        args.iso = game_cfg.iso_path
    # dolphin_home: CLI > config
    if getattr(args, "dolphin_home", None) is None and game_cfg and game_cfg.dolphin_home:
        args.dolphin_home = game_cfg.dolphin_home
    # connect_code for matchmake: CLI > config
    if hasattr(args, "code") and args.code is None and game_cfg and game_cfg.connect_code:
        args.code = game_cfg.connect_code
    # server for matchmake: CLI > config
    if hasattr(args, "server") and args.server is None and nj_cfg.arena_server:
        args.server = nj_cfg.arena_server
    # delay: CLI > config > hardcoded default
    if hasattr(args, "delay") and args.delay is None:
        args.delay = game_cfg.online_delay if game_cfg and game_cfg.online_delay is not None else 6
    # throttle: CLI > config > hardcoded default
    if hasattr(args, "throttle") and args.throttle is None:
        args.throttle = game_cfg.input_throttle if game_cfg and game_cfg.input_throttle is not None else 3


def _require_melee_args(args) -> bool:
    """Validate required Melee args are present after resolution. Returns True if OK."""
    ok = True
    if getattr(args, "dolphin", None) is None:
        logger.error("--dolphin/-d required (run 'nojohns setup melee' or pass explicitly)")
        ok = False
    if getattr(args, "iso", None) is None:
        logger.error("--iso/-i required (run 'nojohns setup melee' or pass explicitly)")
        ok = False
    return ok


# ============================================================================
# Setup Commands
# ============================================================================

def cmd_setup(args):
    """Handle `nojohns setup [melee [phillip]]`."""
    target = args.setup_target

    if not target:
        return _setup_core()
    elif target == ["melee"]:
        return _setup_melee()
    elif target == ["melee", "phillip"]:
        return _setup_melee_phillip()
    else:
        logger.error(f"Unknown setup target: {' '.join(target)}")
        logger.error("Usage: nojohns setup [melee [phillip]]")
        return 1


def _setup_core():
    """Phase 1: Create ~/.nojohns/ directory and bare config."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if CONFIG_PATH.exists():
        print(f"Config already exists: {CONFIG_PATH}")
        print("To reconfigure Melee: nojohns setup melee")
        return 0

    CONFIG_PATH.write_text(
        "# No Johns configuration\n"
        "# See: nojohns setup melee\n"
        "\n"
        "[arena]\n"
        '# server = "http://localhost:8000"\n'
    )

    print(f"Created {CONFIG_PATH}")
    print("Next: run 'nojohns setup melee' to configure Melee")
    return 0


def _setup_melee():
    """Phase 2: Interactive Melee configuration."""
    import tomllib

    # Ensure config dir exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Read existing config if present
    existing_raw = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "rb") as f:
                existing_raw = tomllib.load(f)
        except Exception:
            pass

    # Current values for defaults
    current = existing_raw.get("games", {}).get("melee", {})

    print("Melee Setup")
    print("=" * 40)
    print()

    # Dolphin path
    default_dolphin = current.get(
        "dolphin", "~/Library/Application Support/Slippi Launcher/netplay"
    )
    dolphin = input(f"Dolphin path [{default_dolphin}]: ").strip() or default_dolphin

    # Validate
    dolphin_expanded = str(Path(dolphin).expanduser())
    if not Path(dolphin_expanded).exists():
        print(f"  Warning: {dolphin_expanded} does not exist")

    # ISO path
    default_iso = current.get("iso", "")
    iso_prompt = f"Melee ISO path [{default_iso}]: " if default_iso else "Melee ISO path: "
    iso = input(iso_prompt).strip() or default_iso
    if not iso:
        logger.error("ISO path is required")
        return 1

    iso_expanded = str(Path(iso).expanduser())
    if not Path(iso_expanded).exists():
        print(f"  Warning: {iso_expanded} does not exist")

    # Connect code (optional)
    default_code = current.get("connect_code", "")
    code_prompt = f"Slippi connect code (optional) [{default_code}]: " if default_code else "Slippi connect code (optional): "
    code = input(code_prompt).strip() or default_code

    # Dolphin home (optional)
    default_home = current.get("dolphin_home", "")
    home_prompt = (
        f"Dolphin home dir (optional, for netplay) [{default_home}]: "
        if default_home
        else "Dolphin home dir (optional, for netplay): "
    )
    dolphin_home = input(home_prompt).strip() or default_home

    # Build the melee section
    melee_lines = []
    melee_lines.append(f'dolphin = "{dolphin}"')
    melee_lines.append(f'iso = "{iso}"')
    if code:
        melee_lines.append(f'connect_code = "{code}"')
    if dolphin_home:
        melee_lines.append(f'dolphin_home = "{dolphin_home}"')

    # Delay and throttle — keep existing or use defaults
    delay = current.get("online_delay", 6)
    throttle = current.get("input_throttle", 3)
    melee_lines.append(f"online_delay = {delay}")
    melee_lines.append(f"input_throttle = {throttle}")

    melee_section = "\n".join(melee_lines)

    # Preserve arena section
    arena_section = ""
    arena_data = existing_raw.get("arena", {})
    if arena_data.get("server"):
        arena_section = f'\n[arena]\nserver = "{arena_data["server"]}"\n'
    else:
        arena_section = (
            "\n[arena]\n"
            '# server = "http://localhost:8000"\n'
        )

    # Write config
    config_content = (
        "# No Johns configuration\n"
        "\n"
        "[games.melee]\n"
        f"{melee_section}\n"
        f"{arena_section}"
    )
    CONFIG_PATH.write_text(config_content)

    print()
    print(f"Saved to {CONFIG_PATH}")
    print()
    print("Ready! Try: nojohns fight random do-nothing")
    print("Want Phillip (neural net AI)? Run: nojohns setup melee phillip")
    return 0


def _setup_melee_phillip():
    """Phase 3: Install Phillip fighter dependencies and model."""
    import subprocess

    project_root = Path(__file__).resolve().parent.parent
    fighters_phillip = project_root / "fighters" / "phillip"
    slippi_ai_dir = fighters_phillip / "slippi-ai"
    models_dir = fighters_phillip / "models"
    model_path = models_dir / "all_d21_imitation_v3.pkl"

    # Determine the pip/python to use
    venv_pip = Path(sys.executable).parent / "pip"
    python = sys.executable

    print("Phillip Setup")
    print("=" * 40)
    print()

    # Step 1: Install Python deps
    print("Installing Phillip Python dependencies...")
    result = subprocess.run(
        [str(venv_pip), "install", "-e", f"{project_root}[phillip]"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"pip install failed:\n{result.stderr}")
        return 1
    print("  Python dependencies installed")

    # Step 2: Clone slippi-ai if not present
    if not slippi_ai_dir.exists():
        print("Cloning slippi-ai...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/vladfi1/slippi-ai.git", str(slippi_ai_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"git clone failed:\n{result.stderr}")
            return 1
        print("  slippi-ai cloned")
    else:
        print("  slippi-ai already present")

    # Install slippi-ai
    print("Installing slippi-ai...")
    result = subprocess.run(
        [str(venv_pip), "install", "-e", str(slippi_ai_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"slippi-ai install failed:\n{result.stderr}")
        return 1
    print("  slippi-ai installed")

    # Step 3: Download model weights if not present
    if not model_path.exists():
        print("Downloading model weights (~40 MB)...")
        models_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "curl", "-L", "-o", str(model_path),
                "https://dl.dropbox.com/scl/fi/bppnln3rfktxfdocottuw/all_d21_imitation_v3?rlkey=46yqbsp7vi5222x04qt4npbkq&st=6knz106y&dl=1",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"Model download failed:\n{result.stderr}")
            return 1
        print("  Model downloaded")
    else:
        print("  Model weights already present")

    # Step 4: Verify
    print()
    print("Verifying...")

    # Check TF import
    result = subprocess.run(
        [python, "-c", "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"  {result.stdout.strip()} OK")
    else:
        logger.warning(f"TensorFlow import failed: {result.stderr.strip()}")

    # Check model loads
    result = subprocess.run(
        [
            python, "-c",
            "from slippi_ai import saving; "
            f"s = saving.load_state_from_disk('{model_path}'); "
            "print(f'Model OK (delay={s[\"config\"][\"policy\"][\"delay\"]})')",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"  {result.stdout.strip()}")
    else:
        logger.warning(f"Model load check failed: {result.stderr.strip()}")

    # Check fighter registry
    result = subprocess.run(
        [python, "-m", "nojohns.cli", "list-fighters"],
        capture_output=True,
        text=True,
    )
    if "phillip" in result.stdout.lower():
        print("  Phillip shows up in fighter registry")
    else:
        logger.warning("Phillip not found in fighter registry (may need fighter.toml)")

    print()
    print("Phillip setup complete!")
    return 0


# ============================================================================
# Game Commands
# ============================================================================

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

    logger.info(f"No Johns - Fight!")
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
        logger.info(f"Match Complete!")
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

    # Validate matchmake-specific required args
    if args.code is None:
        logger.error("--code/-c required (run 'nojohns setup melee' or pass explicitly)")
        return 1
    if args.server is None:
        logger.error("--server required (run 'nojohns setup melee' or pass explicitly)")
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
        input_throttle=args.throttle,
        dolphin_home_path=args.dolphin_home,
    )

    logger.info(f"No Johns - Netplay!")
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
        logger.info(f"Netplay Complete!")
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

    logger.info(f"No Johns - Netplay Test (two local Dolphins)")
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
        logger.info(f"Netplay Test Complete!")
        logger.info(f"   Side 1 sees: P{result1.winner_port} won ({result1.score})")
        logger.info(f"   Side 2 sees: P{result2.winner_port} won ({result2.score})")

        return 0

    except KeyboardInterrupt:
        logger.info("\nNetplay test cancelled.")
        return 130
    except Exception as e:
        logger.error(f"Netplay test error: {e}")
        return 1


# ============================================================================
# Non-Game Commands
# ============================================================================

def cmd_list_fighters(args):
    """List available fighters."""
    from nojohns.registry import list_fighters as registry_list

    fighters = registry_list()

    print("\nAvailable Fighters\n")
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

    print(f"\n{info.display_name} v{info.version}")
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


# ============================================================================
# Fighter Loading
# ============================================================================

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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="nojohns",
        description="Melee AI tournaments for Moltbots",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # setup command
    setup_parser = subparsers.add_parser(
        "setup", help="Configure No Johns (setup / setup melee / setup melee phillip)"
    )
    setup_parser.add_argument(
        "setup_target", nargs="*", default=[],
        help="What to set up: nothing=core, melee=game config, 'melee phillip'=install Phillip",
    )
    setup_parser.set_defaults(func=cmd_setup)

    # fight command
    fight_parser = subparsers.add_parser("fight", help="Run a fight between two fighters")
    fight_parser.add_argument("fighter1", help="Fighter for P1")
    fight_parser.add_argument("fighter2", help="Fighter for P2")
    fight_parser.add_argument("--dolphin", "-d", default=None, help="Path to Slippi Dolphin")
    fight_parser.add_argument("--iso", "-i", default=None, help="Path to Melee ISO")
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
    netplay_parser.add_argument("--dolphin", "-d", default=None, help="Path to Slippi Dolphin")
    netplay_parser.add_argument("--iso", "-i", default=None, help="Path to Melee ISO")
    netplay_parser.add_argument("--char", default="FOX", help="Character (default: FOX)")
    netplay_parser.add_argument("--stage", default="FINAL_DESTINATION", help="Stage (default: FINAL_DESTINATION)")
    netplay_parser.add_argument("--games", "-g", type=int, default=1, help="Number of games (default: 1)")
    netplay_parser.add_argument("--stocks", "-s", type=int, default=4, help="Stocks per game (default: 4)")
    netplay_parser.add_argument("--time", "-t", type=int, default=8, help="Time limit in minutes (default: 8)")
    netplay_parser.add_argument("--delay", type=int, default=None, help="Online input delay in frames (default: 6)")
    netplay_parser.add_argument("--throttle", type=int, default=None, help="AI input throttle (default: 3)")
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
    nptest_parser.add_argument("--dolphin", "-d", default=None, help="Path to Slippi Dolphin")
    nptest_parser.add_argument("--iso", "-i", default=None, help="Path to Melee ISO")
    nptest_parser.add_argument("--games", "-g", type=int, default=1, help="Number of games (default: 1)")
    nptest_parser.add_argument("--stage", default="FINAL_DESTINATION", help="Stage (default: FINAL_DESTINATION)")
    nptest_parser.add_argument("--p1-char", default="FOX", help="Side 1 character (default: FOX)")
    nptest_parser.add_argument("--p2-char", default="FOX", help="Side 2 character (default: FOX)")
    nptest_parser.set_defaults(func=cmd_netplay_test)

    # matchmake command
    mm_parser = subparsers.add_parser("matchmake", help="Join arena queue, get matched, play netplay")
    mm_parser.add_argument("fighter", help="Fighter to run")
    mm_parser.add_argument("--code", "-c", default=None, help="Your Slippi connect code (e.g. SCAV#382)")
    mm_parser.add_argument("--server", default=None, help="Arena server URL (e.g. http://localhost:8000)")
    mm_parser.add_argument("--dolphin", "-d", default=None, help="Path to Slippi Dolphin")
    mm_parser.add_argument("--iso", "-i", default=None, help="Path to Melee ISO")
    mm_parser.add_argument("--char", default="FOX", help="Character (default: FOX)")
    mm_parser.add_argument("--stage", default="FINAL_DESTINATION", help="Stage (default: FINAL_DESTINATION)")
    mm_parser.add_argument("--stocks", "-s", type=int, default=4, help="Stocks per game (default: 4)")
    mm_parser.add_argument("--time", "-t", type=int, default=8, help="Time limit in minutes (default: 8)")
    mm_parser.add_argument("--delay", type=int, default=None, help="Online input delay in frames (default: 6)")
    mm_parser.add_argument("--throttle", type=int, default=None, help="AI input throttle (default: 3)")
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

    # Load config and resolve args for game commands
    game_commands = {"fight", "netplay", "netplay-test", "matchmake"}
    if args.command in game_commands:
        nj_cfg = load_config()
        game_cfg = nj_cfg.games.get("melee")
        _resolve_args(args, game_cfg, nj_cfg)

        if not _require_melee_args(args):
            sys.exit(1)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
