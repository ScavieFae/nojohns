"""
nojohns/netplay.py - Single-sided Slippi netplay runner

Runs ONE fighter on ONE Dolphin, connecting to a remote opponent via
Slippi direct connect. Each side runs its own NetplayRunner independently.

For local testing, netplay_test() launches two runners in separate threads
with swapped connect codes.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import melee
from melee import Character, Stage

from .fighter import (
    Fighter,
    FighterConfig,
    MatchConfig,
    ControllerState,
    MatchResult as FighterMatchResult,
)
from .runner import GameResult, MatchResult

logger = logging.getLogger(__name__)


# ============================================================================
# Errors
# ============================================================================

class NetplayDisconnectedError(Exception):
    """Raised when the opponent disconnects mid-match."""
    pass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class NetplayConfig:
    """Configuration for a single-sided netplay session."""

    # Required
    dolphin_path: str
    iso_path: str
    opponent_code: str  # Slippi connect code, e.g. "ABCD#123"

    # Match params (each side brings their own character)
    character: Character = Character.FOX
    stage: Stage = Stage.FINAL_DESTINATION
    stocks: int = 4
    time_minutes: int = 8

    # Netplay tuning
    online_delay: int = 2  # Frames of input delay for rollback

    # Paths
    dolphin_home_path: str | None = None
    slippi_replay_dir: str | None = None

    # Display
    fullscreen: bool = False

    # Port — needed for local two-Dolphin testing where both
    # instances run on the same machine and need different ports
    slippi_port: int = 51441


# ============================================================================
# Netplay Runner
# ============================================================================

class NetplayRunner:
    """
    Runs one fighter on one Dolphin instance over Slippi netplay.

    Unlike MatchRunner which controls both sides, NetplayRunner controls
    only the local fighter. The opponent's inputs arrive through Slippi
    automatically.

    Usage:
        config = NetplayConfig(
            dolphin_path="/path/to/dolphin",
            iso_path="/path/to/melee.iso",
            opponent_code="ABCD#123",
        )
        runner = NetplayRunner(config)
        result = runner.run_netplay(my_fighter)
    """

    def __init__(self, config: NetplayConfig):
        self.config = config
        self._console: melee.Console | None = None
        self._controller: melee.Controller | None = None

    def run_netplay(
        self,
        fighter: Fighter,
        fighter_config: FighterConfig | None = None,
        games: int = 1,
        on_frame: Callable[[melee.GameState], None] | None = None,
        on_game_end: Callable[[GameResult], None] | None = None,
    ) -> MatchResult:
        """
        Run a netplay session (one or more games).

        Args:
            fighter: The local fighter to run
            fighter_config: Optional config for the fighter
            games: Number of games (Bo1, Bo3, Bo5)
            on_frame: Callback for each frame
            on_game_end: Callback after each game

        Returns:
            MatchResult with outcomes from our perspective
        """
        result = MatchResult(winner_port=0)
        games_to_win = (games // 2) + 1

        logger.info(f"Starting netplay: {fighter.metadata.display_name}")
        logger.info(f"Opponent code: {self.config.opponent_code}")
        logger.info(f"Format: Bo{games}, {self.config.stocks} stock")

        try:
            self._setup_console()
            self._setup_controller()
            self._launch_and_connect()

            game_num = 0
            while result.p1_games_won < games_to_win and result.p2_games_won < games_to_win:
                game_num += 1
                logger.info(f"Starting game {game_num}")

                game_result = self._run_game(
                    fighter, fighter_config, on_frame,
                )

                result.games.append(game_result)

                if game_result.winner_port == 1:
                    result.p1_games_won += 1
                else:
                    result.p2_games_won += 1

                logger.info(f"Game {game_num} winner: P{game_result.winner_port}")
                logger.info(f"Score: {result.score}")

                if on_game_end:
                    on_game_end(game_result)

            result.winner_port = 1 if result.p1_games_won > result.p2_games_won else 2
            logger.info(f"Netplay complete! Winner: P{result.winner_port} ({result.score})")

        except NetplayDisconnectedError:
            logger.warning("Opponent disconnected")
            raise
        finally:
            self._cleanup()

        return result

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _setup_console(self) -> None:
        """Initialize Dolphin console with netplay settings."""
        logger.debug("Setting up Dolphin console for netplay")

        self._console = melee.Console(
            path=self.config.dolphin_path,
            dolphin_home_path=self.config.dolphin_home_path,
            tmp_home_directory=self.config.dolphin_home_path is None,
            copy_home_directory=False,
            fullscreen=self.config.fullscreen,
            blocking_input=True,
            save_replays=self.config.slippi_replay_dir is not None,
            online_delay=self.config.online_delay,
            slippi_port=self.config.slippi_port,
        )

    def _setup_controller(self) -> None:
        """Set up a single virtual controller on port 1."""
        logger.debug("Setting up controller (port 1)")

        self._controller = melee.Controller(
            console=self._console,
            port=1,
            type=melee.ControllerType.STANDARD,
        )

    def _launch_and_connect(self) -> None:
        """Launch Dolphin and connect console + controller."""
        logger.info("Launching Dolphin for netplay")
        self._console.run(iso_path=self.config.iso_path)

        logger.debug("Connecting to console")
        if not self._console.connect():
            raise RuntimeError("Failed to connect to Dolphin")

        logger.debug("Connecting controller")
        if not self._controller.connect():
            raise RuntimeError("Failed to connect controller on port 1")

    def _run_game(
        self,
        fighter: Fighter,
        fighter_config: FighterConfig | None,
        on_frame: Callable[[melee.GameState], None] | None,
    ) -> GameResult:
        """Run a single game with one local fighter."""

        # Set up fighter — local player is always port 1
        match_config = MatchConfig(
            character=self.config.character,
            port=1,
            opponent_port=2,
            stage=self.config.stage,
            stocks=self.config.stocks,
            time_minutes=self.config.time_minutes,
        )
        fighter.setup(match_config, fighter_config)

        # Game tracking
        damage_dealt = {1: 0.0, 2: 0.0}
        last_percent = {1: 0.0, 2: 0.0}
        start_frame = None
        game_started = False
        consecutive_nones = 0

        # Main game loop
        while True:
            state = self._console.step()

            if state is None:
                consecutive_nones += 1
                if consecutive_nones > 600:  # ~10 seconds at 60fps
                    raise NetplayDisconnectedError(
                        "Lost connection (no game state for ~10 seconds)"
                    )
                continue
            consecutive_nones = 0

            # In game — the hot path
            if state.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                if not game_started:
                    game_started = True
                    start_frame = state.frame
                    logger.info("Game started")

                # Track damage dealt
                for port in [1, 2]:
                    player = state.players.get(port)
                    if player:
                        opponent_port = 2 if port == 1 else 1
                        if player.percent > last_percent[port]:
                            damage_dealt[opponent_port] += player.percent - last_percent[port]
                        last_percent[port] = player.percent

                # Get action from our fighter only
                try:
                    action = fighter.act(state)
                except Exception as e:
                    logger.error(f"Fighter error: {e}")
                    action = ControllerState()

                # Apply to our controller (port 1 only)
                action.to_libmelee(self._controller)

                # Frame callback
                if on_frame:
                    on_frame(state)

                # Check for game end
                p1 = state.players.get(1)
                p2 = state.players.get(2)

                if p1 and p2 and (p1.stock == 0 or p2.stock == 0):
                    winner = 1 if p2.stock == 0 else 2

                    game_result = GameResult(
                        winner_port=winner,
                        p1_stocks=p1.stock,
                        p2_stocks=p2.stock,
                        p1_damage_dealt=damage_dealt[1],
                        p2_damage_dealt=damage_dealt[2],
                        duration_frames=state.frame - (start_frame or 0),
                        stage=self.config.stage,
                    )

                    # Notify fighter
                    fighter.on_game_end(self._to_fighter_result(game_result))

                    return game_result

                continue

            # Postgame scores — skip through
            if state.menu_state == melee.Menu.POSTGAME_SCORES:
                melee.MenuHelper.skip_postgame(
                    controller=self._controller,
                    gamestate=state,
                )
                continue

            # Any other menu state — navigate toward online match
            self._handle_menu(state)

    def _handle_menu(self, state: melee.GameState) -> None:
        """Navigate menus via Slippi direct connect.

        The key difference from MatchRunner: we pass the opponent's connect
        code to menu_helper_simple, which navigates MAIN_MENU -> ONLINE_PLAY
        -> DIRECT -> enter code -> SLIPPI_ONLINE_CSS instead of local VS.
        """
        melee.MenuHelper.menu_helper_simple(
            gamestate=state,
            controller=self._controller,
            character_selected=self.config.character,
            stage_selected=self.config.stage,
            connect_code=self.config.opponent_code,
            cpu_level=0,
            autostart=True,
            swag=False,
        )

    def _to_fighter_result(self, game: GameResult) -> FighterMatchResult:
        """Convert GameResult to fighter's MatchResult format.

        Local player is always port 1.
        """
        won = game.winner_port == 1
        return FighterMatchResult(
            won=won,
            stocks_remaining=game.p1_stocks,
            opponent_stocks=game.p2_stocks,
            damage_dealt=game.p1_damage_dealt,
            damage_taken=game.p2_damage_dealt,
            duration_frames=game.duration_frames,
            replay_path=game.replay_path,
        )

    def _cleanup(self) -> None:
        """Clean up Dolphin and controller."""
        logger.debug("Cleaning up netplay session")

        if self._console:
            process = self._console._process
            self._console.stop()

            if process is not None:
                try:
                    process.wait(timeout=5)
                except Exception:
                    logger.warning("Dolphin didn't exit after SIGTERM, sending SIGKILL")
                    process.kill()
                    process.wait()

            self._console = None

        self._controller = None


# ============================================================================
# Local Two-Dolphin Test
# ============================================================================

def netplay_test(
    fighter1: Fighter,
    fighter2: Fighter,
    dolphin_path: str,
    iso_path: str,
    code1: str,
    code2: str,
    home1: str,
    home2: str,
    games: int = 1,
    character1: Character = Character.FOX,
    character2: Character = Character.FOX,
    stage: Stage = Stage.FINAL_DESTINATION,
) -> tuple[MatchResult, MatchResult]:
    """
    Run two fighters on two local Dolphins connected via Slippi direct.

    Each Dolphin home dir must have Slippi Launcher configured with a
    different account (the connect codes come from there).

    Args:
        fighter1: Fighter for side 1
        fighter2: Fighter for side 2
        dolphin_path: Path to Slippi Dolphin
        iso_path: Path to Melee ISO
        code1: Slippi connect code for side 1's account
        code2: Slippi connect code for side 2's account
        home1: Dolphin home dir for side 1 (with Slippi account)
        home2: Dolphin home dir for side 2 (with Slippi account)
        games: Number of games
        character1: Side 1's character
        character2: Side 2's character
        stage: Stage to play on

    Returns:
        Tuple of (side1_result, side2_result)
    """
    # Side 1 connects to side 2's code, and vice versa
    config1 = NetplayConfig(
        dolphin_path=dolphin_path,
        iso_path=iso_path,
        opponent_code=code2,  # Side 1 connects to side 2
        character=character1,
        stage=stage,
        dolphin_home_path=home1,
        slippi_port=51441,
    )

    config2 = NetplayConfig(
        dolphin_path=dolphin_path,
        iso_path=iso_path,
        opponent_code=code1,  # Side 2 connects to side 1
        character=character2,
        stage=stage,
        dolphin_home_path=home2,
        slippi_port=51442,  # Different port for second instance
    )

    runner1 = NetplayRunner(config1)
    runner2 = NetplayRunner(config2)

    results: dict[int, MatchResult | Exception] = {}

    def _run_side(side: int, runner: NetplayRunner, fighter: Fighter):
        try:
            results[side] = runner.run_netplay(fighter, games=games)
        except Exception as e:
            results[side] = e

    t1 = threading.Thread(target=_run_side, args=(1, runner1, fighter1))
    t2 = threading.Thread(target=_run_side, args=(2, runner2, fighter2))

    logger.info("Launching two-Dolphin netplay test")
    t1.start()
    # Stagger slightly so Dolphins don't race on resources
    time.sleep(2)
    t2.start()

    t1.join()
    t2.join()

    # Check for errors
    for side in [1, 2]:
        if isinstance(results.get(side), Exception):
            raise RuntimeError(f"Side {side} failed: {results[side]}") from results[side]

    return results[1], results[2]


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "NetplayConfig",
    "NetplayRunner",
    "NetplayDisconnectedError",
    "netplay_test",
]
