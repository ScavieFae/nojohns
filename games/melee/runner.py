"""
nojohns/runner.py - Match execution engine

Runs games between two fighters using Dolphin and libmelee.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import melee
from melee import Stage, Character

from nojohns.fighter import Fighter, MatchConfig, FighterConfig, MatchResult, ControllerState

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DolphinConfig:
    """Configuration for Dolphin emulator."""

    dolphin_path: str  # Path to Slippi Dolphin .app or directory
    iso_path: str      # Path to Melee ISO/CISO

    # Display options
    fullscreen: bool = False

    # Paths
    dolphin_home_path: str | None = None  # Dolphin user config dir (None = auto)
    slippi_replay_dir: str | None = None  # Where to save replays


@dataclass
class MatchSettings:
    """Settings for a match (set of games)."""
    
    # Format
    games: int = 1             # Number of games (1, 3, 5 for Bo1/Bo3/Bo5)
    stocks: int = 4
    time_minutes: int = 8
    
    # Stage selection
    stage: Stage = Stage.FINAL_DESTINATION
    # Future: stage_list for counterpicks
    
    # Characters
    p1_character: Character = Character.FOX
    p2_character: Character = Character.FOX

    # CPU levels (0 = human/bot controlled, 1-9 = CPU)
    p1_cpu_level: int = 0
    p2_cpu_level: int = 0


@dataclass
class GameResult:
    """Result of a single game."""
    
    winner_port: int  # 1 or 2
    
    p1_stocks: int
    p2_stocks: int
    p1_damage_dealt: float
    p2_damage_dealt: float
    
    duration_frames: int
    stage: Stage
    
    replay_path: str | None = None


@dataclass
class MatchResult:
    """Result of an entire match (multiple games)."""
    
    winner_port: int  # 1 or 2 (whoever won more games)
    
    games: list[GameResult] = field(default_factory=list)
    
    p1_games_won: int = 0
    p2_games_won: int = 0
    
    @property
    def score(self) -> str:
        return f"{self.p1_games_won}-{self.p2_games_won}"


# ============================================================================
# Match Runner
# ============================================================================

class MatchRunner:
    """
    Runs matches between two fighters.
    
    Usage:
        runner = MatchRunner(dolphin_config)
        result = runner.run_match(fighter1, fighter2, settings)
    """
    
    def __init__(self, dolphin: DolphinConfig):
        self.dolphin = dolphin
        self._console: melee.Console | None = None
        self._controllers: dict[int, melee.Controller] = {}
    
    def run_match(
        self,
        fighter1: Fighter,
        fighter2: Fighter,
        settings: MatchSettings,
        config1: FighterConfig | None = None,
        config2: FighterConfig | None = None,
        on_frame: Callable[[melee.GameState], None] | None = None,
        on_game_end: Callable[[GameResult], None] | None = None,
    ) -> MatchResult:
        """
        Run a complete match between two fighters.
        
        Args:
            fighter1: Fighter for port 1
            fighter2: Fighter for port 2
            settings: Match rules
            config1: Optional config for fighter 1
            config2: Optional config for fighter 2
            on_frame: Callback for each frame (for streaming/commentary)
            on_game_end: Callback after each game
            
        Returns:
            MatchResult with all game outcomes
        """
        result = MatchResult(winner_port=0)
        games_to_win = (settings.games // 2) + 1
        
        logger.info(f"Starting match: {fighter1.metadata.name} vs {fighter2.metadata.name}")
        logger.info(f"Format: Bo{settings.games}, {settings.stocks} stock, {settings.time_minutes} min")
        
        try:
            self._setup_console()
            self._setup_controllers()
            self._launch_and_connect()

            game_num = 0
            while result.p1_games_won < games_to_win and result.p2_games_won < games_to_win:
                game_num += 1
                logger.info(f"Starting game {game_num}")

                game_result = self._run_game(
                    fighter1, fighter2, settings,
                    config1, config2,
                    on_frame,
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
            logger.info(f"Match complete! Winner: P{result.winner_port} ({result.score})")

        finally:
            self._cleanup()
        
        return result
    
    def run_single_game(
        self,
        fighter1: Fighter,
        fighter2: Fighter,
        settings: MatchSettings,
        config1: FighterConfig | None = None,
        config2: FighterConfig | None = None,
        on_frame: Callable[[melee.GameState], None] | None = None,
    ) -> GameResult:
        """Run a single game. Convenience wrapper around run_match."""
        settings.games = 1
        match_result = self.run_match(
            fighter1, fighter2, settings,
            config1, config2,
            on_frame,
        )
        return match_result.games[0]
    
    # ========================================================================
    # Internal Methods
    # ========================================================================
    
    def _setup_console(self) -> None:
        """Initialize Dolphin console."""
        logger.debug("Setting up Dolphin console")

        self._console = melee.Console(
            path=self.dolphin.dolphin_path,
            dolphin_home_path=self.dolphin.dolphin_home_path,
            tmp_home_directory=self.dolphin.dolphin_home_path is None,
            copy_home_directory=False,
            fullscreen=self.dolphin.fullscreen,
            blocking_input=True,
            save_replays=self.dolphin.slippi_replay_dir is not None,
        )

    def _setup_controllers(self) -> None:
        """Set up virtual controllers for both ports."""
        logger.debug("Setting up controllers")

        for port in [1, 2]:
            controller = melee.Controller(
                console=self._console,
                port=port,
                type=melee.ControllerType.STANDARD,
            )
            self._controllers[port] = controller

    def _launch_and_connect(self) -> None:
        """Launch Dolphin and connect console + controllers."""
        logger.info("Launching Dolphin")
        self._console.run(iso_path=self.dolphin.iso_path)

        logger.debug("Connecting to console")
        if not self._console.connect():
            raise RuntimeError("Failed to connect to Dolphin")

        logger.debug("Connecting controllers")
        for port, controller in self._controllers.items():
            if not controller.connect():
                raise RuntimeError(f"Failed to connect controller on port {port}")
    
    def _run_game(
        self,
        fighter1: Fighter,
        fighter2: Fighter,
        settings: MatchSettings,
        config1: FighterConfig | None,
        config2: FighterConfig | None,
        on_frame: Callable[[melee.GameState], None] | None,
    ) -> GameResult:
        """Run a single game."""
        
        # Set up fighters
        match1 = MatchConfig(
            character=settings.p1_character,
            port=1,
            opponent_port=2,
            opponent_character=settings.p2_character,
            stage=settings.stage,
            stocks=settings.stocks,
            time_minutes=settings.time_minutes,
        )
        match2 = MatchConfig(
            character=settings.p2_character,
            port=2,
            opponent_port=1,
            opponent_character=settings.p1_character,
            stage=settings.stage,
            stocks=settings.stocks,
            time_minutes=settings.time_minutes,
        )
        
        fighter1.setup(match1, config1)
        fighter2.setup(match2, config2)

        # Game tracking
        damage_dealt = {1: 0.0, 2: 0.0}
        last_percent = {1: 0.0, 2: 0.0}
        start_frame = None
        game_started = False

        # Main game loop
        while True:
            state = self._console.step()

            if state is None:
                continue

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

                # Get actions from fighters
                try:
                    action1 = fighter1.act(state)
                    action2 = fighter2.act(state)
                except Exception as e:
                    logger.error(f"Fighter error: {e}")
                    action1 = ControllerState()
                    action2 = ControllerState()

                # Apply actions
                action1.to_libmelee(self._controllers[1])
                action2.to_libmelee(self._controllers[2])

                # Frame callback
                if on_frame:
                    on_frame(state)

                # Check for game end
                p1 = state.players.get(1)
                p2 = state.players.get(2)

                if p1 and p2 and (p1.stock == 0 or p2.stock == 0):
                    winner = 1 if p2.stock == 0 else 2

                    result = GameResult(
                        winner_port=winner,
                        p1_stocks=p1.stock,
                        p2_stocks=p2.stock,
                        p1_damage_dealt=damage_dealt[1],
                        p2_damage_dealt=damage_dealt[2],
                        duration_frames=state.frame - (start_frame or 0),
                        stage=settings.stage,
                    )

                    # Notify fighters
                    fighter1.on_game_end(self._to_fighter_result(result, 1))
                    fighter2.on_game_end(self._to_fighter_result(result, 2))

                    return result

                continue

            # Postgame scores — skip through
            if state.menu_state == melee.Menu.POSTGAME_SCORES:
                melee.MenuHelper.skip_postgame(
                    controller=self._controllers[1],
                    gamestate=state,
                )
                continue

            # Any other menu state — navigate toward the game
            self._handle_menu(state, settings)
    
    def _handle_menu(self, state: melee.GameState, settings: MatchSettings) -> None:
        """Navigate menus to start the game."""
        for port, char, cpu in [
            (1, settings.p1_character, settings.p1_cpu_level),
            (2, settings.p2_character, settings.p2_cpu_level),
        ]:
            melee.MenuHelper.menu_helper_simple(
                gamestate=state,
                controller=self._controllers[port],
                character_selected=char,
                stage_selected=settings.stage,
                connect_code="",
                cpu_level=cpu,
                autostart=True,
                swag=False,
            )
    
    def _to_fighter_result(self, game: GameResult, port: int) -> "MatchResult":
        """Convert GameResult to fighter's MatchResult format."""
        from nojohns.fighter import MatchResult as FighterMatchResult
        
        won = game.winner_port == port
        our_stocks = game.p1_stocks if port == 1 else game.p2_stocks
        opp_stocks = game.p2_stocks if port == 1 else game.p1_stocks
        our_damage = game.p1_damage_dealt if port == 1 else game.p2_damage_dealt
        opp_damage = game.p2_damage_dealt if port == 1 else game.p1_damage_dealt
        
        return FighterMatchResult(
            won=won,
            stocks_remaining=our_stocks,
            opponent_stocks=opp_stocks,
            damage_dealt=our_damage,
            damage_taken=opp_damage,
            duration_frames=game.duration_frames,
            replay_path=game.replay_path,
        )
    
    def _cleanup(self) -> None:
        """Clean up Dolphin and controllers."""
        logger.debug("Cleaning up")

        if self._console:
            # libmelee's stop() sends SIGTERM, but Dolphin may ignore it
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

        self._controllers.clear()


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_fight(
    fighter1: Fighter,
    fighter2: Fighter,
    dolphin_path: str,
    iso_path: str,
    dolphin_home_path: str | None = None,
    games: int = 1,
) -> "MatchResult":
    """
    Quick way to run a fight with minimal config.

    Example:
        from nojohns import quick_fight, RandomFighter

        result = quick_fight(
            RandomFighter(),
            RandomFighter(),
            dolphin_path="/path/to/dolphin",
            iso_path="/path/to/melee.iso",
        )
        print(f"Winner: P{result.winner_port}")
    """
    dolphin = DolphinConfig(
        dolphin_path=dolphin_path,
        iso_path=iso_path,
        dolphin_home_path=dolphin_home_path,
    )
    settings = MatchSettings(games=games)
    runner = MatchRunner(dolphin)
    return runner.run_match(fighter1, fighter2, settings)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "DolphinConfig",
    "MatchSettings",
    "GameResult",
    "MatchResult",
    "MatchRunner",
    "quick_fight",
]
