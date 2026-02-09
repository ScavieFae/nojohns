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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import melee
from melee import Character, Stage

# Optional: httpx for streaming (not required for basic netplay)
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from nojohns.fighter import (
    Fighter,
    FighterConfig,
    MatchConfig,
    ControllerState,
    MatchResult as FighterMatchResult,
)
from .menu_navigation import SlippiMenuNavigator
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
    connect_code: str | None = None  # Our own connect code (for port detection)

    # Match params (each side brings their own character)
    character: Character = Character.FOX
    stage: Stage = Stage.FINAL_DESTINATION
    stocks: int = 4
    time_minutes: int = 8

    # Netplay tuning
    online_delay: int = 2  # Frames of input delay for rollback
    input_throttle: int = 1  # Only get new AI input every N frames (1=every frame, 3=every 3rd frame)
    max_game_seconds: int = 480  # Auto-end game after this many seconds (0=no limit)

    # Paths
    dolphin_home_path: str | None = None
    slippi_replay_dir: str | None = None

    # Display
    fullscreen: bool = False
    headless: bool = False  # Run without graphics (gfx_backend=Null, disable_audio=True)

    # Port — needed for local two-Dolphin testing where both
    # instances run on the same machine and need different ports
    slippi_port: int = 51441

    # Live streaming (optional) — stream frame data to arena for spectators
    arena_url: str | None = None  # e.g. "http://localhost:8000"
    match_id: str | None = None   # Arena match ID for streaming
    stream_throttle: int = 1      # Stream every Nth frame (1 = 60fps, 2 = 30fps)


# ============================================================================
# Match Streamer (for live spectating)
# ============================================================================


class MatchStreamer:
    """Streams match data to arena for live spectating.

    Posts frame data to arena HTTP endpoints. Arena broadcasts to
    connected WebSocket viewers.

    This runs in a background thread to avoid blocking the game loop.
    """

    def __init__(self, arena_url: str, match_id: str):
        self.arena_url = arena_url.rstrip("/")
        self.match_id = match_id
        self._client: "httpx.Client | None" = None
        self._queue: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        """Start the background streaming thread."""
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not installed — live streaming disabled")
            return

        self._client = httpx.Client(timeout=2.0)
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        logger.info(f"Match streamer started for {self.match_id}")

    def stop(self):
        """Stop streaming and clean up."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._client:
            self._client.close()
            self._client = None

    def send_match_start(self, stage_id: int, players: list[dict]):
        """Send match_start message."""
        logger.info(f"Sending match_start: stage={stage_id}, players={len(players)}")
        self._post("stream/start", {
            "stage_id": stage_id,
            "players": players,
        })

    def send_frame(self, frame: int, players: list[dict]):
        """Queue a frame for streaming."""
        with self._lock:
            # Only keep latest frame to avoid queue buildup
            self._queue = [{"frame": frame, "players": players}]

    def send_game_end(self, game_number: int, winner_port: int, end_method: str = "stocks"):
        """Send game_end message."""
        self._post("stream/game_end", {
            "game_number": game_number,
            "winner_port": winner_port,
            "end_method": end_method,
        })

    def send_match_end(self, winner_port: int, final_score: list[int]):
        """Send match_end message."""
        self._post("stream/end", {
            "winner_port": winner_port,
            "final_score": final_score,
        })

    def _post(self, endpoint: str, data: dict):
        """POST to arena endpoint (blocking, for important messages)."""
        if not self._client:
            logger.warning(f"Stream POST skipped (no client): {endpoint}")
            return
        url = f"{self.arena_url}/matches/{self.match_id}/{endpoint}"
        try:
            resp = self._client.post(url, json=data)
            logger.info(f"Stream POST {endpoint}: {resp.status_code}")
        except Exception as e:
            logger.error(f"Stream POST failed ({endpoint}): {e}")

    def _stream_loop(self):
        """Background thread that sends queued frames."""
        logger.info(f"Stream loop started for {self.match_id}")
        frames_sent = 0
        while not self._stop.is_set():
            frame_data = None
            with self._lock:
                if self._queue:
                    frame_data = self._queue.pop(0)

            if frame_data and self._client:
                url = f"{self.arena_url}/matches/{self.match_id}/stream/frame"
                try:
                    resp = self._client.post(url, json=frame_data)
                    frames_sent += 1
                    if frames_sent <= 3 or frames_sent % 100 == 0:
                        logger.info(f"Stream frame {frame_data.get('frame')} sent ({resp.status_code})")
                except Exception as e:
                    logger.warning(f"Stream frame failed: {e}")

            # ~60fps max, but typically throttled by stream_throttle
            time.sleep(0.016)
        logger.info(f"Stream loop ended for {self.match_id} ({frames_sent} frames sent)")


def extract_player_frame(player: "melee.PlayerState", port: int) -> dict:
    """Extract frame data from a libmelee PlayerState.

    Returns a dict matching the arena's PlayerFrameData schema.
    All numeric values are converted to Python native types for JSON serialization.
    """
    # Shield states: 178=ShieldStart, 179=Shield, 180=ShieldRelease, 182=ShieldStun
    shield_states = {178, 179, 180, 182}
    action_value = int(player.action.value) if player.action else 0

    return {
        "port": int(port),
        "x": float(player.position.x) if player.position else 0.0,
        "y": float(player.position.y) if player.position else 0.0,
        "action_state_id": action_value,
        "action_frame": int(player.action_frame) if player.action_frame else 0,
        "facing_direction": 1 if player.facing else -1,
        "percent": float(player.percent) if player.percent else 0.0,
        "stocks": int(player.stock) if player.stock else 0,
        "shield_health": float(player.shield_strength) if action_value in shield_states else None,
        "is_invincible": bool(player.invulnerable),
        "is_in_hitstun": bool(player.hitstun_frames_left > 0),
    }


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
        self._menu_navigator = SlippiMenuNavigator()
        self._menu_helper = melee.MenuHelper()
        self._our_port = 1  # Updated by port detection when game starts

        # Live streaming setup
        self._streamer: MatchStreamer | None = None
        if config.arena_url and config.match_id:
            self._streamer = MatchStreamer(config.arena_url, config.match_id)

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

            # Start live streaming if configured
            if self._streamer:
                self._streamer.start()

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
            result.our_port = self._our_port
            logger.info(f"Netplay complete! Winner: P{result.winner_port} ({result.score})")

            # Stream match end
            if self._streamer:
                self._streamer.send_match_end(
                    winner_port=result.winner_port,
                    final_score=[result.p1_games_won, result.p2_games_won],
                )

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
            # Headless mode: no graphics, no audio
            gfx_backend="Null" if self.config.headless else "",
            disable_audio=self.config.headless,
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

    def _detect_port(self, state: melee.GameState) -> int:
        """Detect our actual game port via connect code.

        Slippi Online assigns a random port (P1-P4). We identify ourselves
        by matching our connect code against player.connectCode, which is
        populated from the GAME_START event (SLP version >= 3.9.0).

        Returns the detected port, or 1 as fallback.
        """
        if self.config.connect_code:
            for port, player in state.players.items():
                code = getattr(player, "connectCode", None)
                if code and code == self.config.connect_code:
                    logger.info(f"Port detected via connect code: P{port} = {code}")
                    return port
            # Fallback: try case-insensitive match
            for port, player in state.players.items():
                code = getattr(player, "connectCode", None)
                if code and code.upper() == self.config.connect_code.upper():
                    logger.info(f"Port detected via connect code (case-insensitive): P{port} = {code}")
                    return port
            logger.warning(
                f"Could not detect port for {self.config.connect_code} — "
                f"available: {[(p, getattr(pl, 'connectCode', '?')) for p, pl in state.players.items()]}. "
                f"Falling back to port 1."
            )
        return 1

    def _run_game(
        self,
        fighter: Fighter,
        fighter_config: FighterConfig | None,
        on_frame: Callable[[melee.GameState], None] | None,
    ) -> GameResult:
        """Run a single game with one local fighter."""

        # Initial fighter setup — port will be corrected after game starts
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
        our_port = 1
        opp_port = 2
        consecutive_nones = 0
        game_result = None  # Store result when game ends, return after postgame

        # Input throttling - cache last action
        last_action = ControllerState()
        frames_since_input = 0

        # Stream throttling - only send every Nth frame
        frames_since_stream = 0
        game_number = len(getattr(self, '_completed_games', [])) + 1

        # Freeze detection - track when frames stop advancing
        last_frame_number = None
        last_frame_time = time.time()
        freeze_timeout = 10  # seconds without frame advancement = freeze

        # Watchdog: kills Dolphin if console.step() blocks forever
        # (e.g. Dolphin crashes without closing the socket cleanly)
        watchdog_heartbeat = [time.time()]
        watchdog_timeout = 15  # seconds
        watchdog_stop = threading.Event()

        def _watchdog():
            while not watchdog_stop.is_set():
                if time.time() - watchdog_heartbeat[0] > watchdog_timeout:
                    logger.error(
                        f"Watchdog: console.step() blocked for "
                        f"{watchdog_timeout}s, killing Dolphin"
                    )
                    if self._console and self._console._process:
                        self._console._process.kill()
                    return
                watchdog_stop.wait(1)

        watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
        watchdog_thread.start()

        # Main game loop
        try:
            while True:
                state = self._console.step()
                watchdog_heartbeat[0] = time.time()

                if state is None:
                    consecutive_nones += 1
                    if consecutive_nones > 600:  # ~10 seconds at 60fps
                        raise NetplayDisconnectedError(
                            "Lost connection (no game state for ~10 seconds)"
                        )
                    continue
                consecutive_nones = 0

                # Detect freeze by checking if frames are advancing
                if state.frame != last_frame_number:
                    last_frame_number = state.frame
                    last_frame_time = time.time()
                elif game_started and time.time() - last_frame_time > freeze_timeout:
                    logger.error(f"Freeze detected: no frame advancement for {freeze_timeout}s at frame {state.frame}")
                    raise NetplayDisconnectedError(
                        f"Game frozen (no frame advancement for {freeze_timeout} seconds)"
                    )

                # Periodic debug logging (every 5s)
                if game_started and state.frame and start_frame:
                    elapsed_frames = state.frame - start_frame
                    if elapsed_frames > 0 and elapsed_frames % 300 == 0:
                        p1 = state.players.get(1)
                        p2 = state.players.get(2)
                        p1_info = f"stock={p1.stock} pct={p1.percent:.0f} action={p1.action}" if p1 else "None"
                        p2_info = f"stock={p2.stock} pct={p2.percent:.0f} action={p2.action}" if p2 else "None"
                        logger.info(
                            f"[debug] frame={state.frame} menu={state.menu_state} "
                            f"P1=[{p1_info}] P2=[{p2_info}] game_result={'SET' if game_result else 'None'}"
                        )

                # In game — the hot path
                if state.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                    if not game_started:
                        game_started = True
                        start_frame = state.frame

                        # Detect our actual port via connect code
                        our_port = self._detect_port(state)
                        opp_port = 2 if our_port == 1 else 1
                        self._our_port = our_port

                        # Re-setup fighter with correct port
                        if our_port != 1:
                            match_config = MatchConfig(
                                character=self.config.character,
                                port=our_port,
                                opponent_port=opp_port,
                                stage=self.config.stage,
                                stocks=self.config.stocks,
                                time_minutes=self.config.time_minutes,
                            )
                            fighter.setup(match_config, fighter_config)

                        logger.info(f"Game started (we are P{our_port})")

                        # Notify fighter that the game has started
                        if hasattr(fighter, 'on_game_start'):
                            fighter.on_game_start(our_port, state)

                        # Stream match start (for live spectating)
                        if self._streamer and game_number == 1:
                            # Build player info from game state
                            # Note: wrap in int() to convert numpy int64 to JSON-serializable int
                            players = []
                            for port, player in state.players.items():
                                if player:
                                    players.append({
                                        "port": int(port),
                                        "character_id": int(player.character.value) if player.character else 0,
                                        "connect_code": getattr(player, "connectCode", ""),
                                        "display_name": getattr(player, "nametag", None),
                                    })
                            self._streamer.send_match_start(
                                stage_id=int(state.stage.value) if state.stage else 0,
                                players=players,
                            )

                    # Track damage dealt
                    for port in [1, 2]:
                        player = state.players.get(port)
                        if player:
                            opponent_port = 2 if port == 1 else 1
                            if player.percent > last_percent[port]:
                                damage_dealt[opponent_port] += player.percent - last_percent[port]
                            last_percent[port] = player.percent

                    # Get action from our fighter only
                    # Throttle AI inputs to reduce netplay load
                    frames_since_input += 1
                    if frames_since_input >= self.config.input_throttle:
                        frames_since_input = 0
                        # Get fresh input from AI
                        us = state.players.get(our_port)
                        if us and us.stock > 0:
                            try:
                                last_action = fighter.act(state)
                            except Exception as e:
                                logger.error(f"Fighter error: {e}")
                                last_action = ControllerState()
                        else:
                            # Player is dead/respawning, send neutral inputs
                            last_action = ControllerState()
                    # else: reuse last_action (throttled)

                    # Apply to our controller (port 1 only)
                    last_action.to_libmelee(self._controller)

                    # Frame callback
                    if on_frame:
                        on_frame(state)

                    # Stream frame data (throttled)
                    if self._streamer:
                        frames_since_stream += 1
                        if frames_since_stream >= self.config.stream_throttle:
                            frames_since_stream = 0
                            players = []
                            for port, player in state.players.items():
                                if player:
                                    players.append(extract_player_frame(player, port))
                            self._streamer.send_frame(int(state.frame), players)

                    # Check for game end - detect when stocks hit 0 or timeout
                    if game_result is None:  # Only check once
                        p1 = state.players.get(1)
                        p2 = state.players.get(2)

                        # Timeout: move on after max_game_seconds
                        elapsed_frames = state.frame - (start_frame or 0)
                        max_frames = self.config.max_game_seconds * 60
                        if elapsed_frames >= max_frames:
                            logger.info(f"Match timeout ({self.config.max_game_seconds}s) — counting as completed")
                            game_result = GameResult(
                                winner_port=1,  # Arbitrary — both survived
                                p1_stocks=p1.stock if p1 else 0,
                                p2_stocks=p2.stock if p2 else 0,
                                p1_damage_dealt=damage_dealt[1],
                                p2_damage_dealt=damage_dealt[2],
                                duration_frames=elapsed_frames,
                                stage=self.config.stage,
                            )
                            fighter.on_game_end(self._to_fighter_result(game_result))
                            # Stream game end
                            if self._streamer:
                                self._streamer.send_game_end(game_number, 1, "timeout")
                            return game_result

                        # Check if someone ran out of stocks
                        # Don't check action states - in netplay, the stable window is too brief
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
                            logger.info(f"Game end detected: P{winner} wins ({game_result.p1_stocks}-{game_result.p2_stocks})")

                            # Stream game end
                            if self._streamer:
                                self._streamer.send_game_end(game_number, winner, "stocks")

                    # If game ended but stocks reset (new match starting), return immediately
                    elif game_result is not None:
                        p1 = state.players.get(1)
                        p2 = state.players.get(2)
                        if p1 and p2 and p1.stock > 0 and p2.stock > 0:
                            logger.info("Stocks reset after game end - returning result without postgame")
                            return game_result

                    continue

                # If game already ended, return on any non-game state
                # (Slippi netplay may skip POSTGAME_SCORES entirely)
                if game_result is not None:
                    logger.debug(f"Game ended, left IN_GAME to {state.menu_state}")
                    return game_result

                # Postgame scores — skip through
                if state.menu_state == melee.Menu.POSTGAME_SCORES:
                    self._menu_helper.skip_postgame(
                        controller=self._controller,
                        gamestate=state,
                    )
                    continue

                # Any other menu state — navigate toward online match
                self._handle_menu(state)
        except OSError:
            # Watchdog killed Dolphin — step() throws when socket dies
            raise NetplayDisconnectedError(
                "Dolphin process killed by watchdog (console.step() blocked)"
            )
        finally:
            watchdog_stop.set()

    def _handle_menu(self, state: melee.GameState) -> None:
        """Navigate menus via Slippi direct connect.

        Uses libmelee's menu_helper_simple. Now that accessibility permissions
        are enabled, this should work properly.
        """
        self._menu_helper.menu_helper_simple(
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

        Uses detected port (self._our_port) to pick the correct side.
        """
        won = game.winner_port == self._our_port
        if self._our_port == 1:
            our_stocks = game.p1_stocks
            opp_stocks = game.p2_stocks
            our_damage = game.p1_damage_dealt
            opp_damage = game.p2_damage_dealt
        else:
            our_stocks = game.p2_stocks
            opp_stocks = game.p1_stocks
            our_damage = game.p2_damage_dealt
            opp_damage = game.p1_damage_dealt
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
        """Clean up Dolphin and controller."""
        import shutil

        logger.debug("Cleaning up netplay session")

        # Stop live streaming
        if self._streamer:
            self._streamer.stop()
            self._streamer = None

        # Track temp directory for cleanup
        temp_dir = None
        if self._console and hasattr(self._console, 'dolphin_home_path'):
            temp_dir = self._console.dolphin_home_path

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

        # Clean up temp directory if it was created by libmelee
        if temp_dir and 'tmp' in str(temp_dir).lower():
            try:
                import os
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.debug(f"Failed to clean temp directory: {e}")


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
