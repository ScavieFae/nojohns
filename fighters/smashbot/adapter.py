"""
fighters/smashbot/adapter.py - SmashBot adapter for nojohns

Wraps altf4's SmashBot (https://github.com/altf4/SmashBot) as a nojohns Fighter.

SmashBot expects to mutate a real libmelee Controller via press_button(), tilt_analog(),
etc. We intercept those calls with a fake controller, capture the resulting state,
and return it as a ControllerState.
"""

import logging
import sys
from pathlib import Path

import melee
from melee import Button, Character, Stage

from nojohns.fighter import (
    BaseFighter,
    ControllerState,
    FighterConfig,
    FighterMetadata,
    MatchConfig,
    MatchResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# InterceptController
# ============================================================================

class InterceptController:
    """
    A fake melee.Controller that captures SmashBot's inputs.

    SmashBot's chain/tactic/strategy hierarchy calls methods like
    press_button(), tilt_analog(), press_shoulder() on a Controller object.
    This class records those calls and produces a ControllerState snapshot.

    After to_controller_state() is called, internal state resets to neutral
    so the next frame starts clean.

    Also tracks `prev` — a libmelee-style ControllerState from the previous
    frame. SmashBot chains read this for frame-perfect sequencing (e.g.
    "was Y pressed last frame?").
    """

    def __init__(self):
        self._buttons: set[Button] = set()
        self._main_x: float = 0.5
        self._main_y: float = 0.5
        self._c_x: float = 0.5
        self._c_y: float = 0.5
        self._l_shoulder: float = 0.0
        self._r_shoulder: float = 0.0
        # SmashBot chains check controller.prev.button[X], controller.prev.c_stick, etc.
        self.prev = melee.controller.ControllerState()

    # -- Button methods --

    def press_button(self, button: Button) -> None:
        self._buttons.add(button)

    def release_button(self, button: Button) -> None:
        self._buttons.discard(button)

    def release_all(self) -> None:
        """SmashBot chains call this between actions."""
        self._buttons.clear()
        self._main_x = 0.5
        self._main_y = 0.5
        self._c_x = 0.5
        self._c_y = 0.5
        self._l_shoulder = 0.0
        self._r_shoulder = 0.0

    # -- Analog methods --

    def tilt_analog(self, button: Button, x: float, y: float) -> None:
        if button == Button.BUTTON_MAIN:
            self._main_x = x
            self._main_y = y
        elif button == Button.BUTTON_C:
            self._c_x = x
            self._c_y = y

    def tilt_analog_unit(self, button: Button, x: float, y: float) -> None:
        """Some SmashBot code uses tilt_analog_unit (normalized -1..1 → 0..1)."""
        # libmelee's tilt_analog_unit converts from unit coords to 0..1 range.
        # SmashBot may call this directly. Convert: 0.5 + x*0.5, 0.5 + y*0.5
        self.tilt_analog(button, 0.5 + x * 0.5, 0.5 + y * 0.5)

    def press_shoulder(self, button: Button, amount: float) -> None:
        if button == Button.BUTTON_L:
            self._l_shoulder = amount
        elif button == Button.BUTTON_R:
            self._r_shoulder = amount

    def empty_input(self) -> None:
        """Alias for release_all(). SmashBot chains call this constantly."""
        self.release_all()

    # -- No-ops for methods SmashBot might call --

    def flush(self) -> None:
        """Real Controller sends state over pipe. We don't."""
        pass

    def connect(self) -> bool:
        """SmashBot may call connect(). Always succeeds."""
        return True

    def disconnect(self) -> None:
        pass

    # -- Snapshot --

    def to_controller_state(self) -> ControllerState:
        """
        Snapshot the current state as a ControllerState, then reset to neutral.

        Called once per frame after SmashBot's act(). Also updates self.prev
        so SmashBot chains can read last frame's inputs next frame.
        """
        state = ControllerState(
            main_x=self._main_x,
            main_y=self._main_y,
            c_x=self._c_x,
            c_y=self._c_y,
            l_trigger=self._l_shoulder,
            r_trigger=self._r_shoulder,
            a=Button.BUTTON_A in self._buttons,
            b=Button.BUTTON_B in self._buttons,
            x=Button.BUTTON_X in self._buttons,
            y=Button.BUTTON_Y in self._buttons,
            z=Button.BUTTON_Z in self._buttons,
            start=Button.BUTTON_START in self._buttons,
            d_up=Button.BUTTON_D_UP in self._buttons,
            d_down=Button.BUTTON_D_DOWN in self._buttons,
            d_left=Button.BUTTON_D_LEFT in self._buttons,
            d_right=Button.BUTTON_D_RIGHT in self._buttons,
        )

        # Update prev for SmashBot chains that check last frame's inputs.
        # prev must be a libmelee ControllerState (dict buttons, tuple sticks).
        prev = melee.controller.ControllerState()
        for btn in self._buttons:
            if btn in prev.button:
                prev.button[btn] = True
        prev.main_stick = (self._main_x, self._main_y)
        prev.c_stick = (self._c_x, self._c_y)
        prev.l_shoulder = self._l_shoulder
        prev.r_shoulder = self._r_shoulder
        self.prev = prev

        self.release_all()
        return state


# ============================================================================
# Dolphin stub
# ============================================================================

class _DolphinStub:
    """Minimal stand-in for melee.Console that ESAgent expects.

    ESAgent accesses dolphin.logger, and the Bait strategy guards with
    `if self.logger:` before logging. Setting logger=None satisfies both.
    """

    def __init__(self):
        self.logger = None


# ============================================================================
# SmashBotFighter
# ============================================================================

class SmashBotFighter(BaseFighter):
    """
    Wraps SmashBot's ESAgent as a nojohns Fighter.

    SmashBot is loaded from a local clone at runtime — not vendored.
    Pass the path to your SmashBot repo checkout.

    Usage:
        fighter = SmashBotFighter("/path/to/SmashBot")
        # then use with MatchRunner as normal
    """

    def __init__(self, smashbot_path: str):
        super().__init__()
        self._smashbot_path = Path(smashbot_path).resolve()
        self._intercept: InterceptController | None = None
        self._agent = None  # ESAgent instance, created in setup()

    @property
    def metadata(self) -> FighterMetadata:
        return FighterMetadata(
            name="smashbot",
            version="1.0.0",
            display_name="SmashBot",
            author="altf4",
            description=(
                "Rule-based Melee AI by altf4. Plays Fox with hand-crafted "
                "strategies, tactics, and chains. The OG Melee bot."
            ),
            fighter_type="rule-based",
            characters=[Character.FOX],
            gpu_required=False,
            min_ram_gb=1,
            repo_url="https://github.com/altf4/SmashBot",
        )

    def setup(self, match: MatchConfig, config: FighterConfig | None = None) -> None:
        super().setup(match, config)

        self._intercept = InterceptController()

        # Import SmashBot from the provided path
        smashbot_str = str(self._smashbot_path)
        if smashbot_str not in sys.path:
            sys.path.insert(0, smashbot_str)

        try:
            from esagent import ESAgent
        except ImportError as e:
            raise ImportError(
                f"Could not import SmashBot's ESAgent from {self._smashbot_path}. "
                f"Make sure you've cloned https://github.com/altf4/SmashBot there.\n"
                f"Original error: {e}"
            ) from e

        # Map aggression (0.0–1.0) to SmashBot difficulty (1–4)
        aggression = self._config.aggression if self._config else 0.5
        difficulty = int(aggression * 3) + 1  # 1 at 0.0, 4 at 1.0

        # ESAgent expects a dolphin (Console) object for dolphin.logger.
        # SmashBot's Bait strategy checks `if self.logger:` before using it,
        # so a stub with logger=None works. If deeper code needs more, extend this.
        dolphin_stub = _DolphinStub()

        self._agent = ESAgent(
            dolphin=dolphin_stub,
            smashbot_port=match.port,
            opponent_port=match.opponent_port,
            controller=self._intercept,
            difficulty=difficulty,
        )

        logger.info(
            f"SmashBot ready: port={match.port}, difficulty={difficulty}, "
            f"path={self._smashbot_path}"
        )

    def act(self, state: melee.GameState) -> ControllerState:
        if self._agent is None or self._intercept is None:
            raise RuntimeError("setup() not called before act()")

        self._agent.act(state)
        return self._intercept.to_controller_state()
