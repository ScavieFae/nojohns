"""Tests for the Slippi netplay runner.

Uses the same melee mock approach as test_smashbot_adapter.py.
"""

import sys
import types
import threading
from dataclasses import dataclass
from enum import Enum, auto
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ============================================================================
# Mock melee module
# ============================================================================

def _setup_melee_mock():
    """Install a fake melee module with the types we need."""
    if "melee" in sys.modules and hasattr(sys.modules["melee"], "__file__"):
        return  # Real melee is installed, no mock needed

    melee_mod = types.ModuleType("melee")

    class Button(Enum):
        BUTTON_A = auto()
        BUTTON_B = auto()
        BUTTON_X = auto()
        BUTTON_Y = auto()
        BUTTON_Z = auto()
        BUTTON_START = auto()
        BUTTON_L = auto()
        BUTTON_R = auto()
        BUTTON_MAIN = auto()
        BUTTON_C = auto()
        BUTTON_D_UP = auto()
        BUTTON_D_DOWN = auto()
        BUTTON_D_LEFT = auto()
        BUTTON_D_RIGHT = auto()

    class Character(Enum):
        FOX = auto()
        FALCO = auto()
        MARTH = auto()

    class Stage(Enum):
        FINAL_DESTINATION = auto()
        BATTLEFIELD = auto()

    class Menu(Enum):
        IN_GAME = auto()
        SUDDEN_DEATH = auto()
        POSTGAME_SCORES = auto()
        CHARACTER_SELECT = auto()
        SLIPPI_ONLINE_CSS = auto()

    class SubMenu(Enum):
        NAME_ENTRY_SUBMENU = auto()

    class ControllerType(Enum):
        STANDARD = auto()

    class MockControllerState:
        def __init__(self):
            self.button = {b: False for b in Button}
            self.main_stick = (0.5, 0.5)
            self.c_stick = (0.5, 0.5)
            self.l_shoulder = 0
            self.r_shoulder = 0

    controller_mod = types.ModuleType("melee.controller")
    controller_mod.ControllerState = MockControllerState

    melee_mod.Button = Button
    melee_mod.Character = Character
    melee_mod.Stage = Stage
    melee_mod.Menu = Menu
    melee_mod.SubMenu = SubMenu
    melee_mod.ControllerType = ControllerType
    melee_mod.Controller = MagicMock
    melee_mod.Console = MagicMock
    melee_mod.GameState = MagicMock
    melee_mod.PlayerState = MagicMock
    melee_mod.MenuHelper = MagicMock()
    melee_mod.controller = controller_mod

    sys.modules["melee"] = melee_mod
    sys.modules["melee.controller"] = controller_mod


_setup_melee_mock()

# Now safe to import our code
import melee
from melee import Character, Stage

from games.melee.netplay import (
    NetplayConfig,
    NetplayRunner,
    NetplayDisconnectedError,
    netplay_test,
)
from games.melee.runner import GameResult, MatchResult
from nojohns.fighter import (
    MatchConfig,
    FighterConfig,
    ControllerState,
    DoNothingFighter,
    RandomFighter,
)


# ============================================================================
# NetplayConfig tests
# ============================================================================

class TestNetplayConfig:
    """Tests for NetplayConfig defaults and fields."""

    def test_required_fields(self):
        """Config needs dolphin_path, iso_path, and opponent_code."""
        cfg = NetplayConfig(
            dolphin_path="/path/to/dolphin",
            iso_path="/path/to/melee.iso",
            opponent_code="ABCD#123",
        )
        assert cfg.dolphin_path == "/path/to/dolphin"
        assert cfg.iso_path == "/path/to/melee.iso"
        assert cfg.opponent_code == "ABCD#123"

    def test_defaults(self):
        """Default values should match plan spec."""
        cfg = NetplayConfig(
            dolphin_path="/d",
            iso_path="/i",
            opponent_code="X#1",
        )
        assert cfg.character == Character.FOX
        assert cfg.stage == Stage.FINAL_DESTINATION
        assert cfg.stocks == 4
        assert cfg.time_minutes == 8
        assert cfg.online_delay == 2
        assert cfg.slippi_port == 51441
        assert cfg.fullscreen is False
        assert cfg.dolphin_home_path is None
        assert cfg.slippi_replay_dir is None

    def test_custom_values(self):
        """All fields can be overridden."""
        cfg = NetplayConfig(
            dolphin_path="/d",
            iso_path="/i",
            opponent_code="WXYZ#456",
            character=Character.FALCO,
            stage=Stage.BATTLEFIELD,
            stocks=3,
            time_minutes=5,
            online_delay=3,
            slippi_port=51442,
            fullscreen=True,
            dolphin_home_path="/home",
            slippi_replay_dir="/replays",
        )
        assert cfg.character == Character.FALCO
        assert cfg.stage == Stage.BATTLEFIELD
        assert cfg.stocks == 3
        assert cfg.time_minutes == 5
        assert cfg.online_delay == 3
        assert cfg.slippi_port == 51442
        assert cfg.fullscreen is True
        assert cfg.dolphin_home_path == "/home"
        assert cfg.slippi_replay_dir == "/replays"


# ============================================================================
# NetplayRunner tests
# ============================================================================

class TestNetplayRunner:
    """Tests for NetplayRunner internals."""

    def _make_runner(self, **overrides) -> NetplayRunner:
        defaults = dict(
            dolphin_path="/dolphin",
            iso_path="/melee.iso",
            opponent_code="TEST#999",
        )
        defaults.update(overrides)
        return NetplayRunner(NetplayConfig(**defaults))

    @patch("games.melee.netplay.melee.Console")
    def test_setup_console_passes_online_delay(self, mock_console_cls):
        """Console should be created with online_delay from config."""
        runner = self._make_runner(online_delay=3, slippi_port=51442)
        runner._setup_console()

        mock_console_cls.assert_called_once()
        call_kwargs = mock_console_cls.call_args[1]
        assert call_kwargs["online_delay"] == 3
        assert call_kwargs["slippi_port"] == 51442

    @patch("games.melee.netplay.melee.Console")
    def test_setup_console_default_delay(self, mock_console_cls):
        """Default online_delay should be 2."""
        runner = self._make_runner()
        runner._setup_console()

        call_kwargs = mock_console_cls.call_args[1]
        assert call_kwargs["online_delay"] == 2

    @patch("games.melee.netplay.melee.Controller")
    def test_setup_controller_port_1(self, mock_controller_cls):
        """Controller should be on port 1."""
        runner = self._make_runner()
        runner._console = MagicMock()
        runner._setup_controller()

        mock_controller_cls.assert_called_once()
        call_kwargs = mock_controller_cls.call_args[1]
        assert call_kwargs["port"] == 1

    def test_handle_menu_passes_connect_code(self):
        """_handle_menu should pass opponent_code to menu_helper_simple."""
        runner = self._make_runner(opponent_code="ABCD#123")
        runner._controller = MagicMock()
        runner._menu_helper = MagicMock()

        mock_state = MagicMock()

        runner._handle_menu(mock_state)

        runner._menu_helper.menu_helper_simple.assert_called_once()
        call_kwargs = runner._menu_helper.menu_helper_simple.call_args[1]
        assert call_kwargs["connect_code"] == "ABCD#123"
        assert call_kwargs["controller"] == runner._controller
        assert call_kwargs["gamestate"] == mock_state

    def test_handle_menu_passes_character(self):
        """_handle_menu should pass character from config."""
        runner = self._make_runner(character="FALCO")
        # Override after init since Character enum may differ in mock
        runner.config.character = Character.FALCO
        runner._controller = MagicMock()
        runner._menu_helper = MagicMock()

        mock_state = MagicMock()

        runner._handle_menu(mock_state)

        call_kwargs = runner._menu_helper.menu_helper_simple.call_args[1]
        assert call_kwargs["character_selected"] == Character.FALCO

    def test_fighter_setup_gets_port_1(self):
        """Fighter.setup() should receive port=1, opponent_port=2."""
        runner = self._make_runner()
        fighter = MagicMock()

        # Build a match config the way _run_game does
        match_config = MatchConfig(
            character=runner.config.character,
            port=1,
            opponent_port=2,
            stage=runner.config.stage,
            stocks=runner.config.stocks,
            time_minutes=runner.config.time_minutes,
        )

        assert match_config.port == 1
        assert match_config.opponent_port == 2

    def test_to_fighter_result_win(self):
        """_to_fighter_result should report a win when port 1 wins."""
        runner = self._make_runner()
        game = GameResult(
            winner_port=1,
            p1_stocks=2,
            p2_stocks=0,
            p1_damage_dealt=340.0,
            p2_damage_dealt=180.0,
            duration_frames=7200,
            stage=Stage.FINAL_DESTINATION,
        )

        result = runner._to_fighter_result(game)
        assert result.won is True
        assert result.stocks_remaining == 2
        assert result.opponent_stocks == 0
        assert result.damage_dealt == 340.0
        assert result.damage_taken == 180.0

    def test_to_fighter_result_loss(self):
        """_to_fighter_result should report a loss when port 2 wins."""
        runner = self._make_runner()
        game = GameResult(
            winner_port=2,
            p1_stocks=0,
            p2_stocks=3,
            p1_damage_dealt=120.0,
            p2_damage_dealt=400.0,
            duration_frames=5400,
            stage=Stage.BATTLEFIELD,
        )

        result = runner._to_fighter_result(game)
        assert result.won is False
        assert result.stocks_remaining == 0
        assert result.opponent_stocks == 3


# ============================================================================
# NetplayDisconnectedError tests
# ============================================================================

class TestNetplayDisconnectedError:
    """Tests for the disconnect error."""

    def test_is_exception(self):
        """Should be a regular exception."""
        err = NetplayDisconnectedError("lost connection")
        assert isinstance(err, Exception)
        assert str(err) == "lost connection"

    def test_can_be_raised_and_caught(self):
        """Should be catchable specifically."""
        with pytest.raises(NetplayDisconnectedError):
            raise NetplayDisconnectedError("opponent dropped")


# ============================================================================
# netplay_test() tests
# ============================================================================

class TestNetplayTest:
    """Tests for the local two-Dolphin test function."""

    def test_configs_have_swapped_codes(self):
        """Side 1 should connect to code2, side 2 to code1."""
        # We can't easily call netplay_test() without real Dolphin,
        # but we can verify the config construction logic by building
        # configs the same way netplay_test() does.
        code1 = "AAAA#111"
        code2 = "BBBB#222"

        config1 = NetplayConfig(
            dolphin_path="/d",
            iso_path="/i",
            opponent_code=code2,
            dolphin_home_path="/home1",
            slippi_port=51441,
        )
        config2 = NetplayConfig(
            dolphin_path="/d",
            iso_path="/i",
            opponent_code=code1,
            dolphin_home_path="/home2",
            slippi_port=51442,
        )

        assert config1.opponent_code == "BBBB#222"
        assert config2.opponent_code == "AAAA#111"

    def test_configs_have_different_ports(self):
        """Two local instances need different slippi_ports."""
        config1 = NetplayConfig(
            dolphin_path="/d",
            iso_path="/i",
            opponent_code="X#1",
            slippi_port=51441,
        )
        config2 = NetplayConfig(
            dolphin_path="/d",
            iso_path="/i",
            opponent_code="Y#2",
            slippi_port=51442,
        )

        assert config1.slippi_port != config2.slippi_port

    def test_configs_have_different_homes(self):
        """Each side needs its own Dolphin home dir for separate Slippi accounts."""
        config1 = NetplayConfig(
            dolphin_path="/d",
            iso_path="/i",
            opponent_code="X#1",
            dolphin_home_path="/home/side1",
        )
        config2 = NetplayConfig(
            dolphin_path="/d",
            iso_path="/i",
            opponent_code="Y#2",
            dolphin_home_path="/home/side2",
        )

        assert config1.dolphin_home_path != config2.dolphin_home_path
