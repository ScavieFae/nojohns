"""Tests for the SmashBot adapter's InterceptController.

Since libmelee has C dependencies (pyenet) that may not build everywhere,
we mock the melee module so these tests run without it installed.
"""

import sys
import types
from dataclasses import dataclass
from enum import Enum, auto
from unittest.mock import MagicMock

import pytest


# ============================================================================
# Mock melee module — enough to test InterceptController + ControllerState
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

    melee_mod.Button = Button
    melee_mod.Character = Character
    melee_mod.Stage = Stage
    melee_mod.Menu = Menu
    melee_mod.Controller = MagicMock
    melee_mod.Console = MagicMock
    melee_mod.GameState = MagicMock
    melee_mod.PlayerState = MagicMock
    melee_mod.MenuHelper = MagicMock()

    sys.modules["melee"] = melee_mod


_setup_melee_mock()

# Now safe to import our code
from melee import Button

from fighters.smashbot.adapter import InterceptController
from nojohns.fighter import ControllerState


# ============================================================================
# Tests
# ============================================================================

class TestInterceptController:
    """Unit tests for InterceptController — the core of the adapter."""

    def test_neutral_state(self):
        """Fresh controller should produce neutral ControllerState."""
        ic = InterceptController()
        state = ic.to_controller_state()

        assert state.main_x == 0.5
        assert state.main_y == 0.5
        assert state.c_x == 0.5
        assert state.c_y == 0.5
        assert state.l_trigger == 0.0
        assert state.r_trigger == 0.0
        assert not state.a
        assert not state.b
        assert not state.x
        assert not state.y
        assert not state.z
        assert not state.start

    def test_press_button(self):
        """press_button should set the corresponding flag."""
        ic = InterceptController()
        ic.press_button(Button.BUTTON_A)
        ic.press_button(Button.BUTTON_Z)

        state = ic.to_controller_state()
        assert state.a
        assert state.z
        assert not state.b

    def test_release_button(self):
        """release_button should clear a previously pressed button."""
        ic = InterceptController()
        ic.press_button(Button.BUTTON_A)
        ic.press_button(Button.BUTTON_B)
        ic.release_button(Button.BUTTON_A)

        state = ic.to_controller_state()
        assert not state.a
        assert state.b

    def test_tilt_analog_main_stick(self):
        """tilt_analog with BUTTON_MAIN should set main stick."""
        ic = InterceptController()
        ic.tilt_analog(Button.BUTTON_MAIN, 0.3, 0.8)

        state = ic.to_controller_state()
        assert state.main_x == pytest.approx(0.3)
        assert state.main_y == pytest.approx(0.8)
        assert state.c_x == 0.5
        assert state.c_y == 0.5

    def test_tilt_analog_c_stick(self):
        """tilt_analog with BUTTON_C should set c-stick."""
        ic = InterceptController()
        ic.tilt_analog(Button.BUTTON_C, 0.0, 1.0)

        state = ic.to_controller_state()
        assert state.c_x == pytest.approx(0.0)
        assert state.c_y == pytest.approx(1.0)
        assert state.main_x == 0.5
        assert state.main_y == 0.5

    def test_tilt_analog_unit(self):
        """tilt_analog_unit should convert from -1..1 to 0..1 range."""
        ic = InterceptController()
        # Full left: -1.0 → 0.0, neutral Y: 0.0 → 0.5
        ic.tilt_analog_unit(Button.BUTTON_MAIN, -1.0, 0.0)

        state = ic.to_controller_state()
        assert state.main_x == pytest.approx(0.0)
        assert state.main_y == pytest.approx(0.5)

    def test_press_shoulder(self):
        """press_shoulder should set trigger values."""
        ic = InterceptController()
        ic.press_shoulder(Button.BUTTON_L, 0.7)
        ic.press_shoulder(Button.BUTTON_R, 1.0)

        state = ic.to_controller_state()
        assert state.l_trigger == pytest.approx(0.7)
        assert state.r_trigger == pytest.approx(1.0)

    def test_release_all_clears_state(self):
        """release_all should reset everything to neutral."""
        ic = InterceptController()
        ic.press_button(Button.BUTTON_A)
        ic.tilt_analog(Button.BUTTON_MAIN, 0.0, 1.0)
        ic.press_shoulder(Button.BUTTON_L, 1.0)

        ic.release_all()
        state = ic.to_controller_state()

        assert not state.a
        assert state.main_x == 0.5
        assert state.main_y == 0.5
        assert state.l_trigger == 0.0

    def test_to_controller_state_resets(self):
        """After to_controller_state(), next call should be neutral."""
        ic = InterceptController()
        ic.press_button(Button.BUTTON_A)
        ic.tilt_analog(Button.BUTTON_MAIN, 0.0, 0.0)

        first = ic.to_controller_state()
        assert first.a
        assert first.main_x == pytest.approx(0.0)

        second = ic.to_controller_state()
        assert not second.a
        assert second.main_x == 0.5

    def test_release_all_then_set_captures_final(self):
        """SmashBot chains often release_all then set new buttons in the same frame."""
        ic = InterceptController()
        ic.press_button(Button.BUTTON_B)
        ic.tilt_analog(Button.BUTTON_MAIN, 0.0, 0.5)

        # Chain transition: clear then re-issue
        ic.release_all()
        ic.press_button(Button.BUTTON_A)
        ic.tilt_analog(Button.BUTTON_MAIN, 1.0, 0.5)

        state = ic.to_controller_state()
        assert not state.b
        assert state.a
        assert state.main_x == pytest.approx(1.0)

    def test_all_digital_buttons(self):
        """Every digital button should map correctly."""
        ic = InterceptController()

        mapping = {
            Button.BUTTON_A: "a",
            Button.BUTTON_B: "b",
            Button.BUTTON_X: "x",
            Button.BUTTON_Y: "y",
            Button.BUTTON_Z: "z",
            Button.BUTTON_START: "start",
            Button.BUTTON_D_UP: "d_up",
            Button.BUTTON_D_DOWN: "d_down",
            Button.BUTTON_D_LEFT: "d_left",
            Button.BUTTON_D_RIGHT: "d_right",
        }

        for button in mapping:
            ic.press_button(button)

        state = ic.to_controller_state()

        for button, attr in mapping.items():
            assert getattr(state, attr), f"{attr} should be True"

    def test_flush_is_noop(self):
        """flush() should not affect state."""
        ic = InterceptController()
        ic.press_button(Button.BUTTON_A)
        ic.flush()

        state = ic.to_controller_state()
        assert state.a

    def test_connect_returns_true(self):
        """connect() should always return True."""
        ic = InterceptController()
        assert ic.connect() is True
