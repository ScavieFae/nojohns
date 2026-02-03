"""Tests for nojohns.registry — fighter discovery and loading."""

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


# ============================================================================
# Melee mock — same pattern as test_smashbot_adapter.py
# ============================================================================

def _setup_melee_mock():
    """Install a fake melee module if real one isn't available."""
    if "melee" in sys.modules and hasattr(sys.modules["melee"], "__file__"):
        return

    from enum import Enum, auto

    melee_mod = types.ModuleType("melee")

    class Character(Enum):
        FOX = auto()
        FALCO = auto()
        MARTH = auto()

    class Stage(Enum):
        FINAL_DESTINATION = auto()
        BATTLEFIELD = auto()

    class Button(Enum):
        BUTTON_A = auto()
        BUTTON_B = auto()
        BUTTON_X = auto()
        BUTTON_Y = auto()
        BUTTON_Z = auto()
        BUTTON_L = auto()
        BUTTON_R = auto()
        BUTTON_START = auto()
        BUTTON_MAIN = auto()
        BUTTON_C = auto()
        BUTTON_D_UP = auto()
        BUTTON_D_DOWN = auto()
        BUTTON_D_LEFT = auto()
        BUTTON_D_RIGHT = auto()

    class _ControllerState:
        def __init__(self):
            self.button = {b: False for b in Button}
            self.main_stick = (0.5, 0.5)
            self.c_stick = (0.5, 0.5)
            self.l_shoulder = 0.0
            self.r_shoulder = 0.0

    controller_mod = types.ModuleType("melee.controller")
    controller_mod.ControllerState = _ControllerState

    melee_mod.Character = Character
    melee_mod.Stage = Stage
    melee_mod.Button = Button
    melee_mod.Controller = type("Controller", (), {})
    melee_mod.GameState = type("GameState", (), {"players": {}})
    melee_mod.PlayerState = type("PlayerState", (), {})
    melee_mod.controller = controller_mod

    sys.modules["melee"] = melee_mod
    sys.modules["melee.controller"] = controller_mod


_setup_melee_mock()

from nojohns.fighter import DoNothingFighter, RandomFighter
from nojohns.registry import (
    FighterInfo,
    FighterLoadError,
    FighterNotFoundError,
    get_fighter_info,
    list_fighters,
    load_fighter,
    register_builtin,
    reset,
    scan_fighters,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry state before each test."""
    reset()
    yield


# ============================================================================
# Built-in registration
# ============================================================================


class TestBuiltins:
    def test_builtins_registered_on_import(self):
        fighters = list_fighters()
        names = {f.name for f in fighters}
        assert "do-nothing" in names
        assert "random" in names

    def test_load_do_nothing(self):
        fighter = load_fighter("do-nothing")
        assert isinstance(fighter, DoNothingFighter)

    def test_load_random(self):
        fighter = load_fighter("random")
        assert isinstance(fighter, RandomFighter)

    def test_builtin_info_has_source(self):
        info = get_fighter_info("do-nothing")
        assert info is not None
        assert info.source == "builtin"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FighterNotFoundError, match="nonexistent"):
            load_fighter("nonexistent")


# ============================================================================
# TOML scanning
# ============================================================================


def _write_toml(directory: Path, content: str) -> Path:
    """Write a fighter.toml into a fighter subdirectory."""
    directory.mkdir(parents=True, exist_ok=True)
    toml_path = directory / "fighter.toml"
    toml_path.write_text(content)
    return toml_path


class TestScanning:
    def test_scan_finds_toml(self, tmp_path):
        fighters_dir = tmp_path / "fighters"
        _write_toml(
            fighters_dir / "testbot",
            """
            name = "testbot"
            entry_point = "nojohns.fighter:DoNothingFighter"
            display_name = "Test Bot"
            author = "tester"
            description = "A test fighter."
            """,
        )

        scan_fighters(fighters_dir)
        info = get_fighter_info("testbot")

        assert info is not None
        assert info.name == "testbot"
        assert info.source == "manifest"
        assert info.display_name == "Test Bot"
        assert info.author == "tester"

    def test_toml_fields_parsed(self, tmp_path):
        fighters_dir = tmp_path / "fighters"
        _write_toml(
            fighters_dir / "fullbot",
            """
            name = "fullbot"
            version = "2.1.0"
            entry_point = "nojohns.fighter:DoNothingFighter"
            display_name = "Full Bot"
            author = "someone"
            description = "All fields."
            fighter_type = "neural-net"
            characters = ["FOX", "FALCO"]
            stages = ["FINAL_DESTINATION"]
            avg_frame_delay = 3
            repo_url = "https://example.com/fullbot"
            weights_required = true

            [hardware]
            gpu_required = true
            min_ram_gb = 8

            [init_args]
            model_path = "{fighter_dir}/weights.pt"
            """,
        )

        scan_fighters(fighters_dir)
        info = get_fighter_info("fullbot")

        assert info is not None
        assert info.version == "2.1.0"
        assert info.fighter_type == "neural-net"
        assert info.characters == ["FOX", "FALCO"]
        assert info.stages == ["FINAL_DESTINATION"]
        assert info.avg_frame_delay == 3
        assert info.repo_url == "https://example.com/fullbot"
        assert info.weights_required is True
        assert info.hardware == {"gpu_required": True, "min_ram_gb": 8}

    def test_fighter_dir_resolved_in_init_args(self, tmp_path):
        fighters_dir = tmp_path / "fighters"
        bot_dir = fighters_dir / "pathbot"
        _write_toml(
            bot_dir,
            """
            name = "pathbot"
            entry_point = "nojohns.fighter:DoNothingFighter"
            [init_args]
            some_path = "{fighter_dir}/data"
            """,
        )

        scan_fighters(fighters_dir)
        info = get_fighter_info("pathbot")

        assert info is not None
        # Raw init_args still has placeholder
        assert "{fighter_dir}" in info.init_args["some_path"]
        # fighter_dir is set to the actual directory
        assert info.fighter_dir == bot_dir

    def test_bad_toml_skipped(self, tmp_path):
        fighters_dir = tmp_path / "fighters"
        bad_dir = fighters_dir / "badbot"
        bad_dir.mkdir(parents=True)
        (bad_dir / "fighter.toml").write_text("this is not { valid toml")

        # Should not raise
        scan_fighters(fighters_dir)
        assert get_fighter_info("badbot") is None

    def test_missing_name_skipped(self, tmp_path):
        fighters_dir = tmp_path / "fighters"
        _write_toml(
            fighters_dir / "noname",
            """
            entry_point = "nojohns.fighter:DoNothingFighter"
            """,
        )

        scan_fighters(fighters_dir)
        assert get_fighter_info("noname") is None

    def test_missing_entry_point_skipped(self, tmp_path):
        fighters_dir = tmp_path / "fighters"
        _write_toml(
            fighters_dir / "noentry",
            """
            name = "noentry"
            """,
        )

        scan_fighters(fighters_dir)
        assert get_fighter_info("noentry") is None

    def test_builtin_wins_on_duplicate(self, tmp_path):
        """If a manifest has the same name as a built-in, built-in wins."""
        fighters_dir = tmp_path / "fighters"
        _write_toml(
            fighters_dir / "random",
            """
            name = "random"
            entry_point = "nojohns.fighter:DoNothingFighter"
            """,
        )

        scan_fighters(fighters_dir)
        info = get_fighter_info("random")
        assert info is not None
        assert info.source == "builtin"


# ============================================================================
# Loading via entry_point
# ============================================================================


class TestLoading:
    def test_entry_point_import(self, tmp_path):
        """Load a fighter via entry_point from a scanned manifest."""
        fighters_dir = tmp_path / "fighters"
        _write_toml(
            fighters_dir / "testbot",
            """
            name = "testbot"
            entry_point = "nojohns.fighter:DoNothingFighter"
            """,
        )

        scan_fighters(fighters_dir)
        fighter = load_fighter("testbot")
        assert isinstance(fighter, DoNothingFighter)

    def test_failed_import_raises_fighter_load_error(self, tmp_path):
        fighters_dir = tmp_path / "fighters"
        _write_toml(
            fighters_dir / "broken",
            """
            name = "broken"
            entry_point = "nonexistent.module:BrokenFighter"
            """,
        )

        scan_fighters(fighters_dir)
        with pytest.raises(FighterLoadError, match="Cannot import module"):
            load_fighter("broken")

    def test_missing_class_raises_fighter_load_error(self, tmp_path):
        fighters_dir = tmp_path / "fighters"
        _write_toml(
            fighters_dir / "noclass",
            """
            name = "noclass"
            entry_point = "nojohns.fighter:ThisClassDoesNotExist"
            """,
        )

        scan_fighters(fighters_dir)
        with pytest.raises(FighterLoadError, match="has no attribute"):
            load_fighter("noclass")

    def test_init_args_passed_to_constructor(self, tmp_path):
        """init_args with {fighter_dir} should be resolved and passed to the class."""
        fighters_dir = tmp_path / "fighters"
        bot_dir = fighters_dir / "argbot"
        _write_toml(
            bot_dir,
            """
            name = "argbot"
            entry_point = "nojohns.fighter:DoNothingFighter"
            [init_args]
            unexpected_kwarg = "value"
            """,
        )

        scan_fighters(fighters_dir)
        # DoNothingFighter.__init__ doesn't accept kwargs, so this should
        # raise FighterLoadError (not a raw TypeError)
        with pytest.raises(FighterLoadError, match="Failed to instantiate"):
            load_fighter("argbot")

    def test_get_fighter_info_without_loading(self):
        info = get_fighter_info("do-nothing")
        assert info is not None
        assert info.name == "do-nothing"
        # Didn't instantiate anything


# ============================================================================
# Lazy scan
# ============================================================================


class TestLazyScan:
    def test_list_triggers_scan(self, tmp_path):
        """list_fighters() should trigger scan if not already done."""
        fighters_dir = tmp_path / "fighters"
        _write_toml(
            fighters_dir / "lazybot",
            """
            name = "lazybot"
            entry_point = "nojohns.fighter:DoNothingFighter"
            """,
        )

        # Patch the default scan path to our tmp dir
        with patch("nojohns.registry._scanned", False):
            with patch(
                "nojohns.registry.scan_fighters",
                wraps=lambda d=None: scan_fighters(fighters_dir),
            ) as mock_scan:
                # Force re-import state
                import nojohns.registry as reg
                reg._scanned = False
                reg.scan_fighters(fighters_dir)

        info = get_fighter_info("lazybot")
        assert info is not None
