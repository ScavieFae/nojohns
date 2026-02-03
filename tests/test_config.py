"""Tests for nojohns.config — local config file management."""

import textwrap
from pathlib import Path

import pytest

from nojohns.config import (
    GameConfig,
    NojohnsConfig,
    load_config,
    get_game_config,
)


@pytest.fixture
def config_dir(tmp_path):
    """Temporary directory for config files."""
    return tmp_path


def _write_config(config_dir: Path, content: str) -> Path:
    """Write a config.toml and return the path."""
    config_path = config_dir / "config.toml"
    config_path.write_text(textwrap.dedent(content))
    return config_path


class TestLoadConfig:
    def test_missing_file_returns_empty(self, config_dir):
        missing = config_dir / "nonexistent.toml"
        cfg = load_config(missing)
        assert isinstance(cfg, NojohnsConfig)
        assert cfg.games == {}
        assert cfg.arena_server is None

    def test_minimal_melee_config(self, config_dir):
        path = _write_config(config_dir, """\
            [games.melee]
            dolphin = "/path/to/dolphin"
            iso = "/path/to/melee.iso"
        """)
        cfg = load_config(path)
        assert "melee" in cfg.games
        gc = cfg.games["melee"]
        assert gc.dolphin_path == "/path/to/dolphin"
        assert gc.iso_path == "/path/to/melee.iso"
        assert gc.connect_code is None
        assert gc.dolphin_home is None
        assert gc.online_delay is None
        assert gc.input_throttle is None

    def test_tilde_expansion(self, config_dir):
        path = _write_config(config_dir, """\
            [games.melee]
            dolphin = "~/Library/Application Support/Slippi Launcher/netplay"
            iso = "~/games/melee/melee.ciso"
            dolphin_home = "~/Library/Application Support/Slippi Dolphin"
        """)
        cfg = load_config(path)
        gc = cfg.games["melee"]
        home = str(Path.home())
        assert gc.dolphin_path.startswith(home)
        assert gc.iso_path.startswith(home)
        assert gc.dolphin_home.startswith(home)
        # Tilde should be gone
        assert "~" not in gc.dolphin_path
        assert "~" not in gc.iso_path
        assert "~" not in gc.dolphin_home

    def test_full_config(self, config_dir):
        path = _write_config(config_dir, """\
            [games.melee]
            dolphin = "/opt/dolphin"
            iso = "/opt/melee.iso"
            connect_code = "TEST#123"
            dolphin_home = "/opt/dolphin-home"
            online_delay = 6
            input_throttle = 3

            [arena]
            server = "http://example.com:8000"
        """)
        cfg = load_config(path)

        gc = cfg.games["melee"]
        assert gc.dolphin_path == "/opt/dolphin"
        assert gc.iso_path == "/opt/melee.iso"
        assert gc.connect_code == "TEST#123"
        assert gc.dolphin_home == "/opt/dolphin-home"
        assert gc.online_delay == 6
        assert gc.input_throttle == 3

        assert cfg.arena_server == "http://example.com:8000"

    def test_corrupt_toml_returns_empty(self, config_dir):
        path = config_dir / "config.toml"
        path.write_text("this is not [valid toml }{")
        cfg = load_config(path)
        assert cfg.games == {}
        assert cfg.arena_server is None

    def test_partial_config_leaves_none(self, config_dir):
        path = _write_config(config_dir, """\
            [games.melee]
            dolphin = "/path/dolphin"
        """)
        cfg = load_config(path)
        gc = cfg.games["melee"]
        assert gc.dolphin_path == "/path/dolphin"
        assert gc.iso_path is None
        assert gc.connect_code is None
        assert gc.online_delay is None

    def test_multiple_games(self, config_dir):
        path = _write_config(config_dir, """\
            [games.melee]
            dolphin = "/melee/dolphin"
            iso = "/melee/iso"

            [games.rivals]
            dolphin = "/rivals/dolphin"
            iso = "/rivals/iso"
        """)
        cfg = load_config(path)
        assert "melee" in cfg.games
        assert "rivals" in cfg.games
        assert cfg.games["melee"].dolphin_path == "/melee/dolphin"
        assert cfg.games["rivals"].dolphin_path == "/rivals/dolphin"

    def test_empty_file(self, config_dir):
        path = config_dir / "config.toml"
        path.write_text("")
        cfg = load_config(path)
        assert cfg.games == {}
        assert cfg.arena_server is None


class TestGetGameConfig:
    def test_returns_game_config(self, config_dir):
        path = _write_config(config_dir, """\
            [games.melee]
            dolphin = "/path/dolphin"
            iso = "/path/iso"
        """)
        gc = get_game_config("melee", path)
        assert gc is not None
        assert gc.dolphin_path == "/path/dolphin"

    def test_returns_none_for_unconfigured_game(self, config_dir):
        path = _write_config(config_dir, """\
            [games.melee]
            dolphin = "/path/dolphin"
        """)
        gc = get_game_config("rivals", path)
        assert gc is None

    def test_returns_none_for_missing_file(self, config_dir):
        missing = config_dir / "nonexistent.toml"
        gc = get_game_config("melee", missing)
        assert gc is None

    def test_default_game_is_melee(self, config_dir):
        path = _write_config(config_dir, """\
            [games.melee]
            dolphin = "/default/dolphin"
        """)
        gc = get_game_config(path=path)
        assert gc is not None
        assert gc.dolphin_path == "/default/dolphin"
