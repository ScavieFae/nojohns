"""
nojohns/config.py - Local configuration management

Reads user config from a platform-appropriate config directory:
  - macOS/Linux: ~/.nojohns/config.toml
  - Windows: %APPDATA%\\nojohns\\config.toml

Game-specific settings live under [games.<game>] sections. Currently only
melee exists; when a second game arrives, it gets [games.rivals] or similar
— no refactor needed.

Example:
    [games.melee]
    dolphin = "~/Library/Application Support/Slippi Launcher/netplay"
    iso = "~/games/melee/melee.ciso"
    connect_code = "SCAV#382"
    dolphin_home = "~/Library/Application Support/Slippi Dolphin"
    online_delay = 6
    input_throttle = 1

    [arena]
    server = "http://localhost:8000"  # Omit to use the public arena
"""

import logging
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================


def _get_config_dir() -> Path:
    """Get platform-appropriate config directory."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "nojohns"
    return Path.home() / ".nojohns"


def default_dolphin_path() -> str:
    """Platform-appropriate default Dolphin path for setup prompts."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return str(Path(appdata) / "Slippi Launcher" / "netplay")
        return ""
    elif sys.platform == "darwin":
        return "~/Library/Application Support/Slippi Launcher/netplay"
    else:
        return "~/.config/Slippi Launcher/netplay"


CONFIG_DIR = _get_config_dir()
CONFIG_PATH = CONFIG_DIR / "config.toml"

# Public arena server — new users connect here out of the box.
# Operators who run their own arena override this in config.toml [arena] server.
DEFAULT_ARENA_URL = "https://nojohns-arena-production.up.railway.app"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class GameConfig:
    """Per-game paths and settings."""

    dolphin_path: str | None = None
    iso_path: str | None = None
    connect_code: str | None = None
    dolphin_home: str | None = None
    online_delay: int | None = None
    input_throttle: int | None = None
    replay_dir: str | None = None  # Where to save Slippi replays
    wager_amount: float | None = None  # MON per match (auto-wager)


@dataclass
class WalletConfig:
    """Agent wallet for onchain signing."""

    address: str | None = None
    private_key: str | None = None


@dataclass
class ChainConfig:
    """Blockchain network configuration."""

    chain_id: int = 143  # Monad mainnet
    rpc_url: str = "https://rpc.monad.xyz"
    # Contract addresses — mainnet defaults from contracts/deployments.json
    match_proof: str = "0x1CC748475F1F666017771FB49131708446B9f3DF"
    wager: str = "0x8d4D9FD03242410261b2F9C0e66fE2131AE0459d"
    prediction_pool: str = "0x33E65E300575D11a42a579B2675A63cb4374598D"
    # ERC-8004 registries (deployed singletons on mainnet)
    identity_registry: str = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"
    reputation_registry: str = "0x8004BAa17C55a88189AE136b182e5fdA19dE9b63"
    agent_id: int | None = None  # Our registered agent NFT token ID


@dataclass
class MoltbotConfig:
    """Configuration for autonomous agent mode (nojohns auto)."""

    risk_profile: str = "moderate"  # conservative, moderate, aggressive
    cooldown_seconds: int = 30
    min_bankroll: float = 0.01  # MON — stop threshold
    tilt_threshold: int = 3  # consecutive losses before refusing wagers


@dataclass
class NojohnsConfig:
    """Top-level configuration."""

    games: dict[str, GameConfig]
    arena_server: str = DEFAULT_ARENA_URL
    wallet: WalletConfig | None = None
    chain: ChainConfig | None = None
    moltbot: MoltbotConfig | None = None

    def __init__(
        self,
        games: dict[str, GameConfig] | None = None,
        arena_server: str | None = None,
        wallet: WalletConfig | None = None,
        chain: ChainConfig | None = None,
        moltbot: MoltbotConfig | None = None,
    ):
        self.games = games or {}
        self.arena_server = arena_server or DEFAULT_ARENA_URL
        self.wallet = wallet
        self.chain = chain
        self.moltbot = moltbot


# ============================================================================
# Parsing
# ============================================================================

def _expand(path: str | None) -> str | None:
    """Expand ~ in a path string."""
    if path is None:
        return None
    return str(Path(path).expanduser())


def _parse_game_config(data: dict) -> GameConfig:
    """Parse a [games.<game>] section into a GameConfig."""
    return GameConfig(
        dolphin_path=_expand(data.get("dolphin")),
        iso_path=_expand(data.get("iso")),
        connect_code=data.get("connect_code"),
        dolphin_home=_expand(data.get("dolphin_home")),
        online_delay=data.get("online_delay"),
        input_throttle=data.get("input_throttle"),
        replay_dir=_expand(data.get("replay_dir")),
        wager_amount=data.get("wager_amount"),
    )


def load_config(path: Path | None = None) -> NojohnsConfig:
    """
    Read config from TOML file.

    Args:
        path: Override config file path (default: ~/.nojohns/config.toml)

    Returns:
        NojohnsConfig. Missing file or bad TOML returns empty config.
    """
    config_path = path or CONFIG_PATH

    if not config_path.exists():
        return NojohnsConfig()

    try:
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
    except Exception as e:
        logger.warning(f"Failed to parse {config_path}: {e}")
        return NojohnsConfig()

    # Parse [games.*] sections
    games = {}
    for game_name, game_data in raw.get("games", {}).items():
        if isinstance(game_data, dict):
            games[game_name] = _parse_game_config(game_data)

    # Parse [arena] section
    arena_data = raw.get("arena", {})
    arena_server = arena_data.get("server") if isinstance(arena_data, dict) else None

    # Parse [wallet] section
    wallet = None
    if "wallet" in raw and isinstance(raw["wallet"], dict):
        wallet_data = raw["wallet"]
        wallet = WalletConfig(
            address=wallet_data.get("address"),
            private_key=wallet_data.get("private_key"),
        )

    # Parse [chain] section
    chain = None
    if "chain" in raw and isinstance(raw["chain"], dict):
        chain_data = raw["chain"]
        _defaults = ChainConfig()
        chain = ChainConfig(
            chain_id=chain_data.get("chain_id", _defaults.chain_id),
            rpc_url=chain_data.get("rpc_url", _defaults.rpc_url),
            match_proof=chain_data.get("match_proof", _defaults.match_proof),
            wager=chain_data.get("wager", _defaults.wager),
            prediction_pool=chain_data.get("prediction_pool", _defaults.prediction_pool),
            identity_registry=chain_data.get("identity_registry", _defaults.identity_registry),
            reputation_registry=chain_data.get("reputation_registry", _defaults.reputation_registry),
            agent_id=chain_data.get("agent_id"),
        )

    # Parse [moltbot] section
    moltbot = None
    if "moltbot" in raw and isinstance(raw["moltbot"], dict):
        moltbot_data = raw["moltbot"]
        moltbot = MoltbotConfig(
            risk_profile=moltbot_data.get("risk_profile", "moderate"),
            cooldown_seconds=moltbot_data.get("cooldown_seconds", 30),
            min_bankroll=moltbot_data.get("min_bankroll", 0.01),
            tilt_threshold=moltbot_data.get("tilt_threshold", 3),
        )

    return NojohnsConfig(
        games=games, arena_server=arena_server, wallet=wallet, chain=chain,
        moltbot=moltbot,
    )


def get_game_config(game: str = "melee", path: Path | None = None) -> GameConfig | None:
    """
    Convenience: load config and return a specific game's settings.

    Returns None if the game isn't configured.
    """
    config = load_config(path)
    return config.games.get(game)
