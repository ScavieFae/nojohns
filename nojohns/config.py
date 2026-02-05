"""
nojohns/config.py - Local configuration management

Reads user config from ~/.nojohns/config.toml. Game-specific settings
live under [games.<game>] sections. Currently only melee exists; when
a second game arrives, it gets [games.rivals] or similar — no refactor needed.

Config file location: ~/.nojohns/config.toml

Example:
    [games.melee]
    dolphin = "~/Library/Application Support/Slippi Launcher/netplay"
    iso = "~/games/melee/melee.ciso"
    connect_code = "SCAV#382"
    dolphin_home = "~/Library/Application Support/Slippi Dolphin"
    online_delay = 6
    input_throttle = 3

    [arena]
    server = "http://localhost:8000"
"""

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

CONFIG_DIR = Path.home() / ".nojohns"
CONFIG_PATH = CONFIG_DIR / "config.toml"


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


@dataclass
class WalletConfig:
    """Agent wallet for onchain signing."""

    address: str | None = None
    private_key: str | None = None


@dataclass
class ChainConfig:
    """Blockchain network configuration."""

    chain_id: int = 10143  # Monad testnet
    rpc_url: str = "https://testnet-rpc.monad.xyz"
    match_proof: str | None = None  # MatchProof contract address
    wager: str | None = None  # Wager contract address
    # ERC-8004 registries (deployed singletons)
    identity_registry: str | None = None  # IdentityRegistry address
    reputation_registry: str | None = None  # ReputationRegistry address
    agent_id: int | None = None  # Our registered agent NFT token ID


@dataclass
class NojohnsConfig:
    """Top-level configuration."""

    games: dict[str, GameConfig]
    arena_server: str | None = None
    wallet: WalletConfig | None = None
    chain: ChainConfig | None = None

    def __init__(
        self,
        games: dict[str, GameConfig] | None = None,
        arena_server: str | None = None,
        wallet: WalletConfig | None = None,
        chain: ChainConfig | None = None,
    ):
        self.games = games or {}
        self.arena_server = arena_server
        self.wallet = wallet
        self.chain = chain


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
        chain = ChainConfig(
            chain_id=chain_data.get("chain_id", 10143),
            rpc_url=chain_data.get("rpc_url", "https://testnet-rpc.monad.xyz"),
            match_proof=chain_data.get("match_proof"),
            wager=chain_data.get("wager"),
            identity_registry=chain_data.get("identity_registry"),
            reputation_registry=chain_data.get("reputation_registry"),
            agent_id=chain_data.get("agent_id"),
        )

    return NojohnsConfig(
        games=games, arena_server=arena_server, wallet=wallet, chain=chain
    )


def get_game_config(game: str = "melee", path: Path | None = None) -> GameConfig | None:
    """
    Convenience: load config and return a specific game's settings.

    Returns None if the game isn't configured.
    """
    config = load_config(path)
    return config.games.get(game)
