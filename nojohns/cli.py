#!/usr/bin/env python3
"""
nojohns/cli.py - Command line interface for No Johns

Usage:
    nojohns setup                  # Create ~/.nojohns/ config
    nojohns setup melee            # Configure Melee paths
    nojohns setup melee phillip    # Install Phillip fighter
    nojohns setup wallet           # Configure wallet + chain (onchain features)
    nojohns setup identity         # Register agent on ERC-8004 IdentityRegistry
    nojohns fight <f1> <f2>        # Run a local fight
    nojohns netplay <f> --code X   # Netplay against remote opponent
    nojohns matchmake <f>          # Arena matchmaking
    nojohns auto <f>               # Autonomous agent loop
    nojohns list-fighters
    nojohns info <fighter>
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from nojohns.config import (
    CONFIG_DIR,
    CONFIG_PATH,
    GameConfig,
    NojohnsConfig,
    default_dolphin_path,
    load_config,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Arg Resolution
# ============================================================================

# Characters suitable for random selection (excluding clones, glitch chars, and
# characters that break menu navigation — see CLAUDE.md gotchas 20)
# Excluded: Sheik (Zelda down-B transform, can't select from CSS),
# Zelda (shares CSS slot with Sheik, breaks netplay char select),
# ICs/Popo (doesn't exist in libmelee as ICECLIMBERS — see gotcha 20)
_RANDOM_POOL = [
    "FOX", "FALCO", "MARTH", "CPTFALCON", "PEACH", "JIGGLYPUFF",
    "SAMUS", "DK", "PIKACHU", "LUIGI", "LINK", "YLINK", "DOC",
    "MARIO", "GANONDORF", "MEWTWO", "ROY", "GAMEANDWATCH", "NESS",
    "YOSHI", "BOWSER", "KIRBY", "PICHU",
]


def _resolve_character(char_name: str, Character):
    """Resolve a character name, with support for 'RANDOM'."""
    import random
    if char_name.upper() == "RANDOM":
        picked = random.choice(_RANDOM_POOL)
        logger.info(f"Random character: {picked}")
        return Character[picked]
    return Character[char_name.upper()]


def _resolve_args(args, game_cfg: GameConfig | None, nj_cfg: NojohnsConfig):
    """Merge CLI args with config. CLI wins. Mutates args in place."""
    # dolphin: CLI > config
    if getattr(args, "dolphin", None) is None and game_cfg and game_cfg.dolphin_path:
        args.dolphin = game_cfg.dolphin_path
    # iso: CLI > config
    if getattr(args, "iso", None) is None and game_cfg and game_cfg.iso_path:
        args.iso = game_cfg.iso_path
    # dolphin_home: CLI > config
    if getattr(args, "dolphin_home", None) is None and game_cfg and game_cfg.dolphin_home:
        args.dolphin_home = game_cfg.dolphin_home
    # connect_code for matchmake: CLI > config
    if hasattr(args, "code") and args.code is None and game_cfg and game_cfg.connect_code:
        args.code = game_cfg.connect_code
    # server for matchmake: CLI > config
    if hasattr(args, "server") and args.server is None and nj_cfg.arena_server:
        args.server = nj_cfg.arena_server
    # delay: CLI > config > hardcoded default
    if hasattr(args, "delay") and args.delay is None:
        args.delay = game_cfg.online_delay if game_cfg and game_cfg.online_delay is not None else 6
    # throttle: CLI > config > hardcoded default
    if hasattr(args, "throttle") and args.throttle is None:
        args.throttle = game_cfg.input_throttle if game_cfg and game_cfg.input_throttle is not None else 1
    # replay_dir: CLI > config (default None = no replay saving)
    if hasattr(args, "replay_dir") and args.replay_dir is None and game_cfg and game_cfg.replay_dir:
        args.replay_dir = game_cfg.replay_dir
    # wager: CLI > config (default None = no wager)
    if hasattr(args, "wager") and args.wager is None and game_cfg and game_cfg.wager_amount is not None:
        args.wager = game_cfg.wager_amount


def _require_melee_args(args) -> bool:
    """Validate required Melee args are present after resolution. Returns True if OK."""
    ok = True
    if getattr(args, "dolphin", None) is None:
        logger.error("--dolphin/-d required (run 'nojohns setup melee' or pass explicitly)")
        ok = False
    if getattr(args, "iso", None) is None:
        logger.error("--iso/-i required (run 'nojohns setup melee' or pass explicitly)")
        ok = False
    return ok


# ============================================================================
# Setup Commands
# ============================================================================

def cmd_setup(args):
    """Handle `nojohns setup [melee [phillip] | wallet | identity]`."""
    target = args.setup_target

    if not target:
        return _setup_core()
    elif target == ["melee"]:
        return _setup_melee()
    elif target == ["melee", "phillip"]:
        return _setup_melee_phillip()
    elif target in (["wallet"], ["monad"]):  # "monad" kept as alias
        return _setup_monad()
    elif target == ["identity"]:
        return _setup_identity()
    else:
        logger.error(f"Unknown setup target: {' '.join(target)}")
        logger.error("Usage: nojohns setup [melee [phillip] | wallet | identity]")
        return 1


def _setup_core():
    """Phase 1: Create ~/.nojohns/ directory and bare config."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if CONFIG_PATH.exists():
        print(f"Config already exists: {CONFIG_PATH}")
        print("To reconfigure Melee: nojohns setup melee")
        return 0

    CONFIG_PATH.write_text(
        "# No Johns configuration\n"
        "# See: nojohns setup melee\n"
        "\n"
        "[arena]\n"
        '# server = "http://localhost:8000"\n'
    )

    print(f"Created {CONFIG_PATH}")
    print("Next: run 'nojohns setup melee' to configure Melee")
    return 0


def _setup_melee():
    """Phase 2: Interactive Melee configuration."""
    import tomllib

    # Ensure config dir exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Read existing config if present
    existing_raw = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "rb") as f:
                existing_raw = tomllib.load(f)
        except Exception:
            pass

    # Current values for defaults
    current = existing_raw.get("games", {}).get("melee", {})

    print("Melee Setup")
    print("=" * 40)
    print()

    # Dolphin path
    default_dolphin = current.get("dolphin", default_dolphin_path())
    dolphin = input(f"Dolphin path [{default_dolphin}]: ").strip() or default_dolphin

    # Validate
    dolphin_expanded = str(Path(dolphin).expanduser())
    if not Path(dolphin_expanded).exists():
        print(f"  Warning: {dolphin_expanded} does not exist")

    # ISO path
    default_iso = current.get("iso", "")
    iso_prompt = f"Melee ISO path [{default_iso}]: " if default_iso else "Melee ISO path: "
    iso = input(iso_prompt).strip() or default_iso
    if not iso:
        logger.error("ISO path is required")
        return 1

    iso_expanded = str(Path(iso).expanduser())
    if not Path(iso_expanded).exists():
        print(f"  Warning: {iso_expanded} does not exist")

    # Connect code (optional)
    default_code = current.get("connect_code", "")
    code_prompt = f"Slippi connect code (optional) [{default_code}]: " if default_code else "Slippi connect code (optional): "
    code = input(code_prompt).strip() or default_code

    # Dolphin home (optional)
    default_home = current.get("dolphin_home", "")
    home_prompt = (
        f"Dolphin home dir (optional, for netplay) [{default_home}]: "
        if default_home
        else "Dolphin home dir (optional, for netplay): "
    )
    dolphin_home = input(home_prompt).strip() or default_home

    # Build the melee section
    melee_lines = []
    melee_lines.append(f'dolphin = "{dolphin}"')
    melee_lines.append(f'iso = "{iso}"')
    if code:
        melee_lines.append(f'connect_code = "{code}"')
    if dolphin_home:
        melee_lines.append(f'dolphin_home = "{dolphin_home}"')

    # Delay and throttle — keep existing or use defaults
    delay = current.get("online_delay", 6)
    throttle = current.get("input_throttle", 1)
    melee_lines.append(f"online_delay = {delay}")
    melee_lines.append(f"input_throttle = {throttle}")

    melee_section = "\n".join(melee_lines)

    # Preserve arena section
    arena_section = ""
    arena_data = existing_raw.get("arena", {})
    if arena_data.get("server"):
        arena_section = f'\n[arena]\nserver = "{arena_data["server"]}"\n'
    else:
        arena_section = (
            "\n[arena]\n"
            '# server = "http://localhost:8000"\n'
        )

    # Write config
    config_content = (
        "# No Johns configuration\n"
        "\n"
        "[games.melee]\n"
        f"{melee_section}\n"
        f"{arena_section}"
    )
    CONFIG_PATH.write_text(config_content)

    print()
    print(f"Saved to {CONFIG_PATH}")
    print()
    print("Ready! Try: nojohns fight random do-nothing")
    print("Want Phillip (neural net AI)? Run: nojohns setup melee phillip")
    return 0


def _setup_melee_phillip():
    """Phase 3: Install Phillip fighter dependencies and model."""
    import subprocess

    project_root = Path(__file__).resolve().parent.parent
    fighters_phillip = project_root / "fighters" / "phillip"
    slippi_ai_dir = fighters_phillip / "slippi-ai"
    models_dir = fighters_phillip / "models"
    model_path = models_dir / "all_d21_imitation_v3.pkl"

    # Determine the pip/python to use
    venv_pip = Path(sys.executable).parent / "pip"
    python = sys.executable

    print("Phillip Setup")
    print("=" * 40)
    print()

    # Step 1: Install Python deps
    print("Installing Phillip Python dependencies...")
    result = subprocess.run(
        [str(venv_pip), "install", "-e", f"{project_root}[phillip]"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"pip install failed:\n{result.stderr}")
        return 1
    print("  Python dependencies installed")

    # Step 2: Clone slippi-ai if not present
    if not slippi_ai_dir.exists():
        print("Cloning slippi-ai...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/vladfi1/slippi-ai.git", str(slippi_ai_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"git clone failed:\n{result.stderr}")
            return 1
        print("  slippi-ai cloned")
    else:
        print("  slippi-ai already present")

    # Install slippi-ai
    print("Installing slippi-ai...")
    result = subprocess.run(
        [str(venv_pip), "install", "-e", str(slippi_ai_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"slippi-ai install failed:\n{result.stderr}")
        return 1
    print("  slippi-ai installed")

    # Step 3: Download model weights if not present
    if not model_path.exists():
        print("Downloading model weights (~40 MB)...")
        models_dir.mkdir(parents=True, exist_ok=True)
        import urllib.request
        model_url = "https://dl.dropbox.com/scl/fi/bppnln3rfktxfdocottuw/all_d21_imitation_v3?rlkey=46yqbsp7vi5222x04qt4npbkq&st=6knz106y&dl=1"
        try:
            urllib.request.urlretrieve(model_url, str(model_path))
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return 1
        print("  Model downloaded")
    else:
        print("  Model weights already present")

    # Step 4: Verify
    print()
    print("Verifying...")

    # Check TF import
    result = subprocess.run(
        [python, "-c", "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"  {result.stdout.strip()} OK")
    else:
        logger.warning(f"TensorFlow import failed: {result.stderr.strip()}")

    # Check model loads
    result = subprocess.run(
        [
            python, "-c",
            "from slippi_ai import saving; "
            f"s = saving.load_state_from_disk('{model_path}'); "
            "print(f'Model OK (delay={s[\"config\"][\"policy\"][\"delay\"]})')",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"  {result.stdout.strip()}")
    else:
        logger.warning(f"Model load check failed: {result.stderr.strip()}")

    # Check fighter registry
    result = subprocess.run(
        [python, "-m", "nojohns.cli", "list-fighters"],
        capture_output=True,
        text=True,
    )
    if "phillip" in result.stdout.lower():
        print("  Phillip shows up in fighter registry")
    else:
        logger.warning("Phillip not found in fighter registry (may need fighter.toml)")

    print()
    print("Phillip setup complete!")
    return 0


def _setup_monad():
    """Configure agent wallet and Monad chain settings."""
    import tomllib

    # Ensure config dir exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Read existing config if present
    existing_raw = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "rb") as f:
                existing_raw = tomllib.load(f)
        except Exception:
            pass

    current_wallet = existing_raw.get("wallet", {})
    current_chain = existing_raw.get("chain", {})

    print("Monad Setup")
    print("=" * 40)
    print()

    # Check if wallet already exists
    if current_wallet.get("address"):
        print(f"Wallet already configured: {current_wallet['address']}")
        print()
        overwrite = input("Overwrite existing wallet? [y/N]: ").strip().lower()
        if overwrite != "y":
            print("Keeping existing wallet.")
            address = current_wallet["address"]
            private_key = current_wallet.get("private_key", "")
        else:
            address, private_key = _monad_wallet_prompt()
    else:
        address, private_key = _monad_wallet_prompt()

    # Chain config
    print()
    print("Chain Configuration")
    print("-" * 40)
    default_chain_id = current_chain.get("chain_id", 10143)
    default_rpc = current_chain.get("rpc_url", "https://testnet-rpc.monad.xyz")

    chain_choice = input(f"Network — testnet (default) or mainnet? [testnet]: ").strip().lower()
    if chain_choice == "mainnet":
        chain_id = 143
        rpc_url = "https://rpc.monad.xyz"
        print("  Using Monad mainnet (chain 143)")
    else:
        chain_id = 10143
        rpc_url = "https://testnet-rpc.monad.xyz"
        print("  Using Monad testnet (chain 10143)")

    # Contract addresses (optional — ScavieFae deploys these)
    match_proof = current_chain.get("match_proof", "")
    wager = current_chain.get("wager", "")

    # Build updated config, preserving existing sections
    lines = ["# No Johns configuration\n"]

    # Preserve [games.*]
    games_data = existing_raw.get("games", {})
    for game_name, game_settings in games_data.items():
        if isinstance(game_settings, dict):
            lines.append(f"[games.{game_name}]")
            for k, v in game_settings.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                else:
                    lines.append(f"{k} = {v}")
            lines.append("")

    # Preserve [arena]
    arena_data = existing_raw.get("arena", {})
    if arena_data.get("server"):
        lines.append("[arena]")
        lines.append(f'server = "{arena_data["server"]}"')
    else:
        lines.append("[arena]")
        lines.append('# server = "http://localhost:8000"')
    lines.append("")

    # [wallet]
    lines.append("[wallet]")
    lines.append(f'address = "{address}"')
    lines.append(f'private_key = "{private_key}"')
    lines.append("")

    # [chain]
    lines.append("[chain]")
    lines.append(f"chain_id = {chain_id}")
    lines.append(f'rpc_url = "{rpc_url}"')
    if match_proof:
        lines.append(f'match_proof = "{match_proof}"')
    else:
        lines.append('# match_proof = "0x..."  # set after contract deploy')
    if wager:
        lines.append(f'wager = "{wager}"')
    else:
        lines.append('# wager = "0x..."  # set after contract deploy')
    lines.append("")

    CONFIG_PATH.write_text("\n".join(lines))

    print()
    print(f"Saved to {CONFIG_PATH}")
    print()
    print(f"Agent address: {address}")
    print()
    print("IMPORTANT: Your private key is stored in plaintext in config.toml.")
    print("           Back it up securely. Do not commit it to git.")
    print()
    print("Next steps:")
    print("  1. Fund your agent wallet with testnet MON")
    print("  2. Contract addresses will be added after deploy")
    return 0


def _setup_identity():
    """Register agent on ERC-8004 IdentityRegistry."""
    import base64
    import json
    import tomllib

    # Load existing config
    if not CONFIG_PATH.exists():
        logger.error("No config found. Run: nojohns setup wallet")
        return 1

    with open(CONFIG_PATH, "rb") as f:
        existing_raw = tomllib.load(f)

    # Check wallet is configured
    wallet_data = existing_raw.get("wallet", {})
    if not wallet_data.get("private_key"):
        logger.error("No wallet configured. Run: nojohns setup wallet")
        return 1

    chain_data = existing_raw.get("chain", {})
    chain_id = chain_data.get("chain_id", 10143)

    # Check if already registered
    if chain_data.get("agent_id"):
        print(f"Already registered as agent #{chain_data['agent_id']}")
        print()
        overwrite = input("Register a new agent? [y/N]: ").strip().lower()
        if overwrite != "y":
            return 0

    # Get melee config for connect code
    melee_data = existing_raw.get("games", {}).get("melee", {})
    connect_code = melee_data.get("connect_code", "")

    print("Identity Registration (ERC-8004)")
    print("=" * 40)
    print()

    # Determine registry address based on chain
    if chain_id == 143:
        identity_registry = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"
        print("Network: Monad mainnet")
    else:
        identity_registry = "0x8004A818BFB912233c491871b3d84c89A494BD9e"
        print("Network: Monad testnet")
    print()

    # Prompt for agent details
    default_name = "NoJohnsAgent"
    name = input(f"Agent name [{default_name}]: ").strip() or default_name

    default_desc = "Melee competitor on No Johns arena"
    description = input(f"Description [{default_desc}]: ").strip() or default_desc

    if not connect_code:
        connect_code = input("Slippi connect code (e.g. SCAV#382): ").strip()

    # Build registration JSON
    registration = {
        "type": "https://eips.ethereum.org/EIPS/eip-8004#registration-v1",
        "name": name,
        "description": description,
        "image": "",
        "services": [
            {"name": "nojohns-arena", "version": "v1"}
        ],
        "active": True,
        "games": {
            "melee": {"slippi_code": connect_code}
        }
    }

    # Encode as data URI
    json_bytes = json.dumps(registration, separators=(",", ":")).encode("utf-8")
    b64 = base64.b64encode(json_bytes).decode("ascii")
    agent_uri = f"data:application/json;base64,{b64}"

    print()
    print("Registration JSON:")
    print(json.dumps(registration, indent=2))
    print()

    confirm = input("Register this agent onchain? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("Cancelled.")
        return 0

    # Call register() on IdentityRegistry
    try:
        from web3 import Web3
        from eth_account import Account
    except ImportError:
        logger.error("web3 and eth-account required: pip install nojohns[wallet]")
        return 1

    rpc_url = chain_data.get("rpc_url", "https://testnet-rpc.monad.xyz")
    w3 = Web3(Web3.HTTPProvider(rpc_url))

    if not w3.is_connected():
        logger.error(f"Cannot connect to {rpc_url}")
        return 1

    private_key = wallet_data["private_key"]
    account = Account.from_key(private_key)

    print(f"Registering from {account.address}...")

    # IdentityRegistry ABI (just register function)
    abi = [
        {
            "inputs": [{"name": "agentURI", "type": "string"}],
            "name": "register",
            "outputs": [{"type": "uint256"}],
            "stateMutability": "nonpayable",
            "type": "function"
        }
    ]

    contract = w3.eth.contract(address=identity_registry, abi=abi)

    try:
        # Estimate gas
        gas_estimate = contract.functions.register(agent_uri).estimate_gas({
            "from": account.address,
        })
        gas_limit = int(gas_estimate * 1.2)  # 20% buffer
        print(f"Estimated gas: {gas_estimate} (using {gas_limit})")

        # Build transaction
        nonce = w3.eth.get_transaction_count(account.address)
        tx = contract.functions.register(agent_uri).build_transaction({
            "from": account.address,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": w3.eth.gas_price,
            "chainId": chain_id,
        })

        # Sign and send
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"Transaction sent: {tx_hash.hex()}")

        # Wait for receipt
        print("Waiting for confirmation...")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

        if receipt.status != 1:
            logger.error("Transaction failed")
            return 1

        # Parse logs to get agentId (tokenId from Transfer event)
        # Transfer(address from, address to, uint256 tokenId)
        transfer_topic = w3.keccak(text="Transfer(address,address,uint256)")
        for log in receipt.logs:
            if log.topics[0] == transfer_topic:
                agent_id = int(log.topics[3].hex(), 16)
                print(f"Registered as agent #{agent_id}")
                break
        else:
            logger.error("Could not parse agentId from logs")
            return 1

    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return 1

    # Update config with agent_id and registry addresses
    _update_config_identity(existing_raw, agent_id, identity_registry, chain_id)

    print()
    print(f"Saved agent_id to {CONFIG_PATH}")
    print()
    print("Your agent is now registered on ERC-8004 IdentityRegistry!")
    return 0


def _update_config_identity(existing_raw: dict, agent_id: int, identity_registry: str, chain_id: int):
    """Update config.toml with identity registration info."""
    # Determine reputation registry
    if chain_id == 143:
        reputation_registry = "0x8004BAa17C55a88189AE136b182e5fdA19dE9b63"
    else:
        reputation_registry = "0x8004B663056A597Dffe9eCcC1965A193B7388713"

    lines = ["# No Johns configuration\n"]

    # Preserve [games.*]
    games_data = existing_raw.get("games", {})
    for game_name, game_settings in games_data.items():
        if isinstance(game_settings, dict):
            lines.append(f"[games.{game_name}]")
            for k, v in game_settings.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                else:
                    lines.append(f"{k} = {v}")
            lines.append("")

    # Preserve [arena]
    arena_data = existing_raw.get("arena", {})
    if arena_data.get("server"):
        lines.append("[arena]")
        lines.append(f'server = "{arena_data["server"]}"')
    else:
        lines.append("[arena]")
        lines.append('# server = "http://localhost:8000"')
    lines.append("")

    # Preserve [wallet]
    wallet_data = existing_raw.get("wallet", {})
    lines.append("[wallet]")
    lines.append(f'address = "{wallet_data.get("address", "")}"')
    lines.append(f'private_key = "{wallet_data.get("private_key", "")}"')
    lines.append("")

    # [chain] with identity info
    chain_data = existing_raw.get("chain", {})
    lines.append("[chain]")
    lines.append(f"chain_id = {chain_data.get('chain_id', chain_id)}")
    lines.append(f'rpc_url = "{chain_data.get("rpc_url", "https://testnet-rpc.monad.xyz")}"')
    if chain_data.get("match_proof"):
        lines.append(f'match_proof = "{chain_data["match_proof"]}"')
    if chain_data.get("wager"):
        lines.append(f'wager = "{chain_data["wager"]}"')
    lines.append(f'identity_registry = "{identity_registry}"')
    lines.append(f'reputation_registry = "{reputation_registry}"')
    lines.append(f"agent_id = {agent_id}")
    lines.append("")

    CONFIG_PATH.write_text("\n".join(lines))


def _monad_wallet_prompt() -> tuple[str, str]:
    """Prompt user to generate or import a wallet. Returns (address, private_key)."""
    print("Wallet Setup")
    print("-" * 40)
    print("  1. Generate a new wallet")
    print("  2. Import an existing private key")
    print()
    choice = input("Choose [1]: ").strip()

    if choice == "2":
        key = input("Private key (hex, with or without 0x): ").strip()
        if not key:
            logger.error("Private key is required")
            sys.exit(1)
        if not key.startswith("0x"):
            key = "0x" + key
        try:
            from nojohns.wallet import _require_eth_account
            eth_account = _require_eth_account()
            account = eth_account.Account.from_key(key)
            address = account.address
            print(f"  Imported wallet: {address}")
            return (address, key)
        except ImportError:
            logger.error("eth-account required: pip install nojohns[wallet]")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Invalid private key: {e}")
            sys.exit(1)
    else:
        try:
            from nojohns.wallet import generate_wallet
            address, key = generate_wallet()
            print(f"  Generated wallet: {address}")
            return (address, key)
        except ImportError:
            logger.error("eth-account required: pip install nojohns[wallet]")
            sys.exit(1)


# ============================================================================
# Game Commands
# ============================================================================

def cmd_fight(args):
    """Run a fight between two fighters."""
    from games.melee import (
        DolphinConfig,
        MatchSettings,
        MatchRunner,
    )
    from nojohns import DoNothingFighter, RandomFighter
    from melee import Character, Stage

    # Load fighters
    fighter1 = load_fighter(args.fighter1)
    fighter2 = load_fighter(args.fighter2)

    if fighter1 is None or fighter2 is None:
        return 1

    # Set up Dolphin
    dolphin = DolphinConfig(
        dolphin_path=args.dolphin,
        iso_path=args.iso,
    )

    # Match settings
    settings = MatchSettings(
        games=args.games,
        stocks=args.stocks,
        time_minutes=args.time,
        stage=Stage[args.stage.upper()],
        p1_character=_resolve_character(args.p1_char, Character),
        p2_character=_resolve_character(args.p2_char, Character),
    )

    logger.info(f"No Johns - Fight!")
    logger.info(f"   P1: {fighter1.metadata.display_name} ({settings.p1_character.name})")
    logger.info(f"   P2: {fighter2.metadata.display_name} ({settings.p2_character.name})")
    logger.info(f"   Format: Bo{settings.games}, {settings.stocks} stock, {settings.stage.name}")
    logger.info("")

    # Run match
    runner = MatchRunner(dolphin)

    def on_game_end(game):
        logger.info(f"   Game over! P{game.winner_port} wins ({game.p1_stocks}-{game.p2_stocks})")

    try:
        result = runner.run_match(
            fighter1, fighter2, settings,
            on_game_end=on_game_end,
        )

        logger.info("")
        logger.info(f"Match Complete!")
        logger.info(f"   Winner: P{result.winner_port} ({result.score})")

        return 0

    except KeyboardInterrupt:
        logger.info("\nMatch cancelled.")
        return 130
    except Exception as e:
        logger.error(f"Match error: {e}")
        return 1


def _sign_and_submit(match_id: str, outcome: str, _get, _post) -> bool:
    """Sign a match result with EIP-712 and submit to the arena.

    Reads the canonical MatchResult from the arena (deterministic fields
    computed server-side) so both agents sign the exact same data.

    Returns True if signing + submission succeeded, False otherwise.
    Skips silently if wallet or eth-account is not configured.
    """
    import hashlib

    if outcome != "COMPLETED":
        logger.debug(f"Outcome '{outcome}' — skipping match signing (match void)")
        return False

    nj_cfg = load_config()
    if nj_cfg.wallet is None or nj_cfg.wallet.private_key is None:
        logger.debug("No wallet configured — skipping match signing")
        return False
    if nj_cfg.chain is None:
        logger.debug("No chain configured — skipping match signing")
        return False
    if nj_cfg.chain.match_proof is None:
        logger.debug("No MatchProof contract address — skipping match signing")
        return False

    try:
        from nojohns.wallet import load_wallet, sign_match_result
    except ImportError:
        logger.warning("eth-account not installed — skipping match signing (pip install nojohns[wallet])")
        return False

    account = load_wallet(nj_cfg)
    if account is None:
        return False

    # Read the canonical result from the arena — both agents get the same data.
    # Poll briefly: our result may arrive before the opponent's, and the arena
    # only marks the match "completed" once both sides report.
    import time as _time

    match_data = None
    for attempt in range(10):
        try:
            match_data = _get(f"/matches/{match_id}")
        except Exception as e:
            logger.warning(f"Failed to fetch match data for signing: {e}")
            return False

        if match_data.get("status") == "completed":
            break

        if attempt < 9:
            logger.info(f"Waiting for match completion ({attempt + 1}/10)...")
            _time.sleep(1)

    if match_data is None or match_data.get("status") != "completed":
        logger.warning("Match not completed after 10s — skipping signing")
        return False

    winner_wallet = match_data.get("winner_wallet")
    loser_wallet = match_data.get("loser_wallet")
    if not winner_wallet or not loser_wallet:
        logger.debug("Match missing wallet addresses — skipping signing")
        return False

    # Build MatchResult from the arena's canonical fields
    match_result = {
        "matchId": hashlib.sha256(match_id.encode()).digest(),
        "winner": winner_wallet,
        "loser": loser_wallet,
        "gameId": "melee",
        "winnerScore": match_data.get("winner_score", 0),
        "loserScore": match_data.get("loser_score", 0),
        "replayHash": b"\x00" * 32,  # placeholder until replay hashing
        "timestamp": match_data["result_timestamp"],
    }

    try:
        sig = sign_match_result(
            account,
            match_result,
            chain_id=nj_cfg.chain.chain_id,
            contract_address=nj_cfg.chain.match_proof,
        )
        logger.info(f"Match result signed by {account.address}")

        # Submit signature to arena
        _post(f"/matches/{match_id}/signature", {
            "address": account.address,
            "signature": sig.hex(),
        })
        logger.info("Signature submitted to arena.")
    except Exception as e:
        logger.warning(f"Failed to sign/submit match result: {e}")
        return False

    # --- Onchain submission: poll for both signatures, then recordMatch() ---
    _try_record_onchain(match_id, match_result, nj_cfg, account, _get)
    return True


def _try_record_onchain(match_id, match_result, nj_cfg, account, _get):
    """Poll arena for both signatures, then submit recordMatch() onchain.

    Waits up to 30 seconds for the opponent's signature. If both arrive,
    calls recordMatch() on the MatchProof contract. If not, logs and moves on.
    """
    import time

    try:
        from nojohns.contract import record_match, is_recorded
    except ImportError:
        logger.debug("web3 not installed — skipping onchain submission")
        return

    # Check if already recorded (e.g. opponent submitted first)
    try:
        if is_recorded(
            match_result["matchId"],
            rpc_url=nj_cfg.chain.rpc_url,
            contract_address=nj_cfg.chain.match_proof,
        ):
            logger.info("Match already recorded onchain.")
            print("  Onchain: already recorded")
            return
    except Exception as e:
        logger.debug(f"Failed to check recorded status: {e}")

    # Poll for both signatures (opponent may not have signed yet)
    print("  Waiting for opponent signature...", end="", flush=True)
    sig_a = None
    sig_b = None
    for _ in range(15):
        try:
            sigs_data = _get(f"/matches/{match_id}/signatures")
            sigs = sigs_data.get("signatures", [])
            if len(sigs) >= 2:
                sig_a = bytes.fromhex(sigs[0]["signature"])
                sig_b = bytes.fromhex(sigs[1]["signature"])
                break
        except Exception:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    print()

    if sig_a is None or sig_b is None:
        logger.info("Opponent hasn't signed yet — skipping onchain submission")
        print("  Onchain: waiting for opponent (will submit next time)")
        return

    # Check again right before submitting (race condition: opponent may have submitted)
    try:
        if is_recorded(
            match_result["matchId"],
            rpc_url=nj_cfg.chain.rpc_url,
            contract_address=nj_cfg.chain.match_proof,
        ):
            print("  Onchain: recorded (opponent submitted first)")
            return
    except Exception:
        pass

    # Submit to contract
    try:
        print("  Submitting to MatchProof contract...", end="", flush=True)
        tx_hash = record_match(
            match_result,
            sig_a,
            sig_b,
            account,
            rpc_url=nj_cfg.chain.rpc_url,
            contract_address=nj_cfg.chain.match_proof,
        )
        print(f" confirmed!")
        print(f"  tx: 0x{tx_hash}")

        # Post Elo update to ReputationRegistry
        _post_elo_update(match_result, nj_cfg, account)

        # Resolve prediction pool (if one exists for this match)
        _try_resolve_pool(match_id, nj_cfg, account)
    except Exception as e:
        # Check if opponent beat us to it
        try:
            if is_recorded(
                match_result["matchId"],
                rpc_url=nj_cfg.chain.rpc_url,
                contract_address=nj_cfg.chain.match_proof,
            ):
                print(f" already recorded (opponent submitted first)")
                return
        except Exception:
            pass
        logger.warning(f"Failed to record match onchain: {e}")
        print(f" failed: {e}")


def _post_elo_update(match_result: dict, config, account):
    """Post Elo update to ReputationRegistry after match."""
    if config.chain is None or config.chain.reputation_registry is None:
        logger.debug("No reputation registry configured — skipping Elo posting")
        return
    if config.chain.agent_id is None:
        logger.debug("No agent_id configured — skipping Elo posting")
        return

    try:
        from nojohns.reputation import (
            get_current_elo,
            calculate_new_elo,
            post_elo_update,
            STARTING_ELO,
        )
    except ImportError:
        logger.debug("reputation module not available")
        return

    # Determine if we won
    our_address = account.address.lower()
    winner_address = match_result["winner"].lower()
    we_won = our_address == winner_address

    # Get current Elo
    current = get_current_elo(
        config.chain.agent_id,
        config.chain.rpc_url,
        config.chain.reputation_registry,
    )

    # For opponent Elo, use default (we don't know their agent_id easily)
    # TODO: look up opponent's agent_id from IdentityRegistry by wallet
    opponent_elo = STARTING_ELO

    # Calculate new Elo
    new_elo = calculate_new_elo(current.elo, opponent_elo, we_won)
    peak_elo = max(current.peak_elo, new_elo)

    # Update record
    if we_won:
        wins = current.wins + 1
        losses = current.losses
    else:
        wins = current.wins
        losses = current.losses + 1
    record = f"{wins}-{losses}"

    print(f"  Posting Elo update: {current.elo} → {new_elo} ({'+' if we_won else ''}{new_elo - current.elo})")

    tx_hash = post_elo_update(
        config.chain.agent_id,
        new_elo,
        peak_elo,
        record,
        account,
        config.chain.rpc_url,
        config.chain.reputation_registry,
        config.chain.chain_id,
    )

    if tx_hash:
        print(f"  Elo tx: 0x{tx_hash}")
    else:
        print("  Elo update failed (non-critical)")


def _try_resolve_pool(match_id: str, config, account):
    """Resolve the prediction pool for a match after recordMatch() succeeds.

    Reads pool_id from the arena, then calls resolve() on PredictionPool.
    Silently skips if no pool exists or prediction_pool not configured.
    """
    if config.chain is None or config.chain.prediction_pool is None:
        return

    # Get pool_id from arena
    arena_server = config.arena_server
    try:
        import urllib.request
        import json

        url = f"{arena_server}/matches/{match_id}/pool"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        pool_id = data.get("pool_id")
    except Exception as e:
        logger.debug(f"Failed to fetch pool info: {e}")
        return

    if pool_id is None:
        logger.debug("No prediction pool for this match")
        return

    try:
        from nojohns.contract import resolve_pool

        print(f"  Resolving prediction pool {pool_id}...", end="", flush=True)
        tx_hash = resolve_pool(
            pool_id, account,
            rpc_url=config.chain.rpc_url,
            contract_address=config.chain.prediction_pool,
        )
        print(f" resolved! tx: 0x{tx_hash[:16]}...")
    except ImportError:
        logger.debug("web3 not installed — skipping pool resolution")
    except Exception as e:
        # Pool might not exist, or opponent resolved first
        if "already" in str(e).lower() or "revert" in str(e).lower():
            print(" already resolved.")
        else:
            logger.warning(f"Failed to resolve prediction pool: {e}")
            print(f" failed (non-critical): {e}")


def _auto_settle_wager(match_id: str, wager_id: int, wager_amount: int | None, config):
    """Auto-settle a wager after match completion."""
    from nojohns.wallet import load_wallet, settle_wager

    account = load_wallet(config)
    if account is None:
        return

    if config.chain is None or config.chain.wager is None:
        return

    # Convert match_id (UUID string) to bytes32
    # Strip hyphens and pad/truncate to 32 bytes
    match_id_clean = match_id.replace("-", "")
    match_id_bytes = bytes.fromhex(match_id_clean.ljust(64, '0')[:64])

    pot_mon = (wager_amount or 0) * 2 / 10**18
    print(f"  Settling wager {wager_id} ({pot_mon} MON pot)...", end="", flush=True)

    try:
        tx_hash = settle_wager(
            account=account,
            rpc_url=config.chain.rpc_url,
            contract_address=config.chain.wager,
            wager_id=wager_id,
            match_id=match_id_bytes,
        )
        print(f" settled! TX: {tx_hash[:16]}...")
    except Exception as e:
        # Settlement might fail if opponent already settled, or match not recorded yet
        if "already" in str(e).lower():
            print(" already settled.")
        else:
            print(f" failed: {e}")
            print(f"  Settle manually: nojohns wager settle {wager_id} {match_id_clean}")


def _negotiate_wager(
    match_id: str,
    queue_id: str,
    opponent_wallet: str | None,
    config,
    server: str,
    _get,
    _post,
    wager_amount: float | None = None,
) -> tuple[int | None, int | None]:
    """
    Handle wager negotiation after match is made, before game starts.

    Fully autonomous — no interactive prompts. Behavior driven by wager_amount:
      - None → skip wagering entirely
      - float → auto-propose that amount, auto-accept opponent proposals ≤ that amount

    Returns (wager_id, wager_amount_wei) or (None, None) if no wager.
    """
    import time
    import urllib.error

    from nojohns.wallet import load_wallet, propose_wager, accept_wager

    # No wager amount configured → skip entirely
    if wager_amount is None:
        return None, None

    # Check if opponent has a wallet (required for wagers)
    if not opponent_wallet:
        logger.info("Opponent has no wallet — skipping wager.")
        return None, None

    account = load_wallet(config)
    if account is None:
        return None, None

    max_amount_wei = int(wager_amount * 10**18)

    # Check current wager status
    try:
        wager_info = _get(f"/matches/{match_id}/wager")
    except urllib.error.URLError:
        return None, None

    # If opponent already proposed, respond automatically
    if wager_info.get("wager_status") == "proposed":
        return _respond_to_wager(
            match_id, queue_id, wager_info, account, config, _get, _post,
            max_amount_wei=max_amount_wei,
        )

    # Propose our wager amount
    amount_wei = max_amount_wei
    amount_mon = wager_amount

    print(f"Proposing {amount_mon} MON wager...")
    try:
        tx_hash, wager_id = propose_wager(
            account=account,
            rpc_url=config.chain.rpc_url,
            contract_address=config.chain.wager,
            opponent=opponent_wallet,
            game_id="melee",
            amount_wei=amount_wei,
        )
    except Exception as e:
        print(f"Failed to propose wager: {e}")
        return None, None

    print(f"Wager proposed (ID: {wager_id})")

    # Tell arena about the proposal
    try:
        _post(f"/matches/{match_id}/wager/propose", {
            "queue_id": queue_id,
            "amount_wei": amount_wei,
            "wager_id": wager_id,
        })
    except urllib.error.URLError as e:
        print(f"Warning: couldn't register wager with arena: {e}")

    # Wait for opponent to accept (30s timeout)
    print("Waiting for opponent to accept (30s)...", end="", flush=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        time.sleep(1)
        print(".", end="", flush=True)
        try:
            wager_info = _get(f"/matches/{match_id}/wager")
            status = wager_info.get("wager_status")
            if status == "accepted":
                print(" accepted!")
                print(f"Wager locked! Pot: {amount_mon * 2} MON")
                return wager_id, amount_wei
            elif status == "declined":
                print(" declined.")
                # Cancel the onchain wager to get refund
                try:
                    from nojohns.wallet import cancel_wager
                    cancel_wager(account, config.chain.rpc_url, config.chain.wager, wager_id)
                    print("Wager cancelled, MON refunded.")
                except Exception:
                    print(f"Warning: cancel wager {wager_id} manually to get refund.")
                return None, None

            # Check if opponent also proposed — auto-accept if within our max
            opponent_amount = wager_info.get("wager_amount")
            opponent_wager_id = wager_info.get("wager_id")
            proposer = wager_info.get("wager_proposer")
            if (opponent_wager_id and opponent_wager_id != wager_id and
                proposer != account.address and opponent_amount is not None):
                if opponent_amount <= max_amount_wei:
                    print(f" opponent proposed {opponent_amount / 10**18} MON!")
                    print("Auto-accepting opponent's wager...")
                    try:
                        from nojohns.wallet import accept_wager, cancel_wager
                        accept_wager(
                            account=account,
                            rpc_url=config.chain.rpc_url,
                            contract_address=config.chain.wager,
                            wager_id=opponent_wager_id,
                            amount_wei=opponent_amount,
                        )
                        # Cancel our own wager to get refund
                        cancel_wager(account, config.chain.rpc_url, config.chain.wager, wager_id)
                        pot_mon = opponent_amount * 2 / 10**18
                        print(f"Wager locked! Pot: {pot_mon} MON")
                        _post(f"/matches/{match_id}/wager/accept", {"queue_id": queue_id})
                        return opponent_wager_id, opponent_amount
                    except Exception as e:
                        print(f"Auto-accept failed: {e}")
                else:
                    print(f" opponent wants {opponent_amount / 10**18} MON (> our max {wager_amount}).")
                    print("Declining opponent's wager.")
                    try:
                        _post(f"/matches/{match_id}/wager/decline", {"queue_id": queue_id})
                    except urllib.error.URLError:
                        pass

        except urllib.error.URLError:
            continue

    print(" timeout.")
    # Cancel the onchain wager
    try:
        from nojohns.wallet import cancel_wager
        cancel_wager(account, config.chain.rpc_url, config.chain.wager, wager_id)
        print("Wager cancelled, MON refunded.")
    except Exception:
        print(f"Warning: cancel wager {wager_id} manually to get refund.")
    return None, None


def _respond_to_wager(
    match_id: str,
    queue_id: str,
    wager_info: dict,
    account,
    config,
    _get,
    _post,
    max_amount_wei: int | None = None,
) -> tuple[int | None, int | None]:
    """
    Respond to an opponent's wager proposal.

    Fully autonomous — auto-accept if amount ≤ max_amount_wei, decline otherwise.
    """
    import urllib.error

    from nojohns.wallet import accept_wager

    amount_wei = wager_info.get("wager_amount", 0)
    wager_id = wager_info.get("wager_id")
    amount_mon = amount_wei / 10**18

    print(f"Opponent proposed {amount_mon} MON wager.")

    # Decline if over our max (or no max configured)
    if max_amount_wei is None or amount_wei > max_amount_wei:
        max_mon = max_amount_wei / 10**18 if max_amount_wei else 0
        print(f"Declining — exceeds our max ({max_mon} MON).")
        try:
            _post(f"/matches/{match_id}/wager/decline", {"queue_id": queue_id})
        except urllib.error.URLError:
            pass
        return None, None

    # Accept onchain
    print(f"Auto-accepting {amount_mon} MON wager...")
    try:
        tx_hash = accept_wager(
            account=account,
            rpc_url=config.chain.rpc_url,
            contract_address=config.chain.wager,
            wager_id=wager_id,
            amount_wei=amount_wei,
        )
    except Exception as e:
        print(f"Failed to accept wager: {e}")
        try:
            _post(f"/matches/{match_id}/wager/decline", {"queue_id": queue_id})
        except urllib.error.URLError:
            pass
        return None, None

    # Tell arena we accepted
    try:
        _post(f"/matches/{match_id}/wager/accept", {"queue_id": queue_id})
    except urllib.error.URLError as e:
        print(f"Warning: couldn't confirm acceptance with arena: {e}")

    print(f"Wager accepted! Pot: {amount_mon * 2} MON")
    return wager_id, amount_wei


@dataclass
class MatchOutcome:
    """Result of a single arena match. Returned by _run_single_match()."""

    match_id: str
    opponent_code: str
    outcome: str  # COMPLETED, DISCONNECT, ERROR
    we_won: bool
    our_stocks: int
    their_stocks: int
    duration: float
    signed: bool
    wager_id: int | None
    wager_amount_wei: int | None
    opponent_wallet: str | None


def _make_arena_helpers(server: str):
    """Create HTTP helper functions for arena communication."""
    import json
    import urllib.request

    def _post(path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{server}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def _get(path: str) -> dict:
        req = urllib.request.Request(f"{server}{path}")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def _delete(path: str) -> dict:
        req = urllib.request.Request(f"{server}{path}", method="DELETE")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    return _post, _get, _delete


def _run_single_match(
    fighter_name: str,
    fighter,
    args,
    nj_cfg,
    server: str,
    _post,
    _get,
    _delete,
    wager_amount: float | None = None,
) -> MatchOutcome | None:
    """Run a single arena match: join queue → play → report → sign → settle.

    This is the core match loop, shared by cmd_matchmake and cmd_auto.

    Args:
        fighter_name: Name of the fighter (for queue join).
        fighter: Loaded fighter instance.
        args: Parsed CLI args with melee config (dolphin, iso, code, etc).
        nj_cfg: NojohnsConfig with wallet/chain.
        server: Arena server URL (no trailing slash).
        _post, _get, _delete: HTTP helpers from _make_arena_helpers().
        wager_amount: MON to wager (None = no wager).

    Returns:
        MatchOutcome on completion, None if queue timed out or unreachable.
    """
    import time
    import urllib.error

    from games.melee import NetplayConfig, NetplayRunner, NetplayDisconnectedError
    from melee import Character, Stage

    our_wallet = nj_cfg.wallet.address if nj_cfg.wallet and nj_cfg.wallet.address else None
    our_agent_id = nj_cfg.chain.agent_id if nj_cfg.chain and nj_cfg.chain.agent_id else None

    # --- Step 1: Join queue ---
    queue_id = None
    match_id = None
    try:
        try:
            join_body = {
                "connect_code": args.code,
                "fighter_name": fighter_name,
            }
            if our_wallet:
                join_body["wallet_address"] = our_wallet
            if our_agent_id:
                join_body["agent_id"] = our_agent_id
            result = _post("/queue/join", join_body)
        except urllib.error.URLError as e:
            logger.error(f"Cannot reach arena server at {server}: {e}")
            return None

        queue_id = result["queue_id"]
        status = result["status"]

        if status == "matched":
            match_id = result["match_id"]
            opponent_code = result["opponent_code"]
            logger.info(f"Matched! Opponent: {opponent_code}")
        else:
            position = result.get("position", "?")
            logger.info(f"Joined queue (position: {position})")
            logger.info("Waiting for opponent...")

            # --- Step 2: Poll for match ---
            poll_count = 0
            max_polls = 150  # 5 minutes at 2-second intervals
            while status == "waiting" and poll_count < max_polls:
                time.sleep(2)
                poll_count += 1
                try:
                    result = _get(f"/queue/{queue_id}")
                except urllib.error.URLError:
                    logger.warning("Lost connection to arena, retrying...")
                    continue

                status = result["status"]
                if status == "matched":
                    match_id = result["match_id"]
                    opponent_code = result["opponent_code"]
                    logger.info(f"Matched! Opponent: {opponent_code}")
                    break

            if status != "matched":
                logger.error("Queue timed out — no opponent found.")
                _delete(f"/queue/{queue_id}")
                return None

        # --- Step 2.5: Wager negotiation ---
        wager_id = None
        wager_amount_wei = None
        opponent_wallet = result.get("opponent_wallet")

        if wager_amount and nj_cfg.wallet and nj_cfg.wallet.private_key and nj_cfg.chain and nj_cfg.chain.wager:
            wager_id, wager_amount_wei = _negotiate_wager(
                match_id=match_id,
                queue_id=queue_id,
                opponent_wallet=opponent_wallet,
                config=nj_cfg,
                server=server,
                _get=_get,
                _post=_post,
                wager_amount=wager_amount,
            )

        # --- Step 3: Run netplay ---
        if getattr(args, "headless", False):
            logger.warning("--headless requires mainline Dolphin (not Slippi Dolphin)")
            logger.warning("Slippi netplay needs a display. Use Xvfb on headless servers.")
        logger.info("Launching netplay...")

        config = NetplayConfig(
            dolphin_path=args.dolphin,
            iso_path=args.iso,
            opponent_code=opponent_code,
            connect_code=args.code,
            character=_resolve_character(args.char, Character),
            stage=Stage[args.stage.upper()],
            stocks=args.stocks,
            time_minutes=args.time,
            online_delay=args.delay,
            input_throttle=args.throttle,
            dolphin_home_path=args.dolphin_home,
            slippi_replay_dir=getattr(args, "replay_dir", None),
            headless=getattr(args, "headless", False),
            arena_url=server,
            match_id=match_id,
        )

        runner = NetplayRunner(config)
        start_time = time.time()
        outcome = "COMPLETED"
        our_stocks = 0
        their_stocks = 0
        we_won = False

        try:
            match_result = runner.run_netplay(fighter, games=1)
            duration = time.time() - start_time
            port = getattr(match_result, "our_port", 1)
            if match_result.games:
                last_game = match_result.games[-1]
                if port == 1:
                    our_stocks = int(last_game.p1_stocks)
                    their_stocks = int(last_game.p2_stocks)
                else:
                    our_stocks = int(last_game.p2_stocks)
                    their_stocks = int(last_game.p1_stocks)
            we_won = hasattr(match_result, "winner_port") and match_result.winner_port == port
        except NetplayDisconnectedError:
            duration = time.time() - start_time
            outcome = "DISCONNECT"
            logger.warning(f"Match ended with disconnect after {duration:.1f}s")
        except Exception as e:
            duration = time.time() - start_time
            outcome = "ERROR"
            logger.error(f"Match error: {e}")

        # --- Step 4: Report result ---
        try:
            _post(f"/matches/{match_id}/result", {
                "queue_id": queue_id,
                "outcome": outcome,
                "duration_seconds": round(duration, 1),
                "stocks_remaining": our_stocks,
                "opponent_stocks": their_stocks,
            })
        except urllib.error.URLError as e:
            logger.warning(f"Failed to report result: {e}")

        # --- Step 5: Sign canonical match result ---
        signed = _sign_and_submit(match_id, outcome, _get, _post)

        # --- Step 6: Auto-settle wager ---
        if wager_id is not None and outcome == "COMPLETED":
            _auto_settle_wager(match_id, wager_id, wager_amount_wei, nj_cfg)

        return MatchOutcome(
            match_id=match_id,
            opponent_code=opponent_code,
            outcome=outcome,
            we_won=we_won,
            our_stocks=our_stocks,
            their_stocks=their_stocks,
            duration=duration,
            signed=signed,
            wager_id=wager_id,
            wager_amount_wei=wager_amount_wei,
            opponent_wallet=opponent_wallet,
        )

    except KeyboardInterrupt:
        raise  # Let caller handle
    finally:
        if queue_id and match_id is None:
            try:
                _delete(f"/queue/{queue_id}")
            except Exception:
                pass


def _print_match_summary(
    fighter_name: str, result: MatchOutcome, nj_cfg, show_nudge: bool = True
):
    """Print the post-match summary box."""
    print()
    print("=" * 44)
    if result.outcome == "COMPLETED":
        result_line = "WIN" if result.we_won else "LOSS"
        print(f"  {fighter_name} vs {result.opponent_code}  —  {result_line}")
        if result.our_stocks > 0:
            print(f"  Stocks remaining: {result.our_stocks}")
    elif result.outcome == "DISCONNECT":
        print(f"  {fighter_name} vs {result.opponent_code}  —  DISCONNECT")
    else:
        print(f"  {fighter_name} vs {result.opponent_code}  —  ERROR")
    print(f"  Duration: {result.duration:.0f}s  |  Match: {result.match_id[:8]}...")
    if result.signed:
        print(f"  Signed: yes  |  Wallet: {nj_cfg.wallet.address[:10]}...")
    elif nj_cfg.wallet and nj_cfg.wallet.address:
        print(f"  Signed: FAILED  |  Wallet: {nj_cfg.wallet.address[:10]}...")
    if result.wager_id is not None:
        pot_mon = (result.wager_amount_wei or 0) * 2 / 10**18
        print(f"  Wager: {pot_mon} MON pot  |  ID: {result.wager_id}")
    print("=" * 44)
    print()

    if show_nudge and not (nj_cfg.wallet and nj_cfg.wallet.address):
        nudge_marker = CONFIG_DIR / ".wallet-nudged"
        if not nudge_marker.exists():
            print("Tip: Record matches onchain and build your agent's reputation.")
            print("     Run: nojohns setup wallet")
            print()
            try:
                nudge_marker.touch()
            except OSError:
                pass


def cmd_matchmake(args):
    """Join the arena queue, wait for a match, play, and report results."""
    fighter = load_fighter(args.fighter)
    if fighter is None:
        return 1

    if args.code is None:
        logger.error("--code/-c required (run 'nojohns setup melee' or pass explicitly)")
        return 1
    if args.server is None:
        logger.error("--server required (run 'nojohns setup melee' or pass explicitly)")
        return 1

    server = args.server.rstrip("/")
    _post, _get, _delete = _make_arena_helpers(server)
    nj_cfg = load_config()

    try:
        result = _run_single_match(
            fighter_name=args.fighter,
            fighter=fighter,
            args=args,
            nj_cfg=nj_cfg,
            server=server,
            _post=_post,
            _get=_get,
            _delete=_delete,
            wager_amount=args.wager,
        )

        if result is None:
            return 1

        _print_match_summary(args.fighter, result, nj_cfg)
        return 0

    except KeyboardInterrupt:
        logger.info("\nCancelled.")
        return 130


def cmd_auto(args):
    """Autonomous agent loop: scout, wager, play, repeat.

    Composes the agent toolkit (agents/) as a reference implementation.
    This is the "Phillip of agents" — a working demo, not the only way.
    """
    import time

    from agents.bankroll import get_bankroll_state, win_probability_from_elo
    from agents.scouting import scout_opponent, scout_by_wallet
    from agents.strategy import KellyStrategy, MatchContext, SessionStats
    from nojohns.config import MoltbotConfig

    fighter = load_fighter(args.fighter)
    if fighter is None:
        return 1

    if args.code is None:
        logger.error("--code/-c required (run 'nojohns setup melee' or pass explicitly)")
        return 1
    if args.server is None:
        logger.error("--server required (run 'nojohns setup melee' or pass explicitly)")
        return 1

    nj_cfg = load_config()

    # Wallet required for auto mode (need to make strategic decisions about money)
    if not args.no_wager:
        if nj_cfg.wallet is None or nj_cfg.wallet.private_key is None:
            logger.error("Wallet required for auto mode. Run: nojohns setup wallet")
            logger.error("Or use --no-wager to play without wagering.")
            return 1
        if nj_cfg.chain is None or nj_cfg.chain.wager is None:
            logger.error("Wager contract not configured. Check [chain] in config.toml")
            logger.error("Or use --no-wager to play without wagering.")
            return 1

    # Merge config defaults with CLI overrides
    moltbot_cfg = nj_cfg.moltbot or MoltbotConfig()
    risk_profile = args.risk or moltbot_cfg.risk_profile
    cooldown = args.cooldown if args.cooldown is not None else moltbot_cfg.cooldown_seconds
    min_bankroll = args.min_bankroll if args.min_bankroll is not None else moltbot_cfg.min_bankroll
    tilt_threshold = moltbot_cfg.tilt_threshold
    max_matches = args.max_matches

    # Set up strategy
    strategy = KellyStrategy(risk_profile=risk_profile, tilt_threshold=tilt_threshold)

    server = args.server.rstrip("/")
    _post, _get, _delete = _make_arena_helpers(server)

    # Session tracking
    session = SessionStats()

    # Get our Elo
    our_elo = 1500
    if nj_cfg.chain and nj_cfg.chain.reputation_registry and nj_cfg.chain.agent_id:
        try:
            from nojohns.reputation import get_current_elo
            state = get_current_elo(
                nj_cfg.chain.agent_id,
                nj_cfg.chain.rpc_url,
                nj_cfg.chain.reputation_registry,
            )
            our_elo = state.elo
        except Exception:
            pass

    print()
    print("=" * 52)
    print(f"  No Johns — Autonomous Mode")
    print(f"  Fighter: {args.fighter}")
    print(f"  Strategy: {strategy}")
    print(f"  Elo: {our_elo}")
    if not args.no_wager:
        print(f"  Min bankroll: {min_bankroll} MON")
    else:
        print(f"  Wagering: disabled")
    if max_matches:
        print(f"  Max matches: {max_matches}")
    print(f"  Cooldown: {cooldown}s between matches")
    print("=" * 52)
    print()

    match_num = 0

    try:
        while True:
            match_num += 1

            if max_matches and match_num > max_matches:
                print(f"\nReached max matches ({max_matches}). Stopping.")
                break

            print(f"\n--- Match {match_num} ---")

            # 1. Check bankroll
            wager_amount = None
            if not args.no_wager and nj_cfg.wallet and nj_cfg.chain and nj_cfg.chain.wager:
                bankroll = get_bankroll_state(
                    nj_cfg.wallet.address,
                    nj_cfg.chain.rpc_url,
                    nj_cfg.chain.wager,
                )
                print(f"Bankroll: {bankroll.available_mon:.4f} MON available")

                if bankroll.available_mon < min_bankroll:
                    print(f"Below min bankroll ({min_bankroll} MON). Stopping.")
                    break

            # 2. Join queue and get matched (wager decision happens after we know opponent)
            # For now, run match without wager — we'll decide after scouting
            result = _run_single_match(
                fighter_name=args.fighter,
                fighter=fighter,
                args=args,
                nj_cfg=nj_cfg,
                server=server,
                _post=_post,
                _get=_get,
                _delete=_delete,
                wager_amount=None,  # Decided dynamically below
            )

            if result is None:
                print("Failed to get a match. Waiting before retry...")
                time.sleep(cooldown)
                continue

            # 3. Scout opponent (post-match for now — pre-match scouting needs queue refactor)
            opponent_report = scout_by_wallet(
                result.opponent_wallet or "",
                nj_cfg.chain.rpc_url if nj_cfg.chain else "",
                nj_cfg.chain.reputation_registry or "" if nj_cfg.chain else "",
            ) if nj_cfg.chain else None

            # 4. Strategy reasoning (even when not wagering — shows what *would* happen)
            if not args.no_wager and nj_cfg.chain and opponent_report:
                bankroll = get_bankroll_state(
                    nj_cfg.wallet.address,
                    nj_cfg.chain.rpc_url,
                    nj_cfg.chain.wager,
                )
                context = MatchContext(
                    our_elo=our_elo,
                    opponent=opponent_report,
                    bankroll_wei=bankroll.available_wei,
                    session_stats=session,
                )
                decision = strategy.decide(context)
                print(f"  Strategy: {decision.reasoning}")

            # 5. Update session stats
            if result.outcome == "COMPLETED":
                wager_wei = result.wager_amount_wei or 0
                if result.we_won:
                    session.record_win(wager_wei)
                else:
                    session.record_loss(wager_wei)
                if result.opponent_code:
                    session.opponents_faced.add(result.opponent_code)

            # 6. Print match summary
            _print_match_summary(args.fighter, result, nj_cfg, show_nudge=False)

            # 7. Cooldown
            if max_matches is None or match_num < max_matches:
                print(f"Cooling down ({cooldown}s)...")
                time.sleep(cooldown)

    except KeyboardInterrupt:
        print("\n\nAuto mode interrupted.")

    # --- Session Summary ---
    print()
    print("=" * 52)
    print(f"  Session Summary")
    print(f"  Matches: {session.matches_played}")
    print(f"  Record: {session.wins}-{session.losses} ({session.win_rate:.0%})")
    print(f"  Unique opponents: {len(session.opponents_faced)}")
    if session.total_wagered_wei > 0:
        net_mon = session.net_profit_wei / 10**18
        total_mon = session.total_wagered_wei / 10**18
        print(f"  Total wagered: {total_mon:.4f} MON")
        print(f"  Net P&L: {'+' if net_mon >= 0 else ''}{net_mon:.4f} MON")
    print("=" * 52)
    print()

    return 0


def cmd_arena(args):
    """Start the arena matchmaking server."""
    try:
        import uvicorn
    except ImportError:
        logger.error("Arena requires extra dependencies: pip install nojohns[arena]")
        return 1

    from arena.server import app

    # Set DB path on app state so lifespan picks it up
    app.state.db_path = args.db
    logger.info(f"Starting arena server on port {args.port} (db: {args.db})")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    return 0


def cmd_netplay(args):
    """Run a fighter over Slippi netplay against a remote opponent."""
    from games.melee import NetplayConfig, NetplayRunner
    from melee import Character, Stage

    fighter = load_fighter(args.fighter)
    if fighter is None:
        return 1

    # Load our own connect code for port detection
    nj_cfg = load_config()
    our_code = None
    game_cfg = nj_cfg.games.get("melee")
    if game_cfg and game_cfg.connect_code:
        our_code = game_cfg.connect_code

    config = NetplayConfig(
        dolphin_path=args.dolphin,
        iso_path=args.iso,
        opponent_code=args.code,
        connect_code=our_code,
        character=_resolve_character(args.char, Character),
        stage=Stage[args.stage.upper()],
        stocks=args.stocks,
        time_minutes=args.time,
        online_delay=args.delay,
        input_throttle=args.throttle,
        dolphin_home_path=args.dolphin_home,
        slippi_replay_dir=args.replay_dir,
    )

    logger.info(f"No Johns - Netplay!")
    logger.info(f"   Fighter: {fighter.metadata.display_name} ({config.character.name})")
    logger.info(f"   Opponent code: {config.opponent_code}")
    logger.info(f"   Format: Bo{args.games}, {config.stocks} stock, {config.stage.name}")
    logger.info("")

    runner = NetplayRunner(config)

    def on_game_end(game):
        logger.info(f"   Game over! P{game.winner_port} wins ({game.p1_stocks}-{game.p2_stocks})")

    try:
        result = runner.run_netplay(
            fighter,
            games=args.games,
            on_game_end=on_game_end,
        )

        logger.info("")
        logger.info(f"Netplay Complete!")
        logger.info(f"   Winner: P{result.winner_port} ({result.score})")

        return 0

    except KeyboardInterrupt:
        logger.info("\nNetplay cancelled.")
        return 130
    except Exception as e:
        logger.error(f"Netplay error: {e}")
        return 1


def cmd_netplay_test(args):
    """Run two fighters on two local Dolphins connected via Slippi."""
    from games.melee import netplay_test
    from melee import Character, Stage

    fighter1 = load_fighter(args.fighter1)
    fighter2 = load_fighter(args.fighter2)

    if fighter1 is None or fighter2 is None:
        return 1

    logger.info(f"No Johns - Netplay Test (two local Dolphins)")
    logger.info(f"   Side 1: {fighter1.metadata.display_name} (code: {args.code1})")
    logger.info(f"   Side 2: {fighter2.metadata.display_name} (code: {args.code2})")
    logger.info(f"   Format: Bo{args.games}")
    logger.info("")

    try:
        result1, result2 = netplay_test(
            fighter1=fighter1,
            fighter2=fighter2,
            dolphin_path=args.dolphin,
            iso_path=args.iso,
            code1=args.code1,
            code2=args.code2,
            home1=args.home1,
            home2=args.home2,
            games=args.games,
            character1=_resolve_character(args.p1_char, Character),
            character2=_resolve_character(args.p2_char, Character),
            stage=Stage[args.stage.upper()],
        )

        logger.info("")
        logger.info(f"Netplay Test Complete!")
        logger.info(f"   Side 1 sees: P{result1.winner_port} won ({result1.score})")
        logger.info(f"   Side 2 sees: P{result2.winner_port} won ({result2.score})")

        return 0

    except KeyboardInterrupt:
        logger.info("\nNetplay test cancelled.")
        return 130
    except Exception as e:
        logger.error(f"Netplay test error: {e}")
        return 1


# ============================================================================
# Non-Game Commands
# ============================================================================

def cmd_list_fighters(args):
    """List available fighters."""
    from nojohns.registry import list_fighters as registry_list

    fighters = registry_list()

    print("\nAvailable Fighters\n")
    print(f"{'Name':<15} {'Type':<12} {'Characters':<20} {'GPU'}")
    print("-" * 60)

    for f in fighters:
        chars = ", ".join(f.characters[:3])
        if len(f.characters) > 3:
            chars += "..."
        gpu = "Yes" if f.hardware.get("gpu_required") else "No"
        print(f"{f.name:<15} {f.fighter_type:<12} {chars:<20} {gpu}")

    print()
    return 0


def cmd_info(args):
    """Show detailed info about a fighter."""
    from nojohns.registry import get_fighter_info, FighterNotFoundError

    info = get_fighter_info(args.fighter)
    if info is None:
        logger.error(f"Unknown fighter: {args.fighter}")
        return 1

    gpu = "Yes" if info.hardware.get("gpu_required") else "No"
    ram = info.hardware.get("min_ram_gb", "?")

    print(f"\n{info.display_name} v{info.version}")
    print(f"   by {info.author}")
    print()
    print(f"   Type: {info.fighter_type}")
    print(f"   Characters: {', '.join(info.characters)}")
    print(f"   GPU Required: {gpu}")
    print(f"   Min RAM: {ram}GB")
    print(f"   Frame Delay: {info.avg_frame_delay}")
    print()
    print(f"   {info.description}")

    if info.repo_url:
        print(f"\n   Repo: {info.repo_url}")

    print()
    return 0


# ============================================================================
# Wager Commands
# ============================================================================

def cmd_wager(args):
    """Handle `nojohns wager <subcommand>`."""
    return args.wager_func(args)


def cmd_wager_propose(args):
    """Propose a wager by escrowing MON."""
    from nojohns.config import load_config
    from nojohns.wallet import load_wallet, propose_wager

    config = load_config()

    if config.wallet is None or config.wallet.private_key is None:
        logger.error("No wallet configured. Run: nojohns setup wallet")
        return 1

    if config.chain is None or config.chain.wager is None:
        logger.error("No wager contract configured in [chain] section.")
        return 1

    account = load_wallet(config)
    if account is None:
        logger.error("Failed to load wallet.")
        return 1

    # Parse amount (support decimal MON, e.g. "0.01")
    try:
        amount_mon = float(args.amount)
        amount_wei = int(amount_mon * 10**18)
    except ValueError:
        logger.error(f"Invalid amount: {args.amount}")
        return 1

    opponent = args.opponent if args.opponent else None
    game_id = args.game or "melee"

    print(f"Proposing wager: {amount_mon} MON")
    print(f"  Game: {game_id}")
    print(f"  Opponent: {opponent or 'open (anyone can accept)'}")
    print(f"  From: {account.address}")
    print()

    try:
        tx_hash, wager_id = propose_wager(
            account=account,
            rpc_url=config.chain.rpc_url,
            contract_address=config.chain.wager,
            opponent=opponent,
            game_id=game_id,
            amount_wei=amount_wei,
        )
        print(f"Wager proposed!")
        print(f"  Wager ID: {wager_id}")
        print(f"  TX: {tx_hash}")
        return 0
    except Exception as e:
        logger.error(f"Failed to propose wager: {e}")
        return 1


def cmd_wager_accept(args):
    """Accept a wager by escrowing matching MON."""
    from nojohns.config import load_config
    from nojohns.wallet import load_wallet, accept_wager, get_wager_info

    config = load_config()

    if config.wallet is None or config.wallet.private_key is None:
        logger.error("No wallet configured. Run: nojohns setup wallet")
        return 1

    if config.chain is None or config.chain.wager is None:
        logger.error("No wager contract configured in [chain] section.")
        return 1

    account = load_wallet(config)
    if account is None:
        logger.error("Failed to load wallet.")
        return 1

    wager_id = int(args.wager_id)

    # Get wager info to show amount
    try:
        info = get_wager_info(config.chain.rpc_url, config.chain.wager, wager_id)
    except Exception as e:
        logger.error(f"Failed to get wager info: {e}")
        return 1

    if info["status"] != "Open":
        logger.error(f"Wager {wager_id} is not open (status: {info['status']})")
        return 1

    amount_mon = info["amount"] / 10**18

    print(f"Accepting wager {wager_id}")
    print(f"  Amount: {amount_mon} MON")
    print(f"  Game: {info['gameId']}")
    print(f"  Proposer: {info['proposer']}")
    print(f"  From: {account.address}")
    print()

    try:
        tx_hash = accept_wager(
            account=account,
            rpc_url=config.chain.rpc_url,
            contract_address=config.chain.wager,
            wager_id=wager_id,
            amount_wei=info["amount"],
        )
        print(f"Wager accepted!")
        print(f"  TX: {tx_hash}")
        print()
        print("Now play a match and settle with:")
        print(f"  nojohns wager settle {wager_id} <match_id>")
        return 0
    except Exception as e:
        logger.error(f"Failed to accept wager: {e}")
        return 1


def cmd_wager_settle(args):
    """Settle a wager using a recorded match result."""
    from nojohns.config import load_config
    from nojohns.wallet import load_wallet, settle_wager, get_wager_info

    config = load_config()

    if config.wallet is None or config.wallet.private_key is None:
        logger.error("No wallet configured. Run: nojohns setup wallet")
        return 1

    if config.chain is None or config.chain.wager is None:
        logger.error("No wager contract configured in [chain] section.")
        return 1

    account = load_wallet(config)
    if account is None:
        logger.error("Failed to load wallet.")
        return 1

    wager_id = int(args.wager_id)
    match_id_hex = args.match_id

    # Convert match_id to bytes32
    if match_id_hex.startswith("0x"):
        match_id_hex = match_id_hex[2:]
    match_id = bytes.fromhex(match_id_hex.ljust(64, '0'))

    # Get wager info
    try:
        info = get_wager_info(config.chain.rpc_url, config.chain.wager, wager_id)
    except Exception as e:
        logger.error(f"Failed to get wager info: {e}")
        return 1

    if info["status"] != "Accepted":
        logger.error(f"Wager {wager_id} cannot be settled (status: {info['status']})")
        return 1

    amount_mon = info["amount"] / 10**18

    print(f"Settling wager {wager_id}")
    print(f"  Total pot: {amount_mon * 2} MON")
    print(f"  Match ID: {match_id_hex[:16]}...")
    print()

    try:
        tx_hash = settle_wager(
            account=account,
            rpc_url=config.chain.rpc_url,
            contract_address=config.chain.wager,
            wager_id=wager_id,
            match_id=match_id,
        )
        print(f"Wager settled!")
        print(f"  TX: {tx_hash}")
        return 0
    except Exception as e:
        logger.error(f"Failed to settle wager: {e}")
        return 1


def cmd_wager_cancel(args):
    """Cancel an open wager and get refund."""
    from nojohns.config import load_config
    from nojohns.wallet import load_wallet, cancel_wager, get_wager_info

    config = load_config()

    if config.wallet is None or config.wallet.private_key is None:
        logger.error("No wallet configured. Run: nojohns setup wallet")
        return 1

    if config.chain is None or config.chain.wager is None:
        logger.error("No wager contract configured in [chain] section.")
        return 1

    account = load_wallet(config)
    if account is None:
        logger.error("Failed to load wallet.")
        return 1

    wager_id = int(args.wager_id)

    # Get wager info
    try:
        info = get_wager_info(config.chain.rpc_url, config.chain.wager, wager_id)
    except Exception as e:
        logger.error(f"Failed to get wager info: {e}")
        return 1

    if info["status"] != "Open":
        logger.error(f"Wager {wager_id} cannot be cancelled (status: {info['status']})")
        return 1

    if info["proposer"].lower() != account.address.lower():
        logger.error(f"Only the proposer can cancel. You: {account.address}, Proposer: {info['proposer']}")
        return 1

    amount_mon = info["amount"] / 10**18

    print(f"Cancelling wager {wager_id}")
    print(f"  Refund: {amount_mon} MON")
    print()

    try:
        tx_hash = cancel_wager(
            account=account,
            rpc_url=config.chain.rpc_url,
            contract_address=config.chain.wager,
            wager_id=wager_id,
        )
        print(f"Wager cancelled!")
        print(f"  TX: {tx_hash}")
        return 0
    except Exception as e:
        logger.error(f"Failed to cancel wager: {e}")
        return 1


def cmd_wager_status(args):
    """Show status of a wager."""
    from nojohns.config import load_config
    from nojohns.wallet import get_wager_info

    config = load_config()

    if config.chain is None or config.chain.wager is None:
        logger.error("No wager contract configured in [chain] section.")
        return 1

    wager_id = int(args.wager_id)

    try:
        info = get_wager_info(config.chain.rpc_url, config.chain.wager, wager_id)
    except Exception as e:
        logger.error(f"Failed to get wager info: {e}")
        return 1

    amount_mon = info["amount"] / 10**18

    print(f"\nWager {wager_id}")
    print(f"  Status: {info['status']}")
    print(f"  Amount: {amount_mon} MON (pot: {amount_mon * 2} MON)")
    print(f"  Game: {info['gameId']}")
    print(f"  Proposer: {info['proposer']}")
    if info["opponent"] != "0x0000000000000000000000000000000000000000":
        print(f"  Opponent: {info['opponent']}")
    else:
        print(f"  Opponent: open (anyone can accept)")
    if info["matchId"] and info["matchId"] != "0" * 64:
        print(f"  Match ID: {info['matchId']}")
    print()
    return 0


def cmd_wager_list(args):
    """List wagers for the configured wallet."""
    from nojohns.config import load_config
    from nojohns.wallet import get_agent_wagers, get_wager_info

    config = load_config()

    if config.wallet is None:
        logger.error("No wallet configured. Run: nojohns setup wallet")
        return 1

    if config.chain is None or config.chain.wager is None:
        logger.error("No wager contract configured in [chain] section.")
        return 1

    address = args.address or config.wallet.address

    try:
        wager_ids = get_agent_wagers(config.chain.rpc_url, config.chain.wager, address)
    except Exception as e:
        logger.error(f"Failed to get wagers: {e}")
        return 1

    if not wager_ids:
        print(f"No wagers found for {address}")
        return 0

    print(f"\nWagers for {address}:\n")
    print(f"{'ID':>6}  {'Status':<10}  {'Amount':>10}  {'Game':<10}  {'Opponent'}")
    print("-" * 70)

    for wid in wager_ids:
        try:
            info = get_wager_info(config.chain.rpc_url, config.chain.wager, wid)
            amount_mon = info["amount"] / 10**18
            opp = info["opponent"]
            if opp == "0x0000000000000000000000000000000000000000":
                opp = "open"
            else:
                opp = opp[:10] + "..."
            print(f"{wid:>6}  {info['status']:<10}  {amount_mon:>10.4f}  {info['gameId']:<10}  {opp}")
        except Exception:
            print(f"{wid:>6}  (error fetching)")

    print()
    return 0


# ============================================================================
# Fighter Loading
# ============================================================================

def load_fighter(name: str):
    """Load a fighter by name via the registry."""
    from nojohns.registry import (
        load_fighter as registry_load,
        FighterNotFoundError,
        FighterLoadError,
    )

    try:
        return registry_load(name)
    except FighterNotFoundError as e:
        logger.error(str(e))
        return None
    except FighterLoadError as e:
        logger.error(str(e))
        return None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="nojohns",
        description="Melee AI tournaments for Moltbots",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # setup command
    setup_parser = subparsers.add_parser(
        "setup", help="Configure No Johns (setup / setup melee / setup wallet / setup identity)"
    )
    setup_parser.add_argument(
        "setup_target", nargs="*", default=[],
        help="What to set up: melee=game, wallet=onchain, identity=ERC-8004 registration",
    )
    setup_parser.set_defaults(func=cmd_setup)

    # fight command
    fight_parser = subparsers.add_parser("fight", help="Run a fight between two fighters")
    fight_parser.add_argument("fighter1", help="Fighter for P1")
    fight_parser.add_argument("fighter2", help="Fighter for P2")
    fight_parser.add_argument("--dolphin", "-d", default=None, help="Path to Slippi Dolphin")
    fight_parser.add_argument("--iso", "-i", default=None, help="Path to Melee ISO")
    fight_parser.add_argument("--games", "-g", type=int, default=1, help="Number of games (default: 1)")
    fight_parser.add_argument("--stocks", "-s", type=int, default=4, help="Stocks per game (default: 4)")
    fight_parser.add_argument("--time", "-t", type=int, default=8, help="Time limit in minutes (default: 8)")
    fight_parser.add_argument("--stage", default="FINAL_DESTINATION", help="Stage (default: FINAL_DESTINATION)")
    fight_parser.add_argument("--p1-char", default="RANDOM", help="P1 character (default: RANDOM)")
    fight_parser.add_argument("--p2-char", default="RANDOM", help="P2 character (default: RANDOM)")
    fight_parser.add_argument("--headless", action="store_true", help="Run without display (faster)")
    fight_parser.set_defaults(func=cmd_fight)

    # netplay command
    netplay_parser = subparsers.add_parser("netplay", help="Run a fighter over Slippi netplay")
    netplay_parser.add_argument("fighter", help="Fighter to run")
    netplay_parser.add_argument("--code", "-c", required=True, help="Opponent's Slippi connect code (e.g. ABCD#123)")
    netplay_parser.add_argument("--dolphin", "-d", default=None, help="Path to Slippi Dolphin")
    netplay_parser.add_argument("--iso", "-i", default=None, help="Path to Melee ISO")
    netplay_parser.add_argument("--char", default="RANDOM", help="Character (default: RANDOM)")
    netplay_parser.add_argument("--stage", default="FINAL_DESTINATION", help="Stage (default: FINAL_DESTINATION)")
    netplay_parser.add_argument("--games", "-g", type=int, default=1, help="Number of games (default: 1)")
    netplay_parser.add_argument("--stocks", "-s", type=int, default=4, help="Stocks per game (default: 4)")
    netplay_parser.add_argument("--time", "-t", type=int, default=8, help="Time limit in minutes (default: 8)")
    netplay_parser.add_argument("--delay", type=int, default=None, help="Online input delay in frames (default: 6)")
    netplay_parser.add_argument("--throttle", type=int, default=None, help="AI input throttle (default: 1)")
    netplay_parser.add_argument("--dolphin-home", default=None, help="Dolphin home dir (for Slippi account)")
    netplay_parser.add_argument("--replay-dir", default=None, help="Directory to save Slippi replays")
    netplay_parser.set_defaults(func=cmd_netplay)

    # netplay-test command
    nptest_parser = subparsers.add_parser("netplay-test", help="Test two fighters on two local Dolphins via Slippi")
    nptest_parser.add_argument("fighter1", help="Fighter for side 1")
    nptest_parser.add_argument("fighter2", help="Fighter for side 2")
    nptest_parser.add_argument("--code1", required=True, help="Slippi connect code for side 1")
    nptest_parser.add_argument("--code2", required=True, help="Slippi connect code for side 2")
    nptest_parser.add_argument("--home1", required=True, help="Dolphin home dir for side 1")
    nptest_parser.add_argument("--home2", required=True, help="Dolphin home dir for side 2")
    nptest_parser.add_argument("--dolphin", "-d", default=None, help="Path to Slippi Dolphin")
    nptest_parser.add_argument("--iso", "-i", default=None, help="Path to Melee ISO")
    nptest_parser.add_argument("--games", "-g", type=int, default=1, help="Number of games (default: 1)")
    nptest_parser.add_argument("--stage", default="FINAL_DESTINATION", help="Stage (default: FINAL_DESTINATION)")
    nptest_parser.add_argument("--p1-char", default="RANDOM", help="Side 1 character (default: RANDOM)")
    nptest_parser.add_argument("--p2-char", default="RANDOM", help="Side 2 character (default: RANDOM)")
    nptest_parser.set_defaults(func=cmd_netplay_test)

    # matchmake command
    mm_parser = subparsers.add_parser("matchmake", help="Join arena queue, get matched, play netplay")
    mm_parser.add_argument("fighter", help="Fighter to run")
    mm_parser.add_argument("--code", "-c", default=None, help="Your Slippi connect code (e.g. SCAV#382)")
    mm_parser.add_argument("--server", default=None, help="Arena server URL (e.g. http://localhost:8000)")
    mm_parser.add_argument("--dolphin", "-d", default=None, help="Path to Slippi Dolphin")
    mm_parser.add_argument("--iso", "-i", default=None, help="Path to Melee ISO")
    mm_parser.add_argument("--char", default="RANDOM", help="Character (default: RANDOM)")
    mm_parser.add_argument("--stage", default="FINAL_DESTINATION", help="Stage (default: FINAL_DESTINATION)")
    mm_parser.add_argument("--stocks", "-s", type=int, default=4, help="Stocks per game (default: 4)")
    mm_parser.add_argument("--time", "-t", type=int, default=8, help="Time limit in minutes (default: 8)")
    mm_parser.add_argument("--delay", type=int, default=None, help="Online input delay in frames (default: 6)")
    mm_parser.add_argument("--throttle", type=int, default=None, help="AI input throttle (default: 1)")
    mm_parser.add_argument("--dolphin-home", default=None, help="Dolphin home dir (for Slippi account)")
    mm_parser.add_argument("--replay-dir", default=None, help="Directory to save Slippi replays")
    mm_parser.add_argument("--headless", action="store_true", help="Run without display (faster, for servers)")
    mm_parser.add_argument("--wager", type=float, default=None, help="Auto-wager this amount of MON per match")
    mm_parser.set_defaults(func=cmd_matchmake)

    # auto command (autonomous agent loop)
    auto_parser = subparsers.add_parser(
        "auto", help="Autonomous agent: scout, wager, play, repeat"
    )
    auto_parser.add_argument("fighter", help="Fighter to run")
    auto_parser.add_argument("--code", "-c", default=None, help="Your Slippi connect code")
    auto_parser.add_argument("--server", default=None, help="Arena server URL")
    auto_parser.add_argument("--dolphin", "-d", default=None, help="Path to Slippi Dolphin")
    auto_parser.add_argument("--iso", "-i", default=None, help="Path to Melee ISO")
    auto_parser.add_argument("--char", default="RANDOM", help="Character (default: RANDOM)")
    auto_parser.add_argument("--stage", default="FINAL_DESTINATION", help="Stage")
    auto_parser.add_argument("--stocks", "-s", type=int, default=4, help="Stocks per game")
    auto_parser.add_argument("--time", "-t", type=int, default=8, help="Time limit in minutes")
    auto_parser.add_argument("--delay", type=int, default=None, help="Online input delay")
    auto_parser.add_argument("--throttle", type=int, default=None, help="AI input throttle")
    auto_parser.add_argument("--dolphin-home", default=None, help="Dolphin home dir")
    auto_parser.add_argument("--risk", choices=["conservative", "moderate", "aggressive"],
                             default=None, help="Risk profile (default: from config or moderate)")
    auto_parser.add_argument("--max-matches", type=int, default=None, help="Stop after N matches")
    auto_parser.add_argument("--min-bankroll", type=float, default=None, help="MON stop threshold")
    auto_parser.add_argument("--cooldown", type=int, default=None, help="Seconds between matches")
    auto_parser.add_argument("--no-wager", action="store_true", help="Play without wagering")
    auto_parser.set_defaults(func=cmd_auto)

    # arena command
    arena_parser = subparsers.add_parser("arena", help="Start the matchmaking server")
    arena_parser.add_argument("--port", "-p", type=int, default=8000, help="Server port (default: 8000)")
    arena_parser.add_argument("--db", default="arena.db", help="SQLite database path (default: arena.db)")
    arena_parser.set_defaults(func=cmd_arena)

    # list-fighters command
    list_parser = subparsers.add_parser("list-fighters", help="List available fighters")
    list_parser.set_defaults(func=cmd_list_fighters)

    # info command
    info_parser = subparsers.add_parser("info", help="Show info about a fighter")
    info_parser.add_argument("fighter", help="Fighter name")
    info_parser.set_defaults(func=cmd_info)

    # wager command with subcommands
    wager_parser = subparsers.add_parser("wager", help="Manage onchain wagers")
    wager_parser.set_defaults(func=cmd_wager)
    wager_subparsers = wager_parser.add_subparsers(dest="wager_command", required=True)

    # wager propose
    wager_propose = wager_subparsers.add_parser("propose", help="Propose a wager by escrowing MON")
    wager_propose.add_argument("amount", help="Amount of MON to wager (e.g. 0.01)")
    wager_propose.add_argument("--opponent", "-o", help="Opponent wallet address (omit for open wager)")
    wager_propose.add_argument("--game", "-g", default="melee", help="Game ID (default: melee)")
    wager_propose.set_defaults(wager_func=cmd_wager_propose)

    # wager accept
    wager_accept = wager_subparsers.add_parser("accept", help="Accept a wager")
    wager_accept.add_argument("wager_id", help="Wager ID to accept")
    wager_accept.set_defaults(wager_func=cmd_wager_accept)

    # wager settle
    wager_settle = wager_subparsers.add_parser("settle", help="Settle a wager with a match result")
    wager_settle.add_argument("wager_id", help="Wager ID to settle")
    wager_settle.add_argument("match_id", help="Match ID from MatchProof (hex)")
    wager_settle.set_defaults(wager_func=cmd_wager_settle)

    # wager cancel
    wager_cancel = wager_subparsers.add_parser("cancel", help="Cancel an open wager")
    wager_cancel.add_argument("wager_id", help="Wager ID to cancel")
    wager_cancel.set_defaults(wager_func=cmd_wager_cancel)

    # wager status
    wager_status = wager_subparsers.add_parser("status", help="Show wager status")
    wager_status.add_argument("wager_id", help="Wager ID to check")
    wager_status.set_defaults(wager_func=cmd_wager_status)

    # wager list
    wager_list = wager_subparsers.add_parser("list", help="List wagers for an address")
    wager_list.add_argument("--address", "-a", help="Wallet address (default: configured wallet)")
    wager_list.set_defaults(wager_func=cmd_wager_list)

    args = parser.parse_args()

    # Load config and resolve args for game commands
    game_commands = {"fight", "netplay", "netplay-test", "matchmake", "auto"}
    if args.command in game_commands:
        nj_cfg = load_config()
        game_cfg = nj_cfg.games.get("melee")
        _resolve_args(args, game_cfg, nj_cfg)

        if not _require_melee_args(args):
            sys.exit(1)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
