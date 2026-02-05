"""
nojohns/reputation.py - ERC-8004 ReputationRegistry integration

Posts Elo updates to the ReputationRegistry after matches.
"""

import base64
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EloState:
    """Current Elo state for an agent."""
    elo: int
    peak_elo: int
    wins: int
    losses: int

    @property
    def record(self) -> str:
        return f"{self.wins}-{self.losses}"


# Starting Elo for new agents
STARTING_ELO = 1500

# K-factor for Elo calculation (higher = more volatile)
K_FACTOR = 32


def calculate_new_elo(our_elo: int, opponent_elo: int, won: bool) -> int:
    """Calculate new Elo after a match using standard formula."""
    expected = 1 / (1 + 10 ** ((opponent_elo - our_elo) / 400))
    actual = 1.0 if won else 0.0
    return round(our_elo + K_FACTOR * (actual - expected))


def get_current_elo(agent_id: int, rpc_url: str, registry_address: str) -> EloState:
    """Read current Elo state from ReputationRegistry.

    Returns STARTING_ELO if no Elo signals found.
    """
    try:
        from web3 import Web3
    except ImportError:
        logger.debug("web3 not installed")
        return EloState(STARTING_ELO, STARTING_ELO, 0, 0)

    w3 = Web3(Web3.HTTPProvider(rpc_url))

    abi = [
        {
            "inputs": [
                {"name": "agentId", "type": "uint256"},
                {"name": "clientAddresses", "type": "address[]"},
                {"name": "tag1", "type": "string"},
                {"name": "tag2", "type": "string"},
                {"name": "includeRevoked", "type": "bool"},
            ],
            "name": "readAllFeedback",
            "outputs": [
                {"name": "clients", "type": "address[]"},
                {"name": "feedbackIndexes", "type": "uint64[]"},
                {"name": "values", "type": "int128[]"},
                {"name": "valueDecimals", "type": "uint8[]"},
                {"name": "tag1s", "type": "string[]"},
                {"name": "tag2s", "type": "string[]"},
                {"name": "revokedStatuses", "type": "bool[]"},
            ],
            "stateMutability": "view",
            "type": "function"
        },
    ]

    contract = w3.eth.contract(address=registry_address, abi=abi)

    try:
        result = contract.functions.readAllFeedback(
            agent_id,
            [],       # all clients
            "elo",    # tag1 filter
            "melee",  # tag2 filter
            False     # exclude revoked
        ).call()

        clients, indexes, values, decimals, tag1s, tag2s, revoked = result

        if not values:
            return EloState(STARTING_ELO, STARTING_ELO, 0, 0)

        # Get the latest Elo value (highest index)
        latest_elo = values[-1]  # int128

        # Try to parse feedbackURI for peak_elo and record
        # For now, just use the value directly
        # TODO: fetch feedbackURI to get full state
        return EloState(
            elo=int(latest_elo),
            peak_elo=int(latest_elo),  # approximation
            wins=0,  # would need to count from history
            losses=0
        )

    except Exception as e:
        logger.debug(f"Failed to read Elo from registry: {e}")
        return EloState(STARTING_ELO, STARTING_ELO, 0, 0)


def post_elo_update(
    agent_id: int,
    new_elo: int,
    peak_elo: int,
    record: str,
    account,  # eth_account.Account
    rpc_url: str,
    registry_address: str,
    chain_id: int,
) -> str | None:
    """Post Elo update to ReputationRegistry.

    Returns transaction hash on success, None on failure.
    """
    try:
        from web3 import Web3
    except ImportError:
        logger.warning("web3 not installed — skipping Elo posting")
        return None

    w3 = Web3(Web3.HTTPProvider(rpc_url))

    if not w3.is_connected():
        logger.warning(f"Cannot connect to {rpc_url}")
        return None

    # Build feedback URI with full state
    feedback_data = {
        "signal_type": "elo",
        "game": "melee",
        "elo": new_elo,
        "peak_elo": peak_elo,
        "record": record,
    }
    json_bytes = json.dumps(feedback_data, separators=(",", ":")).encode("utf-8")
    b64 = base64.b64encode(json_bytes).decode("ascii")
    feedback_uri = f"data:application/json;base64,{b64}"

    # Hash of the feedback data
    feedback_hash = w3.keccak(json_bytes)

    abi = [
        {
            "inputs": [
                {"name": "agentId", "type": "uint256"},
                {"name": "value", "type": "int128"},
                {"name": "valueDecimals", "type": "uint8"},
                {"name": "tag1", "type": "string"},
                {"name": "tag2", "type": "string"},
                {"name": "endpoint", "type": "string"},
                {"name": "feedbackURI", "type": "string"},
                {"name": "feedbackHash", "type": "bytes32"},
            ],
            "name": "giveFeedback",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
    ]

    contract = w3.eth.contract(address=registry_address, abi=abi)

    try:
        # Estimate gas
        gas_estimate = contract.functions.giveFeedback(
            agent_id,
            new_elo,      # value (Elo as int128)
            0,            # valueDecimals (whole number)
            "elo",        # tag1
            "melee",      # tag2
            "",           # endpoint (not used)
            feedback_uri, # feedbackURI
            feedback_hash # feedbackHash
        ).estimate_gas({"from": account.address})

        gas_limit = int(gas_estimate * 1.2)

        # Build transaction
        nonce = w3.eth.get_transaction_count(account.address)
        tx = contract.functions.giveFeedback(
            agent_id,
            new_elo,
            0,
            "elo",
            "melee",
            "",
            feedback_uri,
            feedback_hash
        ).build_transaction({
            "from": account.address,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": w3.eth.gas_price,
            "chainId": chain_id,
        })

        # Sign and send
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

        # Wait for receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

        if receipt.status != 1:
            logger.warning("Elo posting transaction failed")
            return None

        return tx_hash.hex()

    except Exception as e:
        logger.warning(f"Failed to post Elo update: {e}")
        return None
