"""
nojohns/wallet.py - Agent wallet management and EIP-712 match result signing.

Generates/loads Ethereum wallets and produces EIP-712 typed signatures over
MatchResult structs for onchain proof submission. Uses eth-account (lightweight,
no RPC connection needed for signing).

Install: pip install nojohns[wallet]
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# EIP-712 type definitions matching the MatchResult Solidity struct
MATCH_RESULT_TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "MatchResult": [
        {"name": "matchId", "type": "bytes32"},
        {"name": "winner", "type": "address"},
        {"name": "loser", "type": "address"},
        {"name": "gameId", "type": "string"},
        {"name": "winnerScore", "type": "uint8"},
        {"name": "loserScore", "type": "uint8"},
        {"name": "replayHash", "type": "bytes32"},
        {"name": "timestamp", "type": "uint256"},
    ],
}

DOMAIN_NAME = "NoJohns"
DOMAIN_VERSION = "1"


def _require_eth_account():
    """Import and return eth_account, raising a clear error if not installed."""
    try:
        import eth_account
        return eth_account
    except ImportError:
        raise ImportError(
            "eth-account is required for wallet operations. "
            "Install it with: pip install nojohns[wallet]"
        )


def generate_wallet() -> tuple[str, str]:
    """Generate a new Ethereum wallet.

    Returns:
        (address, private_key_hex) — the private key includes the 0x prefix.
    """
    eth_account = _require_eth_account()
    account = eth_account.Account.create()
    key_hex = account.key.hex()
    if not key_hex.startswith("0x"):
        key_hex = "0x" + key_hex
    return (account.address, key_hex)


def load_wallet(config):
    """Load a LocalAccount from a NojohnsConfig's wallet section.

    Args:
        config: NojohnsConfig with a wallet attribute.

    Returns:
        LocalAccount if wallet is configured with a private key, None otherwise.
    """
    eth_account = _require_eth_account()

    if config.wallet is None or config.wallet.private_key is None:
        return None

    key = config.wallet.private_key
    # Ensure 0x prefix
    if not key.startswith("0x"):
        key = "0x" + key

    return eth_account.Account.from_key(key)


def sign_match_result(
    account,
    match_result: dict[str, Any],
    chain_id: int,
    contract_address: str,
) -> bytes:
    """Sign a MatchResult using EIP-712 typed data.

    Args:
        account: eth_account LocalAccount (from load_wallet or Account.from_key).
        match_result: Dict with keys matching the MatchResult struct fields:
            matchId (bytes), winner (str), loser (str), gameId (str),
            winnerScore (int), loserScore (int), replayHash (bytes),
            timestamp (int).
        chain_id: Target chain ID (10143 for Monad testnet, 143 for mainnet).
        contract_address: Address of the MatchProof contract.

    Returns:
        The 65-byte signature (r + s + v).
    """
    from eth_account.messages import encode_typed_data

    domain_data = {
        "name": DOMAIN_NAME,
        "version": DOMAIN_VERSION,
        "chainId": chain_id,
        "verifyingContract": contract_address,
    }

    # encode_typed_data(domain_data, message_types, message_data)
    # message_types should NOT include EIP712Domain — eth-account adds it.
    message_types = {"MatchResult": MATCH_RESULT_TYPES["MatchResult"]}

    signable = encode_typed_data(
        domain_data=domain_data,
        message_types=message_types,
        message_data=match_result,
    )

    signed = account.sign_message(signable)
    return signed.signature


def recover_signer(
    match_result: dict[str, Any],
    signature: bytes,
    chain_id: int,
    contract_address: str,
) -> str:
    """Recover the signer address from a MatchResult signature.

    Args:
        match_result: Same dict that was signed.
        signature: The 65-byte signature.
        chain_id: Chain ID used when signing.
        contract_address: Contract address used when signing.

    Returns:
        The checksummed Ethereum address of the signer.
    """
    eth_account = _require_eth_account()
    from eth_account.messages import encode_typed_data

    domain_data = {
        "name": DOMAIN_NAME,
        "version": DOMAIN_VERSION,
        "chainId": chain_id,
        "verifyingContract": contract_address,
    }

    message_types = {"MatchResult": MATCH_RESULT_TYPES["MatchResult"]}

    signable = encode_typed_data(
        domain_data=domain_data,
        message_types=message_types,
        message_data=match_result,
    )

    return eth_account.Account.recover_message(signable, signature=signature)
