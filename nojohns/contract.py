"""
nojohns/contract.py - MatchProof contract interaction via web3.py.

Calls getDigest(), recordMatch(), and getMatch() on the deployed MatchProof
contract. Used for the integration test (digest verification) and eventually
for onchain match submission after arena matches.

Install: pip install nojohns[wallet]
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# MatchProof ABI — subset needed for Python interaction.
# Extracted from contracts/src/MatchProof.sol (matches web/src/abi/matchProof.ts).
MATCH_PROOF_ABI = [
    {
        "type": "function",
        "name": "getDigest",
        "inputs": [
            {
                "name": "result",
                "type": "tuple",
                "components": [
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
        ],
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "recordMatch",
        "inputs": [
            {
                "name": "result",
                "type": "tuple",
                "components": [
                    {"name": "matchId", "type": "bytes32"},
                    {"name": "winner", "type": "address"},
                    {"name": "loser", "type": "address"},
                    {"name": "gameId", "type": "string"},
                    {"name": "winnerScore", "type": "uint8"},
                    {"name": "loserScore", "type": "uint8"},
                    {"name": "replayHash", "type": "bytes32"},
                    {"name": "timestamp", "type": "uint256"},
                ],
            },
            {"name": "sigA", "type": "bytes"},
            {"name": "sigB", "type": "bytes"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "getMatch",
        "inputs": [{"name": "matchId", "type": "bytes32"}],
        "outputs": [
            {
                "name": "",
                "type": "tuple",
                "components": [
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
        ],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "recorded",
        "inputs": [{"name": "", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
    },
]

# Default testnet address from contracts/deployments.json
DEFAULT_MATCH_PROOF = "0x1CC748475F1F666017771FB49131708446B9f3DF"
DEFAULT_RPC_URL = "https://testnet-rpc.monad.xyz"


def _require_web3():
    """Import and return web3, raising a clear error if not installed."""
    try:
        from web3 import Web3
        return Web3
    except ImportError:
        raise ImportError(
            "web3 is required for contract operations. "
            "Install it with: pip install nojohns[wallet]"
        )


def get_match_proof_contract(
    rpc_url: str = DEFAULT_RPC_URL,
    contract_address: str = DEFAULT_MATCH_PROOF,
):
    """Get a web3 Contract instance for MatchProof.

    Args:
        rpc_url: JSON-RPC endpoint.
        contract_address: Deployed MatchProof address.

    Returns:
        (web3_instance, contract) tuple.
    """
    Web3 = _require_web3()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=MATCH_PROOF_ABI,
    )
    return w3, contract


def get_digest(
    match_result: dict[str, Any],
    rpc_url: str = DEFAULT_RPC_URL,
    contract_address: str = DEFAULT_MATCH_PROOF,
) -> bytes:
    """Call getDigest() on the MatchProof contract (view, no gas).

    Args:
        match_result: Dict with MatchResult struct fields.
        rpc_url: JSON-RPC endpoint.
        contract_address: Deployed MatchProof address.

    Returns:
        The 32-byte EIP-712 digest from the contract.
    """
    _, contract = get_match_proof_contract(rpc_url, contract_address)
    result_tuple = _match_result_to_tuple(match_result)
    digest = contract.functions.getDigest(result_tuple).call()
    return bytes(digest)


def record_match(
    match_result: dict[str, Any],
    sig_a: bytes,
    sig_b: bytes,
    account,
    rpc_url: str = DEFAULT_RPC_URL,
    contract_address: str = DEFAULT_MATCH_PROOF,
) -> str:
    """Submit a dual-signed match result onchain via recordMatch().

    Args:
        match_result: Dict with MatchResult struct fields.
        sig_a: 65-byte signature from one participant.
        sig_b: 65-byte signature from the other participant.
        account: eth_account LocalAccount (for signing the transaction).
        rpc_url: JSON-RPC endpoint.
        contract_address: Deployed MatchProof address.

    Returns:
        Transaction hash hex string.
    """
    Web3 = _require_web3()
    w3, contract = get_match_proof_contract(rpc_url, contract_address)
    result_tuple = _match_result_to_tuple(match_result)

    tx = contract.functions.recordMatch(result_tuple, sig_a, sig_b).build_transaction(
        {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "chainId": w3.eth.chain_id,
        }
    )

    signed_tx = w3.eth.account.sign_transaction(tx, account.key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    logger.info(f"recordMatch tx sent: {tx_hash.hex()}")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
    if receipt["status"] != 1:
        raise RuntimeError(f"recordMatch reverted: {tx_hash.hex()}")

    logger.info(f"recordMatch confirmed in block {receipt['blockNumber']}")
    return tx_hash.hex()


def get_match(
    match_id: bytes,
    rpc_url: str = DEFAULT_RPC_URL,
    contract_address: str = DEFAULT_MATCH_PROOF,
) -> dict[str, Any]:
    """Read a recorded match result from the contract.

    Args:
        match_id: 32-byte match ID.
        rpc_url: JSON-RPC endpoint.
        contract_address: Deployed MatchProof address.

    Returns:
        Dict with MatchResult fields, or empty-ish dict if not recorded.
    """
    _, contract = get_match_proof_contract(rpc_url, contract_address)
    raw = contract.functions.getMatch(match_id).call()
    return {
        "matchId": bytes(raw[0]),
        "winner": raw[1],
        "loser": raw[2],
        "gameId": raw[3],
        "winnerScore": raw[4],
        "loserScore": raw[5],
        "replayHash": bytes(raw[6]),
        "timestamp": raw[7],
    }


def is_recorded(
    match_id: bytes,
    rpc_url: str = DEFAULT_RPC_URL,
    contract_address: str = DEFAULT_MATCH_PROOF,
) -> bool:
    """Check if a match ID has been recorded onchain."""
    _, contract = get_match_proof_contract(rpc_url, contract_address)
    return contract.functions.recorded(match_id).call()


def _match_result_to_tuple(match_result: dict[str, Any]) -> tuple:
    """Convert a MatchResult dict to the tuple format web3.py expects for structs."""
    return (
        match_result["matchId"],
        match_result["winner"],
        match_result["loser"],
        match_result["gameId"],
        match_result["winnerScore"],
        match_result["loserScore"],
        match_result["replayHash"],
        match_result["timestamp"],
    )
