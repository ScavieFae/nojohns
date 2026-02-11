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

# Default address from contracts/deployments.json
# Updated to mainnet after deploy; override via config.toml [chain] section
DEFAULT_MATCH_PROOF = "0x1CC748475F1F666017771FB49131708446B9f3DF"  # TODO: update after mainnet deploy
DEFAULT_RPC_URL = "https://rpc.monad.xyz"


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


# =============================================================================
# PredictionPool
# =============================================================================

PREDICTION_POOL_ABI = [
    {
        "type": "function",
        "name": "createPool",
        "inputs": [
            {"name": "matchId", "type": "bytes32"},
            {"name": "playerA", "type": "address"},
            {"name": "playerB", "type": "address"},
        ],
        "outputs": [{"name": "poolId", "type": "uint256"}],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "resolve",
        "inputs": [{"name": "poolId", "type": "uint256"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "cancelPool",
        "inputs": [{"name": "poolId", "type": "uint256"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "getPool",
        "inputs": [{"name": "poolId", "type": "uint256"}],
        "outputs": [
            {
                "name": "",
                "type": "tuple",
                "components": [
                    {"name": "matchId", "type": "bytes32"},
                    {"name": "playerA", "type": "address"},
                    {"name": "playerB", "type": "address"},
                    {"name": "totalA", "type": "uint256"},
                    {"name": "totalB", "type": "uint256"},
                    {"name": "status", "type": "uint8"},
                    {"name": "winner", "type": "address"},
                    {"name": "createdAt", "type": "uint256"},
                ],
            }
        ],
        "stateMutability": "view",
    },
]


def get_prediction_pool_contract(rpc_url: str, contract_address: str):
    """Get a web3 Contract instance for PredictionPool.

    Returns:
        (web3_instance, contract) tuple.
    """
    Web3 = _require_web3()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=PREDICTION_POOL_ABI,
    )
    return w3, contract


def create_pool(
    match_id_bytes: bytes,
    player_a: str,
    player_b: str,
    account,
    rpc_url: str,
    contract_address: str,
) -> int:
    """Create a prediction pool for a match. Returns the pool ID."""
    Web3 = _require_web3()
    w3, contract = get_prediction_pool_contract(rpc_url, contract_address)

    fn = contract.functions.createPool(
        match_id_bytes,
        Web3.to_checksum_address(player_a),
        Web3.to_checksum_address(player_b),
    )

    # Simulate first to get the return value (pool ID)
    pool_id = fn.call({"from": account.address})

    tx = fn.build_transaction(
        {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "chainId": w3.eth.chain_id,
            "gas": 300_000,
        }
    )

    signed_tx = w3.eth.account.sign_transaction(tx, account.key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    logger.info(f"createPool tx sent: {tx_hash.hex()}")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
    if receipt["status"] != 1:
        raise RuntimeError(f"createPool reverted: {tx_hash.hex()}")

    logger.info(f"createPool confirmed in block {receipt['blockNumber']}, poolId={pool_id}")
    return pool_id


def resolve_pool(
    pool_id: int,
    account,
    rpc_url: str,
    contract_address: str,
) -> str:
    """Resolve a prediction pool after match is recorded onchain. Returns tx hash."""
    w3, contract = get_prediction_pool_contract(rpc_url, contract_address)

    tx = contract.functions.resolve(pool_id).build_transaction(
        {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "chainId": w3.eth.chain_id,
            "gas": 300_000,
        }
    )

    signed_tx = w3.eth.account.sign_transaction(tx, account.key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    logger.info(f"resolve tx sent: {tx_hash.hex()}")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
    if receipt["status"] != 1:
        raise RuntimeError(f"resolve reverted: {tx_hash.hex()}")

    logger.info(f"resolve confirmed in block {receipt['blockNumber']}")
    return tx_hash.hex()


def cancel_pool(
    pool_id: int,
    account,
    rpc_url: str,
    contract_address: str,
) -> str:
    """Cancel a prediction pool (refunds all bettors). Returns tx hash."""
    w3, contract = get_prediction_pool_contract(rpc_url, contract_address)

    tx = contract.functions.cancelPool(pool_id).build_transaction(
        {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "chainId": w3.eth.chain_id,
            "gas": 300_000,
        }
    )

    signed_tx = w3.eth.account.sign_transaction(tx, account.key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    logger.info(f"cancelPool tx sent: {tx_hash.hex()}")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
    if receipt["status"] != 1:
        raise RuntimeError(f"cancelPool reverted: {tx_hash.hex()}")

    logger.info(f"cancelPool confirmed in block {receipt['blockNumber']}")
    return tx_hash.hex()


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
