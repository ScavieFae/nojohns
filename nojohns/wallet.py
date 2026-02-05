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


# ============================================================================
# Wager Contract Interaction
# ============================================================================

# Minimal ABI for Wager contract — only the functions we call
WAGER_ABI = [
    {
        "name": "proposeWager",
        "type": "function",
        "inputs": [
            {"name": "opponent", "type": "address"},
            {"name": "gameId", "type": "string"},
        ],
        "outputs": [{"name": "wagerId", "type": "uint256"}],
        "stateMutability": "payable",
    },
    {
        "name": "acceptWager",
        "type": "function",
        "inputs": [{"name": "wagerId", "type": "uint256"}],
        "outputs": [],
        "stateMutability": "payable",
    },
    {
        "name": "settleWager",
        "type": "function",
        "inputs": [
            {"name": "wagerId", "type": "uint256"},
            {"name": "matchId", "type": "bytes32"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "name": "cancelWager",
        "type": "function",
        "inputs": [{"name": "wagerId", "type": "uint256"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "name": "claimTimeout",
        "type": "function",
        "inputs": [{"name": "wagerId", "type": "uint256"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "name": "getWager",
        "type": "function",
        "inputs": [{"name": "wagerId", "type": "uint256"}],
        "outputs": [
            {
                "name": "",
                "type": "tuple",
                "components": [
                    {"name": "proposer", "type": "address"},
                    {"name": "opponent", "type": "address"},
                    {"name": "gameId", "type": "string"},
                    {"name": "amount", "type": "uint256"},
                    {"name": "status", "type": "uint8"},
                    {"name": "acceptedAt", "type": "uint256"},
                    {"name": "matchId", "type": "bytes32"},
                ],
            }
        ],
        "stateMutability": "view",
    },
    {
        "name": "getWagersByAgent",
        "type": "function",
        "inputs": [{"name": "agent", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256[]"}],
        "stateMutability": "view",
    },
]

# Wager status enum (matches Solidity)
WAGER_STATUS = {
    0: "Open",
    1: "Accepted",
    2: "Settled",
    3: "Cancelled",
    4: "Voided",
}


def _require_web3():
    """Import and return web3, raising a clear error if not installed."""
    try:
        from web3 import Web3
        return Web3
    except ImportError:
        raise ImportError(
            "web3 is required for wager operations. "
            "Install it with: pip install nojohns[wallet]"
        )


def get_wager_contract(rpc_url: str, contract_address: str):
    """Get a web3 contract instance for the Wager contract.

    Args:
        rpc_url: RPC endpoint URL.
        contract_address: Deployed Wager contract address.

    Returns:
        web3.eth.Contract instance.
    """
    Web3 = _require_web3()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    return w3.eth.contract(address=contract_address, abi=WAGER_ABI)


def propose_wager(
    account,
    rpc_url: str,
    contract_address: str,
    opponent: str | None,
    game_id: str,
    amount_wei: int,
) -> tuple[str, int]:
    """Propose a wager by escrowing MON.

    Args:
        account: eth_account LocalAccount.
        rpc_url: RPC endpoint URL.
        contract_address: Wager contract address.
        opponent: Opponent address, or None for open wager.
        game_id: Game identifier (e.g. "melee").
        amount_wei: Amount to escrow in wei.

    Returns:
        (tx_hash, wager_id) tuple.
    """
    Web3 = _require_web3()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(address=contract_address, abi=WAGER_ABI)

    opponent_addr = opponent if opponent else "0x0000000000000000000000000000000000000000"

    tx = contract.functions.proposeWager(opponent_addr, game_id).build_transaction({
        "from": account.address,
        "value": amount_wei,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 500000,  # Higher gas limit for Monad
        "gasPrice": w3.eth.gas_price,
    })

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    # Parse wagerId from logs (WagerProposed event)
    # Event signature: WagerProposed(uint256 indexed wagerId, ...)
    wager_id = None
    for log in receipt.logs:
        if len(log.topics) >= 2:
            # First topic is event signature, second is indexed wagerId
            wager_id = int(log.topics[1].hex(), 16)
            break

    return (tx_hash.hex(), wager_id)


def accept_wager(
    account,
    rpc_url: str,
    contract_address: str,
    wager_id: int,
    amount_wei: int,
) -> str:
    """Accept a wager by escrowing matching MON.

    Args:
        account: eth_account LocalAccount.
        rpc_url: RPC endpoint URL.
        contract_address: Wager contract address.
        wager_id: The wager ID to accept.
        amount_wei: Amount to escrow (must match wager amount).

    Returns:
        Transaction hash.
    """
    Web3 = _require_web3()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(address=contract_address, abi=WAGER_ABI)

    tx = contract.functions.acceptWager(wager_id).build_transaction({
        "from": account.address,
        "value": amount_wei,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 300000,  # Higher gas limit for Monad
        "gasPrice": w3.eth.gas_price,
    })

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    w3.eth.wait_for_transaction_receipt(tx_hash)

    return tx_hash.hex()


def settle_wager(
    account,
    rpc_url: str,
    contract_address: str,
    wager_id: int,
    match_id: bytes,
) -> str:
    """Settle a wager using a recorded match result.

    Args:
        account: eth_account LocalAccount (or any account — settlement is permissionless).
        rpc_url: RPC endpoint URL.
        contract_address: Wager contract address.
        wager_id: The wager ID to settle.
        match_id: The bytes32 match ID from MatchProof.

    Returns:
        Transaction hash.
    """
    Web3 = _require_web3()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(address=contract_address, abi=WAGER_ABI)

    tx = contract.functions.settleWager(wager_id, match_id).build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 300000,  # Higher gas limit for Monad
        "gasPrice": w3.eth.gas_price,
    })

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    w3.eth.wait_for_transaction_receipt(tx_hash)

    return tx_hash.hex()


def cancel_wager(
    account,
    rpc_url: str,
    contract_address: str,
    wager_id: int,
) -> str:
    """Cancel an open wager and get refund.

    Args:
        account: eth_account LocalAccount (must be the proposer).
        rpc_url: RPC endpoint URL.
        contract_address: Wager contract address.
        wager_id: The wager ID to cancel.

    Returns:
        Transaction hash.
    """
    Web3 = _require_web3()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(address=contract_address, abi=WAGER_ABI)

    tx = contract.functions.cancelWager(wager_id).build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 200000,  # Higher gas limit for Monad
        "gasPrice": w3.eth.gas_price,
    })

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    w3.eth.wait_for_transaction_receipt(tx_hash)

    return tx_hash.hex()


def claim_timeout(
    account,
    rpc_url: str,
    contract_address: str,
    wager_id: int,
) -> str:
    """Claim timeout on an accepted wager with no match result.

    Args:
        account: eth_account LocalAccount (anyone can call).
        rpc_url: RPC endpoint URL.
        contract_address: Wager contract address.
        wager_id: The wager ID to void.

    Returns:
        Transaction hash.
    """
    Web3 = _require_web3()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(address=contract_address, abi=WAGER_ABI)

    tx = contract.functions.claimTimeout(wager_id).build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 300000,  # Higher gas limit for Monad
        "gasPrice": w3.eth.gas_price,
    })

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    w3.eth.wait_for_transaction_receipt(tx_hash)

    return tx_hash.hex()


def get_wager_info(
    rpc_url: str,
    contract_address: str,
    wager_id: int,
) -> dict[str, Any]:
    """Get wager details.

    Args:
        rpc_url: RPC endpoint URL.
        contract_address: Wager contract address.
        wager_id: The wager ID to query.

    Returns:
        Dict with wager fields: proposer, opponent, gameId, amount, status, acceptedAt, matchId.
    """
    Web3 = _require_web3()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(address=contract_address, abi=WAGER_ABI)

    result = contract.functions.getWager(wager_id).call()

    return {
        "proposer": result[0],
        "opponent": result[1],
        "gameId": result[2],
        "amount": result[3],
        "status": WAGER_STATUS.get(result[4], f"Unknown({result[4]})"),
        "status_code": result[4],
        "acceptedAt": result[5],
        "matchId": result[6].hex() if result[6] else None,
    }


def get_agent_wagers(
    rpc_url: str,
    contract_address: str,
    agent_address: str,
) -> list[int]:
    """Get all wager IDs for an agent.

    Args:
        rpc_url: RPC endpoint URL.
        contract_address: Wager contract address.
        agent_address: The agent's wallet address.

    Returns:
        List of wager IDs.
    """
    Web3 = _require_web3()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(address=contract_address, abi=WAGER_ABI)

    return contract.functions.getWagersByAgent(agent_address).call()
