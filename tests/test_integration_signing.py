"""
Integration test: EIP-712 digest verification against the testnet MatchProof contract.

This test generates two wallets, constructs a MatchResult, signs it with both,
then calls getDigest() on the deployed testnet contract and verifies the Python
EIP-712 digest matches the Solidity digest. This is the critical integration
checkpoint — if digests match, recordMatch() will accept our signatures.

Requires: pip install nojohns[wallet]
Requires: network access to Monad testnet RPC

Run:
    .venv/bin/python -m pytest tests/test_integration_signing.py -v -o "addopts="
"""

import os
import time
import uuid

import pytest

# Skip entire module if dependencies aren't installed
try:
    import eth_account
    from eth_account.messages import encode_typed_data
    from web3 import Web3
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="eth-account and web3 required")

from nojohns.wallet import (
    generate_wallet,
    sign_match_result,
    MATCH_RESULT_TYPES,
    DOMAIN_NAME,
    DOMAIN_VERSION,
)
from nojohns.contract import (
    get_digest,
    get_match_proof_contract,
    DEFAULT_MATCH_PROOF,
    DEFAULT_RPC_URL,
)

# Use mainnet values (defaults flipped to mainnet in config)
CHAIN_ID = 143
CONTRACT_ADDRESS = DEFAULT_MATCH_PROOF
RPC_URL = DEFAULT_RPC_URL


def _make_match_result(winner: str, loser: str) -> dict:
    """Construct a realistic MatchResult for testing."""
    match_id = Web3.keccak(text=str(uuid.uuid4()))
    replay_hash = Web3.keccak(text="test-replay-data")
    return {
        "matchId": match_id,
        "winner": Web3.to_checksum_address(winner),
        "loser": Web3.to_checksum_address(loser),
        "gameId": "melee",
        "winnerScore": 3,
        "loserScore": 1,
        "replayHash": replay_hash,
        "timestamp": int(time.time()),
    }


def _python_digest(match_result: dict) -> bytes:
    """Compute the full EIP-712 digest: keccak256(0x19 0x01 + domainSep + structHash).

    eth-account's encode_typed_data returns a SignableMessage where:
    - header = domain separator hash
    - body = struct hash
    The full digest is keccak256(\\x19\\x01 + header + body).
    """
    domain_data = {
        "name": DOMAIN_NAME,
        "version": DOMAIN_VERSION,
        "chainId": CHAIN_ID,
        "verifyingContract": CONTRACT_ADDRESS,
    }
    message_types = {"MatchResult": MATCH_RESULT_TYPES["MatchResult"]}
    signable = encode_typed_data(
        domain_data=domain_data,
        message_types=message_types,
        message_data=match_result,
    )
    # Full EIP-712 digest = keccak256(0x19 0x01 + domainSeparator + structHash)
    return bytes(Web3.keccak(b"\x19\x01" + signable.header + signable.body))


class TestDigestVerification:
    """Verify Python's EIP-712 digest matches the Solidity contract's getDigest()."""

    @pytest.fixture
    def two_wallets(self):
        """Generate two fresh wallets (no funding needed for view calls)."""
        addr_a, key_a = generate_wallet()
        addr_b, key_b = generate_wallet()
        account_a = eth_account.Account.from_key(key_a)
        account_b = eth_account.Account.from_key(key_b)
        return account_a, account_b

    @pytest.fixture
    def match_result(self, two_wallets):
        """Construct a MatchResult using the two test wallets."""
        account_a, account_b = two_wallets
        return _make_match_result(account_a.address, account_b.address)

    def test_contract_is_reachable(self):
        """Sanity check: can we talk to the testnet contract?"""
        w3, contract = get_match_proof_contract(RPC_URL, CONTRACT_ADDRESS)
        assert w3.is_connected(), "Cannot connect to Monad testnet RPC"
        assert w3.eth.chain_id == CHAIN_ID

    def test_digest_matches(self, two_wallets, match_result):
        """The critical test: Python EIP-712 digest must match Solidity's getDigest()."""
        # Get digest from the contract (view call, no gas)
        solidity_digest = get_digest(match_result, RPC_URL, CONTRACT_ADDRESS)

        # Compute digest locally using the same method as wallet.py
        python_digest = _python_digest(match_result)

        assert python_digest == solidity_digest, (
            f"Digest mismatch!\n"
            f"  Python:   {python_digest.hex()}\n"
            f"  Solidity: {solidity_digest.hex()}\n"
            f"  This means EIP-712 signing params differ between Python and Solidity."
        )

    def test_signatures_recover_correctly(self, two_wallets, match_result):
        """Both signatures should recover to the correct signer addresses."""
        account_a, account_b = two_wallets

        sig_a = sign_match_result(account_a, match_result, CHAIN_ID, CONTRACT_ADDRESS)
        sig_b = sign_match_result(account_b, match_result, CHAIN_ID, CONTRACT_ADDRESS)

        assert len(sig_a) == 65, f"sig_a is {len(sig_a)} bytes, expected 65"
        assert len(sig_b) == 65, f"sig_b is {len(sig_b)} bytes, expected 65"

        # Recover signers locally
        from nojohns.wallet import recover_signer

        recovered_a = recover_signer(match_result, sig_a, CHAIN_ID, CONTRACT_ADDRESS)
        recovered_b = recover_signer(match_result, sig_b, CHAIN_ID, CONTRACT_ADDRESS)

        assert recovered_a == account_a.address, (
            f"sig_a recovered to {recovered_a}, expected {account_a.address}"
        )
        assert recovered_b == account_b.address, (
            f"sig_b recovered to {recovered_b}, expected {account_b.address}"
        )

    def test_both_sign_same_digest(self, two_wallets, match_result):
        """Both wallets sign the same match data and produce valid signatures
        that would be accepted by recordMatch()."""
        account_a, account_b = two_wallets

        sig_a = sign_match_result(account_a, match_result, CHAIN_ID, CONTRACT_ADDRESS)
        sig_b = sign_match_result(account_b, match_result, CHAIN_ID, CONTRACT_ADDRESS)

        # Signatures should be different (different keys)
        assert sig_a != sig_b, "Two different keys produced identical signatures"

        # But both should be valid against the same digest
        solidity_digest = get_digest(match_result, RPC_URL, CONTRACT_ADDRESS)
        python_digest = _python_digest(match_result)
        assert python_digest == solidity_digest

        # Both recover to participants
        from nojohns.wallet import recover_signer

        recovered_a = recover_signer(match_result, sig_a, CHAIN_ID, CONTRACT_ADDRESS)
        recovered_b = recover_signer(match_result, sig_b, CHAIN_ID, CONTRACT_ADDRESS)

        participants = {account_a.address, account_b.address}
        assert recovered_a in participants
        assert recovered_b in participants
        assert recovered_a != recovered_b
