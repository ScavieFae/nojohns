"""Tests for nojohns.wallet — wallet management and EIP-712 signing."""

import hashlib
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

# Skip entire module if eth-account isn't installed
eth_account = pytest.importorskip("eth_account", reason="eth-account not installed")

from nojohns.wallet import (
    MATCH_RESULT_TYPES,
    generate_wallet,
    load_wallet,
    sign_match_result,
    recover_signer,
)
from nojohns.config import NojohnsConfig, WalletConfig, ChainConfig


# A fixed test key for deterministic tests
TEST_PRIVATE_KEY = "0x4c0883a69102937d6231471b5dbb6204fe512961708279f3a3e6d8b4f8e2c7e1"
TEST_CONTRACT = "0x1234567890AbcdEF1234567890aBcdef12345678"
TEST_CHAIN_ID = 10143


@pytest.fixture
def test_account():
    """A deterministic test account."""
    return eth_account.Account.from_key(TEST_PRIVATE_KEY)


@pytest.fixture
def sample_match_result():
    """A sample MatchResult dict for signing tests."""
    return {
        "matchId": hashlib.sha256(b"test-match-001").digest(),
        "winner": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
        "loser": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
        "gameId": "melee",
        "winnerScore": 3,
        "loserScore": 1,
        "replayHash": hashlib.sha256(b"replay-data").digest(),
        "timestamp": 1707000000,
    }


class TestGenerateWallet:
    def test_returns_address_and_key(self):
        address, key = generate_wallet()
        assert address.startswith("0x")
        assert len(address) == 42
        assert key.startswith("0x")
        assert len(key) == 66  # 0x + 64 hex chars

    def test_generates_unique_wallets(self):
        addr1, _ = generate_wallet()
        addr2, _ = generate_wallet()
        assert addr1 != addr2

    def test_key_recovers_to_address(self):
        address, key = generate_wallet()
        account = eth_account.Account.from_key(key)
        assert account.address == address


class TestLoadWallet:
    def test_loads_from_config(self):
        config = NojohnsConfig(
            wallet=WalletConfig(
                address="0x1234",
                private_key=TEST_PRIVATE_KEY,
            )
        )
        account = load_wallet(config)
        assert account is not None
        assert account.address == eth_account.Account.from_key(TEST_PRIVATE_KEY).address

    def test_returns_none_without_wallet(self):
        config = NojohnsConfig()
        assert load_wallet(config) is None

    def test_returns_none_without_key(self):
        config = NojohnsConfig(
            wallet=WalletConfig(address="0x1234")
        )
        assert load_wallet(config) is None

    def test_handles_key_without_0x_prefix(self):
        key_no_prefix = TEST_PRIVATE_KEY[2:]  # strip 0x
        config = NojohnsConfig(
            wallet=WalletConfig(
                address="0x1234",
                private_key=key_no_prefix,
            )
        )
        account = load_wallet(config)
        assert account is not None

    def test_loads_from_config_file(self, tmp_path):
        """Test round-trip: write config.toml, load it, get wallet."""
        from nojohns.config import load_config

        config_path = tmp_path / "config.toml"
        config_path.write_text(textwrap.dedent(f"""\
            [wallet]
            address = "0xdeadbeef"
            private_key = "{TEST_PRIVATE_KEY}"
        """))

        cfg = load_config(config_path)
        assert cfg.wallet is not None
        assert cfg.wallet.private_key == TEST_PRIVATE_KEY

        account = load_wallet(cfg)
        assert account is not None


class TestSignMatchResult:
    def test_produces_65_byte_signature(self, test_account, sample_match_result):
        sig = sign_match_result(
            test_account,
            sample_match_result,
            chain_id=TEST_CHAIN_ID,
            contract_address=TEST_CONTRACT,
        )
        assert isinstance(sig, bytes)
        assert len(sig) == 65

    def test_deterministic_signature(self, test_account, sample_match_result):
        sig1 = sign_match_result(
            test_account, sample_match_result,
            chain_id=TEST_CHAIN_ID, contract_address=TEST_CONTRACT,
        )
        sig2 = sign_match_result(
            test_account, sample_match_result,
            chain_id=TEST_CHAIN_ID, contract_address=TEST_CONTRACT,
        )
        assert sig1 == sig2

    def test_different_data_different_sig(self, test_account, sample_match_result):
        sig1 = sign_match_result(
            test_account, sample_match_result,
            chain_id=TEST_CHAIN_ID, contract_address=TEST_CONTRACT,
        )
        modified = {**sample_match_result, "winnerScore": 2}
        sig2 = sign_match_result(
            test_account, modified,
            chain_id=TEST_CHAIN_ID, contract_address=TEST_CONTRACT,
        )
        assert sig1 != sig2

    def test_different_chain_different_sig(self, test_account, sample_match_result):
        sig_testnet = sign_match_result(
            test_account, sample_match_result,
            chain_id=10143, contract_address=TEST_CONTRACT,
        )
        sig_mainnet = sign_match_result(
            test_account, sample_match_result,
            chain_id=143, contract_address=TEST_CONTRACT,
        )
        assert sig_testnet != sig_mainnet


class TestRecoverSigner:
    def test_recovers_correct_address(self, test_account, sample_match_result):
        sig = sign_match_result(
            test_account, sample_match_result,
            chain_id=TEST_CHAIN_ID, contract_address=TEST_CONTRACT,
        )
        recovered = recover_signer(
            sample_match_result, sig,
            chain_id=TEST_CHAIN_ID, contract_address=TEST_CONTRACT,
        )
        assert recovered == test_account.address

    def test_wrong_data_recovers_wrong_address(self, test_account, sample_match_result):
        sig = sign_match_result(
            test_account, sample_match_result,
            chain_id=TEST_CHAIN_ID, contract_address=TEST_CONTRACT,
        )
        modified = {**sample_match_result, "winnerScore": 2}
        recovered = recover_signer(
            modified, sig,
            chain_id=TEST_CHAIN_ID, contract_address=TEST_CONTRACT,
        )
        assert recovered != test_account.address

    def test_roundtrip_with_generated_wallet(self, sample_match_result):
        address, key = generate_wallet()
        account = eth_account.Account.from_key(key)

        sig = sign_match_result(
            account, sample_match_result,
            chain_id=TEST_CHAIN_ID, contract_address=TEST_CONTRACT,
        )
        recovered = recover_signer(
            sample_match_result, sig,
            chain_id=TEST_CHAIN_ID, contract_address=TEST_CONTRACT,
        )
        assert recovered == address


class TestMatchResultTypes:
    def test_types_structure(self):
        assert "EIP712Domain" in MATCH_RESULT_TYPES
        assert "MatchResult" in MATCH_RESULT_TYPES

    def test_domain_fields(self):
        domain_fields = {f["name"] for f in MATCH_RESULT_TYPES["EIP712Domain"]}
        assert domain_fields == {"name", "version", "chainId", "verifyingContract"}

    def test_match_result_fields(self):
        result_fields = {f["name"] for f in MATCH_RESULT_TYPES["MatchResult"]}
        expected = {
            "matchId", "winner", "loser", "gameId",
            "winnerScore", "loserScore", "replayHash", "timestamp",
        }
        assert result_fields == expected


class TestConfigParsing:
    """Test that wallet and chain config round-trip through config.toml."""

    def test_wallet_config_from_toml(self, tmp_path):
        from nojohns.config import load_config

        config_path = tmp_path / "config.toml"
        config_path.write_text(textwrap.dedent("""\
            [wallet]
            address = "0xABCD1234"
            private_key = "0xdeadbeef"

            [chain]
            chain_id = 143
            rpc_url = "https://rpc.monad.xyz"
            match_proof = "0x1111111111111111111111111111111111111111"
        """))

        cfg = load_config(config_path)
        assert cfg.wallet is not None
        assert cfg.wallet.address == "0xABCD1234"
        assert cfg.wallet.private_key == "0xdeadbeef"

        assert cfg.chain is not None
        assert cfg.chain.chain_id == 143
        assert cfg.chain.rpc_url == "https://rpc.monad.xyz"
        assert cfg.chain.match_proof == "0x1111111111111111111111111111111111111111"
        # wager not set in toml → falls through to mainnet default
        assert cfg.chain.wager == "0x8d4D9FD03242410261b2F9C0e66fE2131AE0459d"

    def test_chain_defaults(self, tmp_path):
        from nojohns.config import load_config

        config_path = tmp_path / "config.toml"
        config_path.write_text(textwrap.dedent("""\
            [chain]
        """))

        cfg = load_config(config_path)
        assert cfg.chain is not None
        assert cfg.chain.chain_id == 143
        assert cfg.chain.rpc_url == "https://rpc.monad.xyz"

    def test_no_wallet_section(self, tmp_path):
        from nojohns.config import load_config

        config_path = tmp_path / "config.toml"
        config_path.write_text(textwrap.dedent("""\
            [games.melee]
            dolphin = "/path"
        """))

        cfg = load_config(config_path)
        assert cfg.wallet is None
        assert cfg.chain is None
