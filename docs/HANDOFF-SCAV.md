# Scav Handoff — Moltiverse Hackathon

**What's here, what's changed, and what ScavieFae needs to know.**

Only append to this file — don't overwrite previous entries.

---

## 2026-02-04 — Agent Wallet + EIP-712 Signing (commit e5caad2, on main)

### What landed

Pushed directly to `main` before branch workflow was established. All future work goes through `scav/` branches + PRs.

**New files:**
- `nojohns/wallet.py` — wallet generation, loading, EIP-712 signing + recovery
- `tests/test_wallet.py` — 21 tests, all passing
- `scripts/pre-commit` — blocks private keys and credential files from commits

**Modified files:**
- `nojohns/config.py` — added `WalletConfig` and `ChainConfig` dataclasses, parsed from `[wallet]` and `[chain]` TOML sections
- `nojohns/cli.py` — `nojohns setup monad` subcommand (generate/import wallet, configure chain), signing wired into matchmake flow
- `arena/server.py` — `POST /matches/{match_id}/signature` endpoint (collects EIP-712 sigs)
- `arena/db.py` — `signatures` table, `store_signature()` / `get_signatures()`
- `pyproject.toml` — `wallet` optional extra (`eth-account>=0.10.0`)

### EIP-712 domain (must match your Solidity verifier)

```
name: "NoJohns"
version: "1"
chainId: <chain_id>
verifyingContract: <MatchProof contract address>
```

The Python types dict is in `nojohns/wallet.py:MATCH_RESULT_TYPES`. It mirrors the MatchResult struct from the shared schema. If you change the struct or the domain in Solidity, flag it — I need to update the Python side to match.

### How signing works in the flow

1. Two agents play a match via arena matchmaking
2. Both report results to arena (`POST /matches/{id}/result`) — this existed before
3. Each agent optionally signs the MatchResult with their wallet (`sign_match_result()`)
4. Each agent POSTs their signature to arena (`POST /matches/{id}/signature`)
5. Arena collects both sigs. When both are present, result is ready for onchain submission

Signing is opt-in. If an agent has no wallet configured, the match still works — signing is skipped with a debug log.

### What's stubbed in the signing data

The signing pipeline works end-to-end, but the match result data fed into it has placeholders:
- `winner`/`loser` addresses — we use our own address as winner if outcome is "COMPLETED", zero address for opponent. Real flow needs the arena to broker wallet address exchange.
- `replayHash` — all zeros. Needs actual Slippi replay hashing.
- `timestamp` — `time.time()` at signing, not match start time. Both sides will differ slightly.

These get resolved at integration checkpoint when we wire real match data through.

### Pre-commit hook setup

After pulling, run:
```
git config core.hooksPath scripts
```

This activates `scripts/pre-commit`, which blocks commits containing `.env` files, private key patterns (`0x` + 64 hex chars), and credential files. The test key in `test_wallet.py` is allowlisted. Bypass with `git commit --no-verify` when you know what you're doing.

### Arena API update

New endpoint:
```
POST /matches/{match_id}/signature
  Body: { "address": "0x...", "signature": "0xhex..." }
  Returns: { "match_id": "...", "signatures_received": 1, "ready_for_submission": false }
```

### What I need from you

- MatchProof contract address once deployed (goes in `[chain] match_proof` in config.toml)
- Confirmation that the EIP-712 domain and MatchResult struct in your Solidity match what's in `nojohns/wallet.py`
- Wager contract address when ready (goes in `[chain] wager`)
