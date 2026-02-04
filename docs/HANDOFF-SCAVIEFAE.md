# ScavieFae Handoff — Moltiverse Hackathon

**Read this first.** Then read `contracts/CLAUDE.md` and `web/CLAUDE.md` for your workstreams.

## Context

No Johns is agent competition infrastructure. Autonomous agents compete in skill-based games, wager real tokens, and build verifiable onchain track records. The first game is Melee (via Slippi netplay), but the contracts and protocol are game-agnostic.

**What already works (Python side — Scav owns this):**
- Fighter protocol, registry, CLI, config system
- Phillip (neural net fighter) and SmashBot (rule-based) both working
- Arena matchmaking server (FastAPI + SQLite) — code in `arena/`
- Netplay: two machines can match, fight, and report results
- Phillip has won matches live over netplay

**What you're building:**
1. Solidity contracts — MatchProof + Wager on Monad
2. Website — landing page, leaderboard, match history

## Your Workstreams

### 1. Contracts (`contracts/`)

See `contracts/CLAUDE.md` for full details. Summary:

- **MatchProof.sol** — records match results onchain with dual EIP-712 signatures + replay hash
- **Wager.sol** — escrow, accept, settle (reads from MatchProof), timeout for no-shows
- Deploy to Monad testnet first, then mainnet
- Forge tests for both contracts
- ERC-8004 IdentityRegistry and ReputationRegistry are already deployed on Monad — you interact with them, not deploy them

### 2. Website (`web/`)

See `web/CLAUDE.md` for full details. Summary:

- Landing page: what is No Johns, the moltbot/fighter concept
- Leaderboard: reads from ReputationRegistry on Monad
- Match history: reads from MatchProof events + arena REST API
- Tech stack is your call — Next.js, Astro, plain Vite+React, whatever ships fastest

## Shared Schema — MatchResult Struct

This is the contract between our workstreams. Both agents sign this struct (EIP-712 typed data). You write the Solidity that verifies signatures over it. Scav writes the Python that produces them.

```solidity
struct MatchResult {
    bytes32 matchId;      // unique match identifier (from arena)
    address winner;       // winner's agent wallet
    address loser;        // loser's agent wallet
    string gameId;        // "melee", "chess", etc. — game-agnostic
    uint8 winnerScore;    // games won (e.g., 3 in a Bo5)
    uint8 loserScore;     // games lost
    bytes32 replayHash;   // keccak256 of replay file(s)
    uint256 timestamp;    // unix timestamp of match completion
}
```

If you need to change this struct, flag it — Scav needs to update the Python signing code to match.

## ERC-8004 Contract Addresses

**Monad Mainnet (chain 143):**
- IdentityRegistry: `0x8004A169FB4a3325136EB29fA0ceB6D2e539a432`
- ReputationRegistry: `0x8004BAa17C55a88189AE136b182e5fdA19dE9b63`

**Monad Testnet (chain 10143):**
- IdentityRegistry: `0x8004A818BFB912233c491871b3d84c89A494BD9e`
- ReputationRegistry: `0x8004B663056A597Dffe9eCcC1965A193B7388713`

**Monad RPC:**
- Mainnet: `https://rpc.monad.xyz` (chain 143)
- Testnet: `https://testnet-rpc.monad.xyz` (chain 10143)

## Arena API (Scav's side — your website reads this)

The arena server runs on port 8000. Current endpoints:

```
GET  /health                       → { status, queue_size, active_matches }
POST /queue/join                   → { queue_id, status, match_id?, opponent_code? }
GET  /queue/{queue_id}             → { queue_id, status, position?, match_id?, opponent_code? }
DELETE /queue/{queue_id}           → { success }
POST /matches/{match_id}/result    → { success }
GET  /matches/{match_id}           → { id, status, p1_connect_code, p2_connect_code, results... }
```

Scav will add new endpoints for onchain integration:
```
GET  /matches/recent               → list of recent completed matches
GET  /agents                       → list of registered agents with onchain IDs
GET  /leaderboard                  → ranked agents with Elo
```

## Coordination

### Branch strategy
- You: `scaviefae/contracts`, `scaviefae/website`
- Scav: `scav/onchain-integration`
- Both PR into `main` when ready
- No overlapping file edits — you own `contracts/` and `web/`, Scav owns `nojohns/`, `games/`, `arena/`

### Shared artifacts
- You produce: contract ABIs in `contracts/out/`, deployed addresses in `contracts/deployments.json`
- Scav produces: arena API additions, Python signing code that matches your contract's EIP-712 domain

### Integration checkpoint (day 3-4)
When your contracts are on testnet:
1. Push to branch, PR with deployed addresses
2. Scav wires Python to call your contracts
3. We run an end-to-end test: matchmake → fight → sign result → onchain proof → wager settles

## Key Docs

- `docs/SPEC.md` — full system spec with milestone plan
- `docs/ERC8004-ARENA.md` — how ERC-8004 maps to the arena (identity, reputation, match proofs)
- `docs/FIGHTERS.md` — fighter interface spec (for context, not your workstream)
- `contracts/CLAUDE.md` — your Solidity working reference
- `web/CLAUDE.md` — your website working reference

---

## Implementation Status (updated day 3)

### Contracts — DONE, ready for testnet deploy

Both contracts are written, tested (30/30 passing), and have a deploy script.

**Files:**
- `contracts/src/MatchProof.sol` — match result recording with EIP-712 dual-sig verification
- `contracts/src/Wager.sol` — escrow + settlement, reads from MatchProof
- `contracts/test/MatchProof.t.sol` — 9 tests
- `contracts/test/Wager.t.sol` — 21 tests
- `contracts/script/Deploy.s.sol` — deploys both contracts

**No deviations from the spec.** The `MatchResult` struct is implemented exactly as defined above.

### EIP-712 Signing Details (client needs this)

**Domain:**
```
name: "NoJohns"
version: "1"
chainId: <chain-id>  (10143 for testnet, 143 for mainnet)
verifyingContract: <MatchProof deployed address>
```

**Type string:**
```
MatchResult(bytes32 matchId,address winner,address loser,string gameId,uint8 winnerScore,uint8 loserScore,bytes32 replayHash,uint256 timestamp)
```

**Important:** The `gameId` field is a `string`, which means it gets hashed as `keccak256(bytes(gameId))` in the struct hash per EIP-712 rules. All other fields are encoded directly.

**Python signing pseudocode (eth_account / web3.py):**
```python
from eth_account import Account
from eth_account.messages import encode_typed_data

typed_data = {
    "types": {
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
    },
    "primaryType": "MatchResult",
    "domain": {
        "name": "NoJohns",
        "version": "1",
        "chainId": 10143,  # testnet
        "verifyingContract": "<MatchProof address>",
    },
    "message": {
        "matchId": match_id_bytes32,
        "winner": winner_address,
        "loser": loser_address,
        "gameId": "melee",
        "winnerScore": 3,
        "loserScore": 1,
        "replayHash": replay_hash_bytes32,
        "timestamp": unix_timestamp,
    },
}

signed = Account.sign_typed_data(private_key, full_message=typed_data)
signature = signed.signature  # bytes, ready for recordMatch()
```

**Verification helper:** The protocol exposes `getDigest(MatchResult)` as a public view. The client can call this on-chain to verify its digest matches before submitting. Useful for debugging signature mismatches.

**Protocol / client interaction (post-match flow):**
1. Both agents sign the same `MatchResult` using EIP-712 (client)
2. Either agent (or anyone) calls `matchProof.recordMatch(result, sigA, sigB)` (protocol)
3. If there's a wager, call `wager.settleWager(wagerId, matchId)` — permissionless, anyone can trigger (protocol)

### Deploy

Waiting on a funded wallet. To deploy to testnet:
```bash
export PRIVATE_KEY=<deployer-private-key>
export MONAD_TESTNET_RPC_URL=https://testnet-rpc.monad.xyz
cd contracts && forge script script/Deploy.s.sol --rpc-url $MONAD_TESTNET_RPC_URL --broadcast
```

After deploy, addresses go in `contracts/deployments.json` for the client to wire up.

### Architecture Diagram

```
                        PROTOCOL (onchain — Monad)
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ┌──────────────────────┐                           │
│  │     MatchProof       │                           │
│  │                      │                           │
│  │  recordMatch(        │◄──── Agent A signs ───┐   │
│  │    result,           │◄──── Agent B signs ───┤   │
│  │    sigA, sigB)       │                       │   │
│  │                      │                       │   │
│  │  MatchRecorded ──────┼──────────────────┐    │   │
│  │  (event)             │                  │    │   │
│  └──────────┬───────────┘                  │    │   │
│             │                              │    │   │
│             │ reads (one-way)              │    │   │
│             ▼                              │    │   │
│  ┌──────────────────────┐                  │    │   │
│  │     Wager            │                  │    │   │
│  │     (optional)       │                  │    │   │
│  │                      │                  │    │   │
│  │  proposeWager() ←─── MON in            │    │   │
│  │  acceptWager()  ←─── MON in            │    │   │
│  │  settleWager()  ───► MON to winner     │    │   │
│  │  cancelWager()  ───► MON refund        │    │   │
│  │  claimTimeout() ───► MON refund both   │    │   │
│  └──────────────────────┘                  │    │   │
│                                            │    │   │
└────────────────────────────────────────────┼────┼───┘
                                             │    │
                                             │    │
┌────────────────────────┐                   │    │
│  FRONTEND (web/)       │◄──────────────────┘    │
│                        │  indexes events        │
│  - Landing page        │                        │
│  - Leaderboard         │                        │
│  - Match history       │                        │
└────────────────────────┘                        │
                                                  │
┌─────────────────────────────────────────────────┤
│  CLIENT (nojohns/, arena/, games/)              │
│                                                 │
│  Arena ──► Match ──► Both agents ──► EIP-712 ───┘
│  matchmake   fight     sign result    signatures
│                                       submitted to
│                                       recordMatch()
└─────────────────────────────────────────────────┘
```

**Boundary:** The `recordMatch()` call is the interface between protocol and client. The protocol doesn't know what language the client is written in, how matches are played, or what game it is. The client doesn't need to understand contract internals — just the ABI and EIP-712 domain.
