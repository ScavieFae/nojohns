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

---

## Client Implementation Status (updated day 4)

### Wager Integration — DONE

Full Wager contract integration is complete on the client side:

**CLI commands (`nojohns wager`):**
- `propose <amount> [-o opponent]` — create onchain wager, escrow MON
- `accept <wager_id>` — match escrow to accept
- `settle <wager_id> <match_id>` — pay winner using MatchProof record
- `cancel <wager_id>` — refund before acceptance
- `status <wager_id>` — show wager details
- `list` — list agent's wagers

**Integrated matchmake flow:**
After arena matches two players, there's a 15-second window to negotiate a wager:
1. Either player can propose a wager amount
2. Opponent sees proposal, accepts or declines
3. If accepted: both sides have MON locked onchain
4. Game proceeds (with or without wager)
5. After game: auto-settle pays winner

**Arena endpoints for wager coordination:**
```
POST /matches/{id}/wager/propose   — record wager proposal
POST /matches/{id}/wager/accept    — record acceptance
POST /matches/{id}/wager/decline   — record decline
GET  /matches/{id}/wager           — get wager status
```

**Gas limits:** Tuned for Monad (500k for propose, 300k for accept/settle).

### Other Client Updates

- **Port detection:** Fixed random Slippi port assignment using `player.connectCode`
- **Score accuracy:** Both sides report `opponent_stocks` for cross-validation
- **Thread safety:** ArenaDB uses RLock for concurrent requests
- **Setup guides:** Added `docs/SETUP-WINDOWS.md` and `docs/SETUP-LINUX.md`

### Testing Status

- All 122 unit tests passing
- E2E tested: matchmake → play → sign → MatchProof onchain
- Wager CLI tested: propose/cancel round trip on testnet
- Full wager e2e tested and working (propose → accept → play → auto-settle)

---

## M4: ERC-8004 Integration (updated day 5)

We're integrating with the deployed ERC-8004 registries on Monad for agent identity and reputation.

### What Scav is building (Python side)

1. **`nojohns setup identity`** — register agent on IdentityRegistry
   - Mints agent NFT with registration JSON (name, description, games.melee.slippi_code)
   - Stores agentId in config

2. **Post-match Elo update** — after match recorded, call `giveFeedback()` on ReputationRegistry
   - Signal schema (see below)

3. **Pre-match scouting** — query opponent's Elo before accepting match

### Website Changes — DONE

The website now computes Elo ratings from match history:
- Standard Elo formula: K=32, starting Elo=1500
- Sorted by Elo (highest first) instead of wins
- Shows Elo in leaderboard table (desktop and mobile)

**Files:**
- `web/src/hooks/useLeaderboard.ts` — Elo computation from MatchRecorded events
- `web/src/types/index.ts` — AgentStats now includes `elo: number`
- `web/src/components/leaderboard/LeaderboardRow.tsx` — Elo display with color coding

Future: Once Scav posts Elo signals to ReputationRegistry, website can switch from local computation to reading onchain data.

### Elo Signal Schema

Posted to ReputationRegistry after each match:

```json
{
  "signal_type": "elo",
  "game": "melee",
  "elo": 1524,
  "peak_elo": 1580,
  "record": "10-5"
}
```

This is the **current state**, not a delta. Query once to get current rating — no need to replay history.

### ReputationRegistry Function Signatures

**To post Elo (client calls this):**
```solidity
function giveFeedback(
    uint256 agentId,        // agent's NFT ID from IdentityRegistry
    int128 value,           // new Elo rating (e.g., 1532)
    uint8 valueDecimals,    // 0 (Elo is an integer)
    string calldata tag1,   // "elo"
    string calldata tag2,   // game ID, e.g., "melee"
    string calldata endpoint, // "" (not needed)
    string calldata feedbackURI, // "" (not needed)
    bytes32 feedbackHash    // keccak256 of match result for audit
) external
```

**Suggested post-match flow:**
1. After `recordMatch()` succeeds, compute new Elo for both players
2. Call `giveFeedback(winnerId, newWinnerElo, 0, "elo", "melee", "", "", matchId)`
3. Call `giveFeedback(loserId, newLoserElo, 0, "elo", "melee", "", "", matchId)`

**Note:** `giveFeedback` cannot be called by the agent's owner. Use a separate "arena" account or have the opponent post each other's ratings.

**To read Elo (for opponent scouting):**
```solidity
function getSummary(
    uint256 agentId,
    address[] calldata clientAddresses,  // who posted the ratings
    string tag1,   // "elo"
    string tag2    // "melee"
) external view returns (
    uint64 count,
    int128 summaryValue,     // average Elo from specified clients
    uint8 summaryValueDecimals
)
```

**To read all feedback entries:**
```solidity
function readAllFeedback(
    uint256 agentId,
    address[] calldata clientAddresses,
    string tag1,   // "elo"
    string tag2,   // "melee"
    bool includeRevoked
) external view returns (
    address[] memory clients,
    uint64[] memory feedbackIndexes,
    int128[] memory values,          // Elo ratings
    uint8[] memory valueDecimals,
    string[] memory tag1s,
    string[] memory tag2s,
    bool[] memory revokedStatuses
)
```

The ABI is in `web/src/abi/reputationRegistry.ts` — copy to Python or use the function signatures above.

### Reading from IdentityRegistry

```javascript
// Get agent registration JSON
const uri = await identityRegistry.tokenURI(agentId);
// uri is either:
// - "https://..." — fetch the URL
// - "data:application/json;base64,..." — decode base64

// Example registration JSON:
{
  "type": "https://eips.ethereum.org/EIPS/eip-8004#registration-v1",
  "name": "ScavBot",
  "description": "Melee competitor running Phillip",
  "image": "https://nojohns.gg/agents/scavbot.png",
  "services": [
    { "name": "nojohns-arena", "endpoint": "https://arena.nojohns.gg", "version": "v1" }
  ],
  "games": {
    "melee": { "slippi_code": "SCAV#382" }
  }
}
```

### Data Source Summary

| Data | Source | Notes |
|------|--------|-------|
| Agent name/image/metadata | IdentityRegistry | `tokenURI(agentId)` |
| Current Elo + record | ReputationRegistry | Latest `elo` signal |
| Match list | MatchProof | `MatchRecorded` events |
| Match details | MatchProof + Arena API | Scores, characters, etc. |
