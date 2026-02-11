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

---

## ScavieFae's Web Commits (for Scav review)

These commits are all in `web/` — Scav please review when you have a chance.

| Commit | Description |
|--------|-------------|
| `e8e9b5b` | Add Elo ratings to leaderboard (M4) |
| `241a36a` | Rename competition tiers for clarity |
| `9ee9e14` | Fix live viewer character IDs, colors, and camera |
| `04470d5` | Add Live to main navigation with live indicator |
| `a3d7891` | Add debug logging and timeout handling to live match viewer |
| `b2f8f56` | Add live_match_ids to /health endpoint for spectator discovery |
| `399a362` | Add /live route and make "matches live" clickable |
| `9d810d0` | Add Melee replay viewer with SlippiLab animations |
| `d78a9e4` | Add error states, responsive cards, and incremental getLogs caching |
| `18ecf22` | Add CRT aesthetic and live arena status to hero section |
| `673f669` | Fix getLogs for Monad testnet 100-block query limit |
| `3506dc6` | Add website: landing, leaderboard, match history, compete pages |

**To review the full diff:**
```bash
git diff 3506dc6^..e8e9b5b -- web/
```

### Key Changes Summary

**Live Match Viewer** (`web/src/components/viewer/`, `web/src/hooks/useLiveMatch.ts`)
- WebSocket connection to arena `/ws/match/{id}` for frame streaming
- MeleeViewer renders characters using SlippiLab animation data (MIT licensed)
- Character IDs passed through directly (arena sends libmelee internal IDs)
- Player colors: green (P1), purple (P2) — brand colors
- 5-second timeout if no `match_start` received (match may have ended)
- Known issue #14: stage detection not implemented yet (hardcoded platform)

**Leaderboard Elo** (`web/src/hooks/useLeaderboard.ts`)
- Computes Elo from MatchRecorded events chronologically
- K=32, starting Elo=1500 (standard values)
- Sorts by Elo instead of wins
- Color coding: yellow ≥1600, white ≥1500, gray below
- Will switch to reading from ReputationRegistry once you're posting signals

**Competition Tiers** (`web/src/components/compete/CompeteContent.tsx`)
- Renamed: Play → Friendlies, Onchain → Competitive, Wager → Money Match
- Updated descriptions to match new names

**Navigation** (`web/src/components/layout/Header.tsx`)
- Added "Live" link between Home and Leaderboard
- Red pulsing dot when matches are streaming (reads from `/health` endpoint)

**Data Fetching** (`web/src/lib/getLogs.ts`, `web/src/hooks/useMatchEvents.ts`)
- Batched getLogs to handle Monad's 100-block query limit
- Incremental caching: only scans new blocks on refetch
- Error states with retry buttons on Leaderboard and Match History

**Hero Section** (`web/src/components/landing/Hero.tsx`)
- CRT scanline aesthetic
- Live arena status from `/health` (queue size, active matches)

---

## Scav's M4 Implementation Status (updated day 5, evening)

### Completed

**1. `nojohns setup identity`** ✅
- Registers agent on ERC-8004 IdentityRegistry
- Builds registration JSON with name, description, games.melee.slippi_code
- Encodes as data URI (fully onchain, no IPFS dependency)
- Saves agentId to config
- Tested: ScavBot registered as agent #9 on testnet

**2. Arena-based Elo posting** ✅
- Arena server posts Elo updates to ReputationRegistry (not players)
- Triggered when both signatures received
- Posts for winner (+Elo) and loser (-Elo) separately
- Uses standard K=32 formula

**Security model for arena wallet:**
- Key loaded from `ARENA_PRIVATE_KEY` env var (not in config file)
- Arena can only post reputation signals — cannot forge match results (those need player sigs)
- Cannot access player funds
- Keep minimal MON in arena wallet (just enough for gas)

**3. agent_id tracking** ✅
- CLI passes `agent_id` in queue join request
- Matches track `p1_agent_id`, `p2_agent_id`
- Arena maps winner/loser wallets to agent_ids for Elo posting

### New files

- `nojohns/reputation.py` — Elo calculation and ReputationRegistry interaction
  - `get_current_elo(agent_id, rpc, registry)` → EloState
  - `post_elo_update(agent_id, elo, peak, record, account, ...)` → tx_hash
  - `calculate_new_elo(our_elo, opponent_elo, won)` → new_elo

### Environment variables for arena

```bash
# Required for Elo posting (optional — arena works without these)
export ARENA_PRIVATE_KEY="0x..."  # funded arena wallet
export MONAD_RPC_URL="https://testnet-rpc.monad.xyz"
export MONAD_CHAIN_ID="10143"
export REPUTATION_REGISTRY="0x8004B663056A597Dffe9eCcC1965A193B7388713"
```

### What this means for website

Once we run matches with the new arena config, Elo signals will appear in ReputationRegistry. Website can then:

1. **Read from ReputationRegistry** instead of computing locally
   - Filter by tag1="elo", tag2="melee"
   - Take the latest value per agent
   - Authoritative source: arena-attested ratings

2. **Show agent identity from IdentityRegistry**
   - `tokenURI(agentId)` → registration JSON
   - Display agent name, description (instead of truncated wallet address)

3. **Data source transition:**
   - Current: compute Elo from MatchProof events (local)
   - Future: read Elo from ReputationRegistry (onchain, arena-attested)

### Not yet done

- **Pre-match scouting UI** — query opponent Elo before accepting match
- **Full e2e test with arena Elo posting** — need to fund arena wallet and test

### ERC-8004 quirks discovered

- `giveFeedback()` cannot be called by the agent owner (self-feedback not allowed)
- This is why arena posts, not players
- Makes sense: you shouldn't rate yourself

### Commits

| Commit | Description |
|--------|-------------|
| `5eb2800` | Add nojohns setup identity command |
| `923157a` | Add ERC-8004 ReputationRegistry integration for Elo tracking |

---

## M4 Continued: Autonomous Agent Toolkit (updated day 7)

### What's New

A full agent decision-making toolkit in `agents/` (new top-level package, parallel to `fighters/`, `games/`, `arena/`). This is the "agent autonomy" layer the Gaming Arena Agent bounty requires.

**Key insight:** This is a *toolkit*, not a monolith. We provide building blocks and a strategy protocol that agent builders compose themselves. `KellyStrategy` is a reference implementation. `nojohns auto` is a reference agent that composes the tools.

### New: `agents/` Package

```
agents/
├── __init__.py          # Re-exports everything
├── bankroll.py          # Balance queries + Kelly criterion math
├── scouting.py          # Opponent lookup utilities
├── strategy.py          # WagerStrategy protocol + KellyStrategy reference
└── moltbook.py          # Optional Moltbook posting
```

#### `agents/bankroll.py`

Pure utility functions for financial state:

```python
from agents.bankroll import get_bankroll_state, kelly_wager, win_probability_from_elo

# Query balance + active wager exposure
state = get_bankroll_state(address, rpc_url, wager_contract)
# → BankrollState(balance_mon=2.5, active_exposure_wei=..., available_mon=2.3)

# Elo → win probability (standard formula)
p = win_probability_from_elo(1600, 1400)  # → 0.76

# Kelly criterion wager sizing
amount_wei = kelly_wager(win_prob=0.65, bankroll_wei=10**18, multiplier=1.0, max_pct=0.10)
```

#### `agents/scouting.py`

Opponent lookup — wraps `reputation.get_current_elo()`:

```python
from agents.scouting import scout_opponent, scout_by_wallet

report = scout_opponent(agent_id=12, rpc_url=rpc, reputation_registry=reg)
# → ScoutReport(elo=1540, peak_elo=1580, record="10-5", is_unknown=False)

# By wallet (returns unknown until reverse lookup is implemented)
report = scout_by_wallet("0x...", rpc_url, registry)
```

#### `agents/strategy.py`

`WagerStrategy` protocol (like the `Fighter` protocol) plus `KellyStrategy` reference:

```python
from agents.strategy import KellyStrategy, MatchContext, SessionStats, WagerDecision

strategy = KellyStrategy(risk_profile="moderate", tilt_threshold=3)
# risk profiles: conservative (0.5x Kelly, 5% cap), moderate (1.0x, 10%), aggressive (1.5x, 25%)

context = MatchContext(
    our_elo=1540,
    opponent=scout_report,
    bankroll_wei=available_wei,
    session_stats=session,  # tracks wins, losses, consecutive losses, P&L
)

decision = strategy.decide(context)
# → WagerDecision(should_wager=True, amount_wei=..., amount_mon=0.15,
#     reasoning="Elo 1540 vs 1400, P(win)=0.68, moderate profile, wagering 0.15 MON")
```

**Strategy behaviors:**
- Unknown opponent → no wager (play for experience)
- Tilt protection: `consecutive_losses >= threshold` → no wager
- Even or unfavorable matchup → no wager (no edge)
- Kelly criterion with risk profile caps

**Custom strategies:** Implement `decide(context: MatchContext) -> WagerDecision` — that's it. See `skill/references/wager-strategy.md` for examples (flat bet, percentage, confidence-based).

### New: `nojohns auto` CLI Command

Reference autonomous agent that composes the toolkit:

```bash
nojohns auto phillip --risk moderate --max-matches 10
nojohns auto phillip --risk aggressive --cooldown 60 --min-bankroll 0.5
nojohns auto phillip --no-wager  # play without wagering
```

**Loop per match:**
1. Check bankroll (stop if below `--min-bankroll`)
2. Join queue, get matched
3. Scout opponent (Elo lookup)
4. Strategy decides wager amount (with reasoning)
5. Play match (existing netplay code)
6. Report + sign + settle (existing code)
7. Update session stats, cooldown, repeat

**Session summary at end:**
```
  Session Summary
  Matches: 10
  Record: 7-3 (70%)
  Unique opponents: 4
  Total wagered: 0.8500 MON
  Net P&L: +0.3200 MON
```

**Config from `[moltbot]` section (new):**
```toml
[moltbot]
risk_profile = "moderate"
cooldown_seconds = 30
min_bankroll = 0.01
tilt_threshold = 3
```

### New: OpenClaw Skill Updates

`skill/SKILL.md` rewritten with real capabilities. New shell scripts for Moltbot integration:

- `skill/scripts/bankroll.sh --status` — JSON bankroll snapshot
- `skill/scripts/bankroll.sh --kelly --opponent-elo 1400` — wager recommendation
- `skill/scripts/scout.sh --agent-id 12` — opponent lookup
- `skill/scripts/matchmake.sh --fighter phillip` — matchmake wrapper
- `skill/references/wager-strategy.md` — documentation on writing custom strategies

### Refactored: `cmd_matchmake`

The matchmake command was refactored to extract reusable pieces:

- `_run_single_match()` — core match loop (queue → play → report → sign → settle)
- `_make_arena_helpers()` — HTTP helper factory
- `_print_match_summary()` — summary box printer
- `MatchOutcome` dataclass — structured return value

Both `cmd_matchmake` and `cmd_auto` call `_run_single_match()`. Same behavior as before, less code.

### Tests

44 new tests (all pure math, no RPC):
- `tests/test_bankroll.py` — win probability symmetry, Kelly fraction edge cases, cap enforcement, floor
- `tests/test_strategy.py` — unknown opponent handling, tilt protection, edge detection, risk profiles, reasoning strings, SessionStats

**166 total tests passing** (zero regressions).

### What This Means for Website

The `agents/` package is Python-side only — no direct website changes needed. But for the hackathon pitch:

1. **Strategy reasoning is visible** in session output — great for demo recordings
2. **Session stats** (matches, W-L, P&L) could be shown on website if we add a session endpoint to arena
3. **Risk profiles** are a nice talking point: "agents choose their own risk tolerance"

### Files Changed

| File | Change |
|------|--------|
| `agents/__init__.py` | NEW — re-exports |
| `agents/bankroll.py` | NEW — balance + Kelly math |
| `agents/scouting.py` | NEW — opponent lookup |
| `agents/strategy.py` | NEW — protocol + KellyStrategy |
| `agents/moltbook.py` | NEW — optional Moltbook posting |
| `nojohns/cli.py` | MODIFIED — refactored matchmake, added `nojohns auto` |
| `nojohns/config.py` | MODIFIED — added `MoltbotConfig`, `[moltbot]` parsing |
| `pyproject.toml` | MODIFIED — added `agents*` to package discovery |
| `skill/SKILL.md` | REWRITTEN — real capabilities |
| `skill/scripts/*.sh` | NEW — Moltbot shell wrappers |
| `skill/references/wager-strategy.md` | NEW — custom strategy guide |
| `tests/test_bankroll.py` | NEW — 23 tests |
| `tests/test_strategy.py` | NEW — 21 tests |

---

## Playtest Infrastructure (updated day 8)

### Arena is going public

The arena is deploying to Railway so external playtesters can connect. The public URL will be:

```
https://nojohns-arena-production.up.railway.app
```

New users who don't configure `[arena] server` in their config will connect to this URL by default.

### Your machine as auto-opponent

We need your machine running an always-on opponent so there's someone to fight when external testers connect. Here's how:

**1. Update your arena config:**

Either remove the `[arena] server` line from `~/.nojohns/config.toml` (defaults to the public Railway arena), or set it explicitly:

```toml
[arena]
server = "https://nojohns-arena-production.up.railway.app"
```

**2. Run the auto-opponent in tmux:**

```bash
tmux new -s opponent './scripts/auto-opponent.sh phillip'
```

Or directly:

```bash
nojohns auto phillip --no-wager --cooldown 15
```

This joins the public queue, waits for an opponent, plays a match with Phillip, and re-queues. `--no-wager` means it plays for experience only (no MON at stake). `--cooldown 15` gives 15 seconds between matches.

Detach tmux with `Ctrl-B D`. Reattach with `tmux attach -t opponent`.

**3. Verify it's working:**

```bash
curl https://nojohns-arena-production.up.railway.app/health
```

You should see `queue_size: 1` (your auto-opponent waiting).

### SRE monitoring

An SRE agent in `nojohns-community` monitors the arena health. It writes `status/arena.json` which other fleet agents can read. You don't need to do anything for this — it runs on its own.

### Website update needed

The website currently connects to a hardcoded arena URL. It should point at the Railway URL for:
- Live match viewer WebSocket: `wss://nojohns-arena-production.up.railway.app/ws/match/{id}`
- Health check for hero section: `https://nojohns-arena-production.up.railway.app/health`

Check `web/` for where the arena URL is configured and update it.

---

## Live Stream Optimization (day 8)

The live viewer was choppy when streaming from Railway due to HTTP latency. We're fixing it in two parts:

### Done (ScavieFae): Client-side frame buffer

Added to `web/src/hooks/useLiveMatch.ts`:
- Buffers 8 frames (~133ms at 60fps) before starting playback
- Plays back at steady 60fps regardless of network jitter
- Adds `bufferHealth` to state for UI feedback

This is deployed to production.

### TODO (Scav): Batch frame uploads

**Problem**: Each frame is a separate HTTP POST to Railway. With ~50-100ms latency per request, frames arrive unevenly.

**Solution**: Batch 4 frames per request (4x fewer HTTP requests).

#### `games/melee/netplay.py`

In `_stream_loop` (around line 177), batch frames before sending:

```python
BATCH_SIZE = 4  # Send 4 frames per request (~66ms worth at 60fps)

def _stream_loop(self):
    """Background thread that sends queued frames."""
    logger.info(f"Stream loop started for {self.match_id}")
    frames_sent = 0
    while not self._stop.is_set():
        batch = []
        with self._lock:
            while self._queue and len(batch) < BATCH_SIZE:
                batch.append(self._queue.pop(0))

        if batch and self._client:
            url = f"{self.arena_url}/matches/{self.match_id}/stream/frame"
            try:
                resp = self._client.post(url, json={"frames": batch})
                frames_sent += len(batch)
                if frames_sent <= 12 or frames_sent % 100 == 0:
                    logger.info(f"Stream batch of {len(batch)} frames sent ({resp.status_code})")
            except Exception as e:
                logger.warning(f"Stream batch failed: {e}")

        # Wait proportional to batch size
        time.sleep(0.016 * BATCH_SIZE)
    logger.info(f"Stream loop ended for {self.match_id} ({frames_sent} frames sent)")
```

#### `arena/server.py`

Update `/matches/{id}/stream/frame` to accept both single frame and batch:

```python
@app.post("/matches/{match_id}/stream/frame")
async def stream_frame(match_id: str, req: Request) -> dict[str, Any]:
    """Stream frame(s) — accepts single frame or batch."""
    data = await req.json()

    # Handle batch or single frame
    if "frames" in data:
        frames = data["frames"]
    else:
        frames = [data]  # Wrap single frame in list

    # Broadcast each frame to WebSocket clients
    for frame_data in frames:
        message = {
            "type": "frame",
            "frame": frame_data.get("frame"),
            "players": frame_data.get("players", []),
        }
        await _manager.broadcast(match_id, message)

    return {"success": True, "frames_received": len(frames)}
```

#### Impact

| Approach | Requests/sec | Latency impact |
|----------|--------------|----------------|
| Current (1 frame/req) | 60 | Each request adds ~50-100ms jitter |
| Batched (4 frames/req) | 15 | 4x fewer requests, smoother delivery |

### NOW: WebSocket Upload (priority fix for jitter)

The HTTP batching approach is fundamentally causing bursty frame delivery. Even with client-side buffering, the viewer experiences jitter. We need WebSocket for continuous frame streaming.

**Current (bursty):**
```
Client ──HTTP POST (16 frames)──► Arena ──WebSocket──► Viewer
        [4 req/sec, ~250ms gaps]        [bursts of 16]
```

**Target (smooth):**
```
Client ◄────────WebSocket────────► Arena ──WebSocket──► Viewer
        [continuous 60fps stream]         [steady 60fps]
```

#### Changes needed in `arena/server.py`

Add a new WebSocket endpoint for frame upload:

```python
@app.websocket("/ws/stream/{match_id}")
async def stream_upload(websocket: WebSocket, match_id: str):
    """WebSocket endpoint for frame upload from game client."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "frame":
                # Broadcast to viewers immediately (no batching)
                message = {
                    "type": "frame",
                    "frame": data.get("frame"),
                    "players": data.get("players", []),
                }
                await _manager.broadcast(match_id, message)

            elif data.get("type") == "game_end":
                await _manager.broadcast(match_id, data)

            elif data.get("type") == "match_end":
                await _manager.broadcast(match_id, data)
                break

    except WebSocketDisconnect:
        pass
```

#### Changes needed in `games/melee/netplay.py`

Replace HTTP frame posting with WebSocket:

```python
import websockets
import asyncio

class FrameStreamer:
    def __init__(self, arena_url: str, match_id: str):
        self.ws_url = arena_url.replace("http", "ws") + f"/ws/stream/{match_id}"
        self.ws = None
        self._queue = asyncio.Queue()

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url)

    async def send_frame(self, frame_data: dict):
        if self.ws:
            await self.ws.send(json.dumps({"type": "frame", **frame_data}))

    async def close(self):
        if self.ws:
            await self.ws.close()
```

In `_game_loop`, instead of queuing frames for HTTP batch:

```python
# Old: self._frame_queue.append(frame_data)
# New: await self._streamer.send_frame(frame_data)
```

#### Why this fixes jitter

| Aspect | HTTP Batch | WebSocket |
|--------|------------|-----------|
| Delivery pattern | 16 frames every ~250ms | 1 frame every ~16ms |
| Network overhead | New connection per batch | Single persistent connection |
| Latency variance | High (HTTP round-trip) | Low (streaming) |
| Viewer experience | Choppy, requires large buffer | Smooth, minimal buffer needed |

Once WebSocket upload is working, we can reduce the viewer buffer from 24 frames back to 8.

---

## Live Viewer Updates (day 9)

### Stage Detection — DONE (issue #14 closed)

The live viewer now renders actual stage platforms instead of a hardcoded single platform.

**New file: `web/src/lib/stageData.ts`**
- Platform positions for 6 legal stages: Battlefield, FD, Dreamland, Yoshi's Story, FoD, Pokemon Stadium
- Hardcoded as SVG rects — no external assets needed
- Stage name shown in HUD (top right)

**How it works:**
1. `match_start` message includes `stageId` (libmelee internal ID)
2. `MeleeViewer` looks up stage data by ID
3. Renders main platform + side/top platforms as colored rectangles

**Stage IDs (libmelee):**
| Stage | ID |
|-------|-----|
| Fountain of Dreams | 2 |
| Pokemon Stadium | 3 |
| Yoshi's Story | 8 |
| Dreamland | 28 |
| Battlefield | 31 |
| Final Destination | 32 |

Unknown stages fall back to a generic platform.

### Frame Buffer Fix

Fixed a bug where `startPlayback()` was called inside a `setState` callback, causing nested state updates. Moved outside to avoid React batching issues.

Also added `startPlayback` to `handleMessage`'s dependency array to fix potential stale closure.

### Debug Logging

Added console logging to help diagnose stream issues:
- `[useLiveMatch] Connected to wss://...`
- `[useLiveMatch] match_start received: {...}`
- `[useLiveMatch] Frame X, buffer: Y, buffering: true/false`
- `[useLiveMatch] Starting playback, buffer has 8 frames`

Check browser console if the viewer isn't working.

### Commits

| Commit | Description |
|--------|-------------|
| `934e9c8` | Add stage detection and rendering to live viewer |
| `c4f5d72` | Fix live viewer frame buffer and add debug logging |

---

## WebSocket Upload — DONE (day 8, Scav)

The server-side WebSocket upload endpoint is implemented and tested. The arena now accepts
frame data over a persistent WebSocket connection in addition to the legacy HTTP POST endpoints.

### What changed

**`arena/server.py`** — new endpoint:
```
WS /ws/stream/{match_id}
```
Accepts JSON messages with `type` field: `match_start`, `frame`, `game_end`, `match_end`.
Each message is broadcast to viewers immediately (no batching). Connection closes after `match_end`.

**`games/melee/netplay.py`** — `MatchStreamer` refactored:
- Tries WebSocket first (`ws://arena/ws/stream/{match_id}`)
- Falls back to HTTP batching if WebSocket fails
- Sends frames individually over WebSocket (1 frame per message, continuous 60fps)
- No change to the public API — `send_frame()`, `send_match_start()`, etc. work the same

**`pyproject.toml`** — added `websockets>=12.0` to `[arena]` extras (already installed via uvicorn[standard]).

### What you can do now

With WebSocket upload live, viewer jitter should improve significantly. You can:
1. **Reduce `BUFFER_TARGET`** from 24 to 8 frames — WebSocket delivers continuously, less buffer needed
2. **Reduce `MAX_BUFFER_SIZE`** from 60 to 30 — overflow is less likely with steady delivery
3. The `requestAnimationFrame` playback loop you wrote is perfect for this — no changes needed there

### Testing

Verified locally:
- WebSocket connects, sends 10 frames at ~60fps, viewer receives all 10 individually
- match_start + frames + match_end all go through the same WebSocket
- HTTP fallback still works if WebSocket is unavailable
- 166 tests pass

### Legacy HTTP endpoints still work

The HTTP POST endpoints (`/stream/start`, `/stream/frame`, `/stream/frames`, etc.) are unchanged.
Old clients can still use them. The WebSocket path is preferred but not required.

---

## Prediction Markets (day 9, ScavieFae)

### What's New

Parimutuel prediction pools for live matches. Spectators (humans and agents) bet on match outcomes. Proportional payout from losing side to winning side.

**GitHub issue**: #18

### Contract: `PredictionPool.sol`

**Files:**
- `contracts/src/PredictionPool.sol` — parimutuel prediction pool
- `contracts/test/PredictionPool.t.sol` — 45 tests (all passing)
- `contracts/script/DeployPrediction.s.sol` — deploy script

**Interface:**
```solidity
// Arena creates pools when matches queue
function createPool(bytes32 matchId, address playerA, address playerB) external returns (uint256 poolId);

// Anyone bets on a player (payable with MON)
function bet(uint256 poolId, bool betOnA) external payable;

// Anyone resolves once MatchProof has the result
function resolve(uint256 poolId) external;

// Winners claim proportional payout
function claim(uint256 poolId) external;

// Arena cancels if match doesn't happen
function cancelPool(uint256 poolId) external;

// Anyone cancels after 2h timeout
function cancelStalePool(uint256 poolId) external;

// Bettors reclaim from cancelled pools
function claimRefund(uint256 poolId) external;

// View functions
function getPool(uint256 poolId) external view returns (Pool memory);
function getUserBets(uint256 poolId, address user) external view returns (uint256 onA, uint256 onB);
function getClaimable(uint256 poolId, address user) external view returns (uint256);
function poolCount() external view returns (uint256);
```

**Constructor:**
```solidity
constructor(address _matchProof, address _arena, uint256 _feeBps)
// _feeBps: fee in basis points (0 = no fee, 100 = 1%, max 1000 = 10%)
// Start at 0% — add revenue later without redeploying
```

**Design:**
- `arena` is the only address that can call `createPool` and `cancelPool` — prevents spam
- Betting stays open while match is live (parimutuel odds shift as bets come in)
- Resolves via `MatchProof.recorded()` + `getMatch()` — same trust model as Wager.sol
- Fee is configurable at deploy time, deducted from total pool on resolution
- 2-hour timeout: anyone can cancel a stale pool after timeout
- Proportional payout: `payout = (distributable * userBet) / winningSideTotal`

**Edge cases tested:**
- One-sided pool (only bets on A, A wins) → everyone gets their money back
- One-sided pool (only bets on B, A wins) → no one can claim
- Empty pool resolve → valid, no payouts
- Bet on both sides → only winning side bet counts for claim
- Multiple bettors, proportional math verified
- Fee at 0% and 5%

### Deploy

```bash
export PRIVATE_KEY=<deployer-key>
export MONAD_TESTNET_RPC_URL=https://testnet-rpc.monad.xyz
export ARENA_ADDRESS=<arena-wallet-address>  # defaults to deployer
export FEE_BPS=0  # defaults to 0

cd contracts && forge script script/DeployPrediction.s.sol \
  --rpc-url $MONAD_TESTNET_RPC_URL --broadcast \
  --sig "run(address)" 0x1CC748475F1F666017771FB49131708446B9f3DF
```

After deploy, update `web/src/config.ts` with the PredictionPool address.

### Arena Integration (Scav's part)

The arena needs to:

1. **Create pool when match starts:**
   ```python
   pool_id = prediction_pool.createPool(match_id_bytes32, player_a_addr, player_b_addr)
   ```

2. **Expose pool info:**
   ```
   GET /matches/{id}/pool → { poolId, totalA, totalB, status }
   ```

3. **Resolve after match result is recorded:**
   ```python
   prediction_pool.resolve(pool_id)
   ```

4. **Cancel if match doesn't happen:**
   ```python
   prediction_pool.cancelPool(pool_id)
   ```

The arena already has `ARENA_PRIVATE_KEY` for Elo posting — same key signs pool creation/resolution.

**ABI:** Available in `web/src/abi/predictionPool.ts` or from `contracts/out/PredictionPool.sol/PredictionPool.json`.

### Website Widget

Added `PredictionWidget` sidebar to the live match viewer page:

**New files:**
- `web/src/components/prediction/PredictionWidget.tsx` — main widget
- `web/src/components/prediction/OddsBar.tsx` — visual odds bar (green/red)
- `web/src/components/prediction/BetForm.tsx` — amount input + bet buttons
- `web/src/hooks/usePredictionPool.ts` — read pool state from contract
- `web/src/hooks/usePlaceBet.ts` — write: place bet transaction
- `web/src/hooks/useClaimPayout.ts` — write: claim payout transaction
- `web/src/hooks/useWallet.ts` — minimal wallet connection (no wagmi)
- `web/src/lib/wallet.ts` — viem wallet client helper
- `web/src/abi/predictionPool.ts` — contract ABI

**Modified:**
- `web/src/config.ts` — added `predictionPool` to CONTRACTS
- `web/src/pages/LivePage.tsx` — two-column layout (viewer + prediction sidebar)

**Features:**
- Shows current odds as green/red percentage bar
- Shows total pool size in MON
- "Connect Wallet" button for users without wallet
- Bet input with quick amount buttons (0.01, 0.05, 0.1, 0.5 MON)
- User's position: amount bet, potential payout at current odds
- Claim button after resolution (or refund button if cancelled)
- Polls every 5s for odds updates
- Shows "Coming soon" placeholder until contract is deployed (address is zero)
- Responsive: sidebar on desktop, stacked on mobile

**Wallet connection:** Uses viem `createWalletClient` + `window.ethereum` (MetaMask/injected). No wagmi dependency added — keeps the bundle minimal.

### Agent Prediction (Scav's part, future)

Once the arena creates pools, idle agents can bet on other agents' matches:

1. Query arena for active pools
2. Look up both players' Elo from ReputationRegistry
3. Compute implied probability from Elo differential
4. Compare to pool odds — if edge exists, bet (Kelly criterion)
5. Claim payouts after resolution

This extends the existing `agents/bankroll.py` (has `win_probability_from_elo()` and `kelly_fraction()`) and `agents/strategy.py` (has `WagerStrategy` protocol).

---

## Day 11: Mainnet Migration (updated Feb 11, 2026)

4 days until deadline. We're flipping everything from testnet to mainnet. Python defaults already updated.

### CRITICAL — Website Mainnet Config

Once Scav deploys contracts to mainnet, update `web/src/config.ts`:

```typescript
// Chain
export const CHAIN_ID = 143;
export const RPC_URL = "https://rpc.monad.xyz";
export const BLOCK_EXPLORER_URL = "https://monadexplorer.com"; // verify this URL

// Contracts — fill in after Scav deploys
export const CONTRACTS = {
  matchProof: "<mainnet MatchProof address>",
  wager: "<mainnet Wager address>",
  predictionPool: "<mainnet PredictionPool address>",
};

// ERC-8004 registries (already on mainnet)
export const ERC8004 = {
  identityRegistry: "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432",
  reputationRegistry: "0x8004BAa17C55a88189AE136b182e5fdA19dE9b63",
};

// IMPORTANT: update DEPLOY_BLOCK to mainnet deploy block (reduces getLogs scan range)
export const DEPLOY_BLOCK = ???;  // fill in after deploy
```

Scav will update `contracts/deployments.json` with mainnet addresses once deployed. Read from there.

### CRITICAL — Vercel SPA Routing Fix (issue #27)

Direct-linking to `/leaderboard`, `/live`, `/compete` returns 404 on Vercel because it's an SPA.

**Fix:** Create `web/vercel.json`:
```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

Then redeploy. One-line fix.

### Claim Hooks Needed

`useClaimPayout.ts` and `useClaimRefund.ts` were referenced in the prediction widget but may not be fully wired. Bettors need these to collect winnings:

- `useClaimPayout(poolId)` → calls `PredictionPool.claim(poolId)`
- `useClaimRefund(poolId)` → calls `PredictionPool.claimRefund(poolId)` (for cancelled pools)

Pattern: same as `usePlaceBet.ts` but calls different contract function.

### Nice to Have

- Show "No active pool" state in PredictionWidget instead of hiding entirely when pool doesn't exist
- Replace O(n) pool lookup with event log filtering by indexed `matchId` (flagged in PR #23 review)
- Consider showing mainnet vs testnet indicator in UI header

### Python-Side Changes Already Made

- `nojohns/config.py`: Default `chain_id=143`, `rpc_url=https://rpc.monad.xyz`
- `nojohns/contract.py`: Default RPC changed to mainnet
- `arena/server.py`: All env var fallbacks now use mainnet defaults (chain 143, mainnet RPC, mainnet ReputationRegistry)
- `_setup_monad()`: Now defaults to mainnet in the interactive prompt
- `contracts/deployments.json`: Added `monad_mainnet` placeholder section + `prediction_pool` to testnet section
- Arena startup: Logs full config on boot (wallet, pool, registry — warns if misconfigured)
- Arena: `_try_create_pool()` now logs warnings instead of silently returning
- Arena: `_try_resolve_pool()` added — resolves pools server-side when both signatures received

### Railway Env Vars (Scav will update)

```bash
MONAD_RPC_URL=https://rpc.monad.xyz
MONAD_CHAIN_ID=143
REPUTATION_REGISTRY=0x8004BAa17C55a88189AE136b182e5fdA19dE9b63
PREDICTION_POOL=<mainnet address after deploy>
ARENA_PRIVATE_KEY=<same key, funded on mainnet>
```

### Deploy Sequence

1. **ScavieFae deploys contracts to mainnet** (you own `contracts/`)
2. ScavieFae updates `contracts/deployments.json` with mainnet addresses
3. Scav updates Railway env vars + Python defaults with new addresses
4. ScavieFae updates `web/src/config.ts` with mainnet addresses
5. ScavieFae adds `web/vercel.json` for SPA routing
6. ScavieFae redeploys to Vercel
7. Both test: match → pool created → bets → pool resolved → website shows it

### Deploy Commands (mainnet)

**Step 1: MatchProof + Wager**
```bash
export PRIVATE_KEY=<deployer-key-for-0xF04366a3...0420>
export MONAD_MAINNET_RPC_URL=https://rpc.monad.xyz

cd contracts && forge script script/Deploy.s.sol \
  --rpc-url $MONAD_MAINNET_RPC_URL --broadcast
```

**Step 2: PredictionPool**

IMPORTANT: Set `ARENA_ADDRESS` to the arena wallet, NOT the deployer. The arena wallet is what signs pool creation/resolution txs at runtime.

```bash
export ARENA_ADDRESS=0xd8A01dE3D27BD6Fb0df99E7ca94eC69c5dAbe38C
export FEE_BPS=0

cd contracts && forge script script/DeployPrediction.s.sol \
  --rpc-url $MONAD_MAINNET_RPC_URL --broadcast \
  --sig "run(address)" <MAINNET_MATCH_PROOF_ADDRESS>
```

If `ARENA_ADDRESS` is not set, it defaults to the deployer — and the arena server won't be able to create pools.

### Wallet Funding Status

Both wallets are funded on mainnet (~271 MON each):
- Deployer (`0xF04366a3...0420`): ready to deploy
- Arena (`0xd8A01dE3...38C`): ready for runtime gas

### Mainnet Deploy — DONE (ScavieFae, Feb 11)

All three contracts deployed to **Monad mainnet (chain 143)**:

| Contract | Mainnet Address | Notes |
|----------|----------------|-------|
| **MatchProof** | `0x1CC748475F1F666017771FB49131708446B9f3DF` | Same as testnet (same deployer nonce) |
| **Wager** | `0x8d4D9FD03242410261b2F9C0e66fE2131AE0459d` | Same as testnet |
| **PredictionPool** | `0x33E65E300575D11a42a579B2675A63cb4374598D` | Different from testnet |

**Website updated:**
- `web/src/config.ts` → chain 143, mainnet RPC, all three mainnet addresses, DEPLOY_BLOCK=54717354
- `web/src/viem.ts` → renamed `monadTestnet` to `monad`, uses mainnet explorer
- `web/src/lib/wallet.ts` → uses mainnet chain definition
- `web/vercel.json` → SPA rewrite (fixes #27 — direct link 404s)
- Snapshot regenerated for mainnet (0 events — fresh deploy)
- `contracts/deployments.json` → mainnet section filled in

**What Scav needs to do:**
1. Update Railway env vars:
   - `PREDICTION_POOL=0x33E65E300575D11a42a579B2675A63cb4374598D`
   - `MONAD_CHAIN_ID=143` (if not already set)
   - `MONAD_RPC_URL=https://rpc.monad.xyz` (if not already set)
2. Verify `nojohns/contract.py` defaults match mainnet addresses
3. Run a test match → verify it records on mainnet MatchProof
4. Verify prediction pool creation fires during match

**Snapshot script now defaults to mainnet:**
```bash
node web/scripts/generate-snapshot.mjs           # mainnet (default)
node web/scripts/generate-snapshot.mjs --testnet  # testnet
```

---

## Branch Strategy Change (Feb 11, 2026)

**We now have two long-lived branches.** Switch your default working branch to `dev`.

| Branch | Purpose | Audience |
|--------|---------|----------|
| **`main`** | Public-facing releases. Clean, no internal docs. | Users, judges |
| **`dev`** | All agent work. Handoff docs, hooks, dev artifacts. | Us three + Mattie |

### What changed

- **Handoff docs (`docs/HANDOFF-*.md`)** removed from `main`, live only on `dev`
- **`.claude/` hooks and settings** removed from `main`, live only on `dev`
- **`CODEOWNERS`** added on `dev` — Mattie must approve changes to `.claude/`, handoff docs, CLAUDE.md files, and contract source. This protects against prompt injection via external PRs.

### Your workflow now

1. **Pull from `dev`** (not `main`):
   ```bash
   git checkout dev
   git pull origin dev
   ```

2. **Branch off `dev`** for new work:
   ```bash
   git checkout -b scaviefae/mainnet-deploy dev
   ```

3. **PR into `dev`** (not `main`)

4. **Releases to `main`** are curated via `scripts/release-to-main.sh` — run from `dev`, it squash-merges and strips internal files. Mattie or Scav will run this.

### Why

- `main` is the public repo. Judges, users, and potential contributors see it. Handoff docs and agent hooks don't belong there.
- Handoff docs are a prompt injection surface — if someone PRs a modified handoff to a public branch, an agent could execute it. CODEOWNERS + dev-only mitigates this.

### Post-pull hook

A hook now fires after `git pull` and checks if handoff docs changed. If they did, you'll see the diff and a reminder to review + add any items you need to communicate back (planning changes, questions, blockers — not status updates).

---

## Action: Switch Railway to Deploy from `dev`

**Priority: Do this now.** Railway is currently deploying from `main`, but all hackathon work is on `dev`. Until this is switched, arena changes don't go live on push.

1. Open Railway dashboard → nojohns project
2. Click the arena service → Settings → Source (or "Source Repo" section)
3. Change **Branch** from `main` to `dev`
4. Save. Railway will immediately redeploy from `dev`.

After the hackathon we'll switch back to `main`.

**Verify it worked:**
```bash
curl https://nojohns-arena-production.up.railway.app/health
```

Should return `{"status":"ok",...}`. The new `/pools` endpoint should also be live:
```bash
curl https://nojohns-arena-production.up.railway.app/pools
```

See `docs/DEPLOY.md` for the full deploy reference.

---

## Day 11: Onchain Pipeline Fix (Feb 11, 2026)

**Issues:** #31 (match results not recording), #32 (ERC-8004 not active on mainnet)

### What Scav Just Did

**1. Arena now submits `recordMatch()` itself** (commit 3526d05 on `dev`)

Previously: each player's CLI polled for both signatures, then the *player* submitted `recordMatch()` and paid gas. This failed silently because the player wallet had 0 MON on mainnet.

Now: when both EIP-712 signatures arrive at the arena, it calls `recordMatch()` using the arena's own wallet (`ARENA_PRIVATE_KEY`). Players still sign off-chain (free). The arena pays gas and submits.

New env var required: **`MATCH_PROOF`** — the MatchProof contract address.

Execution order when both signatures arrive:
1. `_try_record_match()` — submit to MatchProof (NEW)
2. `_post_elo_updates()` — post Elo to ReputationRegistry (existing)
3. `_try_resolve_pool()` — resolve prediction pool (existing)

**2. Funded player wallet on mainnet**

Sent 5 MON from arena wallet to player wallet `0x1F36B4c388CA19Ddb5b90DcF6E8f8309652Ab3dE`. Not strictly needed anymore (arena pays gas for recording), but useful for wagers and identity registration.

**3. Flipped local config to mainnet**

`~/.nojohns/config.toml` and `~/.nojohns/arena.env` both updated:
- `chain_id = 143`
- `rpc_url = "https://rpc.monad.xyz"`
- All contract addresses → mainnet versions
- All ERC-8004 addresses → mainnet versions

### What ScavieFae Needs To Do

#### 1. Railway env vars (CRITICAL — do this first)

Add/update these env vars in the Railway arena service dashboard:

```
MATCH_PROOF=0x1CC748475F1F666017771FB49131708446B9f3DF
PREDICTION_POOL=0x33E65E300575D11a42a579B2675A63cb4374598D
MONAD_RPC_URL=https://rpc.monad.xyz
MONAD_CHAIN_ID=143
REPUTATION_REGISTRY=0x8004BAa17C55a88189AE136b182e5fdA19dE9b63
```

`ARENA_PRIVATE_KEY` should already be set. If not:
```
ARENA_PRIVATE_KEY=<the key for 0xd8A01dE3D27BD6Fb0df99E7ca94eC69c5dAbe38C>
```

After setting these, Railway will redeploy. Check the deploy logs — you should see:
```
Arena startup config:
  Chain: 143 | RPC: https://rpc.monad.xyz
  Arena wallet: 0xd8A01dE3D27BD6Fb0df99E7ca94eC69c5dAbe38C
  MatchProof: 0x1CC748475F1F666017771FB49131708446B9f3DF
  Prediction pool: 0x33E65E300575D11a42a579B2675A63cb4374598D
  Reputation registry: 0x8004BAa17C55a88189AE136b182e5fdA19dE9b63
```

If any line says "NOT configured", something's missing.

#### 2. Railway branch = `dev` (if not already done)

The arena code with `_try_record_match()` is on `dev`, not `main`. Railway must deploy from `dev`. See the section above ("Switch Railway to Deploy from `dev`") for steps.

Verify after redeploy:
```bash
curl https://nojohns-arena-production.up.railway.app/pools
# Should return {"pools": [...], "count": N} — if 404, Railway is still on main
```

#### 3. Flip your `~/.nojohns/config.toml` to mainnet

Your player's EIP-712 signatures include `chainId` in the domain. If your config still says `chain_id = 10143`, your signatures won't verify against the mainnet contract.

```toml
[chain]
chain_id = 143
rpc_url = "https://rpc.monad.xyz"
match_proof = "0x1CC748475F1F666017771FB49131708446B9f3DF"
wager = "0x8d4D9FD03242410261b2F9C0e66fE2131AE0459d"
prediction_pool = "0x33E65E300575D11a42a579B2675A63cb4374598D"
identity_registry = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"
reputation_registry = "0x8004BAa17C55a88189AE136b182e5fdA19dE9b63"
```

If your player wallet also has 0 MON on mainnet, fund it from the deployer wallet. (Player signing is free now, but wagers and identity registration still need gas.)

#### 4. Verify with a test match

Once Railway is updated, both machines run:
```bash
nojohns auto phillip --no-wager --cooldown 30
```

After a match completes, check:
- Railway logs: should show `Match {id} recorded onchain: tx=...`
- Monad explorer: `https://monadexplorer.com/address/0x1CC748475F1F666017771FB49131708446B9f3DF` should show `MatchRecorded` events
- Website: match should appear in match history

### Current Mainnet Balances

| Wallet | Address | Balance |
|--------|---------|---------|
| Deployer | `0xF04366a3...0420` | 270.60 MON |
| Arena | `0xd8A01dE3...38C` | 265.55 MON |
| Player (Scav) | `0x1F36B4c3...3dE` | 5.00 MON |

### What's Still Not Working (Expected)

- **Elo posting** — ReputationRegistry `name()` reverts on mainnet (issue #32). `giveFeedback()` might also revert. Arena will log a warning and continue — non-blocking.
- **Identity registration** — Neither player is registered on mainnet IdentityRegistry yet. Will do after confirming match recording works.
- **Spectator fleet** — Ready to launch once pools are being created (pools appear automatically when arena has PREDICTION_POOL set and matches are running)
