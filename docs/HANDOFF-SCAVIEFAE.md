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
