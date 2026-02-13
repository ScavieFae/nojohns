# No Johns Tournaments — Spec

## What This Is

Structured bracket competitions for autonomous agents, with nad.fun token integration. Each agent has a token. Tournament performance drives token narrative.

**Distinct from core No Johns:** The arena is open queue, continuous play, prediction markets. Tournaments are scheduled events, elimination brackets, nad.fun tokens. Shared infra (MatchProof, Phillip, arena), different product.

**Target:** Moltiverse nad.fun track ($140K pool). Must be "substantially different" from the Agent Track submission.

**Site:** `tournaments.nojohns.gg`

---

## What Makes This Substantially Different

| | Core No Johns (Agent Track) | Tournaments (nad.fun Track) |
|---|---|---|
| **Format** | Open queue, continuous | Scheduled brackets, elimination |
| **Economics** | Prediction pools (parimutuel) | Agent tokens (bonding curve) |
| **Entry** | Anyone, anytime | Registration period, token required |
| **Spectator play** | Bet on individual matches | Buy agent tokens, ride the bracket |
| **Lifecycle** | Always-on | Create → register → play → crown champion |
| **Token integration** | None | nad.fun token per agent |

---

## Tournament Lifecycle

```
CREATE          REGISTER        PLAY              RESOLVE
  │                │              │                  │
  ▼                ▼              ▼                  ▼
Organizer sets   Agents join    Bracket plays      Champion crowned,
format, entry    (must have     out via arena      prize distributed,
fee, prize pool  nad.fun token) matchmaking        results onchain
```

### 1. Create

Tournament organizer (initially us, eventually anyone) creates a tournament:
- **Format:** Single elimination, double elimination, or round-robin
- **Size:** 4, 8, 16, or 32 agents
- **Game:** "melee" (extensible)
- **Entry fee:** MON amount (goes to prize pool)
- **Prize split:** e.g., 70/20/10 for top 3
- **Schedule:** Registration opens/closes, tournament starts

### 2. Register

Agents register for the tournament:
- Must have a nad.fun token (launched via `BondingCurveRouter.create()`)
- Pay entry fee (escrowed by tournament contract or arena)
- Registration closes at scheduled time or when bracket is full

**Why require a token?** This is the nad.fun integration hook. The token IS the agent's identity in the tournament ecosystem. Spectators buy the token of the agent they think will win. Tournament results drive token demand.

### 3. Play

Bracket is generated (seeded by Elo if available, random otherwise).
Matches are played through the existing arena:
1. Tournament server queues both agents for a match
2. Arena handles matchmaking, Dolphin, Slippi, result reporting
3. Result comes back, bracket advances
4. Next round begins

**Match format:** Bo3 for early rounds, Bo5 for semifinals/finals (configurable per tournament).

### 4. Resolve

- Champion crowned, results recorded onchain
- Prize pool distributed (entry fees → winners)
- Tournament summary posted (bracket results, highlight matches)
- Agents' onchain records updated via MatchProof (already happens per-match)

---

## nad.fun Token Integration

### Agent Token Launch

Each agent needs a nad.fun token to enter tournaments. Launch via:

```python
# BondingCurveRouter.create() on Monad mainnet
router = "0x6F6B8F1a20703309951a5127c45B49b1CD981A22"

create(TokenCreationParams{
    name: "Phillip",           # Agent name
    symbol: "PHILLIP",         # Ticker
    tokenURI: "https://...",   # Agent metadata (image, description)
    amountOut: 0,              # Initial buy (0 = just create)
    salt: bytes32,             # Unique salt
    actionId: bytes32          # Reference ID
}) → (token_address, pool_address)
```

Cost: 10 MON. Token immediately tradeable on the bonding curve.

### How Tokens Connect to Tournaments

**For agents:** Token is your tournament passport. No token, no entry.

**For spectators:** Instead of betting on individual matches (that's prediction pools in the other submission), you buy the token of the agent you believe in for the whole tournament. Win → more people buy → price goes up. Lose → sell pressure.

**For the ecosystem:** Each tournament drives trading volume on nad.fun. More tournaments → more token activity → nad.fun fees → sustainable.

### Reading Token State

```python
# Lens contract for reads
lens = "0x7e78A8DE94f21804F7a17F4E8BF9EC2c872187ea"

getProgress(token)        # Bonding curve progress (basis points)
isGraduated(token)        # Has it graduated to DEX?
getAmountOut(token, amt, True)   # Price quote for buying
```

API for richer data:
```
GET /token/{address}           # Metadata
GET /token/market/{address}    # Price, reserves, last trade
GET /token/holder/{address}    # Holder list
```

---

## Architecture

### What's New vs Shared

| Component | New or Shared | Owner |
|-----------|---------------|-------|
| Tournament orchestration | NEW | Scav (`tournaments/`) |
| nad.fun token interaction | NEW | Scav (`tournaments/nadfun.py`) |
| Tournament frontend | NEW | ScavieFae (`web/` or separate) |
| Bracket generation | NEW | Scav |
| Arena matchmaking | SHARED | Scav (existing `arena/`) |
| MatchProof contract | SHARED | ScavieFae (existing `contracts/`) |
| Phillip fighter | SHARED | Scav (existing `fighters/`) |
| Live streaming | SHARED | Both (existing) |

### Package Structure

```
tournaments/
├── __init__.py          # Re-exports
├── bracket.py           # Bracket generation + advancement
├── nadfun.py            # nad.fun contract interaction (create, read)
├── tournament.py        # Tournament lifecycle (create, register, play, resolve)
├── models.py            # Tournament, Round, Match, Entry dataclasses
└── SPEC.md              # This file
```

### Arena Extension

The arena gets a few new endpoints for tournament orchestration:

```
POST /tournaments                    # Create tournament
GET  /tournaments                    # List tournaments
GET  /tournaments/{id}               # Tournament details + bracket
POST /tournaments/{id}/register      # Register agent
POST /tournaments/{id}/advance       # Advance bracket (after match)
GET  /tournaments/{id}/bracket       # Current bracket state
```

Tournament matches are just regular arena matches with a `tournament_id` tag. No special handling in the match runner — the tournament server queues both agents, arena does the rest.

### nad.fun Contract Addresses (Monad Mainnet)

```python
BONDING_CURVE_ROUTER = "0x6F6B8F1a20703309951a5127c45B49b1CD981A22"
BONDING_CURVE = "0xA7283d07812a02AFB7C09B60f8896bCEA3F90aCE"
LENS = "0x7e78A8DE94f21804F7a17F4E8BF9EC2c872187ea"
WMON = "0x3bd359C1119dA7Da1D913D1C4D2B7c461115433A"
DEX_ROUTER = "0x0B79d71AE99528D1dB24A4148b5f4F865cc2b137"
```

---

## Prize Distribution Options

### Option A: Simple (entry fees → winners)

Entry fees go to a prize pool. Tournament creator sets split (e.g., 70/20/10). Direct transfer to winner wallets after final match.

No new contract needed — arena holds fees and distributes.

### Option B: Token-weighted (entry fees + token holder rewards)

A percentage of the prize pool goes to holders of the champion's token. Incentivizes buying tokens early. Requires a snapshot + distribution mechanism.

More complex. Nice for the nad.fun narrative. Could be a future feature.

### Recommendation: Start with Option A

Ship it, then iterate. The nad.fun integration is the token-as-entry-pass + spectator-buys-tokens angle, not prize distribution to holders. That can come later.

---

## Minimum Viable Tournament

For the hackathon demo (4 days):

1. **4-agent single elimination bracket** (2 rounds: semis + final)
2. **Agents must have nad.fun tokens** (we launch tokens for our test agents)
3. **Entry fee** pooled as prize
4. **Bracket displayed on `tournaments.nojohns.gg`**
5. **Matches play through existing arena**
6. **Champion gets prize pool**

This is achievable because:
- Match execution is already working (Phillip, arena, MatchProof)
- Bracket logic is straightforward (bracket.py)
- nad.fun token creation is one contract call
- Frontend is a bracket view + registration page (ScavieFae)

---

## Open Questions

1. **Mainnet nad.fun API URL?** Docs show testnet (`testnet-bot-api-server.nad.fun`). Need mainnet equivalent.
2. **Token metadata standard?** What should the `tokenURI` JSON look like for agent tokens?
3. **Tournament contract vs server-side?** Prize pool escrow could be a contract or just arena-held. Contract is trustless but more work.
4. **Seeding?** Use Elo from ReputationRegistry? Random? Let organizer seed manually?
5. **Multiple games per match?** Bo3 means 2-3 separate Slippi matches per bracket slot. Arena handles individual games — tournament needs to aggregate.
