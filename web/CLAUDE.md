# web/ — CLAUDE.md

ScavieFae owns this directory. Scav should not edit files here.

## What to Build

A public-facing website that makes the system legible and impressive to hackathon judges, potential competitors, and spectators.

### Pages

#### 1. Landing Page (`/`)
- What is No Johns — one-paragraph pitch
- The moltbot/fighter split: LLMs handle strategy, neural nets handle execution
- How it works: matchmake → fight → result onchain → wager settles
- Visual: the architecture diagram from SPEC.md, but pretty
- CTA: "Start competing" → links to setup docs or GitHub

#### 2. Leaderboard (`/leaderboard`)
- Ranked list of agents by Elo
- Data source: ReputationRegistry on Monad (read `giveFeedback` / `readAllFeedback` events, compute Elo)
- Show: rank, agent name, Elo, record (W-L), games played
- Link each agent to their match history

#### 3. Match History (`/matches`)
- List of recent completed matches
- Data source: MatchProof contract events (`MatchRecorded`) + arena REST API
- Show: date, agents, result, game, link to block explorer for onchain proof
- Detail view: game-by-game scores, replay hash

#### 4. How to Compete (`/compete`)
- Clean install flow: clone, setup, fight
- How to register your agent onchain
- How wagering works
- Link to GitHub, docs

### Stretch Goals

- **Live match viewer** — embed Dolphin video stream (Twitch or Livepeer)
- **Prediction interface** — pre-match predictions for spectators
- **Agent profile pages** — `/agent/{address}` with match history, Elo chart, fighters

## Data Sources

### Onchain (Monad)
- **MatchProof contract** — `MatchRecorded` events for match history
- **Wager contract** — `WagerCreated`, `WagerSettled` events for wager activity
- **IdentityRegistry** (ERC-8004) — registered agents, tokenURI for metadata
- **ReputationRegistry** (ERC-8004) — Elo signals

Read these via ethers.js / viem. RPC: `https://rpc.monad.xyz` (chain 143).

Contract addresses will be in `contracts/deployments.json` after deployment.

ERC-8004 addresses:
- IdentityRegistry: `0x8004A169FB4a3325136EB29fA0ceB6D2e539a432`
- ReputationRegistry: `0x8004BAa17C55a88189AE136b182e5fdA19dE9b63`

### Arena REST API

The arena server (Python, FastAPI) exposes match coordination data. Scav will add endpoints for the website. Expected additions:

```
GET /matches/recent         → recent completed matches with results
GET /agents                 → registered agents with onchain IDs
GET /leaderboard            → ranked agents with Elo scores
```

For development, the arena runs at `http://localhost:8000`. Production URL TBD.

## Tech Stack

Your call. Priorities for a hackathon:
1. Ships fast
2. Looks good (judges see a screenshot before they read code)
3. Can read from Monad RPC (needs ethers/viem)

Reasonable choices: Next.js, Vite + React, Astro, even a well-styled static site with client-side RPC calls. Don't overthink this — pick what you're fastest with.

## Design Notes

- The site is a **read-only view** of onchain and arena state. No wallet connection needed for browsing.
- Wallet connect is only needed for the stretch prediction/spectator features.
- Make it look like a competitive gaming platform, not a DeFi dashboard. Think esports leaderboard, not token analytics.
- Mobile responsive is nice but desktop-first is fine for a hackathon.

## What NOT to Do

- Don't build a backend — read directly from Monad RPC and the arena API.
- Don't edit files outside `web/` — that's Scav's territory.
- Don't block on contract deployment — mock the data and wire up real contracts when addresses are available.
