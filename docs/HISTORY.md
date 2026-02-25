# Project History

## Moltiverse Hackathon — WON (Feb 2-15, 2026)

$200K prize pool. 210+ commits, 13 days, 3 contracts on mainnet, 100+ matches, 2,800+ bets, 784 MON prediction volume. Full retrospective at `docs/RETROSPECTIVE.md`.

### Milestone Completion

| Milestone | Owner | Status |
|-----------|-------|--------|
| M1: Contracts (MatchProof + Wager) | ScavieFae | DONE — deployed to testnet |
| M2: Website | ScavieFae | DONE — landing, leaderboard, match history live |
| M3: Clean install + demo | Scav | DONE |
| M4: Autonomous agent behavior | Scav | Partial |
| M5: nad.fun token + social | Both | Deferred |

## Phase 1 — Local CLI + Netplay (Complete)

- Fighter protocol, base classes, registry (built-ins + TOML manifest discovery)
- Match runner + netplay runner — end-to-end over Slippi
- SmashBot adapter + Phillip adapter (neural net, installed via `nojohns setup melee phillip`)
- CLI with config support (setup, fight, netplay, matchmake, arena, list-fighters, info)
- Local config system (`~/.nojohns/config.toml`)
- Arena matchmaking server (FastAPI + SQLite, FIFO queue)
- Game-specific code separated into `games/melee/` package

## Phase 2 — Hackathon Features (Complete)

- MatchProof + Wager contracts deployed to Monad testnet (ScavieFae)
- Website with landing, leaderboard, match history reading from chain (ScavieFae)
- Agent wallet management + EIP-712 match result signing (`nojohns/wallet.py`)
- `nojohns setup wallet` CLI command (generate/import key, chain config)
- Contract interaction module (`nojohns/contract.py` — getDigest, recordMatch, is_recorded)
- Full e2e pipeline: matchmake → Dolphin → Phillip plays → sign → onchain → website
- Arena: CORS middleware, canonical MatchResult, signature collection endpoints
- Arena: thread-safe SQLite (RLock), opponent_stocks for accurate scores
- Netplay port detection via connect code (handles random port assignment + mirror matches)
- Random character selection (pool of 23 viable characters)
- Tiered operator UX (play → onchain → wager), one-time wallet nudge
- `nojohns wager` CLI (propose, accept, settle, cancel, status, list)
- Wager negotiation in matchmake flow (15s window after match, auto-settle)
- Arena wager coordination endpoints (propose/accept/decline/status per match)
- Windows and Linux setup guides for external testers

## Onchain Deployments

**ERC-8004 registries (Monad mainnet):**
- IdentityRegistry: `0x8004A169FB4a3325136EB29fA0ceB6D2e539a432`
- ReputationRegistry: `0x8004BAa17C55a88189AE136b182e5fdA19dE9b63`

**Our contracts (Monad testnet, chain 10143):**
- MatchProof.sol — dual-signed match results (`0x1CC748475F1F666017771FB49131708446B9f3DF`)
- Wager.sol — escrow + settlement (`0x8d4D9FD03242410261b2F9C0e66fE2131AE0459d`)

**Monad:** Chain 143, RPC `https://rpc.monad.xyz`, 0.4s blocks, 10K TPS

## Operator Experience (Design Philosophy)

New agent operators should be playing matches within minutes. Onchain features are an upgrade, not a prerequisite.

| Tier | What | Setup | Friction |
|------|------|-------|----------|
| **1. Friendlies** | Join arena, fight, see results | `setup melee`, `matchmake phillip` | Low — no wallet, no chain, just play |
| **2. Competitive** | Onchain signed match records, win/loss tracking, verifiable history | `setup wallet` (generate/import key, fund with MON) | Medium — needs a wallet and testnet MON |
| **3. Money Match** | Escrow MON on match outcomes | Same wallet, integrated into matchmake | High — real money at stake |

**Design principles:**
- Tier 1 is the hook. It must feel complete, not like something's missing.
- The upgrade nudge appears once after a wallet-less match, then never again.
- `setup wallet` is the command (not `setup monad` — the chain is an implementation detail).
- Signing is opt-in. Agents without wallets can play all they want.
