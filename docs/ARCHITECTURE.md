# No Johns — Architecture

## Layers

| Layer | What | Where |
|-------|------|-------|
| **Protocol** | Onchain contracts — match proofs, wagering | `contracts/` |
| **Client** | Off-chain tooling — arena, fighters, signing, CLI | `nojohns/`, `arena/`, `games/`, `fighters/` |
| **Frontend** | Website — leaderboard, match history, landing page | `web/` |

## System Diagram

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

## Boundaries

**Protocol / Client boundary:** The `recordMatch()` call. The protocol doesn't know what language the client is written in, how matches are played, or what game it is. The client doesn't need to understand contract internals — just the ABI and the EIP-712 domain.

**Protocol / Frontend boundary:** `MatchRecorded` events and contract view functions. The frontend is read-only — no wallet connection needed for browsing.

**Wagering is optional.** MatchProof is the core primitive. Wager is a separate opt-in layer that reads from MatchProof. Dependency is one-way. Matches work without wagers. No admin key, no owner — settlement is purely mechanical.

## The Protocol is Client-Agnostic

No Johns' Python tooling (the `nojohns/` package) is *one* client. The protocol doesn't depend on it. Anyone can build a completely independent client for any game, in any language, and use the same contracts.

What a third-party client needs:
- The MatchProof contract address and ABI
- The EIP-712 domain (`name: "NoJohns"`, `version: "1"`, chain ID, contract address)
- Two agents with wallets that can sign typed data

What it does *not* need:
- Our Python code, CLI, or arena server
- Any dependency on this repo
- Permission from anyone

A Starcraft client in Rust, a chess client in Go, a card game client in JavaScript — they all submit to the same MatchProof with their own `gameId` string. Same wager escrow. Same leaderboard. Every game's match history is on the same contract, queryable by `gameId` in the emitted events.

```
Protocol (one deployment, all games)
         │
         ├── Client: No Johns (Python, Melee)      gameId: "melee"
         ├── Client: ???      (Rust, Starcraft)     gameId: "starcraft"
         ├── Client: ???      (Go, Chess)           gameId: "chess"
         └── Client: ???      (JS, Poker)           gameId: "poker"
```

## Contracts

Both contracts are immutable. No proxy, no owner, no upgrade mechanism.

- **MatchProof** — records dual-signed match results. Game-agnostic (`gameId` is a string).
- **Wager** — escrow in native MON. Settles by reading from MatchProof. Timeout refunds both sides after 1 hour.

If the protocol needs changes, new contracts are deployed alongside old ones. Old match history is preserved in events.
