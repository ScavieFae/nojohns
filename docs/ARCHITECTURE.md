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

## Live Streaming Architecture

Spectators can watch matches in real-time via the live viewer on the website.

### Current Flow

```
Client (Dolphin)                Arena (Railway)              Viewers (Browser)
      │                               │                            │
      │── HTTP POST /stream/frame ───►│                            │
      │   (60fps, or batched 4x)      │── WebSocket broadcast ────►│
      │                               │   (fan-out to N viewers)   │
```

1. Client extracts frame data from Dolphin (positions, actions, stocks)
2. Client POSTs frames to arena (batched: 4 frames per request)
3. Arena broadcasts to all WebSocket viewers of that match
4. Website buffers 8 frames (~133ms) then plays back at steady 60fps

### Scaling Considerations

**At hackathon scale (1-10 matches):** No problem. ~150 req/sec is trivial.

**At production scale (100+ concurrent matches):** Needs optimization.

#### Quick Wins (TODO)

1. **Stream only when watched**
   - Client asks arena "any viewers?" before streaming
   - No viewers = no frames sent
   - Probably drops load by 90%+ (most matches have 0 spectators)

   ```python
   # In netplay.py, before starting stream:
   if arena.get(f"/matches/{match_id}/viewer_count")["count"] == 0:
       skip_streaming = True
   ```

2. **Lower frame rate for spectators**
   - 30fps is fine for watching, 60fps is overkill
   - Arena can sample every 2nd frame before broadcasting

3. **Adaptive streaming**
   - Start at low rate, increase if viewers join
   - Reduce if buffer health is good

#### Medium-Term: WebSocket Upload

Replace HTTP POST with bidirectional WebSocket for frame streaming.

**Current (HTTP):**
```
Client ──POST──► Arena    (new connection per batch)
```

**Proposed (WebSocket):**
```
Client ◄──────► Arena     (persistent connection, frames flow continuously)
```

Benefits:
- No connection overhead per frame/batch
- Lower latency
- Server can signal "pause streaming" instantly
- Natural backpressure

Implementation notes:
- Client connects to `ws://arena/ws/stream/{match_id}` at match start
- Sends frame messages: `{"type": "frame", "data": {...}}`
- Arena broadcasts to viewer WebSockets (already implemented)
- Same auth as HTTP endpoints (match_id + queue_id)

#### Long-Term: Dedicated Streaming Service

Separate frame ingestion from matchmaking/coordination:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────►│   Stream    │────►│   Viewers   │
│  (Dolphin)  │     │   Service   │     │  (Browser)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │    Arena    │
                    │ (matchmake, │
                    │  results)   │
                    └─────────────┘
```

- Stream service handles high-throughput frame data
- Arena handles coordination (lower volume, higher importance)
- Can scale independently
- Could use specialized infra (WebRTC, Livepeer, etc.)
