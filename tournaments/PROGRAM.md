# Fight Night Build Program

**Event:** Agentic Smash Fight Night
**When:** Wednesday March 11, 7:00–10:30 PM PT
**Where:** Frontier Tower Floor 7, 995 Market St SF
**Format:** Single elimination, Final Destination, 4 stock / 8 min, $10 entry, 70/20/10 pot split
**Site:** `tournaments.nojohns.gg`
**Build window:** ~48 hours (Monday evening → Wednesday afternoon)

---

## What Already Works

These are deployed, tested across 100+ matches during Moltiverse:

- **Arena server** — matchmaking, result reporting, EIP-712 signing (Railway)
- **Netplay** — Phillip vs Phillip over Slippi, watchdog, disconnect detection
- **Contracts** — MatchProof, Wager, PredictionPool on Monad mainnet
- **Prediction pool flow** — arena auto-creates pools, web components for betting (PredictionWidget, BetForm, OddsBar, LiveBetFeed), CLI `nojohns bet`
- **Spectator agent** — autonomous bettor with Kelly criterion
- **Website** — leaderboard, match history, live viewer (Vercel)
- **Fighter registry** — phillip, smashbot, random, do-nothing

## What We're Building

### Workstream 1: Tournament Bracket (P0)

No bracket system exists. Need: generation, display, advancement, and the orchestration to queue bracket matches through the arena.

**Files:**
- `tournaments/bracket.py` — bracket generation, seeding, advancement logic
- `tournaments/models.py` — Tournament, Round, Match, Entry dataclasses
- `tournaments/tournament.py` — lifecycle orchestration (create → register → play → resolve)
- `arena/server.py` — new endpoints: `POST /tournaments`, `GET /tournaments/{id}/bracket`, `POST /tournaments/{id}/advance`

**Bracket viewer:**
Self-contained HTML page served at `tournaments.nojohns.gg`. Shows live bracket state, current match, upcoming matches. Updates via polling or WebSocket. Displayed on secondary monitor at the event.

**Decisions:**
- Single elimination, flexible size (expect ≤16 but not hard-capped — nearest power of 2, byes fill gaps)
- Seeding: random (no Elo history for most entrants)
- Bo1 per round, 4 stock, 8-minute timer (standard tournament rules)
- Bracket advances when arena reports match result
- Admin controls: manual advance, manual re-match, force bye
- Fallback: pen and paper bracket if software fails. Visualization is what matters, not the orchestration being software-dependent.

### Workstream 2: Agent Signup (P0)

Currently agents are identified by Slippi connect codes. For the event we need names, character choices, and a clean registration flow.

**Signup flow:**
1. Entrant approaches registration table
2. Operator enters: **agent name**, **character** (from 23 viable), **strategy** (phillip / random / do-nothing)
3. System assigns a connect code from a pre-generated pool (avoids Sequoia '9' bug)
4. Agent appears in bracket viewer with their chosen name

**Implementation:**
- `tournaments/models.py` — `Entry(name, character, strategy, connect_code, wallet_address?)`
- Arena DB gets a `tournament_entries` table
- `POST /tournaments/{id}/register` accepts name + character + strategy
- Bracket viewer shows names, not codes
- Pre-generate 20 connect codes (all digits 0-8) and validate on Sequoia before event

**On connect codes:**
Since we're in one room, we can pre-configure both machines with all the connect codes. An entrant's "agent" is really a config slot on one of our machines. The operator swaps config (code + character + fighter) between matches.

### Workstream 3: Character Selection Hardening (P0)

Getting stuck at character select = dead air on stage. Known issues:

1. **Sequoia '9' bug** — position 3 unreachable on macOS 15.x
2. **General CSS loop** — navigator can cycle if character position math is off
3. **Loop detection fires too late** — 30 attempts before fallback, that's ~2 seconds of visible struggle

**Fixes:**
- Pre-assign connect codes that avoid '9' (Workstream 2 handles this)
- Reduce loop detection threshold from 30 → 15 attempts
- Add character validation at registration: reject Sheik and Ice Climbers (CSS-unselectable)
- Test all 23 characters on both machines Tuesday evening
- Fallback: if character select fails, retry with a random character from the safe pool

### Workstream 4: Disconnect Recovery (P0)

Disconnects cause cascading stuck state: match stuck → wager stuck → prediction pool stuck. Current timeout is 30 minutes — unacceptable for live event.

**Changes:**
- Reduce match expiration from 1800s → 300s (5 min) for tournament matches
- Add admin endpoint: `POST /matches/{id}/force-expire` — immediately expires match + cancels pool + refunds wagers
- Add admin endpoint: `POST /tournaments/{id}/matches/{round}/{slot}/rematch` — re-queues same bracket matchup
- Arena wallet: pre-fund with 0.5 MON, add balance check at startup
- Pool cancellation on disconnect: if match expires, `cancelPool()` fires and all bettors can `claimRefund()`

**Admin panel:**
Simple HTML page (not public) with buttons for force-expire, rematch, advance. Operator's phone or laptop. Endpoints protected by a shared secret / bearer token.

### Workstream 5: Prediction Markets for Spectators (P1)

The infrastructure exists. The gap is the spectator-facing flow: how does someone in the room go from "I want to bet on this match" to money on the line?

**Flow:**
1. QR code displayed on bracket viewer links to `tournaments.nojohns.gg/bet`
2. Page shows current match, odds bar, bet form
3. Spectator connects Monad wallet (needs MON)
4. Places bet via existing PredictionWidget components
5. After match resolves, claim payout on same page

**The hard part is wallet onboarding.** Most people in the room won't have a Monad wallet with MON. Options:

- **Option A: Pre-funded wallets.** We generate 20 burner wallets, each loaded with 0.1 MON, printed as QR codes on cards. Hand them out at the door. Simple, works, we eat the cost (~2 MON total).
- **Option B: Real wallets.** Spectators install a wallet app, we send them MON. Slower but they keep the wallet. Better for "this is crypto" narrative.
- **Option C: Hybrid.** Offer both. Cards for casual bettors, real wallet instructions for crypto-native attendees.

**Recommendation: Option C.** Pre-fund 20 burner cards (cost: ~2 MON ≈ ~$2), print wallet instructions for those who want to set up real wallets. The burner cards make the prediction market actually work for non-crypto attendees.

**Refund safety:**
- Every pool has a `cancelPool()` path — if match doesn't resolve, all bets refundable
- `claimRefund()` is permissionless — bettors call it themselves
- We add a "Refund" button to the betting page that appears when pool is cancelled
- Arena wallet must have gas to call `cancelPool()` — part of the pre-fund budget

### Workstream 6: Pre-Seeded Markets (P2)

Once the bracket is generated, we know the matchups. Open prediction pools before matches start so bettors can assess the bracket.

**Implementation:**
- When bracket is generated, create pools for Round 1 matches immediately
- When a round completes, create pools for next round
- Bracket viewer shows pool status + odds for upcoming matches
- "Bet Now" link on each bracket slot

**Depends on:** Workstreams 1 + 5 complete.

### Workstream 7: "Beat the Agents" (P2 — stretch)

Track which spectators make the best predictions. Leaderboard of bettors by profit.

**Implementation:**
- Read `Bet` events from PredictionPool contract for the tournament's pools
- Aggregate profit/loss per address
- Display leaderboard on bracket viewer
- Announce "best human predictor" at end of night

**Depends on:** Workstream 5 (spectators actually betting). Lightweight — could be a post-event script if we run out of time.

### Workstream 8: Human vs Agent Station (P2 — stretch)

Separate station where humans walk up and play Phillip. Track wins/losses on a scoreboard. The side attraction while the tournament runs.

**Setup:**
- One machine running `nojohns fight phillip human` (or a mode where P2 is a real controller)
- Scoreboard: simple tally of human wins vs agent wins, displayed on a monitor or whiteboard
- Optional: track individual human names ("Alex: 1-3 vs Phillip")

**Implementation:**
- This mostly works already — `nojohns fight` supports two local players
- P2 needs to be a real controller (not an AI). Check if libmelee supports a human-controlled port alongside a bot port
- Scoreboard could be a simple HTML page that polls the arena, or just pen-and-paper
- If we log results to arena, they show up in match history for free

**Open question:** Does libmelee support having one port be a real human controller and the other be bot-controlled? If not, this might need the human on a separate Dolphin instance (netplay against the bot on the same machine). Or we could do it fully local with two controllers and just observe.

---

## Build Schedule

### Monday Evening (Mar 9)

**Goal:** Core bracket system working end-to-end in dev.

- [ ] `tournaments/models.py` — dataclasses
- [ ] `tournaments/bracket.py` — single elimination generation + advancement
- [ ] `tournaments/tournament.py` — lifecycle orchestration
- [ ] Arena endpoints for tournament CRUD + bracket
- [ ] Bracket viewer HTML (static, polling arena)
- [ ] Admin panel HTML (force-expire, rematch, advance)

### Tuesday (Mar 10)

**Goal:** Full dress rehearsal — simulated 8-agent tournament, prediction markets, all hardening.

Morning:
- [ ] Character selection hardening (loop detection, safe pool, validation)
- [ ] Disconnect recovery (fast timeouts, force-expire, auto pool cancel)
- [ ] Agent signup flow (name, character, strategy → bracket entry)
- [ ] Pre-generate connect code pool (20 codes, no '9' digit)

Afternoon:
- [ ] Spectator betting page (`/bet` route with PredictionWidget)
- [ ] QR code generation (link to betting page per match)
- [ ] Burner wallet generation (20 wallets, fund with 0.1 MON each)
- [ ] Pre-seeded markets (pools created from bracket)

Evening:
- [ ] **Dress rehearsal:** Run 8-agent simulated tournament on two machines
- [ ] Test every path: signup → bracket → match → disconnect → recover → bet → resolve → claim
- [ ] Fund arena wallet (0.5 MON)
- [ ] Deploy bracket viewer to `tournaments.nojohns.gg` (Vercel subdomain)
- [ ] Print burner wallet QR cards

### Wednesday (Mar 11)

**Goal:** Event day. No new features. Fix what broke in dress rehearsal.

Morning:
- [ ] Fix anything from Tuesday night's rehearsal
- [ ] Final deploy
- [ ] Regenerate website snapshot for current leaderboard

Afternoon (at venue, before doors):
- [ ] Test WiFi at Frontier Tower (both machines, 3 test matches)
- [ ] Set up secondary display (bracket viewer)
- [ ] Verify arena server is running (Railway)
- [ ] Verify Monad RPC is responsive
- [ ] Load bracket viewer, admin panel on operator devices
- [ ] Lay out burner wallet cards at door

---

## Architecture Decisions

### One Machine or Two?

We have both options at the venue. **Use two machines** as primary, one machine as fallback.

Two-machine benefits:
- More authentic (separate Dolphins, real netplay)
- Proven in 100+ matches
- Each machine runs one side — clean separation

One-machine fallback:
- Eliminates WiFi risk entirely
- Uses `nojohns fight` (local mode, no netplay)
- Loses Slippi online features but the game still plays
- Have this ready as a backup

### Tournament Matches Through Arena vs Direct

**Use the arena.** Tournament matches are regular arena matches tagged with `tournament_id`. The arena handles matchmaking, result reporting, pool creation, and signing. The tournament server just orchestrates who plays when.

This means:
- No duplicate match execution code
- Prediction pools "just work" (arena creates them on match start)
- Match results flow into the existing onchain pipeline
- Bracket advancement triggers on result webhook from arena

### DNS: tournaments.nojohns.gg

Options:
- **Separate Vercel project** in `tournaments/web/` — clean but more deploy infra
- **Route in existing `web/` app** — less overhead but couples tournament UI to main site

**Decision:** Add tournament routes to existing `web/` app. Add `/tournament`, `/tournament/bracket`, `/tournament/bet` routes. Set up `tournaments.nojohns.gg` as a Vercel domain alias that routes to `/tournament`. Less infra to manage on event day.

### Real Money Hardening

This is the first time real MON is at stake in prediction markets. Safety measures:

1. **Pool size caps** — limit max bet per pool (0.5 MON?) to bound exposure
2. **Arena wallet monitoring** — log balance at each operation, alert if < 0.05 MON
3. **Refund-first mentality** — if anything goes wrong, cancel pool and refund everyone. Nobody loses money on a broken match.
4. **No manual pool resolution** — `resolve()` reads from MatchProof contract. Either both players signed or the pool gets cancelled. No admin override that pays out.
5. **Operator runbook** — printed checklist for common failure modes and their resolution steps

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| WiFi blocks UDP at venue | Matches can't start | Medium | Test Tuesday, have one-machine fallback |
| Agent stuck at character select | Dead air, match delay | Medium | Hardened loop detection, safe character pool, reduced threshold |
| Disconnect mid-match | Stuck wager/pool, audience confusion | Medium | 5-min timeout, admin force-expire, auto pool cancel |
| Arena wallet out of gas | Pools don't create/resolve, results don't post | Low | Pre-fund 0.5 MON, balance logging |
| Monad RPC down | All onchain ops fail | Low | Cache recent state, degrade gracefully (matches still play, just no betting) |
| Spectator loses MON on cancelled pool | Bad vibes | Low | `claimRefund()` always available, refund button prominent |
| More signups than bracket slots | Awkward at door | Low | Cap at 16, waitlist, first-come-first-served |
| Dolphin crash | Match interrupted | Low | Watchdog auto-kills, rematch via admin panel |

---

## Open Questions

1. **Entry fee collection** — Cash/Venmo at the door (per Luma listing). Onchain entry adds friction for no benefit at this scale.
2. ~~**Commentators**~~ — No commentators. Mattie operates the bracket solo.
3. **Exhibition matches** — Separate computer, `nojohns fight` on the side. No bracket integration. Pre-configure fun matchups (Phillip vs do-nothing, random vs random).
4. **nad.fun tokens** — Skip for Wednesday. Not doing the full implementation. Prediction markets are the spectator play.
5. **Bet size limits** — What's the max we're comfortable with per pool? 0.5 MON? 1 MON? Depends on how much MON is worth.
6. **"Beat the agents" tracking** — Build it or announce it post-hoc from chain data?

## Resolved Decisions

- **Stage:** Random stage every match (not FD-only).
- **Character:** Locked for whole tournament. Picked at registration.
- **Do-nothing vs do-nothing:** Coinflip. Skip the match, advance a random winner.
- **Pacing:** Manual. Operator (Mattie) triggers each match from admin panel.
- **One match at a time.** One station, audience focused.
- **Exhibitions:** Separate machine, no bracket.

## Go-Live Checklist

Everything needed by Wednesday 7 PM. Checked off as completed.

### Code — Must Have (P0)

**Bracket System (Brief 001)**
- [ ] `tournaments/models.py` — Tournament, Bracket, Round, Match, Entry dataclasses
- [ ] `tournaments/bracket.py` — single elimination generation, seeding, byes, advancement
- [ ] `tournaments/tournament.py` — lifecycle orchestration, state persistence (SQLite)
- [ ] Arena endpoints — POST /tournaments, GET /tournaments/{id}/bracket, POST /tournaments/{id}/advance, POST /tournaments/{id}/next
- [ ] `tournaments/viewer.html` — bracket viewer, polls arena, projector-ready at 1920x1080
- [ ] `tournaments/admin.html` — operator panel (create tournament, start next match, force-advance, rematch)

**Disconnect Recovery (Brief 003)**
- [ ] ADMIN_TOKEN auth on all /admin/* endpoints (including existing /admin/cleanup)
- [ ] MATCH_TIMEOUT env var (default 300s, replaces hardcoded 1800s)
- [ ] Decouple match expiration from pool cancellation (rename to `_expire_stale_matches()`)
- [ ] POST /admin/matches/{id}/expire — expire one match (does NOT cancel pool)
- [ ] POST /admin/pools/{id}/cancel — cancel one pool onchain
- [ ] POST /admin/matches/{id}/rematch — re-queue same players
- [ ] GET /admin/wallet — address + MON balance
- [ ] Wallet balance logging after onchain ops, WARNING below 0.05 MON

**Character Hardening (Brief 002 — Manual Session Tuesday)**
- [ ] Test all 23 characters on both machines (5 rounds each or random)
- [ ] Build safe character pool from test results
- [ ] Pre-generate 20 connect codes (no '9' digit)
- [ ] Reduce loop detection threshold from 30 → 15

### Code — Should Have (P1)

**Prediction Markets (Brief 004)**
- [ ] Privy integration in web app (email sign-in → embedded wallet)
- [ ] `/bet` page — current match, odds bar, "Bet on X" / "Bet on Y" buttons
- [ ] Claim payout + claim refund buttons
- [ ] Faucet endpoint — auto-fund new Privy wallets (0.1 MON, cap 50)
- [ ] QR code on bracket viewer linking to /bet

### Code — Nice to Have (P2)

**Human vs Agent (Brief 006)**
- [ ] Challenge mode in MatchRunner (port 1 bot, port 2 GCN adapter)
- [ ] `nojohns challenge <fighter>` CLI command
- [ ] match_type field on arena matches (tournament/challenge/exhibition)
- [ ] `tournaments/scoreboard.html` — human vs agent tally
- [ ] Hardware test: GCC adapter + Dolphin + Phillip on one machine

### Ops — Must Do

- [ ] Fund arena wallet with 0.5 MON
- [ ] Deploy bracket viewer to tournaments.nojohns.gg (Vercel)
- [ ] Deploy updated arena to Railway (ADMIN_TOKEN, MATCH_TIMEOUT env vars)
- [ ] Dress rehearsal: 8-agent simulated tournament Tuesday evening
- [ ] Test WiFi at Frontier Tower (both machines, 3 test matches)

### Ops — Day Of (Wednesday)

- [ ] Fix anything from Tuesday dress rehearsal
- [ ] Final deploy (arena + web)
- [ ] Set up secondary display at venue (bracket viewer)
- [ ] Verify arena server running (Railway)
- [ ] Verify Monad RPC responsive
- [ ] Load admin panel on operator device (Mattie's phone/laptop)

### Outreach & Logistics

- [ ] Ask Monad about sponsoring the event
- [ ] Reach out to NorCal Melee for extra setups
- [ ] Find Smash communities to advertise Fight Night
- [ ] Look for sponsors (Burner Wallet, others)
- [ ] Print betting instruction cards (QR code to /bet)

---

## Success Criteria

Wednesday night is a success if:

1. **16 agents play a full single-elimination bracket** — no stuck matches, no manual intervention needed for more than 1 match
2. **Bracket viewer runs on secondary display** — audience can follow the tournament
3. **At least 5 spectators place bets** — prediction markets are live and used
4. **All bets resolve or refund cleanly** — nobody loses money to a bug
5. **It's fun** — people are trash-talking, cheering, the commentators have something to work with
