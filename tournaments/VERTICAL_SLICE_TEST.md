# Vertical Slice Test — Fight Night End-to-End Checklist

Run this against a **local arena** before every event. Takes ~20 minutes.

---

## Setup

### Start local arena

```bash
cd ~/claude-projects/nojohns
source .venv/bin/activate
ADMIN_TOKEN=test123 uvicorn arena.server:app --port 8000
```

### Start local web

```bash
cd web
VITE_ARENA_URL=http://localhost:8000 npm run dev
# Web is now at http://localhost:5173
```

### Open tools

- **Terminal A** — arena logs (watch for errors)
- **Browser tab 1** — `tournaments/admin.html` (operator view)
- **Browser tab 2** — `tournaments/viewer.html` (projector view)
- **Browser tab 3** — `http://localhost:5173/bet` (spectator view, use incognito if possible)

---

## Phase 1 — Tournament Creation

### Step 1.1 — Connect admin panel

- [ ] Open `tournaments/admin.html` in tab 1
- [ ] Enter `http://localhost:8000` as Arena URL
- [ ] Enter `test123` as Admin Token → click **Connect**
- [ ] Status bar shows green "Connected"

### Step 1.2 — Create tournament

- [ ] Click **Create Tournament**
- [ ] Fill in:
  - Name: `Test Tournament`
  - Format: `single-elimination`
  - Entry deadline: any future time
- [ ] Click **Create** → tournament appears in list
- [ ] Note the tournament ID (e.g. `t-001`)

### Step 1.3 — Register 4 entries

Register four entries via the admin panel (or curl):

```bash
TOKEN=test123
ARENA=http://localhost:8000
T_ID=<tournament_id>

curl -s -X POST $ARENA/tournaments/$T_ID/entries \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Phillip (Marth)", "character": "MARTH", "connect_code": "PHIL#001"}'

curl -s -X POST $ARENA/tournaments/$T_ID/entries \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "SmashBot (Fox)", "character": "FOX", "connect_code": "SBOT#002"}'

curl -s -X POST $ARENA/tournaments/$T_ID/entries \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Phillip (Falco)", "character": "FALCO", "connect_code": "PHIL#003"}'

curl -s -X POST $ARENA/tournaments/$T_ID/entries \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "SmashBot (Sheik)", "character": "ZELDA", "connect_code": "SBOT#004"}'
```

- [ ] All 4 entries appear in the entrants list in admin.html
- [ ] Entrant count shows `4`

### Step 1.4 — Seed bracket

- [ ] Click **Seed Bracket** (or equivalent button)
- [ ] Bracket shows semifinal pairings: Match 1 and Match 2
- [ ] Open viewer.html (tab 2) → same bracket structure is visible
- [ ] Both matches show status `pending`

---

## Phase 2 — Match 1 Live

### Step 2.1 — Start match 1

- [ ] In admin.html, click **Start Next Match** for Match 1
- [ ] Arena logs show match created (match ID visible)
- [ ] Viewer.html refreshes → Match 1 shows **LIVE** indicator
- [ ] Note Match 1 ID

### Step 2.2 — Bet page shows active match

- [ ] Open `http://localhost:5173/bet` in tab 3 (incognito recommended)
- [ ] Bet page shows explainer: "AI agents are fighting in Melee. Pick a winner."
- [ ] Sign-in prompt is visible

### Step 2.3 — Sign in and place bet

- [ ] Click **Sign in with Google** (or Privy modal)
- [ ] After sign-in, faucet triggers (check arena logs: `faucet` entry)
- [ ] Active match shows both fighters with odds
- [ ] Click one fighter's bet button (e.g. "Bet on Marth")
- [ ] Confirm transaction in Privy modal
- [ ] Page shows **"You're in!"** confirmation

### Step 2.4 — Verify bet recorded

```bash
curl -s $ARENA/matches/<match_id>/pool | python3 -m json.tool
# Should show: totalBets > 0, at least one bettor
```

- [ ] Pool shows your bet address and amount

---

## Phase 3 — Result & Payout

### Step 3.1 — Report match result

In admin.html (or curl):

```bash
curl -s -X POST $ARENA/tournaments/$T_ID/advance \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"match_id": "<match_1_id>", "winner": "PHIL#001", "scores": {"PHIL#001": 3, "SBOT#002": 1}}'
```

- [ ] Arena logs show: match resolved, pool resolving
- [ ] Viewer.html refreshes → Match 1 shows winner, bracket advances to final

### Step 3.2 — Verify pool resolved

```bash
curl -s $ARENA/matches/<match_1_id>/pool | python3 -m json.tool
# Should show: status = "resolved", winning_side set
```

- [ ] Pool status is `resolved`

### Step 3.3 — Claim payout on bet page

- [ ] Return to bet page (tab 3)
- [ ] Completed match appears with result
- [ ] If you bet on the winner: **Claim** button is visible
- [ ] Click **Claim** → transaction confirms
- [ ] Payout amount shown (should be > original bet)

---

## Phase 4 — Match 2 & Final

### Step 4.1 — Start match 2

- [ ] In admin.html, click **Start Next Match** for Match 2
- [ ] Viewer.html → Match 2 shows **LIVE**

### Step 4.2 — Bet page auto-updates

- [ ] Return to bet page → shows Match 2 as active (no manual refresh needed)
- [ ] Previous match is listed as completed with result

### Step 4.3 — Report match 2 result

```bash
curl -s -X POST $ARENA/tournaments/$T_ID/advance \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"match_id": "<match_2_id>", "winner": "PHIL#003", "scores": {"PHIL#003": 3, "SBOT#004": 0}}'
```

- [ ] Bracket advances → final match shows Match 1 winner vs Match 2 winner

### Step 4.4 — Run the final

- [ ] Start final match, place a bet, report result
- [ ] Viewer.html shows **CHAMPION** for the tournament winner
- [ ] Bracket shows tournament complete

---

## Phase 5 — Edge Cases

These aren't part of normal flow but should be checked before event day:

### Step 5.1 — Force-advance (stuck agent simulation)

```bash
curl -s -X POST $ARENA/admin/tournaments/$T_ID/force-advance \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"match_id": "<any_pending_match>", "winner": "PHIL#001", "reason": "agent_stuck"}'
```

- [ ] Match advances without error
- [ ] Viewer.html updates correctly

### Step 5.2 — Rematch

- [ ] Start a match, then click **Rematch** in admin.html
- [ ] Confirm dialog shows correct player names
- [ ] After confirm, same match shows as `pending` again (not yet started)

### Step 5.3 — Pool cancel (betting system failure)

```bash
# Get pool ID from match first
POOL_ID=$(curl -s $ARENA/matches/<match_id>/pool | python3 -c "import sys,json; print(json.load(sys.stdin)['pool_id'])")

curl -s -X POST $ARENA/admin/pools/$POOL_ID/cancel \
  -H "Authorization: Bearer $TOKEN"
```

- [ ] Pool shows cancelled
- [ ] No payouts issued

---

## Pass Criteria

All of the following must be true to call the vertical slice **PASSED**:

- [ ] Tournament created and entries registered via admin panel
- [ ] Bracket viewer shows live/pending/done states correctly
- [ ] Bet page shows explainer text before sign-in
- [ ] Bet placed successfully after sign-in
- [ ] Faucet fires on first sign-in (arena log shows faucet event)
- [ ] Match result advances bracket in viewer
- [ ] Pool resolves and payout is claimable
- [ ] Bet page auto-updates to next match
- [ ] Tournament champion shown on viewer after final

---

## Known Limitations (don't fail the slice for these)

- Privy embedded wallet requires internet — local-only setup can't fully test Privy. Test with testnet connection or skip the actual transaction, just verify UI state.
- Pool creation requires `PREDICTION_POOL` env var and funded arena wallet. If not configured, pools silently skip — bracket still advances, just no betting.
- QR code scanning: test from a real phone; simulator cameras are unreliable.
