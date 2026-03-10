# Fight Night Operator Runbook

**Agentic Smash Fight Night — Wednesday March 11, 7–10:30 PM PT**
Frontier Tower Floor 7, 995 Market St SF

Keep this open on your phone or print it. Everything is here.

---

## Pre-Event Checklist

### 1. Arena server
- [ ] Open Railway dashboard → verify service is **Running** (green)
- [ ] `curl https://your-arena.railway.app/health` → should return `{"status":"ok"}`
- [ ] Check wallet: `GET /admin/wallet` (or open admin.html → scroll to bottom)
  - If balance is low, top up from personal wallet before doors open
- [ ] Faucet cap: capped at 50 wallets. Reset if needed between events.

### 2. Tournament setup (admin.html)
- [ ] Open `admin.html` in Chrome on the host machine
- [ ] Enter Arena URL + Admin Token in the config bar at top → **Connect**
- [ ] Click **Create Tournament** → fill in:
  - Name: "Fight Night #1"
  - Format: single-elimination
  - Entry deadline: time you want registration to close
- [ ] Share the registration link with helpers (they only need the URL + token)
- [ ] Verify entries appear in the list as people register

### 3. Bracket viewer on projector
- [ ] Open `viewer.html` in a second browser window (or the projector machine)
- [ ] Enter the same Arena URL → **Load**
- [ ] Confirm the bracket displays correctly
- [ ] Zoom the browser to fill the projector screen (Cmd +/-)
- [ ] Leave this window running — it auto-refreshes

### 4. Bet page
- [ ] On your phone: scan the QR code shown on viewer.html
- [ ] Confirm `/bet` loads and shows the sign-in prompt
- [ ] Sign in with email → confirm you see "2 free bets"
- [ ] Verify the pool is visible (it appears once the first match is announced)
- [ ] Done. You don't need to place a real bet unless you want to.

---

## During the Tournament

### Normal match flow

1. **Admin panel → "Start Next Match"** — queues the next bracket match
2. Agents connect via Slippi and start playing automatically
3. Bracket viewer shows **LIVE** for the active match
4. When the match ends, result posts to the arena and bracket advances
5. Prediction pool resolves automatically — winners can claim from `/bet`
6. Repeat

### Agent stuck at character select

Wait 60 seconds. If they're still stuck:

- **Option A:** Force-advance with a random winner (fair if it's a glitch)
  - Admin panel → Force Advance → pick either player → confirm
- **Option B:** Rematch (if you suspect a real technical failure)
  - Admin panel → Rematch → confirm the re-queue

### Agent disconnect / Slippi drop

- Arena auto-expires after **300 seconds** (5 minutes) of no result
- You'll see the match expire in the admin panel
- Once expired, decide: **Rematch** or **Force Advance**
  - Rematch if the opponent might reconnect quickly
  - Force Advance if you need to keep the event moving

### Dolphin crash

1. Kill the crashed Dolphin process: `kill <pid>` or force-quit in Activity Monitor
2. Restart the fight: **Admin panel → Rematch**
3. The operator runs `nojohns matchmake phillip` again on their machine

### Arena unreachable

1. Check Railway logs (dashboard → Deployments → latest → Logs)
2. Look for Python errors or OOM crashes
3. Redeploy if needed: Railway dashboard → **Redeploy**
4. While arena is down, bracket viewer will show stale data — tell the crowd

### Faucet / wallet balance low

- Check: `GET /admin/wallet` returns current balance
- Send MON from personal wallet to the arena wallet address shown
- 0.1 MON per user × 50 cap = 5 MON needed maximum

---

## Emergency Fallback Procedures

### WiFi fails — switch to one-machine local mode

See `tournaments/LOCAL_FALLBACK.md` for the full procedure. Short version:

1. Finish or force-advance the current match
2. Identify the two fighters and their characters from admin panel
3. On the single machine: `nojohns fight phillip phillip --p1-char MARTH --p2-char FOX`
4. Watch match, note winner
5. Force-advance the bracket from admin panel (phone hotspot is fine)
6. Repeat

### Betting completely broken

1. Cancel all open pools: Admin panel → each open pool → **Cancel Pool**
2. Announce to crowd: "Betting is paused for this match"
3. Continue bracket normally — results still advance, just no pools
4. Pools can be re-enabled for later matches once the issue is fixed

### Bracket viewer dies

1. Try refreshing — it's a static HTML file, should come back
2. If Arena is down: use the admin panel to track matches and announce results verbally
3. Backup: screenshot the bracket state and share on a TV/screen

### Do-nothing vs do-nothing (both fighters fail to load)

Per spec: coinflip, random winner. Use Force Advance → pick either player.

---

## Quick Reference

### Key URLs

| Thing | URL |
|---|---|
| Arena health | `https://your-arena.railway.app/health` |
| Admin wallet | `GET /admin/wallet` (header: `Authorization: Bearer <token>`) |
| Force advance | `POST /admin/tournaments/{id}/force-advance` |
| Expire match | `POST /admin/matches/{id}/expire` |
| Cancel pool | `POST /admin/pools/{id}/cancel` |
| Rematch | `POST /admin/matches/{id}/rematch` |

### curl cheatsheet

```bash
ARENA=https://your-arena.railway.app
TOKEN=your-admin-token
TID=tournament-id

# Force advance a match
curl -X POST "$ARENA/admin/tournaments/$TID/force-advance" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"round": 0, "slot": 0, "winner_name": "Marth Bot"}'
# round/slot are 0-indexed. Round 0 = quarters, 1 = semis, 2 = finals.

# Expire a stale match
curl -X POST "$ARENA/admin/matches/$MATCH_ID/expire" \
  -H "Authorization: Bearer $TOKEN"

# Cancel a prediction pool
curl -X POST "$ARENA/admin/pools/$POOL_ID/cancel" \
  -H "Authorization: Bearer $TOKEN"

# Check wallet balance
curl "$ARENA/admin/wallet" -H "Authorization: Bearer $TOKEN"
```

### Common admin token header

All `/admin/*` routes require:
```
Authorization: Bearer <your-admin-token>
```

---

## Event Format

- **Format:** Single elimination, best of 1
- **Rules:** 4 stock, 8-minute timer, random stage each match
- **Characters:** Locked at registration — cannot change mid-tournament
- **Entry fee:** $10
- **Do-nothing bots:** Coinflip winner (force-advance with either player)

---

## Contact / Escalation

If something breaks that's not covered here: call Mattie.
