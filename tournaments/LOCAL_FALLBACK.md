# One-Machine Local Fallback

If WiFi fails at the venue, you can run matches on a single machine using local mode.
Two AI fighters play head-to-head with no network — both run inside the same Dolphin process.

---

## How It Works

`nojohns fight` runs two fighters locally (no arena, no netplay). Each fighter gets a controller,
Dolphin starts, they play, result is printed to the terminal. No Slippi account needed.

```bash
# Simplest: phillip vs phillip (uses characters/settings from ~/.nojohns/config.toml)
nojohns fight phillip phillip

# With explicit character choices:
nojohns fight phillip phillip --p1-char MARTH --p2-char FOX

# Full flags (if config isn't set up):
nojohns fight phillip phillip \
  -d ~/Library/Application\ Support/Slippi\ Launcher/netplay \
  -i ~/games/melee/melee.ciso
```

The result is printed to stdout:

```
P1 wins: 3 stocks remaining
P2 wins: 1 stocks remaining
Winner: P1 (phillip)
```

---

## Reporting the Result to the Arena

Local mode bypasses the arena matchmaking flow. Once you see the winner, manually advance the
bracket using the admin panel.

### Admin panel (recommended)

1. Open `admin.html` → load your tournament
2. Scroll to **Force Advance** section
3. Fill in: `round`, `slot`, and `Winner Name` (exact name as registered)
4. Click **Force Advance** → confirm

This calls `POST /admin/tournaments/{id}/force-advance` with admin auth.

### curl (if admin panel is unreachable)

```bash
ARENA=https://your-arena.railway.app    # or http://localhost:8000
TOKEN=your-admin-token
TOURNAMENT_ID=abc123
ROUND=1
SLOT=0
WINNER="Marth Bot"

curl -X POST "$ARENA/admin/tournaments/$TOURNAMENT_ID/force-advance" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"round\": $ROUND, \"slot\": $SLOT, \"winner_name\": \"$WINNER\"}"
```

Round and slot are zero-indexed. Round 0 = quarterfinals, Round 1 = semis, Round 2 = finals.
Slot 0 = top match in the round, Slot 1 = next, etc.

---

## Switching Mid-Tournament

If you're mid-tournament and WiFi drops:

1. **Finish the current match however it ends** (or force-advance if needed)
2. **Switch fighters to local mode:**
   - Identify the two fighters for the next match from the admin panel
   - Note their characters (registered at entry time)
   - Run: `nojohns fight <strategy1> <strategy2> --p1-char <char1> --p2-char <char2>`
3. **Watch the match**, note the winner
4. **Force-advance the bracket** via admin panel (WiFi on phone hotspot is fine for this)
5. Repeat for each subsequent match

The arena doesn't need to stay connected during the fight — it only needs to be reachable when you
force-advance (which can be done from a phone hotspot).

---

## Verified Command

This has been verified against the CLI and runner code:

- `runner.run_match(fighter1, fighter2, settings)` accepts any two `Fighter` instances
- No netplay code paths are invoked in local mode
- Both `phillip` and `random` work as fighters
- `do-nothing` also works (useful for testing bracket advancement without real fights)

To test without Dolphin/ISO: `nojohns fight random do-nothing` with a mock config or in the test env.

---

## Notes

- Local mode uses the same Dolphin + ISO as netplay — you need both installed
- Both fighters share one machine's CPU — expect slightly slower frame pacing
- Phillip may play worse than usual if the machine is under load
- Onchain recording and prediction pools **do not work** in local mode (no arena match ID)
- Betting should be paused during a fallback match (admin: cancel the pool, resolve manually later)
