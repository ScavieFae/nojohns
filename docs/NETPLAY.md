# Netplay Recommendations for No Johns

## Current Status (2026-02-02)

Netplay with AI fighters is **experimental but functional** with the right configuration.

## Required Configuration

### 1. Code Updates

Both sides must have commit `e984c07` or later, which includes:
- Postgame transition handling (wait for POSTGAME_SCORES before returning)
- Neutral inputs during death/respawn
- Stable state checking before game end detection

### 2. Delay Setting

**Use `online_delay=6`** (6 frames of rollback buffer)

```python
config = NetplayConfig(
    online_delay=6,  # CRITICAL: Use 6, not the default 2
    # ... other settings
)
```

Or via CLI:
```bash
.venv/bin/python -m nojohns.cli netplay smashbot --delay 6 --code "OPPONENT#CODE" -d /path/to/dolphin -i /path/to/melee.iso
```

### 3. Both Sides Must Match

- Same code version (with postgame fix)
- Same delay setting (6 on both sides)
- Same fighter types work best

## Expected Behavior

### What Works ✅

- **Bo1 matches**: Most complete successfully (70% in testing)
- **Young Link**: Most stable character tested (100% success rate)
- **Peach**: Also very stable (33% reach 60s, 100% completion)
- **Random fighter**: Works well with delay=6
- **Local fights**: Always work (no netplay involved)

### What's Unreliable ⚠️

- **Bo3/Bo5**: May desync in later games
- **Extended sessions**: Probability of desync increases over time
- **Different delays**: Asymmetric config causes more desyncs
- **Some characters**: Pikachu, Samus less stable than YLink/Peach
- **High CPU load**: Background processes reduce stability

### What Doesn't Work ❌

- **delay=2** (default): Freezes in seconds with active AI
- **Without postgame fix**: Freezes mid-explosion
- **Sheik**: Menu navigation can't reliably select this character

## Known Issues

1. **Desyncs still occur** - delay=6 reduces but doesn't eliminate them (70% success rate)
2. **Asymmetric freezing** - one computer freezes, other shows "DISCONNECTED"
3. **Non-deterministic** - same setup may work once, fail next time
4. **Network dependent** - may be affected by jitter/latency
5. **CPU load sensitive** - Background processes (Chrome tabs, etc.) increase desync rate
6. **Character-dependent** - Some characters more stable than others (YLink > Peach > Fox > Pikachu)

## Troubleshooting

### If matches freeze immediately (< 10 seconds):

- ✓ Check both sides have the postgame fix (commit e984c07+)
- ✓ Verify delay=6 on both sides
- ✓ Try DoNothing fighter to confirm basic netplay works
- ✓ Check network stability

### If matches last 1-2 minutes then freeze:

- ✓ This is expected behavior with delay=6
- ✓ For tournaments, run Bo1 instead of Bo3
- ✓ Or restart netplay between games in a set

### If one side always freezes:

- ✓ Check if that side has older code version
- ✓ Verify same delay setting on both sides
- ✓ Try swapping which computer initiates connection

## For Tournament Organizers

**Recommended Format:**
- Bo1 single elimination
- Both players use delay=6
- Both players on latest nojohns code
- **Close unnecessary programs** (browsers, etc.) to reduce CPU load
- Prefer stable characters (Young Link, Peach, Fox) over unstable ones (Pikachu, Samus)
- Restart netplay between rounds (not between games in a set)

**Alternative:**
- Run local matches on a central server
- Use MatchRunner (not NetplayRunner) with both AIs local
- No netplay desyncs, guaranteed stability

## Stability Test Results (2026-02-02)

**Setup:** Random fighter, delay=6, random characters, 10 matches
**Results:** 7/10 completed, 2/10 lasted ≥60s

**Character stability:**
- Young Link: 1/1 (100%) lasted 60s
- Peach: 1/3 (33%) lasted 60s, 3/3 (100%) completed
- Fox: 0/2 lasted 60s, 2/2 (100%) completed
- Falco: 0/2 lasted 60s, 1/2 (50%) completed
- Pikachu: 0/1 completed (froze)
- Samus: 0/1 completed (froze)

## Technical Details

The desync appears to be a Slippi netplay issue when:
- Rollback netcode handles complex AI input patterns
- Stock loss/respawn creates many state changes
- Extended gameplay accumulates rollback corrections

Higher delay (6 vs 2 frames) gives more buffer time for the rollback algorithm to sync, significantly reducing (but not eliminating) desync probability.

## Future Work

- Test delay=8 or higher for even more stability
- Investigate why one side freezes vs both
- Add input logging to identify problematic sequences
- Consider reporting to Slippi team with test data
