# Netplay Stability Test Plan

## Goal

Run 10 matches across different characters to measure netplay stability with delay=6.
Success = match lasts ≥60 seconds of in-game time before freeze/completion.

## Setup

**Both sides must:**
1. Pull latest code (includes freeze detection)
2. Run the test script with the same character order
3. Use delay=6
4. Random fighter AI on both sides

## Commands

**ScavieFae side:**
```bash
cd /Users/queenmab/claude-projects/nojohns
git pull
.venv/bin/python test_netplay_stability.py \
  --opponent "SCAV#382" \
  --label scaviefae \
  -d "/Users/queenmab/Library/Application Support/Slippi Launcher/netplay/Slippi Dolphin.app" \
  -i "/Users/queenmab/claude-projects/games/melee/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"
```

**Scav side:**
```bash
cd /path/to/nojohns
git pull
.venv/bin/python test_netplay_stability.py \
  --opponent "SCAVIEFAE#XXX" \
  --label scav \
  -d "/path/to/Slippi Dolphin.app" \
  -i "/path/to/melee.iso"
```

## Character Order

Both sides test the same 10 characters in order:
1. FOX
2. FALCO
3. MARTH
4. SHEIK
5. JIGGLYPUFF
6. PEACH
7. CAPTAIN FALCON
8. PIKACHU
9. SAMUS
10. YOUNG LINK

## Behavior

- Each side launches match independently
- If match freezes → auto-detected after 10s, Dolphin killed, next character
- If match completes normally → move to next character
- 5 second pause between matches
- Both sides log to timestamped file: `netplay_test_{label}_{timestamp}.log`

## What Gets Logged

Per match:
- Character
- Outcome (COMPLETED / FREEZE/DISCONNECT / ERROR)
- Duration in game seconds
- Duration in real seconds (for freezes)
- Success (true if ≥60s)

Summary at end:
- Total successes
- Per-character breakdown
- Full match list

## Post-Test

Compare logs from both sides:
- Did both see the same freezes?
- Any asymmetric behavior (one froze, other completed)?
- Which characters are most stable?

## Notes

- It's OK if sides don't sync perfectly on timing
- The 5-second pause helps sides catch up
- Auto freeze detection means no manual intervention needed
- Both sides should get through all 10 matches automatically
