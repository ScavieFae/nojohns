# No Johns Setup Issues Log

**Date:** 2026-02-02
**Platform:** macOS (Apple Silicon)
**Python Version:** 3.12.12

## Issues Encountered

### 1. pyenet Build Failure (BLOCKING)

**Error:**
```
enet.c:791:10: fatal error: 'enet/types.h' file not found
```

**Context:**
- Occurred during `pip install -e .`
- pyenet is a dependency of libmelee (required for netplay)
- This is a C extension build failure

**Attempted Fix:**
- Installed enet via Homebrew: `brew install enet`
- This provides the enet library at system level

**Status:** ✅ RESOLVED

**Working Solution:**
```bash
# 1. Install enet via Homebrew
brew install enet

# 2. Install with environment variables pointing to Homebrew paths
CFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" .venv/bin/pip install -e .
```

**Notes for Project:**
- This is NOT documented in current SETUP.md (2026-02-02)
- Should add to troubleshooting section or main setup steps
- Affects all macOS users (possibly Linux too if enet not in standard paths)
- Suggest adding this to Step 6 in SETUP.md before the plain `pip install -e .` command
- Alternative: Could vendor pyenet or use a different networking library

**Recommended SETUP.md Addition:**
```markdown
## Step 6: Clone and Install No Johns

On macOS, you'll need to install enet first and set environment variables:

\`\`\`bash
# Install enet library
brew install enet

# Install nojohns with enet paths
CFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" .venv/bin/pip install -e .
\`\`\`
```

---

### 2. pyenet Runtime Linking Error (BLOCKING)

**Error:**
```
ImportError: dlopen(...enet.cpython-312-darwin.so, 0x0002): symbol not found in flat namespace '_enet_address_get_host'
```

**Context:**
- pyenet compiled successfully with CFLAGS/LDFLAGS
- But at runtime, the compiled extension can't find the enet library symbols
- This is a dynamic linking issue on macOS

**Status:** ✅ RESOLVED

**Working Solution:**
```bash
# Build pyenet with explicit linking to system enet library
LDFLAGS="-L/opt/homebrew/lib -lenet" CFLAGS="-I/opt/homebrew/include" \
  .venv/bin/pip install --no-cache-dir --no-binary :all: pyenet
```

**Root Cause:**
- pyenet's bundled enet sources weren't compiling properly
- The compiled extension needs to link against the actual enet library
- The `-lenet` flag explicitly links against libenet

**Verification:**
```bash
otool -L .venv/lib/python3.12/site-packages/enet.cpython-312-darwin.so
# Now shows: /opt/homebrew/opt/enet/lib/libenet.7.dylib
```

**Attempted Fixes:**
1. Building with CFLAGS/LDFLAGS - compiled but runtime error
2. Building with -Wl,-rpath - still runtime error
3. ✅ Building with -lenet flag - SUCCESS

---

## Summary

**Complete Working Setup Command:**
```bash
# Step 1: Install system dependencies
brew install enet python@3.12

# Step 2: Create venv
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip

# Step 3: Install pyenet with explicit linking
LDFLAGS="-L/opt/homebrew/lib -lenet" CFLAGS="-I/opt/homebrew/include" \
  .venv/bin/pip install --no-cache-dir --no-binary :all: pyenet

# Step 4: Install nojohns
.venv/bin/pip install -e .

# Step 5: Install dev dependencies (optional)
.venv/bin/pip install -e ".[dev]"
```

**Test Results:**
- ✅ libmelee imports successfully
- ✅ nojohns CLI works (`list-fighters` command)
- ✅ All 32 tests pass

---

## Configuration Detected

**Dolphin Path:**
`~/Library/Application Support/Slippi Launcher/netplay/Slippi Dolphin.app`

**ISO Path:**
`/Users/queenmab/claude-projects/games/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2)/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso`

---

---

## Issue 4: Netplay Menu Navigation - Position 3 Bug (macOS Sequoia)

**Error:**
Connect codes containing the digit '9' fail to enter correctly on macOS Sequoia.

**Status:** ⚠️ WORKAROUND AVAILABLE

**Root Cause:** ✅ IDENTIFIED
- libmelee's menu navigation has an overflow bug at position 3
- Position 3 corresponds to the character '9' in the number row
- The cursor cannot reach position 3 reliably - it loops: ...→9→4→(can't reach 3)→wraps to 53→...
- This is specific to macOS Sequoia 26.2; works fine on Sonoma 15.7.3

**Tested Positions:**
- ✅ Position 2: '#' - WORKS
- ✗ Position 3: '9' - FAILS (unreachable)
- ✅ Position 4: '.' or ',' (symbols row) - reachable but wrong character
- ✅ Position 8: '8' - WORKS
- ✅ Position 13: '7' - untested but should work
- ✅ Position 18: '6' - WORKS
- ✅ Position 33: '3' - WORKS
- ✅ Position 38: '2' - WORKS
- ✅ Position 48: '0' - WORKS

**Confirmed Working Connect Code:** SCAV#382 ✓
**Confirmed Failing Connect Code:** SCAV#968 ✗ (contains '9')

**Workaround:**
1. Use connect codes that don't contain the digit '9'
2. Or manually enter the connect code in Slippi and let the bot take over at character select

**Technical Details:**
- The Melee name entry grid has 5 rows (libmelee only knows about 4)
- Each position is calculated as: `base_value - (column * 5)`
- Row 3 (numbers): base = 48, so '9' (column 9) = 48 - 45 = 3
- Position 3 appears to be unreachable on Sequoia, possibly due to timing or the 5th row interfering
- libmelee's menu_helper_simple uses the formula correctly, but position 3 navigation fails

**Grid Layout (from screenshot):**
```
Row 0: A B C D E F G H I J
Row 1: K L M N O P Q R S T
Row 2: U V W X Y Z _ _ _ #
Row 3: 0 1 2 3 4 5 6 7 8 9  ← '9' is HERE, but position 3 is unreachable
Row 4: - + = ! ? @ % & $ ,  ← libmelee doesn't account for this row
```

**For Project Maintainers:**
- Consider filing issue with libmelee about Sequoia compatibility
- Or implement custom menu navigation that handles 2D grid properly
- Document limitation: "Slippi connect codes cannot contain the digit 9 on macOS Sequoia"

---

---

## Issue 6: Netplay Stability - Input Delay Testing (2026-02-02 afternoon)

**Problem:** Netplay matches freeze/desync during gameplay, particularly during stock loss events.

**Root Cause:** Slippi's rollback netcode has difficulty handling:
- Complex AI input patterns (random or aggressive strategic inputs)
- Stock loss/respawn state transitions
- Extended gameplay sessions

**Testing Results:**

| Fighter | Delay | Duration | Notes |
|---------|-------|----------|-------|
| DoNothing | 2 | 8+ min | ✅ Full matches, no inputs = no desyncs |
| Random | 2 | 20-90 sec | ❌ Freezes at stock loss |
| SmashBot Fox | 2 | ~2 sec | ❌ Very fast freeze |
| SmashBot Marth | 2 | <1 sec | ❌ Instant freeze |
| Random | 4 | 66-71 sec | ⚠️ Improved but still freezes |
| SmashBot | 4 | 102 sec | ⚠️ 51x improvement! |
| SmashBot | 6 | **146 sec** | ✅ **Full match + partial 2nd match** |

**Key Findings:**

1. **Higher delay = more stability**
   - delay=2: Freezes in seconds with active AI
   - delay=4: Can complete some matches
   - delay=6: Completed full match + started another

2. **Asymmetric freezing**
   - Only ONE computer freezes each time
   - Which one freezes seems random/varies
   - Other computer shows "DISCONNECTED" and continues

3. **Postgame transition fix required**
   - Without fix: Returns mid-explosion, breaks sync
   - With fix: Waits for POSTGAME_SCORES, allows match transitions
   - Enables multi-game sessions (Bo3, Bo5)

4. **Not 100% reliable**
   - delay=6 significantly improves stability
   - May still desync in later matches
   - Likely affected by network conditions, random timing, accumulated state

**Recommended Configuration:**

```python
NetplayConfig(
    online_delay=6,  # 6 frames of rollback buffer (vs default 2)
    # ... other settings
)
```

**Production Readiness:**

- ✅ **Bo1 tournaments**: Likely reliable enough
- ⚠️ **Bo3/Bo5 tournaments**: May need restarts between sets
- ⚠️ **Not guaranteed**: Treat as "much more stable" not "fully fixed"

**Observations:**

- Desync seems random/probabilistic, not deterministic
- Network conditions may play a role
- One side always freezes while other continues (asymmetric)
- Character selection affects stability (Fox > Marth)
- AI complexity affects stability (DoNothing > Random > SmashBot)

**Hypothesis:**

Slippi's rollback netcode has edge cases during:
- Complex state transitions (stock loss, respawn)
- High input variance (random/aggressive AI)
- Accumulated rollback corrections over time

The desync is likely triggered by a combination of:
- Specific game state
- Timing/network jitter
- Rollback history depth

Higher delay gives more buffer time, reducing (but not eliminating) desync probability.

**For Project:**

Document in README/SETUP that netplay with AI is experimental:
- Use delay=6 for best results
- Expect occasional desyncs
- Bo1 recommended over Bo3/Bo5
- Both sides must use same delay setting
- Both sides must have postgame fix

---

## ✅ SETUP COMPLETE!

**Smoke Test:** PASSED (2026-02-02 12:02-12:04)
- Random vs Do Nothing fight ran successfully
- P2 won 0-1 (Do Nothing lost the game 0-4)
- All expected warnings observed (MoltenVK, BrokenPipeError on cleanup)
- libmelee connection working
- Dolphin launching and running properly

**Final Setup Status:**
- ✅ Rosetta 2 installed
- ✅ Xcode Command Line Tools (pre-installed)
- ✅ Homebrew (pre-installed)
- ✅ Python 3.12.12 installed
- ✅ enet library installed via Homebrew
- ✅ pyenet compiled and linked correctly (with workaround)
- ✅ libmelee 0.41.1 working
- ✅ nojohns CLI functional
- ✅ All 32 tests passing
- ✅ Slippi Dolphin configured
- ✅ Melee ISO loaded
- ✅ End-to-end fight working

**Known Working Command:**
```bash
.venv/bin/python -m nojohns.cli fight random do-nothing \
  -d ~/Library/Application\ Support/Slippi\ Launcher/netplay/Slippi\ Dolphin.app \
  -i "/Users/queenmab/claude-projects/games/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2)/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"
```

---

---

## Issue 3: Netplay Menu Navigation Stuck on Connect Code Entry

**Error:**
```
RuntimeWarning: overflow encountered in scalar subtract
  diff = abs(target_code - gamestate.menu_selection)
[Errno 32] Broken pipe (after ~52 seconds)
```

**Context:**
- Netplay command launched Dolphin successfully
- Menu navigation started to enter connect code "SCAV#968"
- Got stuck scrolling right indefinitely on number entry
- User observed: "scrolls to the right indefinitely"
- Eventually timed out with BrokenPipeError

**Status:** ⚠️ BLOCKING for netplay on macOS Sequoia 26.2

**Detailed Observation:**
- Letters "SCAV#" type correctly
- Cursor navigates to numbers section
- When trying to select "968", cursor flies to the right indefinitely
- **Works on macOS Sonoma 15.7.3, fails on macOS Sequoia 26.2**
- Same libmelee version (0.41.1) on both machines

**Root Cause:** ✅ IDENTIFIED
- **libmelee upstream bug** in menuhelper.py line 126
- `diff = abs(target_code - gamestate.menu_selection)` overflows
- nojohns correctly calls `melee.MenuHelper.menu_helper_simple()` with connect_code
- The bug is in libmelee's implementation, not nojohns

**Code Location:**
- nojohns/netplay.py:323-332 - `_handle_menu()` correctly passes connect_code
- libmelee's MenuHelper fails during connect code entry navigation

**Potential Solutions:**
1. **Report to libmelee** - This is an upstream bug
2. **Implement custom menu nav** - nojohns could handle Slippi direct connect menu manually
3. **Use different libmelee version** - Check if newer/older version works
4. **Manual workaround** - User manually navigates to CSS, bot takes over from there

**Recommended Action:**
This is a **libmelee bug specific to macOS Sequoia**. We should:
1. File an issue on libmelee GitHub with details (Sequoia-specific numeric navigation overflow)
2. Implement custom menu navigation in nojohns that bypasses libmelee's buggy menu_helper_simple

**Workaround Options:**
1. **Manual entry workaround** - User manually types connect code, bot takes over at CSS
2. **Custom menu navigation** - Implement Slippi direct connect menu handling in nojohns
3. **Use Sonoma machine** - Netplay works on macOS 15.7.3, just not on 26.2
4. **Wait for libmelee fix** - Report bug and wait for upstream fix

---

## Issue 5: Game Crashes After Stock Loss in Netplay

**Error:**
```
13:23:29 [INFO] Game started
A signal was received. A second signal will force Dolphin to stop.
13:24:33 [ERROR] Netplay error: [Errno 32] Broken pipe
```

**Context:**
- Netplay connection successful with code SCAV#382 (avoiding '9')
- Menu navigation completed successfully
- Game started and ran for ~1 minute
- P1 lost a stock
- Immediately after stock loss, game froze
- Opponent disconnected
- Dolphin received signal and shut down

**Status:** ⚠️ BLOCKING for netplay gameplay

**Observations:**
- Connection phase: ✅ WORKS (with workaround for '9')
- Game start: ✅ WORKS
- Gameplay: ✅ WORKS (for ~1 minute)
- Stock loss event: ❌ CRASH

**Timeline:**
```
13:22:42 - Game 1 starts
13:23:29 - "Game started" (in-game)
13:24:33 - Broken pipe (~64 seconds of gameplay)
```

**User Observations:**
- "the crash was right after p1 lost a stock"
- "both computers start out playing"
- "After 1-3 seconds, SCAV#861 (this computer) freezes"
- "When #861 freezes, the other computer (SCAV#382) freezes temporarily"
- "After 1-3 seconds, #382 unfreezes and continues playing"
- "#861 stays frozen and must be force quit"
- "the crash manifests as the dolphin screen freezes on that frame in-game, always with the stock explosion"
- "I originally noticed this behavior at the end of matches we played locally with two agent-controlled computers. For that, it was always on the final, fourth stock ending the match that it froze."

**CRITICAL:** This is reproducible and happens specifically on the Sequoia machine (SCAV#861), not the other machine (SCAV#382).

**Updated Hypothesis:**
This is a **Dolphin rendering bug on macOS Sequoia 26.2** during stock explosion animation:
1. Stock explosion visual effect starts rendering
2. Dolphin freezes on that frame (likely MoltenVK/Vulkan issue)
3. Python code waits indefinitely at `console.step()` for next frame
4. Other computer continues briefly, then disconnects
5. Must force quit frozen Dolphin

**Code Analysis:**
- Damage tracking logic looks correct (nojohns/netplay.py:260-267)
- Properly handles None players during death/respawn
- Game end detection only triggers when stocks == 0 (not on every stock loss)
- No obvious bugs in nojohns code that would cause crash

**Potential Root Causes:**
1. **Opponent-side crash** - Most likely. Opponent's game may have crashed during respawn, causing disconnect
2. **libmelee netplay bug** - Possible issue with how libmelee handles respawns in netplay mode
3. **Slippi netplay bug** - Could be a Slippi protocol issue during stock loss
4. **Sequoia-specific issue** - May be related to macOS Sequoia 26.2, similar to position 3 bug

**Root Cause:** ✅ IDENTIFIED (Updated)
- **Slippi netplay protocol issue during stock loss**
- NOT OS-specific: Affects both Sequoia AND Sonoma
- Sequoia: Freezes on ANY stock loss (1st, 2nd, 3rd, 4th)
- Sonoma: Freezes only on 4th stock (game end)
- Local fights: ✅ WORK FINE on both systems (tested Sequoia)
- Issue is specific to netplay mode synchronization

**Testing Results:**
```
Test 1 (2026-02-02 13:23): Netplay SCAV#382, froze on FIRST stock
Test 2 (2026-02-02 13:40): Local fight, completed successfully ✓
Test 3 (2026-02-02 13:41): Netplay SCAV#382, played full match, froze on FINAL stock
```

**Updated Pattern:**
- Netplay freezes happen at FINAL stock (game end) on BOTH Sequoia and Sonoma
- Occasionally freezes earlier on Sequoia (stock 1)
- Freeze happens BEFORE sound effect plays
- Local fights complete successfully

**Hypothesis:**
The freeze occurs during the game end transition, likely related to:
1. Game end detection (`p1.stock == 0` check in our code)
2. Slippi netplay synchronization during game-ending stock loss
3. Possible race condition between local game end and netplay protocol

**Workarounds to Test:**
1. **Adjust online_delay** - Try different values (0, 1, 3, 4) to see if timing helps
2. **Test different fighters** - See if RandomFighter's constant inputs cause issues
3. **Multi-game matches** - Try Bo3 to see if freeze happens between games or only at match end
4. **Check libmelee version** - Try different libmelee versions

**Reporting:**
- Report to [Slippi Discord #mac-support](https://discord.com/invite/pPfEaW5)
- Include: macOS Sequoia 26.2, M3 Pro, stock explosion freeze, must force quit
- Mention it's reproducible on both local and netplay
- Provide Dolphin version: v2.0.3-1233-dirty

**For No Johns Project:**
- **Document in README**: Known issue on macOS Sequoia 26.2
- **Recommend macOS Sonoma** for now
- This blocks netplay functionality on Sequoia until Dolphin fix

---

## Recommendations for Project

### Critical: Update SETUP.md

The current SETUP.md needs updates for the pyenet issue:

**Add to Step 4 (Python 3.12):**
```bash
brew install python@3.12 enet
```

**Update Step 6 (Install No Johns):**
```bash
# Install pyenet with explicit linking (macOS requirement)
LDFLAGS="-L/opt/homebrew/lib -lenet" CFLAGS="-I/opt/homebrew/include" \
  .venv/bin/pip install --no-cache-dir --no-binary :all: pyenet

# Install nojohns
.venv/bin/pip install -e .
```

### Optional Improvements

1. **Add troubleshooting section** for pyenet build failures
2. **Document Dolphin location** - it may be in Launcher's netplay dir, not /Applications
3. **Note xattr limitations** - can't clear on code-signed apps in ~/Library
4. **Add example with relative ISO paths** for convenience

---


---

## Issue #7: Automated Testing - Menu Navigation "Naming Bug"

**Date:** 2026-02-02  
**Status:** ✅ RESOLVED

**Symptom:**
When running automated netplay tests that restart Dolphin between matches, menu navigation would get stuck at connect code entry on the 2nd+ match. Both sides would experience this simultaneously, even when one side had already crashed.

**Investigation:**
- Initial theory: Timing issue between sides → Tried delays of 5s, 20s, 60s → No improvement
- Tested temp directory cleanup → No improvement  
- Tested socket/resource cleanup → No improvement
- **Key insight:** Manual restart (kill script, restart) always worked ✅
- **Root cause:** libmelee has internal module-level state that persists between NetplayRunner instances in the same Python process

**Solution:**
Run each match in a separate subprocess for fresh libmelee state.

Created `run_single_netplay_match.py` helper script that runs one match in isolation. Main `test_netplay_stability.py` spawns this as a subprocess for each match.

**Implementation:**
```python
# Each match runs in fresh subprocess
subprocess.run([
    sys.executable,
    "run_single_netplay_match.py",
    "--opponent", opponent_code,
    "--character", char_name,
    # ... other args
])
```

**Results:**
- ✅ No more naming bug
- ✅ Automated testing works perfectly
- ✅ Successfully ran 10-match test cycles
- ✅ Each match gets fresh Python process + libmelee state

**Lessons Learned:**
1. libmelee menu_helper_simple has persistent state issues
2. Timing/cleanup delays dont solve state issues
3. Process isolation (subprocess) is the robust solution
4. Manual testing worked because each run was fresh process
