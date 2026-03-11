# Brief: Fight Night Hardening

**Branch:** 008-hardening
**Model:** sonnet

## Goal

Close the product gaps identified in the post-build review (#43). Focus on: things that will break the event if untested, things that make Mattie's job easier on the night, and things that make the spectator experience feel cohesive.

This is a hardening pass, not new features. Fix what's built, test what's untested, write what's undocumented.

## Tasks

### 1. Rematch button in admin.html

The `/admin/matches/{id}/rematch` endpoint exists but admin.html has no UI for it. Add a "Rematch" button next to force-advance. Should:
- Show current match's players
- Confirmation dialog ("Re-queue Marth vs Fox?")
- Call the rematch endpoint
- Refresh bracket state after

### 2. Registration on a second computer

Right now only admin.html can register entrants. For the event, Mattie may delegate registration to a helper. The admin panel needs to work from any computer with the arena URL + admin token.

- Verify admin.html works when opened from a different machine (just needs arena URL + token in the config bar at top)
- Add a "Registration Mode" view that shows ONLY the tournament creation form and entry list — hides the match control buttons so a helper can't accidentally force-advance. Toggle between full admin and registration-only.
- If the admin panel already works remotely (it should — it's a static HTML file hitting the arena API), just verify and document.

### 3. One-machine local fallback

If WiFi fails at the venue, we need to run matches on one machine without netplay. `nojohns fight` already supports local mode. Verify:
- `nojohns fight phillip phillip` works on one machine (two AI fighters, no network)
- Results can be manually reported to the arena via admin panel or curl
- Document the fallback procedure: how to switch from two-machine netplay to one-machine local mid-tournament

### 4. Operator runbook

Write `tournaments/RUNBOOK.md`. Printed checklist for Mattie. Covers:

**Pre-event:**
- Arena server checks (Railway health, wallet balance)
- Create tournament in admin panel (entry format, character list)
- Verify bracket viewer on projector
- Verify bet page works (scan QR from phone, sign in, see odds)

**During tournament:**
- Match flow: Start Next → wait → result auto-advances
- If agent stuck at character select: wait 60s, then force-advance with random winner or rematch
- If disconnect: wait for 300s timeout, then expire match + decide: rematch or force-advance
- If Dolphin crashes: kill process, rematch from admin panel
- If arena unreachable: check Railway logs, restart if needed
- If wallet low: check /admin/wallet, top up from personal wallet

**Emergency fallback:**
- Switch to one-machine mode (see task 3)
- If betting completely breaks: cancel all pools, continue bracket without betting
- If bracket viewer dies: use admin panel to track matches, announce results verbally

### 5. Vertical slice test

Write a test script or manual checklist that exercises the full path:
1. Create 4-entry tournament via admin panel
2. Start match 1 → verify bracket viewer shows LIVE
3. Scan QR → sign in on bet page → place bet → verify "You're in!"
4. Report result → verify bracket advances → verify pool resolves
5. Claim payout on bet page
6. Start match 2 → verify next match auto-shows on bet page
7. Complete tournament → verify champion shows on viewer

This should run against a local arena (not production). Document it as a repeatable checklist.

### 6. Bet page onboarding screen

Add a brief splash/explainer before the sign-in prompt on `/bet`:
- "AI agents are fighting in Melee. Pick a winner."
- "Sign in to get 2 free bets."
- "No crypto knowledge needed."
- Keep it to 3 lines max. Not a tutorial, just context.

### 7. Fix known issues from review

- Character list: fix `YOUNKLINK` → `YOUNGLINK` typo in admin.html
- Character list: remove duplicate `MEWTWO` in admin.html
- Character list in viewer.html: verify coverage of all 23 viable characters

## Completion Criteria

- [ ] Rematch button works in admin.html
- [ ] Admin panel works from a second computer (tested)
- [ ] One-machine fallback documented and tested
- [ ] Operator runbook written (`tournaments/RUNBOOK.md`)
- [ ] Vertical slice exercised end-to-end (local arena)
- [ ] Bet page has 3-line explainer before sign-in
- [ ] Character list typos/duplicates fixed

## Verification

- `.venv/bin/python -m pytest tests/ -v -o "addopts="` passes
- Open admin.html from a different machine → can create tournament and register entries
- Run vertical slice checklist → all steps pass
- RUNBOOK.md exists and covers pre-event, during, and emergency fallback
