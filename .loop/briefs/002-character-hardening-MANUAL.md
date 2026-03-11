# Brief: Character Selection Hardening

**Branch:** scav/char-hardening
**Model:** sonnet

## Goal

Eliminate character selection failures for Fight Night. We've seen agents get stuck at the CSS (character select screen), but we don't know exactly which characters or conditions cause it. Step one is test, step two is fix what we find.

Reference: `games/melee/menu_navigation.py` (our navigator), `.venv/lib/python3.12/site-packages/melee/menuhelper.py` (libmelee's `choose_character`), gotcha 20 in CLAUDE.md.

## Design Decisions

- **Test first, fix second.** Don't assume the cause — run all 23 characters through matches and see what actually fails.
- **Known exclusions:** Sheik (Zelda transform, no CSS slot), Ice Climbers (POPO in libmelee, hangs), Zelda (raises NotImplementedError in netplay). These are excluded from the pool before testing.
- **Random selection via swag mode is intentional.** slippi-ai found it's the fastest approach for netplay CSS. Don't replace it without evidence.
- **Connect code pool:** Pre-generate codes with no '9' digit to eliminate the separate Sequoia name-entry bug.

## Tasks

1. Create `scripts/test_character_select.py` — automated test harness that runs through all characters in `_RANDOM_POOL` (cli.py line 55). For each character, run `nojohns fight <char> do-nothing --games 1` and record: did it get past CSS? How long (frames/seconds)? Did it select the correct character? Log results to a JSON file. Run each character 3-5 times to catch intermittent failures.

2. Run the test on both machines (Scav's Mac, ScavieFae's Mac). Document results: which characters pass, which fail, which are slow. This is the real deliverable — a tested character pool we can trust for Wednesday.

3. Based on test results, build the safe character pool:
   - If all 23 pass reliably: pool = full 23, done.
   - If some fail: exclude them, document why, restrict registration to the safe pool.
   - If many fail: investigate the common pattern and fix.

4. Add a timeout + fallback to our menu navigator (`games/melee/menu_navigation.py`): if character selection hasn't completed within 10 seconds (~600 frames), log a warning and report the failure. Don't silently hang — surface it so the operator knows to intervene.

5. Pre-generate a pool of 20 connect codes with no '9' digit. Store in `tournaments/connect_codes.py`. These get assigned at registration, avoiding the Sequoia name-entry bug entirely.

6. Add `validate_character(char_name)` function that tournament registration calls — rejects Sheik, Ice Climbers, Zelda, and any characters that failed testing.

## Completion Criteria

- [ ] Test harness runs all 23 characters 3+ times each on at least one machine
- [ ] Test results documented (pass/fail/slow per character, per machine)
- [ ] Safe character pool defined based on test results
- [ ] Timeout on CSS selection prevents silent hangs (logs warning after 10s)
- [ ] Connect code pool exists with 20 codes, none containing '9'
- [ ] `validate_character()` rejects known-bad characters

## Verification

- `scripts/test_character_select.py` completes without hanging
- `.venv/bin/python -m pytest tests/ -v -o "addopts="` passes
- Test results JSON exists with data for all 23 characters
