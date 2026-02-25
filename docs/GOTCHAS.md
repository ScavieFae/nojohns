# Gotchas

Hard-won lessons. Read the relevant section before touching that subsystem.

## General

1. **Use the venv.** System Python 3.13 cannot install libmelee (pyenet C build fails). The `.venv` has Python 3.12 with everything working. See CLAUDE.md "Local Dev Setup."

2. **Melee ISO**: We can't distribute it. User must provide NTSC 1.02.

3. **libmelee setup**: Needs custom Gecko codes installed in Dolphin. See libmelee docs.

4. **Frame timing**: `act()` must be fast (<16ms). Don't do heavy computation there.

5. **Controller state**: libmelee uses 0-1 for analog (0.5 = neutral), not -1 to 1.

6. **vladfi1's libmelee fork is the default**: pyproject.toml pulls vladfi1's fork v0.43.0. Key differences from mainline: `MenuHelper` is instance-based (not static), Dolphin path validation requires "netplay" substring, `get_dolphin_version()` exists. All our code expects the fork.

7. **Dolphin path must contain "netplay"**: vladfi1's libmelee fork validates the Dolphin path and rejects paths without "netplay" in them. Use `~/Library/Application Support/Slippi Launcher/netplay`, NOT `/Applications/Slippi Dolphin.app`.

8. **Slippi ranked**: Do NOT enable play on Slippi's online ranked. Against ToS.

## Cosmetic Noise (safe to ignore)

9. **MoltenVK errors**: Dolphin spams `VK_NOT_READY` errors via MoltenVK. Cosmetic — the game runs fine.

10. **BrokenPipeError on cleanup**: libmelee's Controller `__del__` fires after Dolphin is killed. Harmless noise from the SIGKILL cleanup path.

## Testing

11. **pyproject.toml addopts**: `pytest` config sets `--cov=nojohns --cov=games` by default. If pytest-cov isn't installed, pass `-o "addopts="` to override.

12. **Tests mock melee**: `test_smashbot_adapter.py` and `test_netplay.py` install a fake `melee` module so tests run even without libmelee. The mock is skipped if real melee is present.

## Netplay

13. **`--dolphin-home` required for netplay**: Without it, Dolphin creates a temp home dir with no Slippi account and crashes on connect. The `nojohns setup melee` wizard stores this in config.

14. **AI input throttle**: Historically defaulted to 3 (20 inputs/sec) to reduce rollback pressure, but this degraded neural net fighters like Phillip whose models expect every-frame input. Now defaults to `input_throttle=1` (60 inputs/sec). Configurable in `~/.nojohns/config.toml`. Increase if you see desyncs on slow connections.

15. **Game-end detection**: libmelee's `action.value` stability check is too strict for netplay. Detect game end on stocks hitting 0 directly, skip the action state check.

16. **Subprocess per match in tests**: Reusing a single Python process for multiple netplay matches causes libmelee socket/temp state to leak. The test script (`run_netplay_stability.py`) spawns `run_single_netplay_match.py` as a fresh subprocess per match.

17. **Watchdog for `console.step()` blocking**: If Dolphin crashes without closing the socket cleanly, `console.step()` blocks forever. The netplay runner has a watchdog thread that kills Dolphin after 15s.

18. **CPU load causes desyncs**: Slippi's rollback is sensitive to frame timing. Background processes eating CPU cause desyncs. Close heavy apps during netplay.

19. **`--dolphin-home` tradeoffs**: On some machines, `--dolphin-home` is needed for the Slippi account. On others, it causes a non-working fullscreen mode. If menu nav gets stuck at name selection on match 2+, `--dolphin-home` may be the issue — or the fix.

20. **Sheik and Ice Climbers**: `Character.SHEIK` can't be selected from the CSS (she's Zelda's down-B transform). `Character.ICECLIMBERS` doesn't exist in libmelee — use `Character.POPO`. Both will hang the menu navigator forever.

## Phillip / TensorFlow

21. **TensorFlow 2.20 crashes on macOS ARM**: `mutex lock failed: Invalid argument` on import. Use TF 2.18.1 with tf-keras 2.18.0. The `[phillip]` extra in pyproject.toml pins these correctly.

22. **Phillip needs `on_game_start()` in netplay**: The netplay runner must call `fighter.on_game_start(port, state)` when the game starts. Without it, Phillip's agent never initializes.

23. **Phillip needs frame -123 for parser init**: slippi-ai's `Agent.step()` creates its `_parser` only on frame -123. The adapter synthesizes this in `on_game_start()`.

24. **Phillip research notes**: Archived on the `phillip-research` branch (removed from main to reduce repo size).

## World Model

25. **MPS single-process**: macOS Metal GPU doesn't share between training processes. Only one training run per machine at a time. Check `ps aux | grep worldmodel` before launching.

26. **Tensor dimensions are config-driven**: `EncodingConfig` properties compute dimensions dynamically. Never hardcode tensor sizes — read them from config. The shape preflight in `trainer.py` catches mismatches at startup.

27. **SSH backgrounding**: Don't launch training via inline `nohup cmd &` over SSH — PID capture fails and you spawn phantom processes. SCP a script to the remote machine and run that instead. See RUNBOOK.md "SSH Launch Lessons."

28. **Parsed data location**: Training data lives outside the repo at `~/claude-projects/nojohns-training/data/parsed-v2`. The `--dataset` flag points to this directory.

## Arena

29. **Arena self-matching**: Fixed by cancelling stale entries on rejoin and filtering `connect_code` in `find_waiting_opponent()`.
