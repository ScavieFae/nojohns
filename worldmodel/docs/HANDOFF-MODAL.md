# Modal Pipeline Handoff — Review Feedback for Scav

**From**: ScavieFae (review of `scav/combat-context-heads`)
**Date**: Feb 25, 2026
**Branch**: `scav/combat-context-heads` (commit `dafe30c`)

## Status

Config validation assertion (Scav-2 priority #1) is already in `modal_train.py:110-116`. The pipeline architecture is sound. Three code fixes and two doc fixes remain before trusting a real training run.

## Code Fixes

### 1. Sweep checkpoint collision (Medium)

All sweep runs write to bare `CHECKPOINT_DIR` (`/data/checkpoints/`). Parallel runs overwrite each other's `best.pt`.

**Fix in `modal_train.py` `train()`** — use `run_name` to namespace:
```python
save_dir = f"{CHECKPOINT_DIR}/{run_name}"
os.makedirs(save_dir, exist_ok=True)
```
And pass `save_dir` instead of `CHECKPOINT_DIR` to `Trainer(save_dir=...)`.

### 2. MeleeDataset.__new__ reconstruction (Medium — structural debt)

`modal_train.py:128-135` manually assigns 7 attributes via `__new__`. If `MeleeDataset.__init__` ever gains a new field, Modal training silently breaks.

**Fix**: Add `MeleeDataset.from_tensors()` classmethod in `dataset.py`:
```python
@classmethod
def from_tensors(cls, floats, ints, game_offsets, game_lengths, num_games, cfg):
    ds = cls.__new__(cls)
    ds.cfg = cfg
    ds.floats = floats
    ds.ints = ints
    ds.game_offsets = game_offsets if isinstance(game_offsets, np.ndarray) else game_offsets.numpy()
    ds.game_lengths = game_lengths
    ds.num_games = num_games
    ds.total_frames = int(ds.game_offsets[-1])
    return ds
```
Then replace the `__new__` block in `modal_train.py` with:
```python
dataset = MeleeDataset.from_tensors(
    floats=payload["floats"], ints=payload["ints"],
    game_offsets=payload["game_offsets"], game_lengths=payload["game_lengths"],
    num_games=payload["num_games"], cfg=enc_cfg,
)
```

### 3. Missing error raise on bad data (Low)

`train()` (line 89-94) and `pre_encode()` print errors and `return` when files are missing. `modal run` exits 0 — looks like success.

**Fix**: Replace `return` with `raise FileNotFoundError(...)` in both functions.

## Doc Fixes (in MODAL-REVIEW-RUNBOOK.md)

### 4. Pre-encode memory estimate

Checklist item 3.6 says "300K games would be ~50GB — this would OOM" but doesn't note the practical cap. The review feedback section corrects it (~25-30K at 16GB), but the checklist should match.

### 5. No-epoch-completed framing

Add a note at the top of the runbook: as of Feb 25, no training epoch has completed on any cloud GPU. The smoke test is a first-ever validation, not a regression check.

## Blocked on ScavieFae's side

- Modal CLI is not installed on this machine. Can't run smoke tests (steps 1-7) or the local-vs-Modal comparison (Scav-2 priority #2) until `pip install modal` + `modal token set`.
- If you want me to run the smoke tests, let me know and I'll get Modal set up.

## Priority Order

1. ~~Checkpoint collision fix (#1)~~ — **Done** (Scav, Feb 25). `save_dir` uses `{CHECKPOINT_DIR}/{run_name}`.
2. ~~`from_tensors()` classmethod (#2)~~ — **Done** (Scav, Feb 25). Added to `MeleeDataset`, modal_train.py updated.
3. ~~Smoke test the full loop~~ — **Done** (Feb 25 ~12:08 PT). First epoch completed: run `mamba2-first-complete`, 61min, loss=0.4698, change_acc=0.519, val_loss=0.3405. Checkpoint saved. Second run (`smoke-nw4-v2`, with `num_workers=4`) in progress.
4. ~~Error raises (#3)~~ — **Done** (Scav, Feb 25). `return` → `raise FileNotFoundError` in both `train()` and `pre_encode()`.
5. Doc fixes (#4, #5) — **Done** (Scav, Feb 25). Memory estimate corrected, "no epoch completed" framing added then updated with real results.
