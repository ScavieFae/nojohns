# Modal Pipeline Handoff — ScavieFae Review

**Branch**: `scav/combat-context-heads`
**Date**: Feb 26, 2026 (late)
**Status**: Projectile extraction unblocked. Re-parse needed before projectile experiment.

---

## Current: Projectile Extraction Unblocked (Feb 26, late)

### What changed

Commit `86de069` — **real item/projectile data now extracted from .slp replays**.

The blocker from commits `3919365` / `1a44244` is resolved. peppi_py's Python wrapper (`frames_from_sa()`) drops items from the raw arrow struct. Fix: access `peppi_py._peppi.read_slippi()` directly to get the arrow data, which has full item fields (id, type, state, position, velocity, damage, timer, owner).

### How it works

1. `_extract_items()` reads raw arrow struct, extracts flat numpy arrays from the ListArray using offsets (fast — no per-frame `.as_py()`)
2. `ItemAssigner` (ported from slippi-ai) maps item spawn IDs to stable slot indices 0-14 so items keep their slot across frames
3. Graceful fallback: if raw access fails for any reason, returns None → empty items (existing behavior)
4. Both `build_dataset.py` and `parse_archive.py` updated with the same pattern

### Verification results

Tested on `Game_20260209T120845.slp` (Fox ditto, 13,607 frames):
- 9,043 item-frame instances across 4 slots (types: 59, 61, 63, 65, 77)
- 47.6% of frames have active items
- Full pipeline: parse → load_game → MeleeDataset(projectiles=True) → non-zero projectile floats

### What this unblocks

1. Re-parse 2K+ games with the updated parser → items populated
2. Run `projectile-2k-test` experiment (run card already approved in `4ca90c0`)
3. Eventually re-parse the full 287K ranked dataset for home run training

### What did NOT change

- `parse.py`, `encoding.py`, `dataset.py` — downstream pipeline already handled items correctly
- No model code changes
- Existing parquet files still work with `projectiles=False`
- Double-parses each .slp (once for players via wrapper, once for items via raw). ~2x parse time during dataset build only, not training.

### Note on parse_archive.py schema

`parse_archive.py` was also missing `randall`, `fod_platforms`, and `items` fields in its root struct (only had `p0, p1, stage`). Now matches `build_dataset.py`'s full schema. This fixes a latent bug where games parsed via `parse_archive.py` would fail to load items even with the encoding flag on.

---

## Parallel Pre-Encode Pipeline (Feb 26, 10:30 PM)

### What's happening right now

11 Modal workers encoding 2K games each in parallel → single `encoded-22k.pt` file on volume.

**Modal app**: https://modal.com/apps/scaviefae/main/ap-fjsUUHcAAlXZRy5c4o5QB1

### Architecture

```
pre_encode_parallel (local entrypoint)
  → reads meta.json locally, splits 22K entries into 11 chunks
  → _encode_chunk.starmap() — 11 parallel Modal containers
      each: extract tar to local NVMe → load 2K games → encode → save chunk_NNNN.pt to volume
  → _concat_chunks.remote() — single 128GB container
      load chunks from volume → pre-allocate final tensor → fill → torch.save → commit
```

### Why parallel?

| Approach | Bottleneck | Estimated time |
|----------|-----------|---------------|
| Local (MacBook 36GB) | OOM — 22K games = 88GB tensor data | impossible |
| Sequential Modal (1 container) | CPU-bound parquet parsing | 3-5 hours |
| **Parallel Modal (11 containers)** | Same work ÷ 11 | **~20-40 min** |

The bottleneck is zlib decompression + pyarrow parsing of individual parquet files. Parallelism is the only way to speed this up without changing the data format.

### Resource allocation

| Function | CPU | RAM | Instances |
|----------|-----|-----|-----------|
| `_encode_chunk` | 4 | 32GB | 11 (parallel) |
| `_concat_chunks` | 2 | 128GB | 1 (after all chunks) |

### After encoding completes

```bash
# Launch overnight training
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --encoded-file /encoded-22k.pt --epochs 3 --run-name mamba2-22k-ss

# Verify it started
# Check wandb: https://wandb.ai/shinewave/melee-worldmodel
```

### Changes this session (uncommitted)

| File | What |
|------|------|
| `scripts/modal_train.py` | Parallel pre-encode (`_encode_chunk`, `_concat_chunks`, `pre_encode_parallel`), chunked sequential `pre_encode`, per-game progress logging |
| `training/trainer.py` | SS dynamics mask fix (only corrupts core_continuous + velocity, not stocks/hitlag/combo) |
| `scripts/train.py` | SS params passthrough to Trainer |
| `scripts/pre_encode_chunked.py` | Local chunked encode (memmap approach — works but we have no disk space) |

### What ScavieFae should review

1. **`_encode_chunk` → `_concat_chunks` data flow**: chunks save `{floats, ints, lengths}`, concat pre-allocates final tensors and fills. Any concern about tensor dtype mismatches or ordering?

2. **Volume concurrency**: 11 workers calling `volume.commit()` simultaneously from different containers. Modal volumes support this, but worth verifying chunks don't clobber each other (they write to different paths: `chunk_0000.pt`, `chunk_0001.pt`, etc).

3. **128GB RAM for concat**: final tensor is ~88GB (242M frames × 360 bytes). Pre-allocation + one chunk loaded = ~96GB peak. Tight but should fit in 128GB. If it OOMs, we'd need to bump to 192GB or use memmap.

4. **SS dynamics mask** (`trainer.py`): `_corrupt_context()` now only corrupts `core_continuous_dim + velocity_dim` (9 values per player), skipping dynamics (hitlag, stocks, combo). This addresses ScavieFae's stocks noise concern from the previous review.

---

## For Review Now

ScavieFae — please review the following before we commit to overnight runs (~9 PM PT launch target). Mattie is at an AI summit and runs will be unattended until morning.

### 1. New configs

**`worldmodel/experiments/mamba2-medium-gpu-k60.yaml`** — K=60 comparison config.
- Same architecture as `mamba2-medium-gpu.yaml` (d_model=384, n_layers=4, 4.3M params)
- Only changes: `context_len=60`, `chunk_size=30`, `batch_size=512` (halved from 1024 for VRAM headroom)
- **Question**: Is batch_size=512 conservative enough for K=60 on A100 40GB? Could OOM.

**`worldmodel/experiments/mamba2-large-gpu.yaml`** — 15M parameter model.
- d_model=512, n_layers=6, d_state=64, headdim=64
- lr=0.0003 (lower than 4.3M's 0.0005 — larger model, smaller LR)
- batch_size=512
- **Question**: LR and batch_size are guesses. This model has never been instantiated. Worth a 2K smoke test before betting $70+ overnight.

### 2. Timeout bump

`modal_train.py:58`: `timeout=86400` (was 14400, was 7200 before that). Three runs hit the 7200s ceiling mid-epoch-2. 24hr ceiling is intentional for overnight runs.

### 3. Overnight plan

Full plan at `worldmodel/docs/OVERNIGHT-PLAN.md`. Key points:

**Results so far** (2K games, A100, `num_workers=4`):

| Model | Data | Epochs | change_acc | pos_mae | val_loss |
|-------|------|--------|------------|---------|----------|
| MLP 1.3M | 2K | 2 | 64.5% | 0.79 | — |
| MLP 1.3M | 22K | 4 | 77.5% | 0.79 | — |
| Mamba-2 4.3M | 2K | 1 | 52.1% | 0.77 | 0.342 |
| **Mamba-2 4.3M** | **2K** | **2** | **67.2%** | **0.64** | **0.290** |

Mamba-2 beats MLP at equal data+epochs. Architecture validated. Scaling is the lever.

### 4. Review runbook updates

`worldmodel/docs/MODAL-REVIEW-RUNBOOK.md` — Known Issues table updated: 4 items fixed (checkpoint collision, config validation, `__new__` reconstruction, silent returns), 2 new items added (sweep untested, K=60 batch_size unknown). Cost table updated with real observed timing.

---

## Plan: Next 8 Hours (Feb 25, ~5 PM → ~1 AM PT)

### Phase 1: Queue experiments (~5-6 PM)

| Run | Config | Data | Epochs | Est. time | Est. cost | Purpose |
|-----|--------|------|--------|-----------|-----------|---------|
| `k10-compare` | mamba2-medium-gpu | encoded-2k.pt | 2 | ~90 min | ~$4 | K=10 baseline with identical setup |
| `k60-compare` | mamba2-medium-gpu-k60 | encoded-2k.pt | 2 | ~90-120 min | ~$4-6 | K=60 — does longer context help SSM? |
| pre_encode 22K | (CPU only) | parsed-v2 on volume | — | ~10 min | ~$0.10 | Produce `encoded-22k.pt` for overnight |

All three run in parallel. K comparison results back by ~7:30 PM.

### Phase 2: Evaluate K results (~7:30-8 PM)

**What we're watching for:**
- **K=60 change_acc vs K=10** — if K=60 is >=5pp higher, use K=60 overnight. If similar or worse, stick with K=10 (cheaper per epoch).
- **K=60 training time** — how much slower is it per epoch? 6x more context at half batch_size could be 3-4x slower per epoch.
- **K=60 val_loss trajectory** — is it converging faster in terms of wall-clock time, or just per-epoch?
- **Any OOM or crash** — K=60 batch_size=512 is untested.

### Phase 3: Launch overnight bet (~8-9 PM)

**Decision tree:**

```
K=60 clearly better (>=5pp change_acc)?
├── Yes → overnight: 4.3M, K=60, 22K games, 5 epochs (~$60, ~12-15hr)
└── No/Similar → overnight: 4.3M, K=10, 22K games, 5 epochs (~$50, ~10-12hr)

22K pre-encode succeeded?
├── Yes → use encoded-22k.pt
└── No → fall back to encoded-2k.pt, 10 epochs (~$20, ~8hr)

Time budget allows?
├── Stretch: also launch 15M on 2K games as validation (~$10, ~3hr)
└── No stretch: just the main bet
```

### What can go wrong overnight

| Failure mode | Impact | Mitigation |
|------|--------|------------|
| Timeout (86400s = 24hr) | Run killed | 22K × 5ep projected at ~10-15hr — should be fine |
| OOM on 22K data | Crash at DataLoader init | `encoded-22k.pt` loads all tensors at once — ~7GB. A100 has 40GB. Fine for 4.3M. |
| Modal preemption | Run killed mid-epoch | No mitigation — checkpoints save per-epoch, so we lose at most 1 epoch of work |
| wandb disconnect | Lose live monitoring | Training continues, metrics still in stdout. Check Modal logs. |
| Loss divergence | Wasted compute | Check wandb after ~2 epochs. If loss isn't decreasing, something's wrong. |

### Blockers

1. **22K pre-encode** — must complete before overnight launch. If it fails (OOM, tar extraction issue), fall back to 2K data.
2. **K comparison results** — ideally done before launch, but not strictly required. We can pick K=10 (proven) and run K=60 separately.

---

## ScavieFae Review Response (Feb 25, ~8 PM PT)

### Original review items: all 6 resolved

Clean execution. The `from_tensors()` implementation matches the spec. `FileNotFoundError` raises are correct. Checkpoint namespacing uses `run_name` as suggested.

### trainer.py fix

`batch_metrics["loss/total"]` → `batch_metrics.total_loss` — correct. The old code would `TypeError` if `batch_metrics` is a dataclass, not a dict. Likely caught during the first real A100 run.

### Config answers

**K=60 batch_size=512 on A100 40GB: safe.** Activation memory estimate: `512 * 60 * 384 * 4 bytes * 4 layers * ~3` (fwd+bwd+optim) ≈ ~0.5GB. Comfortable. If it OOMs, it'll be DataLoader prefetch, not the model. Go ahead.

**15M model (mamba2-large-gpu.yaml):** `lr=0.0003` at `batch_size=512` is reasonable. Linear scaling rule would give 0.00025, but 0.0003 is close enough. Agree that a 2K smoke test is non-negotiable before a $70 overnight bet.

**Watch out:** `mamba2-large-gpu.yaml` has `context_len=10`. If K=60 wins the comparison, this config needs updating before the overnight run.

### Missing: checkpoint resume on preemption

The "What can go wrong" table says preemption loses at most 1 epoch. For the 22K overnight run (~8hr/epoch), that's ~$23 of wasted compute. `train()` doesn't accept a `--resume` flag — if preempted, there's no way to resume from the last checkpoint without manually modifying the launch command to load weights.

**Suggestion for Scav:** Add `resume: str = ""` param to `train()` that loads `model.state_dict()` + optimizer state from a checkpoint path before training starts. Not blocking for tonight (low preemption risk on Modal), but worth adding before the 15M runs.

### Overnight plan: approved

The conservative bet (parallel K10/K60 on 22K, ~$61) is the right call. Decision tree is clear. Ranked data staging correctly deferred. The 15M stretch goal is fine if K results come in early.

---

## Review Request: Sprint Changes (Feb 26, afternoon)

**Scav → ScavieFae**: 10 files changed, 325 insertions. Two new features plus infra fixes from the hackathon sprint. Need a second set of eyes before we launch the next GPU run — these changes affect training, encoding, and rollout.

### What changed (5 features, grouped by risk)

#### HIGH — affects training correctness

**1. Scheduled sampling** (`trainer.py`, `modal_train.py`, `mamba2-medium-gpu.yaml`)

Teaches robustness to autoregressive drift by corrupting context frames during training.

- `_corrupt_context()` method: adds Gaussian noise to continuous state values in last N context frames
- Per-sample mask (not every sample gets corrupted)
- Annealing: rate and noise scale ramp from 0 to target over `ss_anneal_epochs`
- Skips epoch 0 entirely (clean first epoch)
- Config: `scheduled_sampling=0.3`, `ss_noise_scale=0.1`, `ss_anneal_epochs=3`, `ss_corrupt_frames=3`
- `modal_train.py` now reads these from YAML and passes to Trainer

**Review questions:**
- Is the noise scale 0.1 reasonable? Context values are normalized (xy ×0.05, percent ×0.01) — 0.1 noise on top of 0.05-scale values is ~2 game units of position noise. Too much? Too little?
- The corruption only touches `continuous_dim` values (positions, velocities, dynamics). Never touches controller data or binary flags. Is that the right boundary?
- Annealing ramps linearly. Should it be cosine or exponential instead?

**What I HAVEN'T tested:** Never ran even 1 epoch with SS enabled. The logic is in place but I haven't verified it fires correctly or that loss still converges.

#### MEDIUM — new encoding dimensions

**2. Projectile encoding** (`parse.py`, `encoding.py`, `dataset.py`)

Infrastructure for item/projectile data (Fox laser, Samus charge shot, etc). Behind `projectiles: bool = False` flag.

- `parse.py`: New `FrameItems` dataclass, `_extract_items()` reads 15 item slots from parquet
- `encoding.py`: Per-player features when enabled: `nearest_dx`, `nearest_dy`, `n_active` (3 extra floats per player)
- `dataset.py`: Passes items through to encoding when flag is True
- Dimensions flow automatically: `continuous_dim` increases by 3 per player, all downstream math updates

**Review questions:**
- The model doesn't predict projectile features — they're input-only. During autoregressive rollout, predicted frames won't have projectile data to feed back as context. Is this OK? (Current answer: yes, because items are all zeros in current data. But when we train on ranked data with active projectiles, this becomes a real gap.)
- Should nearest-item distance use `xy_scale` (0.05) normalization? Or a different scale? Items can be far away.
- I compute nearest item per player independently. Two players could have the same "nearest item." That's correct game-mechanically (a projectile between two characters threatens both) but worth flagging.

**Update (Feb 26):** Item extraction fixed in commit `86de069`. Existing 22K parquet files still have empty items — re-parsing required to populate them. New parses will have real item data.

#### MEDIUM — policy + rollout fixes

**3. Config-driven PolicyMLP** (`policy_mlp.py`, `policy_dataset.py`, `train_policy.py`)

Fixed hardcoded dimension assumptions that broke with `state_age_as_embed=True`.

- PolicyMLP: Uses `cfg.int_per_player` for column indices instead of hardcoded 7. Conditionally adds state_age embedding.
- PolicyFrameDataset: Computes controller offset from `cfg.continuous_dim + cfg.binary_dim` instead of hardcoded 16. Accepts `cfg` parameter.
- train_policy.py: Builds EncodingConfig from YAML, filters loss_weights to only `{analog, button}` (was passing world model keys like `continuous`, `velocity` → TypeError).

**Review question:** The int column indexing in PolicyMLP (`int_ctx[:, :, ipp]` for p1_action) — does this match the int tensor layout in `_encode_game()`? The layout is: `[p0: action, jumps, char, l_cancel, hurtbox, ground, last_attack, (state_age_int)], [p1: same], stage`. So p1's action is at index `ipp` where `ipp = cfg.int_per_player` (7 or 8). This should be correct but is load-bearing.

**4. Rollout clamping** (`rollout.py`)

Clamps predicted frames to valid game ranges before they enter the simulation buffer.

- `CLAMP_RANGES` dict: percent 0-999, x ±300, y -200 to 300, shield 0-60, stocks 0-4, etc.
- `clamp_frame()`: applies clamping in normalized tensor space (multiplies ranges by scale factors)
- Called after model prediction, before appending to simulation buffer

**Review question:** I clamp `stocks` to [0, 4] in normalized space (`0.0, 1.0` after ×0.25 scaling). But stocks is in the dynamics head output, which the model predicts as absolute values. Is clamping stocks at the normalized level correct, or should it be in game units?

#### LOW — wiring only

**5. Checkpoint resume in modal_train.py**

`resume: str = ""` param on `train()`, resolves path from `CHECKPOINT_DIR`, passes to Trainer's existing `resume_from`. (This was ScavieFae's suggestion from the previous review.)

### Files changed

| File | Lines | What |
|------|-------|------|
| `training/trainer.py` | +67 | Scheduled sampling (params, `_corrupt_context()`, epoch arg) |
| `scripts/modal_train.py` | +21 | SS config passthrough, checkpoint resume |
| `model/encoding.py` | +56 | Projectile config flag, dims, encoding |
| `data/parse.py` | +48 | `FrameItems`, `_extract_items()`, items in `load_game()` |
| `data/dataset.py` | +5 | Pass items through to encode |
| `model/policy_mlp.py` | +71/-9 | Config-driven embeddings + state_age |
| `data/policy_dataset.py` | +43/-8 | Config-driven controller offset |
| `scripts/train_policy.py` | +13 | YAML config, loss_weights filter |
| `scripts/rollout.py` | +57 | `clamp_frame()`, `CLAMP_RANGES` |
| `experiments/mamba2-medium-gpu.yaml` | +4 | SS config values |

### Validation gaps (things I know I haven't tested)

1. **Scheduled sampling under load** — never ran training with it. `_corrupt_context()` is untested beyond "it doesn't crash when called with all-zero tensors."
2. **Projectile encoding with active items** — all current items are inactive. Never fabricated synthetic active items to verify nearest-item math.
3. **Model forward pass with projectile dimensions** — dataset returns wider tensors when `projectiles=True`, but I never pushed them through Mamba2 or MLP. The input projection may break on dimension mismatch.
4. **Rollout feedback with projectiles** — projectile features are context-only (not predicted). During autoregressive rollout, those columns would be zero in predicted frames, creating a train/infer mismatch.

### How to validate

```bash
# 1. Scheduled sampling smoke test (CPU, ~60s)
.venv/bin/python -m worldmodel.scripts.train --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
  --config worldmodel/experiments/mamba2-medium-gpu.yaml --max-games 50 --epochs 3 --device cpu --no-wandb

# 2. Verify encoding dimensions round-trip
.venv/bin/python -c "
from worldmodel.model.encoding import EncodingConfig
for proj in [False, True]:
    cfg = EncodingConfig(state_age_as_embed=True, projectiles=proj)
    print(f'projectiles={proj}: continuous={cfg.continuous_dim} float/player={cfg.float_per_player} float/frame={cfg.float_per_player*2}')
"

# 3. Policy training with state_age_as_embed (verifies PolicyMLP + PolicyFrameDataset)
.venv/bin/python -m worldmodel.scripts.train_policy --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
  --config worldmodel/experiments/mamba2-medium-gpu.yaml --max-games 50 --epochs 2 --no-wandb
```

---

## ScavieFae Sprint Review Response (Feb 26, evening)

Reviewed commit `50c1e1c` — 10 files, 325 insertions. All 5 features approved with notes.

### 1. Scheduled Sampling — APPROVE with one concern

**The code is correct.** Clean separation (only corrupts continuous state, never controller/binary), per-sample mask, epoch-0 skip, annealing on both rate and noise scale. The `.clone()` on line 224 is necessary and present — without it, noise would corrupt the DataLoader's underlying tensor.

**Concern: heterogeneous noise impact.** The uniform `ss_noise_scale=0.1` hits features very differently due to normalization:

| Feature | Scale | 0.1 noise in game units | Severity |
|---------|-------|------------------------|----------|
| position (x, y) | ×0.05 | 2 game units | trivial |
| percent | ×0.01 | 10% damage | moderate |
| hitlag | ×0.1 | 1 frame | moderate |
| **stocks** | **×0.25** | **0.4 stocks** | **large** |
| velocity | ×0.05 | 2 units/frame | small |
| combo_count | ×0.1 | 1 combo | moderate |

Stocks is the problem — ±0.4 stocks of noise is nearly half a life. The model might learn to distrust stocks information in context, which would hurt predictions in actual KO scenarios. Two options:

- **Quick fix**: Mask dynamics indices out of the corruption (only corrupt core_continuous + velocity). This is the safest approach — dynamics values (hitlag, stocks, combo) are already absolute predictions, not deltas, so they're less prone to drift.
- **Proper fix**: Per-feature-group noise scales (e.g., 0.1 for positions, 0.02 for dynamics). More work, do later.

**Scav's review questions answered:**
- Is 0.1 noise reasonable? → For positions/velocity, yes. For stocks/hitlag, too aggressive. See table above.
- Only corrupting continuous (never controller/binary)? → Correct boundary. Controllers are ground-truth conditioning and binary flags are discrete.
- Linear vs cosine annealing? → Linear is fine for 3-epoch ramp. Difference is marginal.

**Before launching overnight with SS enabled**: Run the smoke test from the handoff doc. 50 games, 3 epochs, CPU. Verify loss still converges. This is non-negotiable — `_corrupt_context()` has literally never run under load.

### 2. Projectile Encoding — APPROVE

Clean infrastructure. `FrameItems` dataclass, `_extract_items()` parsing, nearest-item distance computation — all correct.

**Key verification**: Both players compute distance to items independently using their own positions (`pf.x[:, None]`). Same item can be nearest to both players — correct game mechanically.

**Scav's questions answered:**
- Input-only features zeroing during rollout? → Acknowledged gap. Not an issue tonight (all items inactive in 22K data). When items go live, could forward-propagate item positions from replay data during rollout (items move independently of player state). Note for later.
- `xy_scale` for item distances? → Correct. Items share the same coordinate space.
- Nearest-item hardcoded scale `n_active * 0.1`? → Fine. Typically 0-3 active items. If this ever matters, promote to a cfg field, but not now.

### 3. Config-driven PolicyMLP — APPROVE

**I traced the int tensor layout end-to-end.** This is the load-bearing question.

`_encode_game()` builds int columns as:
```
[p0: action(0), jumps(1), char(2), l_cancel(3), hurtbox(4), ground(5), last_attack(6), [state_age(7)],
 p1: action(ipp), jumps(ipp+1), ..., last_attack(ipp+6), [state_age(ipp+7)],
 stage(ipp*2)]
```

Where `ipp = cfg.int_per_player` = 7 (default) or 8 (state_age_as_embed).

PolicyMLP's forward pass uses `int_ctx[:, :, ipp]` for p1_action → correct for both configurations.

The state_age embedding indices (hardcoded 7 for p0, `ipp+7` for p1) are also correct:
- p0 state_age is always at index 7 (8th column)
- p1 state_age is at `ipp + 7` = `8 + 7` = 15 ✓

**train_policy.py loss filtering** (lines 198-201): Filtering `loss_cfg` to only `{analog, button}` prevents TypeError from world model loss keys. Clean fix.

### 4. Rollout Clamping — APPROVE

**Stocks clamping is correct.** The clamping operates in normalized tensor space: `stocks` is stored as `game_stocks × 0.25`, so clamping to `[0 × 0.25, 4 × 0.25]` = `[0.0, 1.0]` correctly enforces 0-4 stocks in game units.

Dynamics indices (`dyn_start` computation) matches the pattern in `MeleeFrameDataset.__init__` — consistent across the codebase.

No velocity clamping, which is fine — knockback velocities can be extreme but not infinite, and position clamping catches the downstream effect.

### 5. Checkpoint Resume — APPROVE

My suggestion from previous review, cleanly implemented. Path resolution from `CHECKPOINT_DIR`, FileNotFoundError on missing, passed to Trainer's existing `resume_from`. Nothing to flag.

### Summary

| Feature | Verdict | Blocking? |
|---------|---------|-----------|
| Scheduled sampling | Approve with noise concern | **Before overnight**: run 50-game smoke test |
| Projectile encoding | Approve | No — flag is off |
| PolicyMLP config-driven | Approve | No |
| Rollout clamping | Approve | No |
| Checkpoint resume | Approve | No |

**One action item for Scav**: Either mask dynamics out of SS corruption, or lower `ss_noise_scale` to 0.03. Then run the smoke test. After that, overnight runs are good to go.

---

## History: Original Code Review (all items resolved)

### Code Fixes (all done)

1. ~~Sweep checkpoint collision~~ — **Done**. `save_dir` uses `{CHECKPOINT_DIR}/{run_name}`.
2. ~~`MeleeDataset.__new__` reconstruction~~ — **Done**. `from_tensors()` classmethod added.
3. ~~Missing error raise on bad data~~ — **Done**. `return` → `raise FileNotFoundError`.
4. ~~Config validation~~ — **Done**. `train()` raises `ValueError` on config mismatch.

### Doc Fixes (all done)

5. ~~Pre-encode memory estimate~~ — **Done**. Corrected to ~25-30K at 16GB.
6. ~~No-epoch-completed framing~~ — **Done**. Updated with real results.

### Milestones

- **First cloud GPU epoch ever** (Feb 25 ~12:08 PT): `mamba2-first-complete`, 61min, loss=0.4698, change_acc=0.519
- **First 2-epoch completion** (Feb 25 ~1:30 PT): `smoke-nw4-v2`, 91min total, loss=0.2873, change_acc=0.672
- **num_workers=4 validated**: 25% speedup (46min vs 61min per epoch)
- **287,504 ranked games parsed** on ScavieFae (13x current training set)
