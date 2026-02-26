# Plan: Parser v3 — Stage Fix + State Flags + Hitstun

*ScavieFae → Scav handoff. Issues #40, #41. Feb 26, 2026.*

## Context

Three parser bugs/gaps are causing the world model's biggest systematic failures. Demo playback error analysis pinpointed specific error patterns traceable to missing input features. All three fixes require a **full re-parse** of training data, so we're bundling them into a single parser revision.

| Issue | Bug | Impact on model |
|-------|-----|-----------------|
| **#40** | Stage ID always 0 in parsed data | Stage embedding is useless — can't distinguish platform layouts, blast zones, Randall |
| **#41** | No death/respawn signal | Largest errors in entire model: 226–316 unit position jumps at death transitions |
| **#41** | No hitstun countdown | Knockback trajectory undershooting (3–22 units), can't predict when control resumes |
| *(bonus)* | `invulnerable` always False | Dead binary feature — `build_dataset.py:92` hardcodes zeros despite `hurtbox_state` being available |

**None of this affects the current trained model.** All new features are behind `EncodingConfig` flags defaulting to `False`. Existing `.pt` caches and checkpoints remain compatible. The re-parse produces richer parquet files; the encoding layer decides what to use.

---

## What's Changing (Summary)

### Parser layer (parquet schema — new columns in player struct)

| New field | Type | Source | Per player |
|-----------|------|--------|------------|
| `state_flags_0..4` | uint8 × 5 | `raw_peppi.frames.field('post').field('state_flags').field('{0,1,2,3,4}')` | Yes |
| `hitstun_remaining` | float32 | `raw_peppi.frames.field('post').field('misc_as')` (bit-recovery needed) | Yes |
| `stage` (fix) | int64 | `game.start.stage` → `_build_parquet_table()` param | Per game |
| `invulnerable` (fix) | bool | `hurtbox_state != 0` (already parsed, just not wired) | Yes |

### Encoding layer (EncodingConfig flags)

| Flag | Features added | Type | Per player |
|------|---------------|------|------------|
| `state_flags=True` | `is_dead`, `is_dying`, `is_respawning`, `in_hitlag`, `in_knockback`, `is_shielding` | 6 binary | +6 floats |
| `hitstun=True` | `hitstun_remaining` | 1 continuous | +1 float |

### Dimension impact

| Config | `float/player` | `player_dim` | `frame_dim` | `float_tgt` | Param Δ (4.3M model) |
|--------|---------------|-------------|------------|------------|---------------------|
| **Current baseline** | 29 | 89 | 182 | 30 | — |
| + `state_flags` | 35 | 95 | 194 | 42 | +~62K (+1.4%) |
| + `hitstun` | 36 | 96 | 196 | 44 | +~72K (+1.7%) |
| + both | 36 | 96 | 196 | 44 | +~72K (+1.7%) |
| + both + `projectiles` | 39 | 99 | 202 | 44 | +~102K (+2.4%) |
| + both + `projectiles` + `state_age_embed` | 38 | 106 | 216 | 44 | +~174K (+4.0%) |

**Breakdown of `float_tgt` growth (30 → 44):**
- Binary targets: 6 → 18 (+12: 6 new state_flags × 2 players)
- Dynamics targets: 6 → 8 (+2: hitstun × 2 players)
- Continuous deltas: 8 (unchanged)
- Velocity deltas: 10 (unchanged)

**Parameter impact is negligible.** The +72K on a 4.3M model is <2%. On the 15M model it's <1%. The main benefit is richer input signal, not more parameters.

---

## Discovery Notes (color for Scav)

### We never told the model about death

The single largest error category in the entire demo playback is death → respawn transitions (226–316 unit position jumps). The model sees a character at the blast zone one frame and at the respawn platform the next, with zero signal that a death occurred. It just treats it as a very wrong position prediction.

`state_flags` byte 4 has a clean 2-bit death encoding:
- `0b11` (192) = dead (action states DEAD_DOWN, DEAD_LEFT, etc.)
- `0b10` (128) = dying/in kill trajectory
- `0b01` (64) = respawning (ENTRY, ON_HALO_DESCENT)
- `0b00` (0) = alive and normal

This was verified on a real 13,607-frame replay. The old decision to skip state_flags ("inferrable from action_state") was wrong — the model proves this by failing catastrophically at every death.

### Hitstun is recoverable despite garbage values

Old notes flagged `misc_as` as garbage (values like 5.6e-45). Turns out ~42% of non-zero values are integers bit-reinterpreted as float32 — they're subnormal floats that look like garbage but are actually clean countdown timers. Reinterpreting the bytes as uint32 recovers them perfectly: frames 330–349 show hitstun counting down 33→14 in a clean sequence.

### invulnerable has been dead since day one

`build_dataset.py:92` says `invulnerable = np.zeros(num_frames, dtype=bool)` with the comment "Not in peppi_py post." But `hurtbox_state` IS in peppi_py post (line 126 extracts it!). `parse_archive.py` already derives invulnerable correctly from hurtbox_state. One-liner fix.

### Stage was computed but never written

`build_dataset.py:296` computes `stage_val = int(start.stage)` and puts it in `meta.json`, but line 220 writes `np.zeros(num_frames, dtype=np.uint8)` to the parquet. The stage value was right there the whole time.

---

## Files to Modify

### Tier 1: Parser (extract data → parquet)

| File | Changes | Risk |
|------|---------|------|
| `scripts/item_utils.py` | Add `extract_combat_context()` — state_flags + hitstun from raw arrow | Low — new function, no existing code modified |
| `scripts/build_dataset.py` | (1) Fix stage: pass `stage_val` to `_build_parquet_table()` (2) Fix invulnerable: derive from `hurtbox_state` (3) Add state_flags + hitstun columns to parquet schema (4) Call `extract_combat_context()` alongside `extract_items()` | Medium — primary parser |
| `scripts/parse_archive.py` | Same schema changes for parity | Low — secondary parser |

### Tier 2: Data loading (parquet → numpy)

| File | Changes | Risk |
|------|---------|------|
| `data/parse.py` | Add 6 fields to `PlayerFrame` dataclass (`state_flags_0..4`, `hitstun_remaining`). Read from parquet with `_safe_field()` fallback for old data. | Low — additive, old data gets zeros |

### Tier 3: Encoding (numpy → tensors)

| File | Changes | Risk |
|------|---------|------|
| `model/encoding.py` | (1) Add `state_flags` and `hitstun` flags to `EncodingConfig` (2) Update `binary_dim`, `continuous_dim`, `dynamics_dim` properties (3) Add `predicted_binary_dim` and update `predicted_dynamics_dim` to be config-driven (4) Extract 6 binary features from state_flags bytes in `encode_player_frames()` (5) Add hitstun to continuous block | Medium — dimension properties drive the whole pipeline |
| `data/dataset.py` | Update `_encode_game()` and `MeleeFrameDataset.__getitem__()` to handle variable binary/dynamics target sizes | Medium — target slicing must match new dimensions |

### Tier 4: Training (loss computation)

| File | Changes | Risk |
|------|---------|------|
| `training/metrics.py` | Make target slice indices config-driven (currently hardcoded `:8`, `8:18`, `18:24`, `24:30`) | Medium — wrong indices = silent training corruption |
| `model/mlp.py` | Change `binary_head` from hardcoded `6` to `cfg.predicted_binary_dim`. Change `dynamics_head` from `cfg.predicted_dynamics_dim` (currently hardcoded to return 6) to config-driven value. | Low — one-line each |
| `model/mamba2.py` | Same as mlp.py | Low |

### Not modified

- `model/base.py` — just docstrings, not functional
- `training/trainer.py` — loss computation is delegated to metrics.py
- `scripts/modal_train.py` — YAML configs drive everything
- `scripts/pre_encode.py` — uses `EncodingConfig` already

---

## Implementation Steps

### Step 1: Fix stage bug (#40) — `build_dataset.py`

One-liner + plumbing. Currently line 220:
```python
stage = pa.array(np.zeros(num_frames, dtype=np.uint8))  # "Filled from metadata"
```
But `stage_val` is computed at line 296 and never passed down. Fix:

1. Add `stage_id: int` parameter to `_build_parquet_table()`
2. Use it: `stage = pa.array(np.full(num_frames, stage_id, dtype=np.int64))`
3. In `parse_single_slp()`, pass `stage_val` to `_build_parquet_table()`

**Also verify** `parse_archive.py` — it already handles stage correctly (line 168 uses `stage_id` param). No fix needed there.

### Step 2: Fix invulnerable bug — `build_dataset.py`

Line 92:
```python
invulnerable = np.zeros(num_frames, dtype=bool)  # Not in peppi_py post
```

Replace with:
```python
invulnerable = (hurtbox_state != 0)  # hurtbox 1=invulnerable, 2=intangible
```

Note: `hurtbox_state` is already extracted at line 126. Just need to move the extraction before `invulnerable` or reorder. `parse_archive.py` already does this correctly (line 130).

### Step 3: Add `extract_combat_context()` to `item_utils.py`

New function using raw arrow access (same pattern as `extract_items()`):

```python
def extract_combat_context(slp_path: str, num_frames: int) -> dict | None:
    """Extract state_flags and hitstun from raw peppi_py arrow struct.

    Returns dict with per-port keys:
        p{0,1}_state_flags_{0..4}: (num_frames,) uint8 arrays
        p{0,1}_hitstun_remaining: (num_frames,) float32 array
    Returns None on failure.
    """
    try:
        import peppi_py._peppi as raw_peppi
        raw_game = raw_peppi.read_slippi(slp_path)
    except Exception:
        return None

    result = {}
    for port_idx in range(2):
        post = raw_game.frames.field('port').field(str(port_idx)) \
                   .field('leader').field('post')
        prefix = f"p{port_idx}_"

        # state_flags: 5 uint8 fields
        sf = post.field('state_flags')
        for byte_idx in range(5):
            arr = sf.field(str(byte_idx)).to_numpy().astype(np.uint8)
            result[f"{prefix}state_flags_{byte_idx}"] = arr[:num_frames]

        # misc_as: float32 field, needs bit-recovery for subnormals
        raw_misc = post.field('misc_as').to_numpy().astype(np.float32)
        subnormal = (np.abs(raw_misc) < 1e-10) & (raw_misc != 0)
        recovered = np.where(
            subnormal,
            np.frombuffer(raw_misc.tobytes(), dtype=np.uint32).astype(np.float32),
            raw_misc,
        )
        result[f"{prefix}hitstun_remaining"] = np.clip(recovered[:num_frames], 0, 200)

    return result
```

**Key detail — hitstun recovery:** peppi_py stores `misc_as` as float32, but some values are integers bit-reinterpreted as floats (producing subnormals like 5.6e-45). We detect subnormals and reinterpret the bytes as uint32 to recover clean countdown values (1, 2, 3, ..., 33, etc.).

**Optimization opportunity:** This function currently does a separate `read_slippi()` call. Could be combined with `extract_items()` to share the raw parse. For now, keep them separate — parse time isn't the bottleneck and it keeps the code cleaner.

### Step 4: Update parquet schema — both parsers

Add new columns to the player struct in `_build_parquet_table()`:

```python
# In the player struct, after existing combat context fields:
pa.array(p["state_flags_0"]), pa.array(p["state_flags_1"]),
pa.array(p["state_flags_2"]), pa.array(p["state_flags_3"]),
pa.array(p["state_flags_4"]),
pa.array(p["hitstun_remaining"]),
```

And add corresponding names to the struct field list. If `extract_combat_context()` returns None, fill with zeros (graceful fallback).

### Step 5: Update `PlayerFrame` dataclass — `parse.py`

Add 6 new fields:

```python
# State flags (5 raw bytes — encoding layer extracts specific bits)
state_flags_0: np.ndarray  # (T,) uint8
state_flags_1: np.ndarray  # (T,) uint8
state_flags_2: np.ndarray  # (T,) uint8
state_flags_3: np.ndarray  # (T,) uint8
state_flags_4: np.ndarray  # (T,) uint8
# Hitstun
hitstun_remaining: np.ndarray  # (T,) float32 — frames of hitstun left
```

In `_extract_player()`, read with `_safe_field()` — old parquet files without these columns will get zeros automatically.

### Step 6: Update `EncodingConfig` — `encoding.py`

```python
# New experiment flags (default False for backward compat)
state_flags: bool = False   # 6 binary features from state_flags bytes
hitstun: bool = False       # 1 continuous feature (hitstun countdown)
hitstun_scale: float = 0.02 # normalization: range 0-50 → 0-1

@property
def binary_dim(self) -> int:
    base = 3  # facing, invulnerable, on_ground
    if self.state_flags:
        base += 6  # is_dead, is_dying, is_respawning, in_hitlag, in_knockback, is_shielding
    return base

@property
def dynamics_dim(self) -> int:
    base = 2 if self.state_age_as_embed else 3  # hitlag, stocks [, state_age]
    if self.hitstun:
        base += 1  # hitstun_remaining
    return base

@property
def predicted_binary_dim(self) -> int:
    return self.binary_dim * 2  # both players

@property
def predicted_dynamics_dim(self) -> int:
    base = 6  # hitlag + stocks + combo, both players
    if self.hitstun:
        base += 2  # hitstun × 2 players
    return base
```

In `encode_player_frames()`, extract 6 binary features from raw bytes:

```python
if cfg.state_flags:
    sf4 = pf.state_flags_4
    is_dead = ((sf4 >> 6) & 3) == 3        # byte4 bits 6-7 = 0b11
    is_dying = ((sf4 >> 7) & 1).astype(bool)  # byte4 bit 7
    is_respawning = ((sf4 >> 6) & 1).astype(bool) & ~is_dying  # byte4 bit 6 only
    in_hitlag = ((pf.state_flags_1 >> 5) & 1).astype(bool)     # byte1 bit 5
    in_knockback = ((pf.state_flags_3 >> 1) & 1).astype(bool)  # byte3 bit 1
    is_shielding = (pf.state_flags_1 & 1).astype(bool)         # byte1 bit 0
    # Append to binary columns
    binary_cols = [pf.facing, pf.invulnerable, pf.on_ground,
                   is_dead, is_dying, is_respawning,
                   in_hitlag, in_knockback, is_shielding]
```

### Step 7: Update target construction — `dataset.py`

Make `MeleeFrameDataset.__init__()` compute target slice offsets from config:

```python
# Target float layout:
#   [cont_delta(8), vel_delta(10), binary(binary_dim*2), dynamics(predicted_dynamics_dim)]
self._tgt_cont_end = 8
self._tgt_vel_end = 18
self._tgt_bin_end = 18 + cfg.binary_dim * 2   # 24 or 36
self._tgt_dyn_end = self._tgt_bin_end + cfg.predicted_dynamics_dim  # 30 or 44
```

Update `__getitem__()` binary/dynamics target extraction to use config-driven slices.

### Step 8: Update metrics — `metrics.py`

Replace hardcoded target indices with config-driven slices:

```python
def __init__(self, cfg, weights=None):
    self.cfg = cfg
    self.weights = weights or LossWeights()
    # Target layout offsets
    self._cont_end = 8
    self._vel_end = 18
    self._bin_end = 18 + cfg.binary_dim * 2
    self._dyn_end = self._bin_end + cfg.predicted_dynamics_dim

def compute_loss(self, predictions, float_tgt, int_tgt, int_ctx=None):
    cont_delta_true = float_tgt[:, :self._cont_end]
    vel_delta_true = float_tgt[:, self._cont_end:self._vel_end]
    binary_true = float_tgt[:, self._vel_end:self._bin_end]
    dynamics_true = float_tgt[:, self._bin_end:self._dyn_end]
```

### Step 9: Update model output heads — `mlp.py`, `mamba2.py`

```python
# Replace:
self.binary_head = nn.Linear(trunk_dim, 6)
# With:
self.binary_head = nn.Linear(trunk_dim, cfg.predicted_binary_dim)

# predicted_dynamics_dim property already used by dynamics_head,
# just needs its return value to be config-aware (Step 6 handles this)
```

---

## Verification Plan

### After parser changes (Steps 1-4):

```bash
# Parse 20 games with updated parser
.venv/bin/python -m worldmodel.scripts.build_dataset \
    --input ~/data/raw-slps --output /tmp/test-v3 --limit 20

# Verify stage is no longer all zeros
.venv/bin/python -c "
from worldmodel.data.parse import load_game
from pathlib import Path
from collections import Counter
games = list(Path('/tmp/test-v3/games').iterdir())[:20]
stages = Counter(load_game(g).stage for g in games)
print(f'Stage distribution: {stages}')
assert 0 not in stages or len(stages) > 1, 'Stage still all zeros!'
"

# Verify state_flags and hitstun are populated
.venv/bin/python -c "
from worldmodel.data.parse import load_game
from pathlib import Path
g = load_game(next(Path('/tmp/test-v3/games').iterdir()))
sf4 = g.p0.state_flags_4
deaths = ((sf4 >> 6) & 3) == 3
print(f'Death frames: {deaths.sum()}/{len(sf4)}')
print(f'Hitstun non-zero frames: {(g.p0.hitstun_remaining > 0).sum()}/{len(sf4)}')
print(f'Invulnerable frames: {(g.p0.invulnerable > 0).sum()}/{len(sf4)}')
"
```

### After encoding changes (Steps 5-9):

```bash
# Full pipeline: parse → encode → dataset
.venv/bin/python -c "
from worldmodel.data.parse import load_games_from_dir
from worldmodel.data.dataset import MeleeDataset, MeleeFrameDataset
from worldmodel.model.encoding import EncodingConfig

cfg = EncodingConfig(state_flags=True, hitstun=True)
games = load_games_from_dir('/tmp/test-v3', max_games=5)
ds = MeleeDataset(games, cfg)
fds = ds.get_frame_dataset(context_len=10)

print(f'binary_dim={cfg.binary_dim} (expect 9)')
print(f'float_per_player={cfg.float_per_player} (expect 36)')
print(f'frame_dim={cfg.frame_dim} (expect 196)')
print(f'predicted_binary_dim={cfg.predicted_binary_dim} (expect 18)')
print(f'predicted_dynamics_dim={cfg.predicted_dynamics_dim} (expect 8)')

# Check a sample
fc, ic, nc, ft, it = fds[0]
print(f'float_ctx shape: {fc.shape} (expect [10, 72])')
print(f'float_tgt shape: {ft.shape} (expect [44])')
print(f'int_tgt shape: {it.shape} (expect [12])')

# Verify state_flags features aren't all zeros
bin_start = cfg.continuous_dim
death_idx = bin_start + 3  # is_dead is 4th binary feature
print(f'Death feature has non-zero: {(ds.floats[:, death_idx] != 0).any()}')
"
```

### Model smoke test:

```bash
.venv/bin/python -c "
from worldmodel.model.mlp import FrameStackMLP
from worldmodel.model.encoding import EncodingConfig
import torch

cfg = EncodingConfig(state_flags=True, hitstun=True)
model = FrameStackMLP(cfg, context_len=10, d_model=256, n_layers=2)
print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')

B = 4
fc = torch.randn(B, 10, cfg.float_per_player * 2)
ic = torch.randint(0, 10, (B, 10, cfg.int_per_frame))
nc = torch.randn(B, cfg.ctrl_conditioning_dim)
out = model(fc, ic, nc)
print(f'binary_logits: {out[\"binary_logits\"].shape} (expect [{B}, {cfg.predicted_binary_dim}])')
print(f'dynamics_pred: {out[\"dynamics_pred\"].shape} (expect [{B}, {cfg.predicted_dynamics_dim}])')
"
```

---

## Re-parse Requirements

**All training data must be re-parsed.** The new parquet schema includes state_flags and hitstun columns that don't exist in current files. While `_safe_field()` provides zeros for old data, zeros defeat the purpose.

- Current dataset: ~22K games in `parsed-v2/`
- Re-parse time estimate: ~2-3 hours (based on original parse performance)
- Re-encoding (`.pt` cache rebuild): ~10 min
- **No model architecture changes needed** — existing checkpoints are incompatible anyway since input dimensions change with new flags

**Recommendation:** Parse to a new `parsed-v3/` directory to avoid clobbering. Old data remains available for comparison runs.

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Raw arrow field names wrong for state_flags | Verified in previous session: `post.state_flags.{0,1,2,3,4}` confirmed present |
| Hitstun bit-recovery produces garbage | Validated on real replay: clean countdown values (33→14 over 20 frames). Clip to [0, 200] as safety. |
| Binary target size mismatch crashes training | Shape preflight in trainer.py catches this before epoch 1 |
| Old parquet files break with new PlayerFrame fields | `_safe_field()` returns zeros — fully backward compatible |
| `predicted_dynamics_dim` change breaks saved checkpoints | Expected — new flags mean new experiments, new checkpoints |
| Double raw parse (items + combat context) | Acceptable: parse is I/O-bound, not compute-bound. Can merge later if needed. |

---

## Deferred (not in this PR)

- **Randall position** (Yoshi's Story moving platform) — deterministic from frame index, no parser change needed. Requires encoding additions.
- **FoD platform heights** — available in raw arrow data but requires new encoding fields.
- **`animation_index`** — available but redundant with `action` state. Low priority.
- **`last_hit_by`** — player port that last hit you. Potentially useful for interaction modeling but requires new encoding category.

---

## Commit Strategy

Three commits for clean review:

1. **Fix stage + invulnerable bugs** — `build_dataset.py` only, two one-liners
2. **Add state_flags + hitstun extraction** — `item_utils.py`, both parsers, `parse.py`
3. **Encoding + training pipeline** — `encoding.py`, `dataset.py`, `metrics.py`, `mlp.py`, `mamba2.py`
