# GPU Training Run Plan — "Play the Model"

**Goal:** Train a 15M-param Mamba-2 world model on 100K+ games so we can plug two virtual controllers in and "play" the model autoregressively.

**Status:** Pre-work. Blocked on: finishing K=10 vs K=60 comparison, projectile encoding, parsing new replays.

## Correction (Feb 25)

The original conversation (`gpu-rent-model-scratch.md`) claimed "the model predicts 18 values but feeds 180+" and that velocity/stocks/hitlag prediction heads needed to be added. **This was wrong.** The model already predicts 34 values (30 float + 4 int):
- `continuous_delta` (8): percent, x, y, shield deltas per player
- `velocity_delta` (10): 5 velocity components per player
- `dynamics_pred` (6): hitlag, stocks, combo per player
- `binary_logits` (6): facing, invuln, on_ground per player
- action logits (2 × 400-class) + jumps logits (2 × 7-class)

The rollout code (`scripts/rollout.py`) already feeds all of these back into the context window, including rules-based state_age updates. The autoregressive drift we observed (characters floating off stage, percent climbing to 500%+) is from **compounding prediction error in a small model**, not missing plumbing. The fix is scale: bigger model, more data, better architecture.

## Combat Context Heads (Feb 25)

We found 4 categorical fields that the model **read** from context but **never predicted** — the same class of problem that had been incorrectly attributed to velocity/dynamics. In autoregressive mode, these go stale (carry forward from the seed, never updating):

| Field | Vocab | What it captures |
|-------|-------|------------------|
| `l_cancel` | 3 | L-cancel success/fail/none |
| `hurtbox_state` | 3 | vulnerable/invulnerable/intangible |
| `ground` | 32 | surface ID (0 = airborne) |
| `last_attack_landed` | 64 | most recent attack that connected |

**Changes made across 6 files (~52K new params, old checkpoints still load):**

1. **`encoding.py`** — Added `target_int_dim` property (returns 12).

2. **`mlp.py` + `mamba2.py`** — 8 new `nn.Linear` heads (4 fields × 2 players), output added to forward dict:
   ```
   p0_l_cancel_head, p1_l_cancel_head       → (B, 3)
   p0_hurtbox_head, p1_hurtbox_head         → (B, 3)
   p0_ground_head, p1_ground_head           → (B, 32)
   p0_last_attack_head, p1_last_attack_head → (B, 64)
   ```

3. **`dataset.py`** — `int_tgt` expanded from `(4,)` to `(12,)`:
   ```
   [p0: action, jumps, l_cancel, hurtbox, ground, last_attack,
    p1: action, jumps, l_cancel, hurtbox, ground, last_attack]
   ```

4. **`metrics.py`** — 4 new CE losses (`l_cancel`, `hurtbox`, `ground`, `last_attack`) with weight 0.3 each. All `int_tgt` column indices updated to match new 12-column layout.

5. **`rollout.py`** — Feeds combat context predictions back into the integer context window, guarded with `if "p0_l_cancel_logits" in preds:` for old-checkpoint compatibility.

**What's now predicted vs carried forward:**

| Field | Status | Notes |
|-------|--------|-------|
| continuous (percent, x, y, shield) | Predicted (delta) | Always was |
| velocity (5 components) | Predicted (delta) | Always was |
| dynamics (hitlag, stocks, combo) | Predicted (absolute) | Always was |
| binary (facing, invuln, on_ground) | Predicted (logits) | Always was |
| action | Predicted (400-class CE) | Always was |
| jumps_left | Predicted (7-class CE) | Always was |
| l_cancel | **Predicted (3-class CE)** | NEW |
| hurtbox_state | **Predicted (3-class CE)** | NEW |
| ground | **Predicted (32-class CE)** | NEW |
| last_attack_landed | **Predicted (64-class CE)** | NEW |
| state_age | Rules-based | Increment if same action, reset on change |
| character, stage | Carried forward | Constants per game |
| controller input | External | Fed from replay or policy |
| **Projectiles** | **NOT IN ENCODING** | Data exists in parquet, not wired yet |

The model now predicts every field it consumes except character/stage (constants) and projectiles (encoding gap). Projectile encoding is the last remaining gap before the model has complete frame information.

## Decisions Made

| Decision | Answer | Rationale |
|----------|--------|-----------|
| GPU provider | **RunPod** | SSH-native, PyTorch templates, ~$1.79/hr for A100 80GB. Vast.ai cheaper but rougher. Lambda smoothest but sold out. |
| Model size | **Large: 6-layer, d_model=512, ~15M params** | 100K games saturate 1.15M. Go for ceiling. |
| Projectiles | **In** | Fox lasers/charge shots are invisible without them. Going big, go complete. |
| Budget | **~$100** | A100 80GB, ~15-20hrs for 10 epochs on 100K games |

## What Needs to Happen Before the Run

### ~~0. Combat context prediction heads~~ DONE (Feb 25)
All 4 stale-in-rollout fields now have prediction heads. See "Combat Context Heads" section above.

### 1. Decide K (context length)
- K=10 running locally (PID 34563, 22K games, resumed from epoch 2)
- K=60 should be on ScavieFae
- Compare results → pick one

### 2. Projectile encoding
Parquet data already has `items` (15 slots × {exists, type, state, x, y}). Our `parse.py` doesn't read them yet — no re-parsing needed, just wire them from parquet → encoding → model.

Approach TBD — probably fixed-size summary ("nearest projectile to each player" or "count + closest position") rather than variable-length.

### 3. Parse new replays
200GB zipped downloading → 80-100K+ games. CPU-bound parsing can run locally while GPU trains.
- Parse pipeline: `worldmodel/data/parse.py`
- Output to: `~/claude-projects/nojohns-training/data/parsed-v2/`

### 4. Write mamba2-large.yaml
```yaml
# Sketch — finalize after K decision
architecture: mamba2
d_model: 512
n_layers: 6
# K: TBD (10 or 60)
# chunk_size: must evenly divide K
state_age_as_embed: true
# projectiles: TBD encoding flags
```

### 5. Training stability for autoregressive use
- **Scheduled sampling**: During training, sometimes feed model its own predictions instead of ground truth. Teaches recovery from errors.
- **Clamping**: Keep positions inside blast zones, stocks 0-4, etc. during rollout.
- Both are training-time / rollout-time decisions — implement before the big run.

## RunPod Workflow

1. Create RunPod pod with PyTorch template (A100 80GB)
2. SSH in, clone repo, pip install
3. Upload parsed data to persistent network volume
4. Launch training, monitor via wandb
5. Same workflow as ScavieFae — we already know how to do this

## Cost Estimates

| Run | GPU | Time/epoch (100K) | 10 epochs | Cost |
|-----|-----|-------------------|-----------|------|
| Conservative | A100 | ~4-5 hrs | ~2 days | ~$80-90 |
| Fast | H100 | ~2-3 hrs | ~1 day | ~$60-70 |

## Open Questions

- Projectile encoding design — 15 slots is too many to embed directly. Fixed-size summary? Top-N nearest?
- Parquet `items` schema confirmed: `{exists: bool, type: uint16, state: uint8, x: float, y: float}` × 15 slots
- Scheduled sampling ratio — what % of training steps use model's own predictions?
- Should we do a medium-scale validation run (5M params, 20K games) before committing $100?
- Existing 22K parsed games do NOT need re-parsing — projectile data is already in the parquet files

## Also Done This Session (Feb 25)

- **ActionBreakdown + category accuracy** — per-action and per-category validation metrics (10 gameplay categories: idle, movement, aerial, ground_attack, aerial_attack, damage, shield_dodge, grab, edge, special). Logs worst-5 actions by error count and per-category accuracy to wandb.
- **CLAUDE.md split** — reduced from 675→172 lines. Gotchas, commands, and history moved to `docs/GOTCHAS.md`, `docs/COMMANDS.md`, `docs/HISTORY.md`.

## References

- Raw conversation: `worldmodel/docs/gpu-rent-model-scratch.md`
- Experiment results: MEMORY.md "Experiment Results" section
- Encoding details: MEMORY.md "Encoding — What We Have vs Don't Have"
- Autoregressive rollout: `worldmodel/scripts/rollout.py`
