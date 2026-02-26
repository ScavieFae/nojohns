# World Model Roadmap

**Single source of truth for improvements, experiments, and moonshots.**
Previously scattered across GPU-RUN-PLAN.md, FEATURE-DIAGNOSTICS.md, ACTION-PREDICTION-DIRECTIONS.md, GGPO-ROLLBACK-WORLDMODEL.md, autonomous-ecs-scratch.md, OVERNIGHT-PLAN.md, and RUNBOOK.md.

Last updated: Feb 25, 2026

---

## Task Calendar — 48hr Sprint to Friday (Feb 27 deadline)

### Tuesday night (Feb 25, now → midnight)
- K10 vs K60 results come in (running)
- 22K pre-encode finishes (running)
- Pick K, launch **4.3M × 22K × 5 epochs** overnight (~$50-60, ~12hr)
- While overnight runs: implement **checkpoint resume (#2)** and **rollout clamping (#4)** — both are small

### Wednesday morning (Feb 26, wake up)
- Check overnight results on wandb
- Evaluate: does 22K move the needle? (MLP went 64.5% → 77.5% on the same jump)
- If yes → data scaling works. Start staging 287K ranked data from ScavieFae.
- If flattening → architecture bottleneck. Pivot to 15M immediately.

### Wednesday day (Feb 26)
- **Projectile encoding (#1)** — biggest encoding gap, data already in parquet
- **Scheduled sampling (#3)** — train the model to recover from its own errors
- **Pre-encode at new scale** — 50K+ games if staging ranked data
- Launch **Wednesday overnight run**: best config so far + new improvements

### Thursday morning (Feb 27 eve)
- Evaluate Wednesday overnight results
- If capacity allows: launch **15M model smoke test** on 2K games (~3hr, ~$10)
- Cherry-pick from experiments list if time: **hierarchical action loss (#6)** is cheapest bang
- **INT8 QAT (#5)** if onchain demo is part of the hackathon deliverable

### Friday (Feb 27 — deadline)
- Results in hand. Best checkpoint downloaded.
- Rollout demos generated with clamping
- Ship whatever we've got

### What we're racing against
Each overnight run is 1 iteration. We get **2 overnight runs** (Tue night, Wed night) and maybe a **daytime run** Wed if configs are ready early. Every improvement that isn't ready before a launch window waits 12+ hours for the next one.

---

## Now: Active (Feb 25 evening)

### K=10 vs K=60 comparison
**Status**: Running on Modal. K=10 done (67.3% change_acc, 2ep). K=60 in progress.
**Purpose**: Does 1 second of context unlock SSM, or is K=10 enough?
**Decision**: If K=60 >= 5pp better, use K=60 overnight. Otherwise K=10 (cheaper).
**Source**: OVERNIGHT-PLAN.md

### 22K pre-encode
**Status**: Running on Modal (CPU). First attempt timed out at 3600s, relaunched with 14400s + 32GB.
**Purpose**: Produce `encoded-22k.pt` for overnight training.
**Source**: OVERNIGHT-PLAN.md

### Overnight scaling bet
**Status**: Waiting for K results + pre-encode.
**Decision tree**: Best K × 22K games × 5 epochs (~$50-60, ~12hr).
**Source**: OVERNIGHT-PLAN.md, HANDOFF-MODAL.md

---

## Next: Before 15M run

### 1. Projectile encoding
**Priority**: High — the biggest encoding gap
**What**: Fox lasers, Samus charge shots, etc. Model sees damage appear from nowhere.
**Data**: Already in parquet files (`items` field: up to 15 × {exists, type, state, x, y}). No re-parsing.
**Approach**: Fixed-size summary per player — nearest projectile (type, distance, velocity) or count + closest position. Variable-length is too complex for now.
**Cost**: ~16 dims per player, small param increase.
**Source**: GPU-RUN-PLAN.md:91-94, RUNBOOK.md encoding section

### 2. Checkpoint resume on preemption
**Priority**: High — $23/epoch wasted on preemption without this
**What**: Add `resume: str = ""` param to `train()` in modal_train.py. Load model state_dict + optimizer state from checkpoint before training starts.
**Cost**: ~20 lines of code.
**Source**: HANDOFF-MODAL.md (ScavieFae suggestion)

### 3. Scheduled sampling
**Priority**: Medium — key for autoregressive quality
**What**: During training, sometimes feed model's own predictions instead of ground truth. Teaches recovery from compounding errors.
**Approach**: Start with 0% self-feeding, anneal to 20-50% over epochs. Only on continuous heads (position, velocity) — action/jumps stay teacher-forced.
**Cost**: Moderate code change in trainer.py.
**Source**: GPU-RUN-PLAN.md

### 4. Rollout clamping
**Priority**: Medium — cheap, prevents obviously wrong rollouts
**What**: Clamp predictions to valid ranges during autoregressive rollout. Positions inside blast zones, stocks 0-4, percent 0-999%, shield 0-60.
**Cost**: ~30 lines in rollout.py.
**Source**: GPU-RUN-PLAN.md

---

## Experiments: Queued

### 5. INT8 quantization-aware training
**Priority**: Medium — serves two purposes
**What**: During training, simulate INT8 quantization noise on activations and/or weights. Fake-quantize: round to INT8 grid, add straight-through estimator for gradients.
**Why dual-purpose**:
- **Onchain path**: INT8 is required for Solana CU budget. Training with quantization noise means the model is pre-adapted — no accuracy cliff at deploy time.
- **Regularization**: Quantization noise acts as a form of dropout/noise injection. May improve generalization independently of the onchain goal.
**Approach**: Add `quantize_noise: bool` flag to config. When enabled, fake-quantize linear layer outputs during forward pass (training only). Per-channel scale factors, INT8 range [-128, 127].
**Benchmark**: Compare float32 vs QAT model on val metrics. If QAT degrades >3%, investigate mixed precision (keep SSM scan in float, quantize projections only).
**Source**: autonomous-ecs-scratch.md:56-99

### 6. Hierarchical action loss (Test B)
**Priority**: Medium — cheapest action prediction improvement
**What**: Keep 400-class CE head, add auxiliary 20-class functional category loss at 2.0× weight. Categories: movement, attack, shield, grab, aerial, special, etc.
**Expected**: +2-4pp change_acc.
**Cost**: Add category mapping table + one extra CE loss. ~50 lines.
**Source**: ACTION-PREDICTION-DIRECTIONS.md (Test B)

### 7. Action embedding regression (Test D)
**Priority**: Medium — smoother errors even if accuracy doesn't jump
**What**: Replace or supplement 400-class CE with 32-dim MSE on action embeddings. Cross-category errors become small embedding distances instead of hard misclassifications.
**Expected**: -2 to 0pp change_acc but qualitatively better errors (similar actions confused, not random).
**Cost**: Moderate — need to decide embedding source (learned vs pretrained).
**Source**: ACTION-PREDICTION-DIRECTIONS.md (Test D)

### 8. 3-frame lookahead (Exp 3b)
**Priority**: Low — interesting but trades accuracy types
**What**: Predict frame t+3 given ctx + ctrl(t:t+3). Model sees 3 frames of future intent.
**Expected**: +6-12pp action change_acc but pos_mae increases to 0.82-0.90 (physics harder to predict 3 frames out).
**Tradeoff**: Better for decision understanding, worse for physics sim.
**Source**: ACTION-PREDICTION-DIRECTIONS.md (Exp 3b)

### 9. Press events in all context frames (Exp 2b)
**Priority**: Low — 2a was marginal, but 2b might be different
**What**: Add 16 binary "just pressed" features to ALL context frames, not just next_ctrl.
**Expected**: +5-8pp (original prediction), but 2a showed only +1.9pp for next_ctrl-only. Full context might unlock more.
**Source**: ACTION-PREDICTION-DIRECTIONS.md (Exp 2b)

---

## Encoding improvements: When error analysis demands

### 10. animation_index
**What**: Character-specific animation ID. Fox kick vs Marth sword on shared action_state.
**Cost**: +13% params, no re-parsing.
**Gate**: Include if mixed-character errors >3pp worse than single-character AND errors concentrate on shared action_states.
**Source**: FEATURE-DIAGNOSTICS.md:13-58

### 11. state_flags (40 bits)
**What**: Melee state flags — reflect active, shield reflect, fastfall, hitbox connected, offscreen, dead, sleep mode.
**Cost**: +7.9% params. Requires re-parsing (peppi_py 0.8.6 doesn't expose flags).
**Gate**: Include if hit-frame vs steady-frame error ratio >2x, or binary head `invulnerable` accuracy 5pp below other binary heads.
**Source**: FEATURE-DIAGNOSTICS.md:61-121

### 12. Hitbox frame data (external game tables)
**What**: Community frame data from ikneedata.com/ssbwiki for active frames, positions, knockback.
**Cost**: ~0% params (lookup table), days of data engineering.
**Gate**: Include if Phase 2 Mamba-2 converged but spatial attack resolution errors persist and accuracy gap between disjointed/close-range characters >5pp.
**Source**: FEATURE-DIAGNOSTICS.md:124-175

### 13. Stale move queue
**What**: Track last 9 moves that hit. Reduces damage by up to 0.45×. Affects knockback calculations.
**Cost**: Preprocessing from last_attack_landed + percent changes. Medium complexity.
**Gate**: Include if damage prediction errors correlate with repeated moves.
**Source**: GPU-RUN-PLAN.md, RUNBOOK.md

### 14. Embed dim tuning
**What**: action=32 (probably right), ground=4 (maybe too large), last_attack=8 (maybe needs 12-16).
**Gate**: Include if per-head error analysis shows specific categorical heads underperforming.
**Source**: FEATURE-DIAGNOSTICS.md:193-250

---

## Infrastructure: As needed

### 15. Chunked pre-encoding for large datasets
**What**: Encode in 25K-game chunks, merge .pt files. Required for >30K games on 16GB Modal instances.
**Alternative**: Request memory=65536 on Modal (~$0.50 extra per run). Handles up to ~50K.
**Gate**: Needed when we go past 50K games.
**Source**: OVERNIGHT-PLAN.md

### 16. Multi-GPU sweep end-to-end test
**What**: `sweep()` exists but is untested. Uses `train.spawn()` for parallel A100 runs.
**Gate**: Useful for hyperparameter searches once we have a stable config.
**Source**: MODAL-REVIEW-RUNBOOK.md

### 17. Local-vs-Modal correctness comparison
**What**: Same 50 games, same config, 1 epoch local vs 1 epoch Modal. Prove the pipeline produces identical results.
**Gate**: Should do before betting >$50 on a single run.
**Source**: MODAL-REVIEW-RUNBOOK.md (Scav-2 feedback)

### 18. Fix hardcoded dimension debt
**What**: policy_mlp.py, policy_dataset.py, rollout.py have hardcoded v2.2 column indices.
**Gate**: Must fix before running rollout or policy training on experiment checkpoints (1a encoding).
**Source**: RUNBOOK.md:525-539

---

## Moonshots: Phase 3+

### 19. GGPO-style speculative execution
**What**: Predict next frame assuming ctrl(t+1) = ctrl(t). Rollback if wrong. ~80% of frames correct (one pass), ~20% need rollback (two passes). Average 1.2× cost.
**Prereqs**: Mamba-2 converged, basic RL loop working, lookahead=0 baseline established.
**Source**: GGPO-ROLLBACK-WORLDMODEL.md

### 20. Paired (state, video) dataset
**What**: Map game state tensors to rendered pixels via headless Dolphin replay. "Sora for Melee."
**Scale**: 22K games → ~300 GB H.264, ~19 days single-Dolphin, ~3 days 6-parallel.
**Source**: RUNBOOK.md:727-769

### 21. Autonomous onchain world model
**What**: Deploy Mamba-2 on Solana via MagicBlock ephemeral rollups. INT8 inference in ~59M CU/frame.
**Status**: Design complete (autonomous-ecs-scratch.md), 7-phase plan. Deadline was Feb 27 (hackathon).
**Source**: autonomous-ecs-scratch.md

### 22. Zero-shot held-out character (Exp 4b)
**What**: Train on all characters except one rare one (Pichu, G&W). Test generalization.
**Expected**: 45-55% change_acc, category_acc 80-85%.
**Source**: ACTION-PREDICTION-DIRECTIONS.md (Exp 4b)

---

## Completed

| Item | Result | Date |
|------|--------|------|
| Combat context heads (l_cancel, hurtbox, ground, last_attack) | 4 new prediction heads, +52K params | Feb 24 |
| Exp 1a (state_age embed) | **+7.2pp** change_acc — Phase 1 winner | Feb 23 |
| Exp 2a (press events) | +1.9pp — marginal, subsumed by 1a | Feb 23 |
| Exp 3a (1-frame lookahead) | +6.3pp but trades action_acc | Feb 23 |
| MLP ceiling (22K, 4ep) | 77.5% change_acc — flattening | Feb 23 |
| Mamba-2 SSD chunked scan | 2.7× faster than sequential on MPS | Feb 24 |
| Modal pipeline (pre_encode, train, sweep) | Working, reviewed, documented | Feb 25 |
| First GPU epoch ever | 61min on A100, change_acc=51.9% | Feb 25 |
| num_workers=4 validation | 25% speedup (61→46 min/epoch) | Feb 25 |
| Mamba-2 beats MLP at equal scale | 67.2% vs 64.5% (2K, 2ep) | Feb 25 |
| 288K ranked games parsed | On ScavieFae, ready for staging | Feb 25 |
