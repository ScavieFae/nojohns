# Overnight Training Plan — Feb 25, 2026

**Goal**: By 9 PM, have enough running to make a strong overnight bet on the world model. Mattie at AI summit, runs unattended until morning.

## Process Readiness

### Data inventory

| Source | Games | Location | Status |
|--------|-------|----------|--------|
| Existing parsed-v2 | 22,162 | `~/nojohns-training/data/parsed-v2/` | Ready, on Modal volume |
| ScavieFae ranked (zip) | 171,405 | `queenmab:~/data/parsed-ranked/games/` | **Done** |
| ScavieFae ranked (7z) | 116,099 | `queenmab:~/data/parsed-ranked/games/` | **Done** |
| **Total ranked** | **287,504 games** | queenmab | Parsed, meta.json written |

The zip batch had a peppi_py panic at the end (assertion failed) but completed — 287,504 game files on disk, 116,099+171,405 parsed with only 149 failures. This is massive. We have **13x more data than our current training set.**

### Staging ranked data to Modal

The 287K games are on ScavieFae's machine. Getting them to Modal:

1. **Tar on ScavieFae**: `ssh queenmab "cd ~/data && tar cf /tmp/parsed-ranked.tar parsed-ranked/"` (~34GB based on log output)
2. **Transfer to Scav**: `scp queenmab:/tmp/parsed-ranked.tar /tmp/` (Tailscale, ~10-15 min)
3. **Upload to Modal**: `.venv/bin/modal volume put melee-training-data /tmp/parsed-ranked.tar /parsed-ranked.tar`
4. **Pre-encode on Modal**: `.venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode --config ... --max-games 100000 --output /encoded-100k.pt`

**Problem**: pre_encode needs ~65GB RAM for 100K games (both raw + encoded tensors in memory). Modal's `memory=16384` caps at ~25-30K. Options:
- **A**: Encode in chunks (25K at a time), merge `.pt` files. Requires new code.
- **B**: Request `memory=65536` on pre_encode. Simple, ~$0.50 for the function call.
- **C**: Pre-encode locally on Scav (36GB RAM, handles ~50K). Upload the .pt.

**Recommendation**: Option B for up to 50K games. For 100K+, we need Option A (chunked encoding). Start with 50K tonight, that's already 2.5x our biggest run.

### SOP durability

| Item | Status | Notes |
|------|--------|-------|
| RUNBOOK GPU section | Current | Updated with real timing data |
| MODAL-REVIEW-RUNBOOK.md | Current | Has Scav-2 feedback + milestone |
| HANDOFF-MODAL.md | Current | All 5 items checked off |
| worldmodel/CLAUDE.md | Current | Agent roles, key files, coordination |
| `/gpu-train` skill | Exists | Not tested recently |
| Timeout | **Needs bump** | Changed to 14400s locally, not yet pushed |

### Multi-GPU

Sweep function exists (`train.spawn()`) but is **untested**. For overnight:
- We don't need sweep for the main bet — one strong run is better than three mediocre ones
- Useful for K10 vs K60 comparison (2 parallel runs, ~$6 each, 2 hours)
- Risk: checkpoint collision was just fixed but sweep hasn't been tested end-to-end

## Model Evaluation

### A100 results vs MPS baselines

| Model | Data | Epochs | change_acc | action_acc | pos_mae | val_loss | Device |
|-------|------|--------|------------|------------|---------|----------|--------|
| MLP 1.3M | 2K | 2 | 64.5% | 96.4% | 0.79 | — | MPS |
| MLP 1.3M | 22K | 4 | **77.5%** | 97.4% | 0.79 | — | MPS |
| Mamba-2 4.3M | 2K | 1 | 52.1% | 95.2% | 0.77 | 0.342 | A100 |
| **Mamba-2 4.3M** | **2K** | **2** | **67.2%** | **96.7%** | **0.64** | **0.290** | **A100** |

**Epoch 2 flips the story.** After 1 epoch, Mamba-2 trailed MLP on change_acc (52.1% vs 64.5%). After 2 epochs, it pulls ahead: **67.2% vs 64.5%** (+2.7pp). And it's better on pos_mae (0.64 vs 0.79) and action_acc (96.7% vs 96.4%).

This is directionally correct and encouraging:
- **Mamba-2 > MLP at equal data+epochs** — the architecture wins even at K=10
- **Still well below MLP@22K ceiling** (77.5%) — data scaling is likely the main lever
- **pos_mae improvement** (0.77→0.64 after epoch 2) suggests the SSM state accumulates useful physics info
- K=60 comparison is still worthwhile but less urgent — Mamba-2 isn't broken, it just needed more training

**Updated key question**: How fast does Mamba-2 close the gap with more data? Does K=60 accelerate convergence or just cost more per epoch?

### K10 vs K60 comparison — queue now

Two parallel runs on Modal, 2K games, 2 epochs each:

```bash
# Run 1: K=10 (current config, for direct comparison)
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
    --encoded-file /encoded-2k.pt --epochs 2 --run-name "k10-compare"

# Run 2: K=60 (needs new pre-encoded file with K=60 context)
# NOTE: encoded-2k.pt was encoded with the medium-gpu config.
# K=60 needs a separate encode because MeleeFrameDataset slices differ.
```

**Wait** — the `.pt` file contains raw game tensors, not frame datasets. The `context_len` is applied at `get_frame_dataset()` time. So the same `encoded-2k.pt` works for both K=10 and K=60. We just need a K=60 config that points to the same data.

**Action**: Create `mamba2-medium-gpu-k60.yaml` (copy of medium-gpu, change context_len=60, chunk_size=30), then launch both in parallel.

### 15M parameter model estimates

From GPU-RUN-PLAN.md sketch: 6-layer Mamba-2, d_model=512, ~15M params.

**Scaling estimates** (extrapolating from 4.3M model at 2K games):

| Scenario | Games | Epochs | Est. per-epoch | Est. total | Est. cost |
|----------|-------|--------|----------------|------------|-----------|
| 4.3M @ 2K (observed) | 2,000 | 1 | 46 min | 46 min | ~$2.13 |
| 4.3M @ 22K (projected) | 22,000 | 1 | ~8.4 hr | 8.4 hr | ~$23 |
| **15M @ 22K** | 22,000 | 1 | ~25 hr* | 25 hr | ~$70 |
| **15M @ 50K** | 50,000 | 1 | ~57 hr* | 57 hr | ~$158 |
| **15M @ 100K** | 100,000 | 1 | ~114 hr* | 114 hr | ~$316 |

*15M is ~3.5x the params of 4.3M. Forward/backward scales roughly linearly with params for SSMs.

**For overnight completion targets:**

| Deadline | Config | Games | Epochs | Feasibility |
|----------|--------|-------|--------|-------------|
| 6 hours | 4.3M | 22K | 1 | **Possible** but tight (8.4 hr projected — maybe with num_workers) |
| 6 hours | 4.3M | 2K | 10 | **Yes** — ~7.7 hr without workers, ~5.7 hr with |
| 12 hours | 4.3M | 22K | 1 | **Yes** — comfortable margin |
| 12 hours | 15M | 2K | 2 | **Maybe** — ~5.4 hr projected, untested |
| 24 hours | 15M | 22K | 1 | **Maybe** — ~25 hr is tight |
| 24 hours | 4.3M | 50K | 3 | **Possible** — if pre-encode + upload complete in time |

### Prerequisites for 15M run

1. **Write `mamba2-large.yaml`** — d_model=512, n_layers=6, d_state=64, headdim=64, chunk_size=10 (or 30 for K=60)
2. **K decision** — K10 vs K60 comparison results needed first
3. **Pre-encode at scale** — 22K games encoded to `.pt`, uploaded to Modal volume (already have `encoded-2k.pt`, need `encoded-22k.pt`)
4. **Timeout**: 14400s (4hr) is too short for 15M@22K. Need 86400s (24hr) for overnight.
5. **Memory**: A100 40GB should handle 15M params at batch_size=512. May need to drop to 256.

## The Overnight Bet

### Conservative (high confidence)
**4.3M Mamba-2, 22K games, 5 epochs, K=10 and K=60 in parallel.**
- Pre-encode 22K games on Modal (~$0.50, 10 min)
- Two parallel A100 runs (~$30 each, ~12 hours)
- Total: ~$61, answers the K question definitively, gets us past MLP ceiling or proves we need more
- **Risk**: 22K encode hasn't been tested on Modal yet

### Moderate (good odds)
**4.3M Mamba-2, 50K games, 5 epochs.**
- Requires staging ranked data from ScavieFae → Modal (~1 hour transfer + upload)
- Pre-encode 50K on Modal with `memory=65536` (~$1, 20 min)
- One A100 run (~$70, ~24 hours)
- **Risk**: New data untested, pre-encode at scale untested

### Aggressive (go big)
**Write 15M config, 22K games, 3 epochs.**
- Write config, pre-encode 22K, launch
- One A100 run (~$70, ~18 hours projected)
- **Risk**: 15M model is completely untested. Could OOM, could diverge, could need LR tuning.

### Recommendation

**Do the conservative bet PLUS write the 15M config.**

1. **Now**: Create K=60 config, launch K10 vs K60 comparison (2K games, 2 epochs, parallel)
2. **By 6 PM**: Pre-encode 22K games on Modal
3. **By 8 PM**: Review K10 vs K60 results (should be done in ~2 hours)
4. **By 9 PM**: Launch overnight: best K × 22K games × 5+ epochs
5. **Stretch**: If K results are clear early, also launch 15M on 2K games as a validation

## Action Items (Priority Order)

1. [x] Create `mamba2-medium-gpu-k60.yaml` — **Done**. Same architecture as medium-gpu, context_len=60, chunk_size=30, batch_size halved to 512.
2. [ ] Launch K10 vs K60 comparison (parallel, ~$6 total)
3. [ ] Pre-encode 22K games on Modal (`pre_encode --max-games 22000 --output /encoded-22k.pt`)
4. [x] Write `mamba2-large-gpu.yaml` (15M params) — **Done**. d_model=512, n_layers=6, lr=0.0003, batch_size=512.
5. [ ] Stage ranked data from ScavieFae (background, for tomorrow)
6. [x] Bump timeout to 86400s for overnight runs — **Done** in modal_train.py.
7. [ ] Launch overnight bet based on K results
