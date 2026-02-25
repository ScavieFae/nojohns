# Modal Pipeline Handoff — ScavieFae Review

**Branch**: `scav/combat-context-heads`
**Date**: Feb 25, 2026 (evening)
**Status**: Pipeline validated. Preparing overnight training runs. Need review before launch.

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
