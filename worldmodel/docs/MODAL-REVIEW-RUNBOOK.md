# Modal Training Pipeline — Review Runbook

**For**: Another Claude instance reviewing the full GPU training pipeline.
**Last updated**: Feb 25, 2026

## Architecture Overview

```
Local machine                          Modal (cloud)
─────────────                          ─────────────
worldmodel/ source code                 Baked into container image via add_local_dir()
                                        ↓
                                   ┌─── pre_encode() ───┐
                                   │  CPU-only (4 core)  │
                                   │  Reads: /data/parsed-v2.tar (raw parquet)
                                   │  Writes: /data/encoded.pt (tensors)
                                   └─────────┬──────────┘
                                             │ volume commit
                                             ↓
                                   ┌─── train() ────────┐
                                   │  A100 40GB GPU      │
                                   │  Reads: /data/encoded.pt
                                   │  Writes: /data/checkpoints/
                                   │  Logs: wandb + stdout
                                   │  DataLoader: num_workers=4, persistent, prefetch=4
                                   └─────────┬──────────┘
                                             │ volume commit
                                             ↓
                                   ┌─── sweep() ────────┐
                                   │  Local entrypoint   │
                                   │  Calls train.spawn() N times
                                   │  Each gets own A100
                                   └────────────────────┘

Volume: melee-training-data (persistent across runs)
Secret: wandb-key (WANDB_API_KEY)
```

## Review Checklist

### 1. Data Pipeline Correctness

- [ ] **pre_encode reads the right data**: Tar auto-extraction uses `tarfile.extractall()` to `/data/`. Verify the tar contains `parsed-v2/meta.json` and `parsed-v2/games/` at the right nesting level.
- [ ] **Encoding config matches between pre_encode and train**: Both read from the same YAML config file. The `encoding_config` dict is saved in the `.pt` payload but **not validated against the config used at train time**. A mismatch here would produce silent wrong results.
- [ ] **MeleeDataset reconstruction in train()**: Uses `__new__` to bypass `__init__`, manually sets fields. Verify all required fields are set: `cfg`, `floats`, `ints`, `game_offsets`, `game_lengths`, `num_games`, `total_frames`.
- [ ] **game_offsets dtype**: Saved as `torch.tensor(dataset.game_offsets)`, loaded with `.numpy()`. Verify int64 precision is preserved (offsets can exceed 2^31 for large datasets).

### 2. DataLoader Configuration

- [ ] **num_workers=4 on CUDA**: Fork-mode workers share parent tensors via copy-on-write on Linux. This is correct and efficient for map-style datasets backed by contiguous tensors.
- [ ] **persistent_workers=True**: Workers stay alive between epochs — avoids re-fork overhead. Memory cost: 4 worker processes × ~200MB each.
- [ ] **prefetch_factor=4**: Each worker pre-loads 4 batches. With batch_size=1024 and 4 workers, that's 16 batches in flight. Verify A100 40GB can handle this (each batch is ~50MB → 800MB prefetch buffer, fine).
- [ ] **MPS/CPU stays at num_workers=0**: MPS doesn't benefit from multiprocess loading (single-device, shared memory is already fast). CPU training is rare and doesn't need it.
- [ ] **IterableDataset path**: StreamingMeleeDataset is IterableDataset — `shuffle=False` is set, `num_workers` still applies but workers split the iterable. Verify this doesn't cause data duplication.

### 3. Modal Configuration

- [ ] **Image**: `debian_slim(python_version="3.11")` + `pip_install("torch", index_url=cu121)`. Verify torch CUDA version matches A100 driver (cu121 is safe for all current Modal A100 images).
- [ ] **GPU type**: `gpu="A100"` — Modal auto-selects 40GB or 80GB. The code doesn't assume 80GB anywhere.
- [ ] **Timeout**: `train()` has 7200s (2hr). For 10-epoch runs this should be sufficient. For 20+ epochs, may need to increase.
- [ ] **pre_encode timeout**: 3600s (1hr). Encoding 22K games takes ~3-5 minutes. Encoding 300K games could take longer — check if this is sufficient for the full dataset.
- [ ] **pre_encode resources**: `cpu=4, memory=16384` (16GB RAM). 22K games encoded into tensors is ~4GB. 300K games would be ~50GB — this would OOM. Document the limit.
- [ ] **Volume commit**: Both `pre_encode` and `train` call `volume.commit()` after writing. Verify no data loss if the function crashes before commit.

### 4. Sweep Configuration

- [ ] **train.spawn()**: Launches asynchronous Modal functions. Each gets its own A100 container. Verify there's no resource contention (volume is read-only for train, write-only for checkpoints).
- [ ] **Checkpoint collision**: All sweep runs write to the same `CHECKPOINT_DIR` (`/data/checkpoints/`). If they all write `best.pt`, they'll overwrite each other. **This is a bug** — sweep runs should write to `{CHECKPOINT_DIR}/{run_name}/`.
- [ ] **wandb run names**: Each sweep run gets a unique name via the `--names` param. Verify wandb doesn't merge runs with different names.
- [ ] **Cost**: N runs × A100 hourly rate. 3 parallel runs × 2hr = ~$17. Document this clearly.

### 5. Error Handling

- [ ] **Missing encoded file**: `train()` prints error and returns. No exception raised — `modal run` exits cleanly with return code 0. Consider raising an exception for clearer failure signaling.
- [ ] **Missing tar in pre_encode**: Same pattern — prints error and returns.
- [ ] **wandb failure**: Caught with `try/except`, training continues without wandb. Good.
- [ ] **volume.commit() failure**: Not caught. If this fails, checkpoints are lost. Low risk (Modal volumes are reliable) but worth noting.

### 6. Backward Compatibility

- [ ] **Trainer num_workers param**: Default `None` → auto-detect. Existing code that doesn't pass `num_workers` gets the same behavior as before (0 on MPS/CPU).
- [ ] **train.py --num-workers**: Optional arg, defaults to None (pass-through to Trainer auto-detect).
- [ ] **modal_train.py train()**: Hardcodes `num_workers=4` (always CUDA). This is correct.
- [ ] **Existing .pt files**: pre_encode writes the same payload format as `pre_encode.py`. Old `.pt` files work with the new `train()`.

### 7. Cost Sanity

| Operation | Instance | Est. time | Est. cost |
|-----------|----------|-----------|-----------|
| pre_encode (22K games) | CPU-4 16GB | ~5 min | ~$0.05 |
| train (1 epoch, 2K games, no workers) | A100 40GB | ~61 min | ~$2.83 |
| train (10 epochs, 2K games) | A100 40GB | ~10 hr (est.) | ~$28 |
| sweep (3 runs, 10 epochs) | 3× A100 | ~1-2 hr | ~$9-18 |
| check_volume | CPU (minimal) | ~10 sec | ~$0.01 |

## How to Know Training Is Actually Running

**"Starting training..." followed by silence proves nothing.** It means init completed. It does not mean batches are computing. We've been fooled by this before.

Possible states after seeing "Starting training" with no further output:

| What you hope | What might actually be happening |
|---------------|----------------------------------|
| Batches computing, waiting for 10% log | DataLoader workers stuck on I/O, GPU idle |
| GPU crunching | `persistent_workers=True` fork hanging or deadlocked |
| Just slow | Silent CUDA OOM (kills process, no traceback reaches stdout) |
| Detach lag | Modal log streaming delayed — output exists but hasn't been forwarded |

**Proof levels (strongest first):**

1. **wandb has a data point** — a metric was logged, meaning a full epoch completed
2. **Batch log line in stdout** — `batch 1730/17300 (10%)` means at least 1,730 batches ran
3. **Modal dashboard shows GPU utilization > 0%** — compute is happening
4. **No crash after N minutes** — weakest signal. Only rules out immediate failure.

**Update Feb 25, 2026 ~12:08 PT: first epoch completed on cloud GPU.** Run `mamba2-first-complete` (2K games, Mamba-2 4.3M params, A100 40GB, no `num_workers` fix): Epoch 1 in 3679.6s (61min), loss=0.4698, action_acc=0.952, change_acc=0.519, val_loss=0.3405. Checkpoint saved. Second run `smoke-nw4-v2` (with `num_workers=4`) still in progress — will give A/B timing comparison.

## Smoke Tests

Run these in order to validate the pipeline end-to-end:

```bash
# 1. Check volume has data
.venv/bin/modal run worldmodel/scripts/modal_train.py::check_volume

# 2. Pre-encode a small subset (~$0.05)
.venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode \
    --max-games 50 --output /encoded-test.pt

# 3. Verify the encoded file landed
.venv/bin/modal run worldmodel/scripts/modal_train.py::check_volume

# 4. Train 1 epoch on the test file (~$0.50)
.venv/bin/modal run worldmodel/scripts/modal_train.py::train \
    --encoded-file /encoded-test.pt --epochs 1 --run-name "smoke-test"

# 5. Check wandb for the smoke-test run
# https://wandb.ai/shinewave/melee-worldmodel

# 6. Verify checkpoints were saved
.venv/bin/modal run worldmodel/scripts/modal_train.py::check_volume

# 7. Local DataLoader test (free, no GPU)
.venv/bin/python -m worldmodel.scripts.train \
    --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
    --max-games 50 --epochs 1 --device cpu --num-workers 2 --no-wandb
```

## Known Issues / Risks

| Issue | Severity | Notes |
|-------|----------|-------|
| Sweep checkpoint collision | **Medium** | All runs write `best.pt` to same dir. Fix: use `{CHECKPOINT_DIR}/{run_name}/` |
| pre_encode memory limit | Low | 16GB RAM caps at ~25-30K games (raw + encoded tensors coexist). Fine for 22K. For 230K+, needs `memory=65536` or chunked encoding. |
| ~~Encoding config not validated~~ | ~~Medium~~ | **Fixed** — `train()` now raises `ValueError` if `.pt` config != YAML config. |
| No retry on volume.commit() | Low | If commit fails, checkpoints are lost. Modal volumes are reliable but not infallible. |
| train() returns 0 on missing data | Low | Missing encoded file prints error but doesn't raise. `modal run` shows success. |

## Architectural Decisions

**Why pre-encode on Modal instead of locally?**
Local encoding works but requires uploading 7GB of tensors. Pre-encoding on Modal reads the 3.4GB tar that's already on the volume, encodes on Modal's fast CPUs, and writes the .pt directly. Saves upload time and bandwidth.

**Why num_workers=4, not higher?**
A100 instances on Modal have 12 vCPUs. 4 workers leaves headroom for the main process + CUDA kernels. Diminishing returns above 4 for tensor-backed datasets (the bottleneck is GPU compute, not data loading — workers just need to stay ahead of the GPU).

**Why persistent_workers?**
Without it, PyTorch forks 4 new processes every epoch. With 10 epochs and ~200MB per worker, that's 40 process spawns. `persistent_workers=True` keeps them alive — small memory cost, big latency win.

**Why sweep uses spawn() not map()?**
`train.spawn()` returns a handle immediately, letting us launch all runs before waiting. `train.map()` would work but doesn't give per-run control over arguments (each run might want different configs in the future).

---

## Review Feedback (Scav-2, Feb 25 2026 ~19:45 PT)

External review of this runbook by a separate Claude instance doing training pipeline research.

### What's right

- Architecture diagram is clean and accurate
- Checkpoint collision in sweeps — real bug, correctly flagged as Medium
- DataLoader config reasoning is solid (COW fork workers, prefetch math, persistent_workers rationale)
- Smoke tests are step-by-step and in the right order
- IterableDataset worker duplication flag is a good catch
- Architectural decisions section is well-reasoned (spawn vs map, num_workers=4 ceiling)

### Severity adjustment: encoding config validation → Medium

Marked Low in Known Issues. Should be **Medium**. This is silent data corruption — if you pre-encode with config A and train with config B (same shapes, different semantics), the model trains happily on garbage. The shape preflight catches dimension mismatches but not semantic ones.

**Concrete fix** — add to `modal_train.py` after loading the payload:
```python
saved_cfg = payload.get("encoding_config", {})
if saved_cfg and saved_cfg != enc_cfg_dict:
    raise ValueError(f"Config mismatch! Encoded with {saved_cfg}, training with {enc_cfg_dict}")
```

**Structural fix** — add `MeleeDataset.from_tensors()` classmethod in `dataset.py` so the reconstruction logic lives next to `__init__` and stays in sync when attributes change.

### Pre-encode memory estimate is wrong

Runbook says "16GB RAM caps at ~70K games." Actual math:

- 70K games × ~150KB raw parsed = ~10.5GB in memory
- Encoding produces contiguous tensors: 70K × ~9,400 frames × (58 floats + 15 ints) × 4 bytes ≈ **~25GB**
- Both live in RAM simultaneously during `MeleeDataset(games, cfg)`
- Total: ~35GB. Hard OOM at 16GB.

Realistic cap is **~25-30K games** at 16GB. For the full 230K+ dataset, `pre_encode` needs `memory=65536` or higher, or it needs to encode in chunks (load N games, encode, append to disk, free, repeat).

### Smoke test doesn't verify correctness

The smoke test proves the pipeline doesn't crash. It doesn't prove it produces correct output. Missing:

1. **Loss sanity check** — after 1 epoch on 50 games, is loss in expected range? Not NaN? Not stuck at init?
2. **Accuracy above random** — action accuracy should be > 0.25% (random for 400 classes). If it's near random after a full epoch, something is wrong.
3. **Local-vs-Modal comparison** — same 50 games, same config, 1 epoch local vs 1 epoch Modal. If val metrics diverge significantly, the `__new__` reconstruction or the config is wrong. This is the real proof.
4. **Checkpoint download + load verification** — the smoke test checks checkpoints exist on the volume but never downloads one. Add:

```bash
# 8. Download and verify checkpoint loads
.venv/bin/modal volume get melee-training-data /checkpoints/best.pt ./test-checkpoint.pt
.venv/bin/python -c "
import torch
c = torch.load('./test-checkpoint.pt', weights_only=False)
print(f'Epoch: {c[\"epoch\"]}, Val loss: {c[\"val_loss\"]:.4f}')
print(f'Keys: {list(c.keys())}')
"
```

### Missing `--detach` for real runs

All `modal run` examples block the terminal. Fine for smoke tests, impractical for 2-4hr training runs. Should document:

```bash
# Long training run (detached — terminal-free)
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
    --encoded-file /encoded-22k.pt --epochs 10 --run-name "v2.2-22k-10ep"

# Check logs later
.venv/bin/modal app logs melee-worldmodel
```

### Training cost estimates are optimistic

"10 epochs, 22K games: ~1-2 hr" assumes good GPU utilization. Observed utilization before the `num_workers` fix was 20-30%, which implies ~20-25min/epoch = 3-4hr for 10 epochs. With `num_workers=4`, probably 1.5-2.5hr. Widen the range until we have real timing data from a completed run.

### Missing context: no epoch has ever completed on cloud GPU

The runbook reads as if the pipeline works. It should state upfront that as of Feb 25, no training epoch has completed on any cloud GPU (RunPod or Modal). The smoke test exists specifically to close that gap for the first time. This isn't a knock on the work — it's important framing so the next Claude running the smoke test treats it as a first-ever validation, not a regression check.

### Summary

**Solid B+.** Structure and coverage are good. The gaps are about rigor — asserting correctness, not just no-crash. Two highest-priority fixes before trusting a real training run:

1. Config validation assertion (prevents silent garbage-in)
2. Local-vs-Modal comparison on same data (proves the pipeline produces correct results)
