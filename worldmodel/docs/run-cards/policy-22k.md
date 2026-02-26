# Policy Run Card: policy-22k

**Created**: Feb 26, 2026
**Status**: PENDING REVIEW

## Goal

Train a fighter policy on 22K ranked games to match the world model's data distribution. Current policies (50 games) produce near-random inputs that the 22K world model hasn't seen — this is the #1 quality bottleneck for the agent-vs-agent demo.

## Distribution Match

| Component | Data scale | Notes |
|-----------|-----------|-------|
| World model | 22K games | mamba2-22k-ss: ep1 done (75.9% change_acc), ep2-3 running as mamba2-22k-ss-resumed (bs=4096) |
| **This policy** | **22K games** | Same dataset, same distribution |
| Previous policy | 50 games | Near-random relative to world model's training data |

## Data

| Field | Value |
|-------|-------|
| Source | `/encoded-22k.pt` on Modal volume (pre-encoded, skip parsing) |
| File size | 74.4 GB |
| Games | 22,000 |
| Total frames | 206,754,081 |
| Train examples | ~185,648,963 (90%) |
| Val examples | ~20,885,118 (10%) |
| Game offsets | Stored in `.pt` — respects game boundaries |
| **Predict player** | **p0 only** — swap perspective for p1 at inference |

### One Policy, Two Slots

Training one player cuts training time in half. The policy sees `[my_state, opponent_state]` and predicts `my_controller`. At inference, `PolicyAgent` swaps the player halves of the context when loaded for p1, so the same checkpoint works for both slots. **This requires a ~5-line code change to PolicyAgent** (not yet implemented).

## Model

| Field | Value |
|-------|-------|
| Architecture | PolicyMLP (imitation learning) |
| Parameters | ~1,153,781 |
| Context len (K) | 10 |
| Hidden dim | 512 |
| Trunk dim | 256 |
| Dropout | 0.1 |
| Output | analog (5) + button_logits (8) = 13-dim controller |

## Training

| Field | Value |
|-------|-------|
| Epochs | 5 (4K-game run converged at ep ~26, 22K should converge faster per epoch) |
| Batch size | 1024 |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Optimizer | AdamW + cosine LR |
| Loss | MSE (analog) + BCE (buttons), equal weight |
| Device | CUDA (T4) |

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | **T4** ($0.59/hr) — 1.15M param MLP, compute is trivial |
| Platform | Modal (detached) |
| Data source | Pre-encoded `.pt` on volume (skip parse+encode) |
| Data load time | ~90s (already measured from world model launches) |
| wandb | `shinewave/melee-worldmodel`, tags: `[policy, imitation]` |

## Timing & Cost

| | Optimistic | 3x calibrated | 5x calibrated |
|---|---|---|---|
| Epoch time | ~30 min | ~90 min | ~150 min |
| Training (5 ep) | 2.5h | 7.5h | 12.5h |
| Data load | ~2 min | ~5 min | ~10 min |
| **Total wall** | **~2.5h** | **~7.5h** | **~12.5h** |
| **Cost** | **~$1.50** | **~$4.50** | **~$7.50** |

Modal timeout: **43200s (12h)** — covers 3x estimate. If 5x hits, resume from checkpoint.

## Prior Baselines

| Run | Games | Player | stick_mae | btn_pressed_acc | val_loss | Notes |
|-----|-------|--------|-----------|-----------------|----------|-------|
| policy-p0-test | 50 | p0 | 0.037 | 96.6% | 0.039 | 5 epochs, MPS, 12s/epoch |
| policy-imitation-4k | 4,000 | p0 | 0.020 | 98.2% | 0.032 | 50 epochs, converged ep 26 |

## Target Metrics

| Metric | Baseline (4K) | Target | Notes |
|--------|--------------|--------|-------|
| val_stick_mae | 0.020 | <0.015 | 5.5x more data should help |
| val_button_pressed_acc | 98.2% | >99% | The hard signal — when buttons are active |
| val_loss/total | 0.032 | <0.025 | Diminishing returns expected |

## Kill Thresholds

| Condition | Action |
|-----------|--------|
| val_loss not decreasing after 2 epochs | Stop — model has converged or data issue |
| val_stick_mae > 0.030 after epoch 1 | Investigate — worse than 50-game baseline |
| val_button_pressed_acc < 95% after epoch 1 | Investigate — regression from baseline |
| GPU OOM | Reduce batch_size to 512, restart |

## Implementation Checklist (before launch)

### 1. Modal policy launcher
Add `train_policy()` to `modal_train.py` (or new `modal_policy_train.py`):
- Load `encoded-22k.pt` from volume (same as world model)
- Build `PolicyFrameDataset` from the pre-encoded tensors + `game_offsets`
- Train `PolicyMLP` with `PolicyTrainer`
- **`volume.commit()` in epoch_callback** — same pattern as mamba2-22k-ss fix (c79a6ae)
- Save `best.pt` + `latest.pt` to volume each epoch
- Use `gpu="T4"`

### 2. PolicyFrameDataset from pre-encoded data
Current `PolicyFrameDataset` takes a `MeleeDataset` (in-memory games). Need a variant or adapter that takes the pre-encoded `(floats, ints, game_offsets)` tensors directly. The controller target is extracted from `floats[t]` using the same `ctrl_slice` logic.

### 3. Perspective swap in PolicyAgent
```python
# In PolicyAgent.get_controller(), when player != training_player:
#   swap float_ctx[:, :fp] and float_ctx[:, fp:2*fp]
#   swap int_ctx[:, :ipp] and int_ctx[:, ipp:2*ipp] (keep stage)
```

## Launch Command

```bash
# After implementation:
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train_policy \
    --encoded-file /encoded-22k.pt --epochs 5 \
    --batch-size 1024 --predict-player 0 \
    --run-name policy-22k
```

## Resume Command

```bash
# If run hits timeout or needs restart:
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train_policy \
    --encoded-file /encoded-22k.pt --epochs 5 \
    --batch-size 1024 --predict-player 0 \
    --resume policy-22k/latest.pt \
    --run-name policy-22k-resumed
```

## After Training

1. Download checkpoint: `modal volume get melee-training-data /checkpoints/policy-22k/best.pt worldmodel/checkpoints/policy-22k/best.pt`
2. Generate agent demo:
```bash
.venv/bin/python -m worldmodel.scripts.play_match \
    --world-model worldmodel/checkpoints/mamba2-22k-ss-ep1.pt \
    --seed-game <game> \
    --p0 policy:worldmodel/checkpoints/policy-22k/best.pt \
    --p1 policy:worldmodel/checkpoints/policy-22k/best.pt \
    --max-frames 20000 --output demo-22k-agents-v2.json -v
```
3. Compare behavior in viewer — agents should look like actual Melee players, not random flailing

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
