# Home Run Training Plan

**Created**: Feb 26, 2026
**Status**: Planning — three experiments before committing

## Vision

A world model good enough that two agents playing inside it look like real Melee. That means:
- Enough data that the model has seen the distribution of real play
- Enough params that it can represent the physics faithfully
- Encoding that captures projectiles (Fox lasers appear from nowhere right now)
- Small enough to quantize for fast inference (and eventually onchain)

## What We Know

### Scaling data (all Mamba-2, 22K games)

| Model | Params | Data | Epochs | change_acc | pos_mae | val_loss | movement |
|-------|--------|------|--------|-----------|---------|----------|----------|
| **MLP** | **1.4M** | 22K | 4 (converged) | **77.5%** | 0.79 | — | ~91% |
| **Mamba-2** | **4.3M** | 22K | 1 | **75.9%** | 0.549 | 0.231 | 91.0% |
| Mamba-2 | 4.3M | 2K | 2 | 67.7% | 0.694 | 0.291 | 90.2% |
| Mamba-2 | 4.3M | 2K (K=60) | 2 | 68.3% | 0.549 | 0.271 | 90.6% |

**Takeaways:**
- **Data is the biggest lever.** 2K→22K = +8pp change_acc on Mamba-2 in 1 epoch.
- **Architecture helps convergence, not ceiling (yet).** MLP hit 77.5% in 4 epochs. Mamba-2 hit 75.9% in 1 epoch. If Mamba-2 plateaus near 77.5% after epochs 2-3, both architectures hit the same data/encoding ceiling.
- **Movement is structurally pinned at ~91%** across every run, every architecture, every data scale. Not a convergence issue — something in the encoding or loss formulation caps it.
- **Context length (K=10 vs K=60) barely matters** (+0.6pp). Don't chase this.

### Model sizes

| Config | Params | Architecture | Cost/epoch (22K, A100) |
|--------|--------|-------------|----------------------|
| MLP baseline | 1.4M | FrameStackMLP (h=512, trunk=256) | ~$1 (CPU/MPS) |
| mamba2-medium | 4.3M | Mamba-2 (d=384, 4 layers) | ~$23 |
| mamba2-large | **10.6M** | Mamba-2 (d=512, 6 layers) | ~$58 (projected) |

Note: the "15M" config is actually 10.6M. Still a 2.5× jump from 4.3M.

### Data inventory

| Source | Games | Location | Status |
|--------|-------|----------|--------|
| Training set (current) | 22,000 | Modal volume (`encoded-22k.pt`, 74.4GB) | In use |
| Full ranked dataset | 287,504 | queenmab (`~/data/parsed-ranked/`) | Parsed, not staged |
| Available for scaling | ~265K more | queenmab | Needs tar + upload + pre-encode |

### Encoding status

| Feature | Dims/player | Status |
|---------|------------|--------|
| Core continuous (pos, shield, vel, hitlag, stocks, combo) | 12-13 | ✓ Active |
| Binary (facing, invuln, on_ground) | 3 | ✓ Active |
| Controller (sticks, buttons) | 13 | ✓ Active |
| Action embedding (400 vocab) | 32 | ✓ Active |
| Combat context (l_cancel, hurtbox, ground, last_attack) | 16 embed | ✓ Active |
| State age embedding (150 vocab) | 8 embed | ✓ Active |
| **Projectiles (nearest dx/dy, n_active)** | **+3 continuous** | **Gated, code exists, never tested** |

Projectile code: `encoding.py:229-257`. Flag: `EncodingConfig(projectiles=True)`.
Impact: float_per_player goes 28→31. Adds 6 total dims per frame.

## Decisions To Make

These three experiments answer the decisions before we commit to the big run.

### Decision 1: Does 10.6M improve over 4.3M at equal data?

If yes → the home run is 10.6M × 200K.
If no → the home run is 4.3M × 200K (cheaper, same accuracy).

**Experiment: 10.6M × 2K smoke test (~$20)**
- Config: `mamba2-large-gpu.yaml` (d=512, 6 layers, lr=0.0003)
- Data: `encoded-2k.pt` (already on volume)
- Epochs: 2
- Compare to: Mamba-2 4.3M × 2K/2ep baseline (67.7% change_acc)
- If 10.6M beats 67.7% significantly → model capacity matters, scale it
- If 10.6M ≈ 67.7% → diminishing returns on params at 2K, data is the bottleneck

### Decision 2: Do projectiles move the needle?

If yes → include in every future run and re-encode the big dataset with them.
If no → skip, save encoding complexity.

**STATUS: BLOCKED — parser doesn't populate item data.**

Scav tested: 0/500 games have any item with `exists=True`. The struct is in the schema but every field is empty. Flipping `projectiles=True` adds 6 dims of zeros. The parsing pipeline (`parse.py`) needs to extract item/projectile data from raw replays (libmelee's `gamestate.projectiles`), then raw .slp files need re-parsing — existing parquets are missing the data.

**This does NOT invalidate any existing training.** All runs used `projectiles=False` (the default). No projectile dims were ever in any encoded data.

**Impact on home run:** The home run can proceed without projectiles — Decisions 1 + 4 still inform model size and data scale. Projectiles only affect a subset of frames (characters with projectiles in relevant matchups). Fixing the parser is real work but not on the critical path. See `run-cards/projectile-2k-test.md` for details.

~~**Experiment: Projectile encoding on 2K ($6)**~~
~~- Re-encode 2K games with `EncodingConfig(projectiles=True, state_age_as_embed=True)`~~
~~- Train Mamba-2 4.3M, 2 epochs~~
~~- Compare pos_mae specifically — projectiles should help position prediction~~
~~- Also watch movement category — projectile dodging might unstick the 91% ceiling~~

### Decision 3: How much accuracy do we lose to INT8?

This is the most important experiment. If INT8 degrades <3%, we train in float and quantize after. If it degrades >5%, we need QAT (quantization-aware training) which changes the entire training pipeline.

**Experiment: Post-training quantization on 4.3M checkpoint (free, ~30 min code)**

Steps:
1. Load `mamba2-22k-ss-ep1.pt` (our best checkpoint)
2. Apply `torch.quantization.quantize_dynamic` (INT8 weights, float activations)
3. Run full validation pass — same val set as training
4. Compare: val_loss, change_acc, pos_mae, per-action-category accuracy
5. Also benchmark: inference time per frame (float32 vs INT8)

**RESULTS (Feb 26, 2026) — PASS:**

Ran `benchmark_quantization.py` on `mamba2-22k-ss-ep1.pt`, 50-game val set, qnnpack engine (ARM):

| Metric | Float32 | INT8 | Degradation |
|--------|---------|------|-------------|
| Val Loss | 0.2588 | 0.2642 | +2.1% (mild) |
| **Change Acc** | **68.82%** | **68.73%** | **-0.1%** |
| Action Acc | 97.04% | 97.04% | 0.0% |
| Position MAE | 0.534 | 0.542 | +1.6% (mild) |
| Velocity MAE | 0.066 | 0.067 | +0.1% |

22/26 linear layers quantized. **Change accuracy barely moves.** Position MAE up 1.6%. Well within "train float32, quantize post-training" territory. **No QAT needed.**

Script: `worldmodel/scripts/benchmark_quantization.py`

**Follow-up: force-quantize all 26/26 layers (onchain mode)**

The 4 skipped layers are tiny prediction heads (l_cancel: 384→3, hurtbox: 384→3, × 2 players = 4,620 params total). PyTorch skipped them as a CPU speed optimization. For onchain ephemeral rollup, ALL math must be integer — no exceptions. Added `--force-all` flag to benchmark script that force-quantizes every Linear layer to INT8, including the heads PyTorch considers too small.

Decision: force INT8 everywhere (not INT32) for uniformity — one integer matmul path in the onchain verifier, not two.

**Force-all results (Feb 26, 2026) — PASS:**

| Metric | Float32 | INT8 (22/26) | INT8-ALL (26/26) | Force-all cost |
|--------|---------|--------------|------------------|----------------|
| Change Acc | 68.82% | 68.65% | 68.65% | **0.0%** |
| Position MAE | 0.534 | 0.541 | 0.541 | **0.0%** |
| Val Loss | 0.2581 | 0.2641 | 0.2641 | **0.0%** |

Quantizing the 4 tiny heads costs literally nothing — they're too small (384→3) for INT8 rounding to matter. **Full INT8, all 26 layers, confirmed safe for onchain.**

**Mamba-2 quantization notes:**
- SSM scan ops (softplus, cumsum) are NOT in the linear layers — they survive quantization untouched
- The 4 non-quantized layers are small prediction heads (in=384, out=3 for l_cancel/hurtbox) — `quantize_dynamic` skips very small layers where INT8 packing isn't beneficial. The heavy hitters — the 8 Mamba in_proj/out_proj layers (~70% of all params) — all got quantized.
- On CUDA with proper INT8 kernels, expect actual speedup (qnnpack on ARM was 0.94x)

**Scrutiny of "0.1% degradation, ship it" (Scav + Mattie discussion, Feb 26):**

The headline is promising but the evidence is narrower than it looks.

*Statistical power:* 0.09pp drop on 62K frames. Standard error on a binary metric at this sample size is ~0.18pp (sqrt(0.69×0.31/62533)). The 0.1% difference is within statistical noise — we can't distinguish it from zero. That's actually good (no detectable degradation), but it also means we can't confidently bound the true degradation.

*What we tested vs what we'd deploy:*
- **Dynamic quantization** (what we tested): weights → INT8, activations stay float32. Each forward pass: float32 input → INT8 weight matmul → float32 output. Cheapest, safest form.
- **Static quantization** (what onchain/rollup needs): weights AND activations → INT8. Requires a calibration dataset to measure activation ranges, then fixed scale factors per layer. Harder, likely degrades more. **Not yet tested.**

*Assumptions we're making:*
1. Dynamic quant is representative of deployment. We only quantized weights. Static quant (needed for ephemeral rollup) could be worse.
2. 50 games is enough for relative delta. Absolute numbers (68.8% vs 75.9% on real val set) don't match because different sample. But float32-vs-INT8 delta should be stable. Probably true, worth confirming on larger set.
3. qnnpack (ARM) ≈ CUDA INT8. Different backends have different rounding behavior.
4. **Single-step accuracy ≈ autoregressive accuracy. Biggest leap.** Benchmark is teacher-forced (one step). In autoregressive rollout, errors compound. If INT8 introduces slightly different error patterns (even at same magnitude), drift could diverge differently. Not tested.
5. 4.3M generalizes to 10.6M. Larger models sometimes quantize better (more redundancy), sometimes worse (more precision-dependent layers).

*What we don't know:*
- Autoregressive drift under INT8 — the real deployment scenario
- Static quantization degradation (needed for onchain)
- Whether a more-trained checkpoint (epochs 2-3) is more or less fragile
- CUDA INT8 kernel behavior specifically

*Honest assessment:* PTQ looks very promising — quantization barely touches the metrics that matter. But "no QAT needed" is a claim about deployment, and we only tested one step of inference on 62K frames with ARM kernels. **Before shipping INT8, run an autoregressive demo with the quantized model and compare drift visually.**

### Decision 4: How much data actually helps? (answered by current runs)

Mamba-2 4.3M × 22K epochs 2-3 are training now (`mamba2-22k-ss-resumed`, wandb: `8za01pdm`).

| If epochs 2-3 result | Interpretation | Next step |
|---------------------|----------------|-----------|
| >80% change_acc | Data scaling still climbing, architecture works | Stage 200K, train at current params |
| 77-80% | Matching MLP ceiling, might be encoding-limited | Try projectiles + 10.6M before scaling data |
| <77% | Plateauing below MLP | Investigate — SS or combat heads might hurt |

## The Home Run

After the three experiments, the home run looks like one of:

**Option A: Scale data (if 4.3M isn't capacity-bottlenecked)**
- 4.3M Mamba-2 × 200K games × 5 epochs
- Single H100: ~11h/epoch, ~$150 total
- INT8 quantized post-training

**Option B: Scale both (if 10.6M shows clear win)**
- 10.6M Mamba-2 × 200K games × 3 epochs
- 4× A100 DDP or 4× H100 DDP
- ~10h/epoch on 4×H100, ~$475 total
- INT8 quantized post-training (or QAT if PTQ degrades)

**Option C: Scale data + fix encoding (if projectiles help)**
- 4.3M or 10.6M × 200K games with projectile encoding
- Re-encode full dataset with projectiles (~$5 CPU cost)
- Then Option A or B

## Multi-GPU Plan (for Option B)

DDP (DistributedDataParallel) — each GPU gets a copy of the model, processes a slice of the batch.

**Wall clock estimates (200K games, 3 epochs):**

| Setup | 4.3M | 10.6M | Cost (3 ep) |
|-------|------|-------|-------------|
| 1× A100 | ~27h/ep | ~67h/ep | $225 / $558 |
| 1× H100 | ~11h/ep | ~27h/ep | $130 / $320 |
| 4× A100 DDP | ~7h/ep | ~17h/ep | $234 / $558 |
| 4× H100 DDP | ~3h/ep | ~7h/ep | $142 / $332 |

**Code changes for DDP (~60 lines in Trainer):**
1. `torch.distributed.init_process_group("nccl")`
2. Wrap model in `DistributedDataParallel`
3. `DistributedSampler` instead of shuffle
4. Gate checkpoints + wandb on `rank == 0`
5. Modal: `gpu="H100:4"`, spawn processes

**Sweet spot:** 4× H100 for 10.6M (overnight runs, ~7h/epoch). For 4.3M, single H100 is plenty.

## Data Pipeline (200K)

1. **Tar on queenmab**: `tar cf /tmp/parsed-ranked.tar parsed-ranked/` (~34GB)
2. **Transfer**: `scp queenmab:/tmp/parsed-ranked.tar /tmp/` (Tailscale, ~15 min)
3. **Upload to Modal**: `modal volume put melee-training-data /tmp/parsed-ranked.tar /parsed-ranked.tar`
4. **Pre-encode**: Chunked parallel, 100 workers × 2K games. ~$5 CPU, ~30 min.
5. **Output**: `encoded-200k.pt` (~620GB projected). Verify volume has space.

## Experiment Priority

Mattie's call: quantization first, then projectiles and 10.6M smoke test.

| # | Experiment | Cost | Time | Status |
|---|-----------|------|------|--------|
| **1a** | **PTQ on 4.3M checkpoint (default)** | **Free** | **~30 min** | **DONE — PASS (0.1% change_acc loss, 22/26 layers)** |
| **1b** | **PTQ force-all (onchain mode)** | **Free** | **~30 min** | **DONE — PASS (26/26 layers, 0.0% additional cost)** |
| ~~2~~ | ~~Projectile encoding test (2K)~~ | ~~$6~~ | ~~2h~~ | **BLOCKED — ScavieFae working on parser fix** |
| 3 | 10.6M × 2K smoke test | ~$20 | ~35 min (2ep on A100) | **LAUNCHING** |

~~Experiments 2 and 3 can run in parallel on Modal.~~
Experiment 3 can launch independently. Experiment 2 blocked until parser fix.

## Current Runs

| Run | GPU | Status | wandb |
|-----|-----|--------|-------|
| mamba2-22k-ss-resumed | A100 | Training ep 2-3 | `8za01pdm` |
| policy-22k-v2 | T4 | Training ep 1/5, batch ~8K/181K | `6ss5y5je` |

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
