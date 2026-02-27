# World Model Research Lab

Research experiments on the Melee world model. Each experiment tests a **structural hypothesis** — we're looking for step-function improvements, not incremental gains.

## Experiment Taxonomy

| Category | What changes | Examples |
|----------|-------------|---------|
| **data** | What the model trains on | Replay count, filtering criteria, data enrichment |
| **encoding** | How game state becomes tensors | New features, normalization, stage geometry, hitbox properties |
| **loss** | What the model optimizes | Focal loss, auxiliary physics losses, hierarchical action loss |
| **architecture** | The model itself | Hybrid attention, diffusion heads, larger/smaller models |
| **training** | How the model learns | Scheduled sampling, curriculum learning, LR schedules |
| **harness** | Inference-time behavior | Threshold tuning, rollout clamping, delay frames |

These categories are **not interchangeable**. Changing replay count is a fundamentally different lever than introducing focal loss. Experiments should test one category at a time where possible.

## Current Baselines

| Name | Checkpoint | Arch | Data | Key Metrics |
|------|-----------|------|------|-------------|
| **mamba2-v3-2k** | `checkpoints/mamba2-v3-2k-test-v2/best.pt` | Mamba-2 4.3M, K=10 | 2K games, 2ep | action=96.7%, change=68.3%, on_ground=77.8%, pos_err=1.02 |
| **mlp-22k** | `checkpoints/baseline-v22/` | MLP | 22K games, 4ep | change=77.5% (ceiling) |

Batch eval methodology: `scripts/batch_eval.py`, 5+ games, teacher-forced. Full baseline eval: `research/baselines/`.

## Experiment Index

| ID | Category | Name | Status | Hypothesis | Target Metric |
|----|----------|------|--------|-----------|---------------|
| E001 | loss | [Focal loss on binary head](experiments/E001-focal-loss-binary/) | planned | Class imbalance causes on_ground recall crisis (52%). Focal loss downweights easy-negative airborne frames. | on_ground recall > 75% |
| E002 | encoding | [Explicit stage geometry](experiments/E002-stage-geometry/) | planned | 4-dim stage embedding can't represent platform positions. 20 floats of explicit geometry replaces guesswork with facts. | on_ground recall, pos_error on platform stages |
| E003 | loss | [Auxiliary physics losses](experiments/E003-auxiliary-physics/) | planned | Soft penalty terms for stock monotonicity, percent monotonicity, blast zone bounds teach rules the model currently hallucinates. | Fewer impossible state transitions in rollout |
| E004 | training | [True scheduled sampling](experiments/E004-scheduled-sampling/) | planned | Feed model's own predictions during training instead of Gaussian noise. Directly attacks autoregressive drift. | Autoregressive rollout quality (visual + metrics) |
| E005 | encoding | [Hitbox properties](experiments/E005-hitbox-properties/) | planned | Damage change_acc ceiling (24%) exists because model can't see attack properties. Lookup table indexed by (action, state_age). | damage change_acc, knockback direction accuracy |
| E006 | loss | [Hierarchical action loss](experiments/E006-hierarchical-action/) | planned | Auxiliary 20-class category loss teaches action grouping before fine-grained 400-class discrimination. | change_acc +2-4pp |
| E007 | harness | [Binary threshold tuning](experiments/E007-threshold-tuning/) | planned | Default sigmoid threshold (0.5) maximizes accuracy but tanks recall. Lowering to 0.3 trades precision for recall. | on_ground recall in autoregressive mode |
| E008 | training | [Focal context](experiments/E008-focal-context/) | implemented (dataset) | Context window extends D frames past prediction target (full state). Model predicts a point inside the window. Four model variants below. | change_acc, on_ground recall |
| E008a | architecture | [Tap focal hidden state](experiments/E008-focal-context/E008a-tap-focal-hidden.md) | planned | Read SSM hidden state at the focal position instead of last position. ~5 lines. | change_acc |
| E008b | architecture | [Positional conditioning](experiments/E008-focal-context/E008b-positional-conditioning.md) | planned | Add scalar/embedding to ctrl telling model which position to predict. ~15 lines. | change_acc |
| E008c | architecture+loss | [Multi-position prediction](experiments/E008-focal-context/E008c-multi-position-prediction.md) | planned | Predict at every position (GPT-style). K× training signal per example. ~100+ lines. | change_acc, training efficiency |
| E008d | architecture | [Bidirectional Mamba](experiments/E008-focal-context/E008d-bidirectional-mamba.md) | planned | Reverse Mamba pass gives every position full past+future context. ~60 lines. | change_acc, on_ground recall |

## Known Structural Walls (from batch eval)

These are the failure modes. Every experiment should state which wall it's attacking.

1. **on_ground recall crisis** — 99% precision, 52% recall. Model defaults to "airborne." Root cause: class imbalance + collision detection is a discontinuity. [Full analysis](BATCH-EVAL-ANALYSIS.md)
2. **facing recall crisis** — 99.5% precision, 51% recall. Coupled to on_ground (facing changes are state-gated by grounded/airborne).
3. **aerial attack confusion** — BAIR/FAIR change_acc 13-21%. Downstream of facing: can't distinguish forward/back aerial without knowing which way character faces.
4. **movement transition ambiguity** — TURNING 28% change_acc, WALK_SLOW 33%. One-frame states at analog dead zone boundaries.
5. **damage unpredictability** — 24% change_acc. Likely a ceiling — requires hitbox/hurtbox geometry we don't encode.
6. **autoregressive compounding** — Teacher-forced errors are constant-rate (~22% on_ground), but compound in autoregressive mode because wrong predictions feed back as context.

## Experiment Protocol

Each experiment gets a directory under `experiments/` with:

```
experiments/E00N-short-name/
  hypothesis.md    # What we expect to change and why
  config.yaml      # Training config (or diff from baseline)
  results.md       # What happened (filled in after)
```

### Rules

1. **One category per experiment.** Don't change the loss function AND the encoding in the same run.
2. **State the target metric.** "Improves the model" is not a hypothesis. "on_ground recall > 75%" is.
3. **Run against baseline.** Same data, same epochs, same hardware. Only the intervention differs.
4. **Cheap first.** 2K games, 2 epochs, A100. ~$6, ~90 min. Enough for trendlines.
5. **Record nulls.** An experiment that doesn't move the target metric is still a finding. Write it up.

## Research Backlog (unprioritized)

See [training-improvements-research.html](literature/training-improvements-research.html) for the full literature review and technique catalog. See [ROADMAP.md](../docs/ROADMAP.md) for the (now stale) sprint calendar — items there that haven't been promoted to experiments above are deferred.

## Findings

| Date | Finding | Source |
|------|---------|--------|
| 2026-02-27 | K=10 covers 14.6% of commitment windows (attack→hitstun). Median commitment is 38 frames. K=60 covers 71.7%. | [commitment-window-analysis.md](notes/commitment-window-analysis.md) |
| 2026-02-27 | on_ground recall 52%, facing recall 51% — one-directional class imbalance bias | [BATCH-EVAL-ANALYSIS.md](BATCH-EVAL-ANALYSIS.md) |
| 2026-02-27 | BAIR/FAIR confusion is downstream of facing failure | [BATCH-EVAL-ANALYSIS.md](BATCH-EVAL-ANALYSIS.md) |
| 2026-02-27 | Teacher-forced error rate is flat across time buckets — not a drift problem | [BATCH-EVAL-ANALYSIS.md](BATCH-EVAL-ANALYSIS.md) |
| 2026-02-25 | Mamba-2 beats MLP at equal scale (67.3% vs 64.5%, 2K/2ep) | ROADMAP.md |
| 2026-02-23 | MLP ceiling at 77.5% change_acc (22K/4ep) — architecture was the bottleneck | ROADMAP.md |
| 2026-02-23 | state_age_as_embed: +7.2pp change_acc — biggest single encoding improvement | ROADMAP.md |
