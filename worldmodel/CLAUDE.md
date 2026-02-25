# worldmodel/CLAUDE.md

## Agent Roles

| Agent | Role | Can edit code? |
|-------|------|----------------|
| **Scav** | Code changes, experiments, training ops | Yes |
| **ScavieFae** | Review, feedback, monitoring | No — review only |

ScavieFae reads Scav's code and writes review feedback. ScavieFae does NOT edit Python files in `worldmodel/`.

## Working Branch

All world model work happens on `scav/combat-context-heads` unless otherwise specified. Both agents should be on this branch when working on world model tasks.

**ScavieFae: stay on this branch.** Do not switch back to `main` after reviewing — you'll just have to switch back. Handoff docs and CLAUDE.md edits go here too.

## Coordination

Use `worldmodel/docs/HANDOFF-MODAL.md` for review feedback, questions, and blockers. Default to the handoff doc when uncertain whether to log to a doc or a GitHub issue.

## Key Files

| File | Role |
|------|------|
| `model/encoding.py` | `EncodingConfig`, `encode_player_frames()`, `StateEncoder` |
| `model/mlp.py` | `FrameStackMLP` — context window + ctrl → state prediction |
| `model/mamba2.py` | `FrameStackMamba2` — SSM sequence model |
| `data/dataset.py` | `MeleeDataset`, `MeleeFrameDataset` — returns 5 tensors per sample |
| `training/trainer.py` | Training loop, checkpointing, shape preflight, wandb |
| `training/metrics.py` | Loss computation, accuracy tracking |
| `scripts/modal_train.py` | Modal launcher (pre_encode, train, sweep) |
| `scripts/pre_encode.py` | Local pre-encoding to `.pt` files |
| `RUNBOOK.md` | Training ops — recipes, experiment tracking, data sources |
| `docs/MODAL-REVIEW-RUNBOOK.md` | Modal pipeline review checklist + Scav-2 feedback |
| `docs/HANDOFF-MODAL.md` | Active review feedback from ScavieFae → Scav |
| `docs/OVERNIGHT-PLAN.md` | Feb 25 overnight training strategy, scaling estimates, K comparison |
| `experiments/mamba2-medium-gpu.yaml` | 4.3M Mamba-2 K=10 — validated on A100 |
| `experiments/mamba2-medium-gpu-k60.yaml` | 4.3M Mamba-2 K=60 — needs review (batch_size?) |
| `experiments/mamba2-large-gpu.yaml` | 15M Mamba-2 — needs review (LR, batch_size, untested) |
