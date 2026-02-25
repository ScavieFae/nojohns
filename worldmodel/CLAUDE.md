# worldmodel/CLAUDE.md

## Agent Roles

| Agent | Role | Can edit code? |
|-------|------|----------------|
| **Scav** | Code changes, experiments, training ops | Yes |
| **ScavieFae** | Review, feedback, monitoring | No — review only |

ScavieFae reads Scav's code and writes review feedback. ScavieFae does NOT edit Python files in `worldmodel/`.

## Working Branch

All world model work happens on `scav/combat-context-heads` unless otherwise specified. Both agents should be on this branch when working on world model tasks.

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
