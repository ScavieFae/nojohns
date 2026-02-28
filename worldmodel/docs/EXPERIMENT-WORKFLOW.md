# Experiment Workflow

How experiments go from hypothesis to results.

## Lifecycle

1. **Hypothesis** — What are we testing and why? Write it in a research note or the experiment YAML header.
2. **Config** — Create `worldmodel/experiments/e{NNN}{letter}-{name}.yaml`. Self-contained: encoding flags, model params, training config, loss weights.
3. **Branch** — One branch per experiment *family*: `scav/research/e{NNN}-{slug}`. All variants (a, b, c) live on the same branch, isolated by config flags.
4. **Code** — Implement on the branch. New `EncodingConfig` flags with `False` defaults so existing configs are unaffected. Smoke test locally.
5. **Run card** — For GPU runs >$5 or >2hr, create `worldmodel/docs/run-cards/e{NNN}{letter}-{name}.md` before launch. Use `/run-card` or copy an existing one.
6. **Train** — Launch on Modal. Check wandb for convergence.
7. **Eval** — Run `batch_eval.py` on the checkpoint. Compare to baseline on the same eval set.
8. **Results** — Update run card with results. Log learnings in `RESEARCH-DIARY.md`.
9. **Decision** — Promote (merge to dev), iterate, or archive.

## Branch Strategy

**Config flags are the isolation mechanism, not branches.**

Orthogonal experiments (touching different parts of the pipeline) go on one branch with separate YAML configs:

```
scav/research/e010-movement-suite     ← one branch, three configs
  experiments/e010a-ctrl-residual.yaml    (ctrl_residual_to_action: true)
  experiments/e010b-transition-weight.yaml (action_change_weight: 5.0)
  experiments/e010c-ctrl-thresholds.yaml   (ctrl_threshold_features: true)
```

**Why one branch?** Orthogonal features behind config flags don't conflict. Separate branches duplicate the shared base, triple the merge work, and hide integration issues until merge time.

**When to split branches:** Only when experiments modify the same code in *incompatible* ways (e.g., two different rewrites of the same function). If changes are gated by config flags, they belong on one branch.

**Graduation:** When an experiment family proves out, its winning flags become defaults in the next experiment's base config. E.g., E009a graduated E008c's `multi_position: true`.

## Naming

| Thing | Pattern | Example |
|-------|---------|---------|
| Experiment family | `E{NNN}` | E010 (movement suite) |
| Variant | `E{NNN}{letter}` | E010a, E010b, E010c |
| Branch | `scav/research/e{NNN}-{slug}` | `scav/research/e010-movement-suite` |
| Config | `worldmodel/experiments/e{NNN}{letter}-{slug}.yaml` | `e010a-ctrl-residual.yaml` |
| Run card | `worldmodel/docs/run-cards/e{NNN}{letter}-{slug}.md` | `e010a-ctrl-residual.md` |
| Checkpoint dir | `checkpoints/e{NNN}{letter}-{slug}` | on Modal volume |
| wandb run | `e{NNN}{letter}-{slug}` | in melee-worldmodel project |

## Config Conventions

- Every experiment YAML is self-contained — copy-paste the base config and change only the new flags.
- New `EncodingConfig` flags default to `False` / `0` / `1.0` so baseline behavior is unchanged.
- Comment the hypothesis at the top of the YAML.
- `save_dir` matches the experiment slug.

## Comparative Eval

Run `batch_eval.py` on all checkpoints with the same seed and game count:
```bash
.venv/bin/python worldmodel/scripts/batch_eval.py \
  --checkpoint checkpoints/e010a-ctrl-residual/best.pt \
  --games 30 --seed 42
```

Primary metrics vary by experiment goal. For E010 (movement):
- `movement_change_acc` — the target
- `overall_change_acc` — must not regress
- `pos_mae` — must not regress
