# Experiment Workflow

How experiments go from hypothesis to results.

## Lifecycle

1. **Hypothesis** — What are we testing and why? Write it in a research note or the experiment YAML header.
2. **Config** — Create `worldmodel/experiments/e{NNN}{letter}-{name}.yaml`. Self-contained: encoding flags, model params, training config, loss weights.
3. **Branch** — One branch per experiment variant: `scav/research/e{NNN}{letter}-{name}`. Branch from the parent experiment's base commit (usually the last merged experiment).
4. **Code** — Implement on the branch. New `EncodingConfig` flags with `False` defaults so existing configs are unaffected. Smoke test locally.
5. **Run card** — For GPU runs >$5 or >2hr, create `worldmodel/docs/run-cards/e{NNN}{letter}-{name}.md` before launch. Use `/run-card` or copy an existing one.
6. **Train** — Launch on Modal. Check wandb for convergence.
7. **Eval** — Run `batch_eval.py` on the checkpoint. Compare to baseline on the same eval set.
8. **Results** — Update run card with results. Log learnings in `RESEARCH-DIARY.md`.
9. **Decision** — Promote (merge to dev), iterate, or archive. Branches stay as references even if not merged.

## Branch Strategy

```
scav/research/e010a-ctrl-residual     ← one branch per experiment variant
scav/research/e010b-transition-weight
scav/research/e010c-ctrl-thresholds
```

**Why separate branches?**
- Independent review and revert
- Clean diffs — each branch shows exactly one idea
- Parallel GPU runs without merge conflicts
- Easy to cherry-pick winners into a combined config

**When to merge:** When an experiment improves the target metric without regressions, merge its branch into the next experiment's base. E.g., E009a graduated E008c's ideas.

**Archival:** Don't delete research branches. They're cheap references and sometimes you need to check what a past experiment actually did.

## Naming

| Thing | Pattern | Example |
|-------|---------|---------|
| Experiment family | `E{NNN}` | E010 (movement suite) |
| Variant | `E{NNN}{letter}` | E010a, E010b, E010c |
| Branch | `scav/research/e{NNN}{letter}-{slug}` | `scav/research/e010a-ctrl-residual` |
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
