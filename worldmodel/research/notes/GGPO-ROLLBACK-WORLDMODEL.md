# GGPO-Style Speculative Execution in World Model Rollouts

Status: **mad science / future exploration**
Date: Feb 24, 2026
Context: Emerged from exp-3a (lookahead) results showing +6.4pp change_acc but circular dependency at inference time.

## The Problem

Lookahead=1 gives better predictions (especially on action transitions) but needs ctrl(t+1) at inference time. In the RL rollout loop, ctrl(t+1) = policy(state(t)), which is what we're trying to predict. Circular dependency.

## The Idea

Use GGPO-style input prediction: assume ctrl(t+1) = ctrl(t), simulate forward, rollback if wrong.

```
1. policy(state(t-1)) → ctrl(t)
2. PREDICT ctrl(t+1) = ctrl(t)               ← GGPO-style
3. worldmodel(context, ctrl(t), predicted) → speculative state(t+1)
4. policy(speculative state(t+1)) → actual ctrl(t+1)
5. if actual ≠ predicted → ROLLBACK:
   worldmodel(context, ctrl(t), actual) → corrected state(t+1)
```

~80% of frames, prediction is right (one forward pass). ~20% transition frames need rollback (two forward passes). Those transition frames are exactly where 3a shines. Average cost: ~1.2x lookahead=0.

## Why It's Interesting

- Structural match to GGPO is exact, not metaphorical
- MuZero/Dreamer don't do this — would be novel
- Better predictions on transition frames = better RL signal on the frames that matter
- Multi-agent case (two policies playing through the world model) maps even more cleanly — one policy's output IS the "remote input"

## What Breaks (Critical Analysis)

### 1. Policy reacts to wrong speculative state
On transition frames, the policy sees speculative state(t+1) computed with wrong inputs. Its ctrl(t+1) response is based on that wrong state. The "corrected" state uses the right inputs but the policy's *decision* was informed by garbage. In GGPO the remote player is a separate human who doesn't see your speculative frames. Here, the policy reacts to them.

### 2. No latency to hide
GGPO's value is hiding network latency — the alternative is blocking. In our loop there's no network latency. The alternative is lookahead=0: one clean forward pass, always. We're adding ~0.2 extra forward passes per frame for better transition predictions, but we could just use the simpler model.

### 3. "Repeat last input" is worse for policies than humans
GGPO predicts *human* inputs where reaction time limits how fast inputs change. A learned policy can produce wildly different outputs on consecutive frames, especially during exploration. The autocorrelation assumption comes from human controller data at 60fps — a policy isn't bound by that.

### 4. Training/inference mismatch
The 3a model was trained on ground-truth ctrl(t+1), never on noisy/predicted ctrl(t+1). It has no robustness to input noise in that slot. Would need to train with predicted future inputs to make this work — which is a different model.

### 5. Continuous rollback threshold
GGPO compares discrete inputs (button pressed or not). Controller sticks are floats. "Not equal" for continuous values means always. Need a threshold — new hyperparameter trading correction accuracy vs compute.

## Where It Maps Cleanly

**Multi-agent rollouts.** Two policies playing each other through the world model. Policy A produces ctrl_A(t), but we need ctrl_B(t) to predict the next state. Policy B hasn't acted yet — its input is genuinely "remote." GGPO pattern maps 1:1: predict B's input, simulate, correct when B's actual input arrives.

This is the real use case. Single-agent rollouts don't have the "waiting for remote input" structure that GGPO is designed for.

## Experimental Prerequisites

Before testing any of this:
1. Finish encoding experiments (exp 3a analysis)
2. Commit 1a to default config
3. Architecture upgrade (Mamba-2) — the MLP ceiling is the bottleneck, not the rollout strategy
4. Build the basic RL loop with lookahead=0 first
5. *Then* try speculative lookahead=1 as an optimization

## References

- `rnd-2026/coding/ggpo-rollback-netcode.md` — full GGPO explainer with section 11 on world model connections
- Exp 3a results: +6.4pp change_acc, -2pp action_acc, better physics (position_mae, damage_mae)
- GGPO source: github.com/pond3r/ggpo
