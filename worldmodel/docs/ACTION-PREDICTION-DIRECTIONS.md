# Action State Prediction — Research Directions

**Date:** February 24, 2026
**Context:** v2.2 world model hitting 72.4% action-change accuracy (22K games, epoch 2, still climbing). The question: can we push further by rethinking what "correct" means?

## The Core Insight

For a simulator, what matters is *physical consequence*, not *action state label*. Predicting FTILT when the answer was JAB is much less wrong than predicting SHIELD when the answer was JAB — the former is "wrong attack" while the latter is "wrong category of behavior entirely." But our current 400-class cross-entropy loss treats both as equally wrong.

The world model doesn't need to be "right" about what the player is doing in the theoretical sense. It needs to produce states that lead to correct downstream physics — positions, velocities, damage, hitstun. If the simulator says "some kind of grounded attack came out" and gets the knockback roughly right, that's a usable simulation even if it picked the wrong specific attack.

## Directions to Test

### 1. Grouped Action States (Softer Targets)

Collapse ~400 action states into functional categories:
- **Movement**: WAIT, WALK, DASH, RUN, TURN, SQUAT, etc.
- **Grounded attacks**: JAB1-3, FTILT, UTILT, DTILT, FSMASH, USMASH, DSMASH
- **Aerial attacks**: NAIR, FAIR, BAIR, UAIR, DAIR
- **Specials**: NEUTRAL_B, SIDE_B, UP_B, DOWN_B (per character)
- **Defensive**: SHIELD, ROLL, SPOTDODGE, AIRDODGE
- **Hitstun/knockback**: DAMAGE_*, TUMBLE, DOWN_*
- **Recovery**: CLIFF_*, EDGE_*, REBOUND
- **Grab/throw**: GRAB, GRABBED, THROW_*

**Test A — Category-only prediction:** Replace the 400-class head with a ~20-class head. Measure category accuracy vs. the physics metrics (pos_mae, velocity error). If physics stays good, the label doesn't matter.

**Test B — Hierarchical loss:** Keep the 400-class head but add a second loss term for category accuracy. Weight category errors higher than within-category errors. Something like:
```
loss = CE(action_pred, action_true) + 2.0 * CE(category_pred, category_true)
```

**Test C — Confusion-weighted CE:** Build a cost matrix where cross-category mistakes cost more than within-category mistakes. Use it to weight the cross-entropy loss.

### 2. Embedding-Space Loss

Instead of predicting a discrete action class, predict a point in the learned action embedding space.

The model already has a 32-dim action embedding. Currently it's used as input only (embed the current action state for context). The idea: also use it as an output target.

**Test D — Embedding regression:** Replace the 400-class CE head with a regression head that predicts a 32-dim embedding vector. Loss = MSE between predicted embedding and the target action's learned embedding. Actions that are mechanically similar (different jabs, different tilts) have nearby embeddings, so "close" becomes almost as good as "right."

**Test E — Embedding + classifier:** Keep the 400-class head but add an auxiliary embedding regression loss. The model learns to get the embedding neighborhood right even when the exact class is wrong:
```
loss = CE(action_pred, action_true) + 1.0 * MSE(embed_pred, embed_true)
```

**Test F — Contrastive embedding:** Train the action embeddings with a contrastive objective so that mechanically similar actions cluster together. Then use embedding regression as the prediction target. This might produce better embeddings than the ones learned purely from state prediction.

### 3. Direct Physics Prediction (Skip the Label)

The radical version: don't predict action state at all. Predict the physical outcomes directly.

**Test G — Physics-only heads:** Remove the action classification head entirely. Predict:
- Position delta (x, y)
- Velocity (air_x, y, ground_x)
- Damage dealt this frame
- Hitstun frames applied
- Shield damage
- Whether the character is actionable next frame (binary)

**Test H — Physics + action embedding (no classifier):** Predict a latent action embedding (32-dim) alongside the physics predictions. The embedding isn't tied to any specific action state — it's a learned intermediate that the model uses to organize its internal representations. At inference time, you can nearest-neighbor the embedding back to an action state for visualization, but the model is optimized purely on physics accuracy.

### 4. Combined Approaches

**Test I — Hierarchical + embedding:** Category classifier + embedding regression + physics heads. The full kitchen sink. Compare total physical accuracy against the simpler approaches to see if the complexity helps.

**Test J — Curriculum:** Start with the current 400-class CE (easy to optimize), then gradually anneal toward embedding-space loss (harder but more forgiving). Let the model learn the discrete structure first, then relax into the softer target.

## Evaluation

For all tests, the metrics that matter are:

1. **Action-change accuracy** (current metric — keep for comparison)
2. **Category accuracy** (new — how often is the functional category right?)
3. **Physics accuracy** (pos_mae, velocity_mae, damage_mae) — the real bottom line
4. **Rollout coherence** (new — how many frames before the autoregressive rollout produces implausible states?)

A model that scores 60% on action-change but 95% on category accuracy and 0.5 pos_mae is probably a better simulator than one that scores 75% action-change but 85% category accuracy and 0.9 pos_mae.

## Implementation Priority

Start with the cheapest experiments:

1. **Test B (hierarchical loss)** — just add a category head, minimal code change
2. **Test D (embedding regression)** — swap one head, reuse existing embeddings
3. **Test A (category-only)** — simplify the head, fast to train, answers "does the label matter?"
4. **Test G (physics-only)** — remove a head, answers "do we need action state at all?"

Then combine winners. Tests C, F, I, J are more involved and should wait for clear signal from the simpler experiments.

## Connection to Rollout Accuracy

This directly addresses the rollout drift problem. In autoregressive simulation:
- A wrong action state that's in the right *category* might produce a plausible next frame
- A wrong action state in the wrong category will produce a physically implausible next frame (shielding when you should be attacking, etc.)
- Physics-focused loss directly optimizes for what matters in rollouts

If we can get the physical consequences right even when the exact action label is wrong, rollouts will stay coherent much longer. That's the path to the distributable simulator.

## Why Isn't Action Change Already ~100%?

Melee's action transitions are **deterministic**. Given current action state + exact animation frame + controller input + environmental context, the next action state is a state machine lookup. So why 72.4% and not 95%+?

### Things we encode but imprecisely

**State age precision.** We normalize `state_age` as float × 0.01. But Melee's interrupt windows are exact frame counts — "you can jump-cancel shine on frames 1-3" requires knowing you're on frame 2, not "0.02." The model has to learn to decode the normalization AND learn the frame windows. Integer or higher-precision encoding would give this information directly.

**Button press vs hold.** We encode "is A pressed: 1/0" but Melee distinguishes PRESSING A (transition from 0→1) vs HOLDING A (staying at 1). Pressing triggers an attack; holding does nothing. The model can infer this from context (compare frame t vs t-1), but it's doing inference work that could be a direct input. That's 8 extra binary features — "was this button pressed THIS frame for the first time."

### Things we don't encode at all

**Hitbox/hurtbox collision.** Whether an attack connects depends on spatial overlap between hitboxes and hurtboxes — geometry we don't model. We have positions but not "Fox's up-smash hitbox extends from y+5 to y+20 on frames 7-12." This directly affects whether the action state transitions to HIT_*, SHIELD_STUN, etc.

**IASA frames (Interruptible As Soon As).** Each move has specific frame windows where it can be canceled. This is the core of Melee's depth — the model has to learn these windows from thousands of examples rather than having them as structured input.

**Stale move queue.** Melee tracks the last 10 moves for damage scaling. Affects knockback calculations, which affects whether kill moves actually kill (action transition to DEAD_*).

### Model capacity

26 characters × ~400 action states × variable frame windows × environmental context = a very large transition table. 1.3M params may not be enough. The Mamba-2 backbone (Phase 2) addresses this.

## Determinism Experiments

A series of targeted runs to close the gap between 72% and theoretical ceiling, ordered by implementation cost.

### Experiment 1: State Age Precision

**Hypothesis:** Integer-precision state_age unlocks transition window learning.

**Change:** Encode state_age as integer (add to int tensor, embed like action state) OR as higher-precision float (× 1.0 instead of × 0.01). Keep the normalized version too — the model might want both.

**Variants:**
- 1a: state_age as raw integer, embedded (vocab ~200, embed dim 8)
- 1b: state_age as float × 1.0 (un-normalized) alongside the × 0.01 version

### Experiment 2: Press Events

**Hypothesis:** Explicit button press events (0→1 transitions) are the primary trigger for action changes.

**Change:** For each of the 8 buttons per player, add a binary "just_pressed" feature: 1 if the button is pressed on frame t but was NOT pressed on frame t-1, 0 otherwise. +16 floats to the controller input (8 buttons × 2 players).

**Variants:**
- 2a: Press events added to `next_ctrl` only (the conditioning signal)
- 2b: Press events added to all context frames too (richer history)

### Experiment 3: Wider Prediction Window (Delay/Lookahead)

**Hypothesis:** Some action transitions don't fully resolve on the same frame as the input — they play out over 3-6 frames. Real netcode (GGPO, Slippi rollback) handles this with input delay for exactly this reason. Expanding the target window might help.

Melee's engine processes input → state on the same frame, but some transitions are multi-frame: a smash attack has startup frames before the hitbox appears, and the action state might not change until 2-3 frames after the input. The model seeing only frame t's input and predicting frame t's state might be asking for a prediction that hasn't fully "landed" yet.

**Change:** Instead of predicting frame t given ctrl(t), predict frame t+d given ctrl(t) through ctrl(t+d), where d is a small delay (1-5 frames).

**Variants:**
- 3a: Predict frame t+1 given context [t-K:t] + ctrl(t) + ctrl(t+1) — one frame of lookahead
- 3b: Predict frame t+3 given context [t-K:t] + ctrl(t:t+3) — three frames of lookahead
- 3c: Predict frame t+5 given context [t-K:t] + ctrl(t:t+5) — five frames (full GGPO-style buffer)

This is interesting because it means the model has MORE information (multiple frames of future input) and the action transition has had time to resolve. The tradeoff: at inference time for the RL pipeline, you'd need the policy to commit to multiple frames of input at once, which is actually how Phillip works (it outputs actions at 10Hz, not 60Hz, for exactly this reason).

### Experiment 4: Generalize to Specialize

**Hypothesis:** Training on more diverse data helps the model understand underlying mechanics, not just memorize specific situations. The "generalize to help specialize" principle.

This isn't a code change — it's a data experiment. The overnight run (22K games, all characters, all stages) is already testing this. But we can make it more targeted:

**Variants:**
- 4a: Compare 22K mixed vs 4K Fox-only — does the mixed model predict Fox better than the Fox-only model?
- 4b: Hold out a rare character entirely (Pichu, Game & Watch), train on everything else, measure zero-shot accuracy on the held-out character. If mechanics generalize, it should predict basic transitions correctly even for unseen characters.

## Pre-Experiment Predictions

Predictions made Feb 24, 2026 before running any experiments. Baseline: v2.2 on 2K games, 2 epochs = **62.4% change_acc**, 0.79 pos_mae, 0.299 val_loss. Compare after.

### Layer 1 — Softer Targets

| Test | change_acc | pos_mae | Reasoning |
|------|-----------|---------|-----------|
| **A** (category-only, ~20 classes) | N/A (no exact prediction) | 0.80-0.85 | Can't measure change_acc since we're not predicting exact states. Physics stays similar — the model still learns dynamics, just with a coarser action signal. Might slightly degrade because exact action state was a useful intermediate for physics. |
| **B** (hierarchical loss) | **64-66%** (+2-4pp) | 0.78-0.80 | Modest win. Extra category gradient steers the model away from cross-category errors. Cheap and additive. Won't hurt, probably helps a little. |
| **C** (confusion-weighted CE) | **64-67%** (+2-5pp) | 0.78-0.80 | Similar magnitude to B, different mechanism. More implementation effort for roughly the same gain. |
| **D** (embedding regression only) | **58-62%** (−2 to 0pp) | 0.76-0.80 | Nearest-neighbor from predicted embedding back to action state will be noisier than direct classification. But embedding distance of errors will be smaller — wrong predictions are at least in the right neighborhood. Physics might *improve* slightly since the loss is smoother. |
| **E** (embedding + classifier) | **63-66%** (+1-4pp) | 0.77-0.80 | The embedding auxiliary loss acts as a regularizer — "when you're wrong, be wrong nearby." Modest improvement, physics stays similar. |
| **F** (contrastive embedding) | Depends on D/E | — | Not directly comparable — this is a precursor that improves the embedding space. Only matters if we're using embedding regression (D or E). If current embeddings already cluster similar actions (haven't checked), gains are small. If they don't, could meaningfully help D/E. |
| **G** (physics-only) | N/A | **0.70-0.80** | All model capacity goes to physics. Might slightly beat baseline on pos_mae. But discontinuous events (getting hit, attack landing) are harder to predict without action state as an organizing intermediate. This is the "do we need the label at all" test — if physics is good, the answer is no. |
| **H** (physics + latent embed) | N/A (nearest-neighbor: ~55-62%) | **0.68-0.78** | Slightly better physics than G — the latent embedding gives the model a way to internally organize state changes without being graded on exact labels. Best physics of any softer-target approach. |
| **I** (kitchen sink) | **64-67%** | 0.75-0.80 | Marginal improvement over best of B/E. More parameters, more loss terms, but diminishing returns from stacking. The complexity probably isn't worth it unless individual components each showed 3+ pp gains. |
| **J** (curriculum) | **64-68%** | 0.76-0.80 | Interesting but hard to predict. The anneal from CE → embedding loss is a bet that discrete structure learned early helps the softer target converge better. Could be the best of this group or could just be extra complexity. |

**Layer 1 overall take:** These experiments are about graceful degradation — being less wrong when wrong. Expected gains are modest (+2-5pp on change_acc). The real value shows up in rollout coherence, not single-frame accuracy. If category accuracy is 90%+ while change_acc is only 65%, rollouts will stay physically plausible much longer.

### Layer 2 — Determinism Experiments

| Experiment | change_acc | pos_mae | Reasoning |
|------------|-----------|---------|-----------|
| **1a** (state_age integer embed) | **65-68%** (+3-6pp) | 0.78-0.80 | The model currently normalizes state_age to 0.02, 0.03, etc. and has to learn that "0.04 means frame 4, which is inside the interrupt window for shine." Integer embedding makes this a lookup instead of a learned decoding. Biggest impact on IASA-window transitions (jump-cancel, L-cancel timing, interrupt windows). Won't help with transitions that don't depend on frame-exact timing. |
| **1b** (state_age float × 1.0) | **64-66%** (+2-4pp) | 0.78-0.80 | Same information as 1a but as a continuous value. Model still has to learn discrete boundaries from a float. Strictly worse than 1a but still better than × 0.01 because the resolution isn't crushed. |
| **2a** (press events, next_ctrl only) | **66-69%** (+4-7pp) | 0.78-0.80 | **Highest-confidence prediction.** Button press is THE trigger for action changes. The model can already infer press vs hold by comparing ctrl(t) to ctrl(t-1) in context, but it has to learn this diffing operation rather than having it given. Making it explicit removes a computation step the model is spending capacity on. Every missed "player just pressed A" directly causes a wrong action prediction. |
| **2b** (press events, all frames) | **67-70%** (+5-8pp) | 0.77-0.79 | Slightly better than 2a — richer history of recent button transitions helps the model understand input sequences (tap vs hold, double-tap, etc.). The marginal gain over 2a is small since the most important press event is the current frame's. |
| **3a** (1-frame lookahead) | **64-67%** (+2-5pp) | 0.80-0.85 | One extra frame of input, one more frame for transition to resolve. Modest improvement on action prediction. But pos_mae might *increase* slightly — predicting t+1 from context through t is a bigger jump than predicting t, and the model doesn't see intermediate states. |
| **3b** (3-frame lookahead) | **68-74%** (+6-12pp) | 0.82-0.90 | **Biggest potential swing.** Multi-frame transitions (smash startup, shield drop, ledge options) have fully resolved by t+3. The model gets 4 frames of input context. But physics gets harder — the state 3 frames out has more variance. This is the tradeoff: action prediction improves because transitions are settled, physics prediction degrades because you're reaching further into the future. |
| **3c** (5-frame lookahead) | **69-75%** (+7-13pp) | 0.85-0.95 | Diminishing returns on action prediction over 3b (most transitions resolve within 3 frames). Physics continues to degrade with the longer horizon. Probably not worth the physics cost over 3b. |
| **4a** (22K mixed vs 4K Fox-only) | Mixed wins by **3-8pp** on Fox matchups | Similar | The 22K dataset has way more than 4K Fox games (Fox is ~20% of competitive Melee). Shared mechanics (grabs, shields, knockback, DI) learned from other characters transfer. The Fox-only model memorizes Fox situations; the mixed model understands *mechanics* that happen to apply to Fox. |
| **4b** (zero-shot held-out character) | **45-55%** change_acc on held-out char | 0.80-0.90 | Basic transitions transfer: movement, getting hit, shielding, grabs — these are shared mechanics. Character-specific moves (specials, unique properties like float, tether) will be mostly wrong. Category accuracy should be much higher (~80-85%) since the model knows "this looks like an attack" even if it picks the wrong specific attack. Rare characters (Pichu, G&W) have simpler movesets, so they might transfer better than complex ones. |

**Layer 2 overall take:** These experiments attack the real ceiling — giving the model the information it needs to be correct. Exp 2 (press events) is the safest bet: low risk, clear mechanism, moderate reward. Exp 3b (3-frame delay) is the biggest swing: high potential on action prediction but physics might suffer. Exp 1a (state_age embed) is a clean targeted fix for a specific class of errors.

### Combined predictions

| Combination | change_acc | Notes |
|-------------|-----------|-------|
| **1a + 2a** (state_age + press events) | **70-76%** | Additive — they address different error sources. This should roughly match the current 22K overnight (72.4%) but on only 2K games and 2 epochs. |
| **1a + 2a + 3b** (all three) | **74-80%** | If the gains are even partially additive. But 3b's physics cost might force tradeoffs. |
| **1a + 2a at 22K scale** | **78-84%** | Scale + better encoding. This is the realistic target for "how high can the current architecture go." |
| **Best config + Mamba-2** | **82-90%** | Speculative. More model capacity resolves the "too many transitions for 1.3M params" problem. The remaining gap would be hitbox collision and stale move queue — things we genuinely don't encode. |

### What a miss looks like

If these predictions are wrong, the most likely failure modes:

1. **Exp 2 (press events) shows < 2pp gain**: Would mean the model is already doing the press-vs-hold diffing effectively, and the bottleneck is elsewhere. This would shift priority toward Exp 3 (delay) and the softer-target approaches.
2. **Exp 3b (delay) shows no action improvement**: Would mean transitions genuinely resolve on the same frame and the delay just makes physics harder. Would kill the GGPO analogy.
3. **Exp 1a (state_age) shows < 1pp gain**: Would mean interrupt windows aren't a significant error source at current accuracy levels — the model already handles them well enough from context.
4. **Exp 4a (mixed model loses to Fox-only on Fox)**: Would mean cross-character training is adding noise, not signal. Would change our data strategy significantly.

## Phase Gate: What Goes Where

### Phase 1 Goal

Validate the encoding and data pipeline. Prove which features matter. Find the MLP ceiling. **Don't optimize what Phase 2 will replace.**

Phase 1 exit criteria:
1. Encoding experiments run — know which features move the needle
2. MLP ceiling established on 22K with best encoding
3. `category_acc` metric implemented — baseline for Phase 2
4. Clear evidence that capacity (not encoding) is the remaining bottleneck

### Phase 1 Experiments (run now, on the MLP)

These answer "is our data right?" — results carry directly into Phase 2.

| Experiment | Data size | Why Phase 1 |
|------------|-----------|-------------|
| **Exp 1a** (state_age embed) | 2K, 2ep | Encoding question. Does the feature matter? |
| **Exp 2a** (press events) | 2K, 2ep | Encoding question. Highest-confidence bet. |
| **Exp 2b** (press events all frames) | 2K, 2ep | Only if 2a shows signal — is richer history worth the dims? |
| **Exp 3b** (3-frame delay) | 2K, 2ep | Frame windowing question. Changes the prediction contract — need to know before Phase 2 if this is the right framing. |
| **Exp 4a** (mixed vs Fox-only) | 4K, 5ep | Data strategy question. Needs variety — can't run on 2K. Answers "does diversity help?" before we invest in more data. |
| **category_acc metric** | n/a | Implement the metric. Needed for everything downstream. |
| **Benchmark run** | 22K, 10ep | Best encoding config from above. Establishes MLP ceiling. **One run, not many.** |

Total: ~5-6 quick runs (1h each) + 1 overnight. Then move on.

### Phase 2 Experiments (run on Mamba-2)

These answer "can we squeeze more from a bigger model?" — not worth testing on 1.3M params.

| Experiment | Why Phase 2 |
|------------|-------------|
| **Test A** (category-only head) | Answers "does the label matter?" — more interesting when the model has capacity to potentially learn fine-grained labels. On MLP, it's already struggling with capacity. |
| **Test B** (hierarchical loss) | Loss function tuning. Small gains on MLP; potentially larger on a model that's not capacity-bottlenecked. |
| **Test C** (confusion-weighted CE) | Same rationale as B. More complex, save for when the model is worth optimizing. |
| **Test D** (embedding regression) | Interesting direction but needs a model with enough capacity that dropping the classifier is a real tradeoff, not just losing signal. |
| **Test E** (embed + classifier) | Auxiliary loss. Worth testing once the primary loss is well-optimized on a bigger model. |
| **Test F** (contrastive embedding) | Research direction. Expensive. Only if D/E show promise on Phase 2. |
| **Test G** (physics-only) | The radical "skip the label" test. Answers a fundamental question about what the model needs — better asked on a model with real capacity. |
| **Test H** (physics + latent embed) | Same as G but with a learned intermediate. Phase 2 territory. |
| **Test I** (kitchen sink) | Combine Phase 2 winners. |
| **Test J** (curriculum) | Training schedule optimization. Only on the model we're actually keeping. |
| **Exp 1b** (state_age float ×1.0) | Strictly weaker than 1a. Only run if 1a fails and we want to understand why. |
| **Exp 3a** (1-frame delay) | Strictly weaker than 3b. Only run if 3b fails and we want to isolate the delay variable. |
| **Exp 3c** (5-frame delay) | Only if 3b shows signal and we want to find the optimal delay. |
| **Exp 4b** (zero-shot held-out) | Cool research question, not blocking progress. Run when we have cycles. |

### Future: Multi-Instance Rollout (Phase 3)

Run three parallel world model instances (inspired by rollback netcode): a "server" and two "players," each doing independent rollouts. Compare predictions frame-by-frame.

**What it buys:**
- **Consensus as confidence.** Where all three agree, you're almost certainly right. Where they diverge, the model is uncertain.
- **Error detection.** Divergence points are natural resync moments — snap to majority vote (action state) or average (continuous values).
- **Uncertainty map.** Log every frame where instances disagree. That's a free hard-example-mining signal for training.
- **Staggered delay windows.** Each instance could use a different lookahead (ctrl(t), ctrl(t-1:t+1), ctrl(t-2:t+2)), then reconcile — closer to how real GGPO handles different views of the input timeline.

**What it doesn't buy:** An authoritative state. No instance is "right" — the server is just the consistent one everyone resyncs to. Consistently wrong in a coherent way > three models independently wrong in different directions.

**Prerequisite:** A model accurate enough that the instances mostly agree (>90% change_acc). Below that, disagreement is too frequent to be a useful signal.

## Test Plan: Two-Machine Run Schedule

All experiments use v2.2 (input-conditioned) as baseline. Same 2K game dataset for controlled comparison except where noted.

### Round 1: Quick validation (2K games, 2 epochs each) — COMPLETE

Both ran on ScavieFae (Scav was running v2.2 overnight). Feb 24, 2026.

| Run | What's different | wandb |
|-----|-----------------|-------|
| Exp 1a: state_age embed | +8 dims per player, integer embedded | `8eck1wxa` |
| Exp 2a: press events | +16 floats in next_ctrl | `89t6zcjt` |

Baseline: 3 runs averaged (wandb `wgk0lluj`, `fpl5kpjk`, `jlvvp198`). All 2K games, 2 epochs, default EncodingConfig.

**Results (val metrics, epoch 2):**

| Config | Val Loss | Val Acc | Val Change Acc | Pos MAE | Delta vs baseline |
|--------|----------|---------|----------------|---------|-------------------|
| **Baseline** (avg×3) | 0.3045 | 0.964 | 0.645 | 0.79 | — |
| **Exp 2a** (press events) | 0.3010 | 0.964 | 0.664 | 0.78 | **+1.9pp** change_acc |
| **Exp 1a** (state_age embed) | 0.2665 | 0.970 | 0.717 | 0.79 | **+7.2pp** change_acc |

**vs predictions:**

| Experiment | Predicted | Actual | Verdict |
|------------|-----------|--------|---------|
| Exp 1a (state_age embed) | +3-6pp (65-68%) | **+7.2pp (71.7%)** | Beat ceiling. State age precision matters more than expected. |
| Exp 2a (press events) | +4-7pp (66-69%) | **+1.9pp (66.4%)** | Below floor. Model already infers press-vs-hold from context. |

**Analysis:**
- Exp 1a is the clear winner. The learned integer embedding for state_age unlocks frame-exact transition windows — the model no longer has to decode `0.04 → frame 4` from a scaled float. Val loss drop (−0.038) is substantial, indicating better fit across all heads, not just action prediction.
- Exp 2a shows signal but is marginal. The +1.9pp is real (consistent across epochs, beats all 3 baseline runs) but fell short of the "highest-confidence" prediction. The failure mode we predicted: "would mean the model is already doing the press-vs-hold diffing effectively." The 10-frame context window gives the model plenty of recent button history to diff against.
- Both experiments improved on baseline — "both" remains a valid combined config.
- Physics (pos_mae) unchanged for both. Neither experiment hurt continuous prediction.

### Round 2: Combined + lookahead (2K games, 2 epochs each)

Both run on ScavieFae sequentially (Scav still finishing v2.2 overnight).

| Order | Run | Config | What's different |
|-------|-----|--------|-----------------|
| 1st | Exp 1a+2a combined | `exp-1a2a-combined.yaml` | Both flags: state_age embed + press events |
| 2nd | Exp 3a: 1-frame lookahead | `exp-3a-lookahead-1.yaml` | Predict t+1 from ctrl(t)+ctrl(t+1). Needs code changes. |

**What we're testing:**
- **1a+2a combined:** Are the gains additive? Pre-experiment prediction: 70-76% change_acc. Given 1a alone = 71.7%, the question is whether 2a's +1.9pp stacks on top or is already subsumed by the state_age improvement.
- **3a (1-frame lookahead):** Does giving the model one extra frame of controller input (and predicting one frame further out) help? This is the gentler version of 3b — tests the delay concept without the physics penalty of a 3-frame horizon. Predicted: +2-5pp over baseline.

### Round 3: Scale test (overnight)

| Machine | Run | What's different |
|---------|-----|-----------------|
| Scav | Best config from Round 2, 22K games streaming, 10 epochs | Full scale |
| ScavieFae | Exp 4a: 4K Fox-only vs 4K mixed, 5 epochs each | Generalization test |

### What to measure

For every run, compare:
1. **action_change_acc** — the headline number
2. **action_change_acc by category** (new) — are we getting the broad strokes right even when the specific state is wrong?
3. **Physics metrics** (pos_mae, velocity error) — unchanged or better?
4. **Val loss** — overfitting check

The goal: find which combination pushes action-change accuracy highest while maintaining physics accuracy. Then take that to Phase 2 (Mamba-2) where model capacity is no longer the bottleneck.

---

*These directions emerged from discussing what "correct" means for a world model that serves as a game simulator rather than a replay predictor. The core insights: (1) agents care about physics, not labels, (2) Melee's transitions ARE deterministic — the gap is in our encoding, not the problem, (3) more diverse training data helps understand mechanics, not just memorize situations.*
