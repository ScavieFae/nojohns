# World Model Glossary

**Date:** February 24, 2026

Reference definitions for the concepts and design levers in our world model. All examples use our actual encoding config values.

---

## Concepts

**Dimensions (dims)** = the width of a number array. A single float like `percent = 0.42` is 1 dim. A position `(x, y)` is 2 dims. When we say "89 dims per player," that's the total width of the vector representing one player's state on one frame.

**Embeddings** = a learned translation from a category to a vector of dims. Think of it as a lookup table that the model builds during training.

Why we need them: `action_state` is a number 0-399, but those numbers are arbitrary labels — state 44 (ATTACK_S_3) isn't "bigger" than state 14 (WAIT). If we fed 44 as a raw number, the model would think ATTACK_S_3 is 3x WAIT. That's meaningless.

An embedding says: "I don't know what action 44 means yet — give me 32 blank floats and I'll learn what values to put there during training." After training, similar actions end up with similar vectors. The model discovers that ATTACK_S_3 and ATTACK_S_4 (forward smash variants) should have nearby embeddings, even though their IDs are arbitrary.

**Vocab** = how many categories the embedding supports. `action_vocab=400` means IDs 0-399. If a new action state appeared with ID 401, the model would crash — it has no row for it.

**Embed_dim** = how many floats each category gets. `action_embed_dim=32` means each of the 400 action states gets a 32-float vector. Bigger = more expressive, but more parameters to learn and more input width downstream.

**The relationship:** dims is the general term. Embed_dim is specifically the width of an embedding's output. A player's 89 dims = 29 raw floats + 60 dims from embeddings (32 action + 4 jumps + 8 character + 2 l_cancel + 2 hurtbox + 4 ground + 8 last_attack).

**Parameters (params)** = the total number of learnable numbers in the model. Our MLP has 1,304,182 params. This includes all embedding tables, all trunk weights, and all prediction head weights. More params = more capacity to learn complex patterns, but also more data needed to train them well.

**Context window** = the chunk of recent history the model looks at. Our context is K=10 frames (167ms at 60fps). The model sees frames [t-10, ..., t-1] plus frame t's controller input, and predicts frame t's state.

**Trunk** = the shared middle of the model that all prediction heads read from. Input (1846 dims) → trunk layer 1 (512 dims) → trunk layer 2 (256 dims) → prediction heads. The 256-dim bottleneck forces the model to compress all game knowledge into 256 floats.

**Prediction heads** = separate output layers, one per type of thing we're predicting. Each head reads the trunk's 256-dim output and predicts something different: continuous_delta (position/percent/shield changes), binary (facing/invulnerable/on_ground), action (next action state), jumps (jumps remaining).

**Delta prediction** = predicting the *change* from one frame to the next, not the absolute value. Model outputs Δx = +2.0, and we add it to the previous x. Works because most frames have tiny changes — the model learns corrections, not coordinates.

**Loss** = how wrong the model is, as a single number. Lower = better. Different heads use different loss functions: MSE (mean squared error) for continuous values, cross-entropy for categoricals, BCE (binary cross-entropy) for binary flags. The total loss is a weighted sum of all heads.

**Validation loss (val_loss)** = loss computed on held-out data the model never trains on. If train_loss drops but val_loss doesn't, the model is memorizing training data instead of learning generalizable patterns (overfitting).

**Action-change accuracy (change_acc)** = our headline metric. On frames where the action state *changes* (not holds), what fraction does the model predict correctly? Most frames (~96%) are action holds — overall accuracy is inflated by those easy predictions. Change_acc measures the hard part.

**Category accuracy (category_acc)** = not yet implemented but planned. Groups the 400 action states into ~15 categories (movement, attack, shield, hitstun, etc.) and measures whether the model gets the right *category* even if the exact action state is wrong. Predicting ATTACK_S_3 when the answer is ATTACK_S_4 is a near-miss; predicting WAIT when the answer is ATTACK_S_3 is a catastrophic miss.

---

## Design Levers

Every number in the encoding config is a decision. Here's the full set of levers we can pull, with current values.

### 1. Embed_dim (per categorical field)

How many floats each category gets:

| Field | Vocab | Embed_dim | Ratio | Reasoning |
|-------|-------|-----------|-------|-----------|
| action_state | 400 | 32 | 1:12.5 | Highest-value feature, most categories |
| character | 33 | 8 | 1:4 | Important but few categories |
| jumps_left | 8 | 4 | 1:2 | Tiny vocab, low information |
| ground | 32 | 4 | 1:8 | Mostly binary (airborne vs not) |
| last_attack | 64 | 8 | 1:8 | Moderate vocab, moderate signal |
| l_cancel | 3 | 2 | 1:1.5 | Basically ternary |
| hurtbox | 3 | 2 | 1:1.5 | Basically ternary |
| stage | 33 | 4 | 1:8 | Low variance (few legal stages) |

Rule of thumb: embed_dim ≈ min(50, vocab^0.25 × 8). But this is a guess — there's no universal formula.

### 2. Normalization scales

Raw floats get multiplied by a constant to keep them in a similar range. Position × 0.05 maps the range (-200, 200) to roughly (-10, 10). Percent × 0.01 maps (0, 999) to (0, 9.99). If one feature has values 1000x larger than another, the model's gradients get dominated by the big one.

Current scales come from slippi-ai's `embed.py` — vladfi1 tuned these for his model. We inherited them. Probably fine but not sacred.

### 3. Context length (K=10)

How many past frames the model sees. K=10 at 60fps = 167ms of history. Longer context = more information but wider input (input_dim grows by 182 per frame added).

### 4. Trunk architecture (hidden_dim=512, trunk_dim=256)

The MLP's width. 512 is the first layer, 256 is the bottleneck before the prediction heads. All information about the game state gets compressed into 256 floats before any prediction happens.

### 5. What's a float vs what's an embedding

`state_age` is currently a float (×0.01). Exp 1a proposes making it an embedding instead. `combo_count` is a float (×0.1) but could be an embedding (vocab ~20). This is a design choice per field: do the numeric relationships between values matter (use float), or are the values arbitrary/categorical (use embedding)?

Float = "the difference between 5 and 10 is meaningful, and it's the same as the difference between 10 and 15." Good for position, percent, velocity.

Embedding = "these are labels, not quantities. The relationship between values is something the model should learn, not assume." Good for action_state, character, ground surface.

Grey area = `state_age`. Frame 7 being "later" than frame 5 is meaningful (float-like), but frame 7 of an attack having active hitboxes while frame 6 doesn't is a discrete threshold (embedding-like).

### 6. What's predicted as delta vs absolute

Continuous targets (percent, x, y, shield) are predicted as deltas: model outputs Δx, and we add it to the previous frame's x. This works because most frames have tiny changes. We could predict absolute values instead, but then the model wastes capacity learning "x is usually around 0" instead of "x changed by 2."

### 7. Loss weights

How much each prediction head matters during training. Currently: continuous=1.0, binary=0.5, action=2.0, jumps=0.5. Action is weighted 2x because it's the hardest and most important target. These weights change what the model prioritizes learning.

---

## See Also

- `../research/literature/FEATURE-DIAGNOSTICS.md` — when to pull each lever (error patterns and decision thresholds)
- `../research/literature/ACTION-PREDICTION-DIRECTIONS.md` — experiment schedule and predictions
- `../research/architecture/INPUT-CONDITIONING.md` — the v2.2 architecture change
