# Feature Diagnostics — When to Expand the Encoding

**Date:** February 24, 2026

We deliberately excluded several available fields from the world model's encoding (see RUNBOOK "Deliberately Excluded" table). Each exclusion was correct for Phase 1, but some become worth revisiting based on specific error patterns observed during training.

This document defines the diagnostic criteria: what to look for, where to look, and what adding the feature would cost. The goal is to avoid speculative feature engineering — add fields only when the error pattern is clear and the existing encoding demonstrably can't cover it.

All diagnostic experiments can run at 2-4K games. We're detecting patterns, not benchmarking final accuracy.

---

## 1. animation_index

**What it is:** A numeric ID mapping 1:1 to the visual animation played during an action state. For most actions, animation_index == action_state. They diverge when characters share action_state IDs but play different animations (Fox and Marth both use ATTACK_S_3, but Fox's is a kick and Marth's is a sword swing — different frame timing, range, hitboxes).

**Why we skipped it:** Redundant with (action_state + character). The model receives both as embeddings and can learn the cross-product.

**Current coverage:** The model has `action_state` (32-dim embed) and `character` (8-dim embed) in every context frame. In principle, 32+8 = 40 dims is enough to represent any character-specific action behavior. The question is whether the MLP trunk can learn the interaction efficiently.

### Error pattern that signals we need it

**Cross-character action confusion.** The model applies one character's frame timing to another character performing the same action_state.

Concrete examples:
- Fox's up-smash (ATTACK_HI_3) has hitboxes active frames 7-13, Marth's up-smash has hitboxes active frames 8-17. If the model predicts damage appearing on frame 7 for Marth (Fox's timing), that's the signal.
- Peach's down-smash (ATTACK_LW_3) hits on frames 5-39 (long, spinning). Most characters' down-smash is 2-4 active frames. If the model treats all down-smashes as short-duration, it's not distinguishing the animation.
- Jigglypuff's rest (ATTACK_LW_3 or SpecialN) has 1 active frame. If the model treats it like a normal attack duration, it's averaging across characters.

### Where to look

1. **Run Exp 4a (mixed characters vs Fox-only) first.** If Fox-only and mixed have similar change_acc, the model handles the cross-product fine and animation_index won't help.

2. **If mixed loses significantly (>3pp change_acc):** Break down errors by (action_state, character) pairs. Sort by error rate. If the worst-performing pairs are shared action_states where characters have very different animations (smashes, specials), that's the signal.

3. **Specific metric:** Per-character action_change accuracy on attack-class actions (action_state 40-80). If one character's attacks are predicted well and another's are systematically wrong, the model is leaking timing across characters through the shared action_state embedding.

4. **Control check:** If the accuracy loss in mixed mode is uniform across all action types (not concentrated on attacks), the problem is model capacity, not encoding — go to Phase 2 (Mamba-2) instead of adding features.

### What adding it costs

- Vocab ~400, embed_dim 16 → +6,400 embedding params
- Input dim: 1846 → 2166 → trunk layer 1 grows by +163,840
- **Total: ~1,474K params (+13%)**
- No re-parsing. peppi_py exposes `post.animation_index` directly; add to `parse.py` extraction and `encoding.py` categorical list.
- Experiment size: 4K games, 2 epochs. Compare mixed-character change_acc with and without animation_index.

### Decision threshold

Add animation_index if:
- Exp 4a shows mixed is >3pp worse than Fox-only on change_acc, AND
- Error analysis shows the gap concentrates on shared action_state IDs with divergent character animations (attacks, specials — not movement states which are universal)

Don't add it if:
- Mixed vs Fox-only gap is <2pp (the model handles the cross-product fine)
- The gap is uniform across action types (capacity problem, not encoding)
- Fox-only experiments are the primary focus anyway (animation_index = action_state for single-character)

---

## 2. state_flags (bit field features)

**What it is:** 5 bytes (40 bits) of per-frame state flags from the Slippi spec. Includes: reflect active, shield reflect, in hitlag (redundant — we have hitlag float), fastfall, above camera, hitbox connected this frame, defensive collision occurred, offscreen, dead, sleep mode.

**Why we skipped it:** Most flags are inferable from (action_state + state_age + velocity). A character in GUARD action with shield > 0 is shielding. A character with downward velocity exceeding terminal fall speed is fastfalling. The model can learn these from existing features.

**Current coverage:** We encode `hurtbox_state` (vulnerable/invulnerable/intangible) as a 2-dim embedding, which captures the most important defensive flag. We also have `hitlag` as a float, which tells the model when hit interactions are happening.

### Error pattern that signals we need them

**State transition errors on interaction frames.** The model correctly predicts what both players are doing in neutral, but makes errors specifically on frames where an attack connects, a projectile reflects, or a shield interaction occurs.

Concrete examples:
- Model predicts shield holding when a grab should have connected (missed the grab-vs-shield interaction, which depends on the "defensive collision" flag logic).
- Model doesn't predict percent increase on the correct frame when two attacks trade (simultaneous hitbox collision — "hitbox connected" flag would mark the exact frame).
- Shield-poke situations: model predicts continued shielding when the attack actually hit through a depleted shield. The flags encode shield sub-states the model might not infer from shield_strength alone.
- Reflect interactions (Fox shine → projectile reversal): the reflect flag tells the model a projectile was reflected this frame. Without it, the model only sees the shine action and the projectile... which it doesn't even track (we don't encode projectiles).

### Where to look

1. **Segment validation errors by action category.** Group frames into: neutral/movement, attack startup, attack active (hitbox out), shielding, hitstun, and ledge. If error rates spike specifically on "attack active" and "shielding" frames but not on movement frames, the model is struggling with interaction resolution — the part where flags carry information the other features don't.

2. **Binary head accuracy breakdown.** The binary prediction head outputs (facing, invulnerable, on_ground) per player. If `invulnerable` accuracy is notably lower than `facing` and `on_ground`, the model can't infer invulnerability from action_state alone — some invulnerability comes from flags (intangible ledge grab frames, etc.). Note: we already have `hurtbox_state` covering this, so if invulnerable prediction is bad, hurtbox_state might not be sufficient.

3. **Percent prediction timing.** Look at frames where percent changes (hit landed). If the model predicts the percent change 1-2 frames early or late relative to ground truth, it's guessing at the interaction frame rather than detecting it. The "hitbox connected" flag would nail the exact frame.

4. **Compare hitlag transition accuracy.** On frames where hitlag goes from 0 → N (a hit just connected), what's the model's error rate vs frames where hitlag is steady or decrementing? If the 0→N transition is much harder to predict, the model needs more signal about when hits connect.

### What adding them costs

Not all 40 flags are useful. A practical subset:

| Flag | Why useful | Inferable from existing? |
|------|-----------|------------------------|
| hitbox_connected | Marks exact frame a hit lands | Partially (hitlag 0→N), but 1 frame late |
| fastfall | Changes fall speed/landing timing | Yes — velocity_y exceeds threshold |
| reflect_active | Projectile reflection | No — action_state alone doesn't distinguish |
| shield_reflect | Shield-based reflection (Yoshi parry) | No |
| dead | Death flag | Yes — stocks decrement |
| offscreen | Blast zone exit | Partially — position extremes |

Realistic subset: **6-10 binary features** per player, not all 40.

- Per-player dim: 89 + 10 = 99
- Input dim: (99×2 + 4)×10 + 26 = 2,046
- Trunk layer 1 growth: +102,400
- **Total: ~1,407K params (+7.9%)**
- **Requires re-parsing.** Need to add bit flag extraction to `parse.py` via `post.state_flags`. peppi_py 0.8.6 returns `None` for flags (per our exclusion table), so this needs investigation — may require a peppi_py update or manual bit unpacking from the raw SLP data.
- Experiment size: 4K games, 2 epochs. Compare change_acc and percent MAE with and without flags.

### Decision threshold

Add state_flags if:
- Percent prediction MAE is significantly worse on hit-connection frames vs steady-state frames (>2x error ratio)
- AND/OR binary head accuracy for `invulnerable` is >5pp below `facing` and `on_ground`
- AND the interaction-frame error pattern persists after Exp 1a (state_age embed) and Exp 2a (press events) — those might already help by giving the model better timing resolution

Don't add them if:
- Error analysis shows the model's mistakes are about WHICH action happens (action prediction), not WHEN interactions resolve (timing). Flags help with timing, not action selection.
- peppi_py can't expose the flags without significant parser work. The re-parsing cost makes this a Phase 2 candidate if the tooling isn't ready.

---

## 3. Hitbox frame data (external game tables)

**What it is:** Melee's character DAT files contain frame-by-frame hitbox data for every action: on which frames hitboxes are active, their position relative to the character, size, knockback, damage. Community resources (ikneedata.com, frame data spreadsheets) have this fully documented.

**Why we skipped it:** IP defensibility. Including Nintendo's hitbox tables in the training data makes the project look like it depends on reverse-engineered game code. The model learning physics from observation is cleaner — it discovers the same information empirically, without copying the source.

**Current coverage:** The model observes the *effects* of hitboxes (percent changes, knockback via velocity, action state transitions into hitstun) without seeing the hitboxes themselves. With enough data, it can learn the correlations: "Fox ATTACK_HI_3 at state_age 7-13, opponent within distance X → percent increase + hitstun transition."

### Error pattern that signals we need it

**Spatial prediction errors during attacks.** The model knows an attack is happening (correct action state) and roughly when (state_age), but gets the *where* wrong — predicting hits at incorrect ranges, or missing hits that should connect based on positioning.

Concrete examples:
- Two characters are close during an attack's active frames, but the model doesn't predict a hit. The hitbox doesn't reach — the model doesn't know the attack's spatial extent. With enough training data this is learnable, but some attacks have unusual disjointed hitboxes (Marth's sword) that are hard to infer from position data alone.
- Model predicts damage from an attack that whiffed. The characters were in range for most attacks but this particular move has a small/narrow hitbox. The model is using position proximity as a proxy for "hit connects" and doesn't distinguish move-specific ranges.
- Knockback direction errors. Melee's knockback depends on the Sakurai angle, which varies per hitbox. The model can only learn this from aggregate statistics — if it hasn't seen enough examples of a specific move hitting at a specific percent, it'll predict average knockback instead of the correct angle.
- Multi-hit move errors. Moves like Fox's drill (ATTACK_AIR_LW) have multiple hitboxes on different frames. The model might predict one hit when two should connect, or vice versa, because it doesn't know the exact active frame windows.

### Where to look

1. **Position MAE conditioned on action type.** Split pos_mae into: movement frames, attack startup, attack active, hitstun. If attack-active frames have significantly higher pos_mae than movement frames, the model struggles with hit resolution physics — which depends on hitbox data.

2. **Percent prediction accuracy by range.** For attack interactions, bin the results by distance between attacker and target. If the model is accurate at close range (where all hitboxes connect) but inaccurate at mid-range (where only disjointed hitboxes like swords reach), it's struggling with move-specific spatial data.

3. **Character-specific attack resolution.** Compare percent prediction accuracy during attacks for disjointed characters (Marth, Samus — hitboxes far from body) vs close-range characters (Fox, Falcon — hitboxes near body). If disjointed characters have significantly higher error rates, the model needs spatial hitbox data to distinguish move reach.

4. **Per-move hit rate calibration.** For the most common attacks (jab, dash attack, aerials), compare the model's predicted hit rate (does it predict percent change?) vs the actual hit rate in training data, binned by distance. Well-calibrated = the model learned the hitbox range empirically. Poorly calibrated = it needs the lookup table.

### What adding it costs

This is a lookup table, not a learned feature — parameter impact is near zero. The cost is in data engineering:

- **Source:** Community frame data (ikneedata.com, ssbwiki, or the 20XX dat extraction). ~26 characters × ~50 actions each × ~60 frames per action = ~78,000 entries.
- **Encoding:** Per frame, add a binary `hitbox_active` flag (or a small set: `hitbox_active`, `hitbox_size`, `hitbox_offset_x`, `hitbox_offset_y`). These are looked up from the table using (character, action_state, state_age) as keys.
- **Dims:** 1-4 floats per player. Per-player dim 89 → 90-93. Input dim ~1,850-1,880. Negligible parameter increase (<1%).
- **The catch:** Building and validating the lookup table is real work. Different move types have 1-4 hitboxes with different IDs, priorities, and positions. Getting this right for 26 characters is a multi-day project.
- **IP catch:** Even using community-sourced frame data (vs raw DAT files), the information ultimately comes from Melee's game code. It's more defensible than distributing DAT files, but still a grey area.

### Decision threshold

Consider hitbox frame data if:
- The model has converged on Phase 2 (Mamba-2, all encoding improvements applied)
- AND spatial attack resolution errors persist — specifically, the model's hit/no-hit predictions are poorly calibrated at mid-range distances
- AND the accuracy gap between close-range characters and disjointed characters is >5pp
- AND the IP implications have been discussed and accepted

This is the last resort, not an early optimization. The whole thesis of the world model is that it learns Melee's physics from observation. If it can't learn hitbox ranges from 22K games of examples, that's important to know — but adding the lookup table means we're not purely learning from replays anymore.

Don't add it if:
- Phase 2 (more model capacity) closes the spatial accuracy gap on its own. The MLP might simply lack the capacity to learn move-specific ranges, and Mamba-2 with 5-10x parameters might handle it.
- The model's errors are about timing (when), not space (where). Timing errors are better addressed by state_age embedding and press events.
- We want to keep "no game code required" as a selling point for the project.

---

## Diagnostic Experiment Sizing

All of these diagnostic checks work at 2-4K games. The experiments are about detecting error *patterns*, not final accuracy numbers.

| Diagnostic | Min data | What to measure |
|-----------|---------|----------------|
| animation_index signal | 4K mixed-character | Per-(action_state, character) change_acc breakdown |
| state_flags signal | 4K any | Hit-frame vs steady-frame error ratio, binary head accuracy split |
| Hitbox data signal | 4K any | Pos_mae by action category, percent accuracy by distance bin |

Mattie's instinct on 4K is right. These are pattern-detection runs, not benchmarks. If the error pattern is real, it'll be visible at 4K. If it's marginal at 4K, the feature probably isn't worth adding.

---

## 4. Tuning Existing Encoding — Embed Dims, Scales, Architecture

The features above are about *adding new fields*. This section is about *changing how we encode what we already have*. Same idea: specific error patterns → specific lever to pull.

### Changing embed_dim on existing categoricals

**When to increase:** The embedding is too small to distinguish categories the model needs to tell apart. Signal: two action states that behave very differently (e.g., WAIT vs GUARD) get confused with each other — the model predicts one when it should predict the other. If you inspect the learned embeddings after training and find that dissimilar categories have very similar vectors (cosine similarity > 0.9), the embedding is too cramped.

**When to decrease:** The embedding is so large that rare categories never get enough training signal to learn useful vectors. Signal: rare action states (low frequency in training data) have high prediction error compared to common ones, AND increasing data doesn't help. Smaller embeddings are easier to learn from few examples.

**Current candidates:**
- `action_embed_dim=32`: Unlikely to need change. 400 vocab into 32 dims is moderate. Would revisit only if the action head plateaus while other heads keep improving — that suggests the embedding can't represent the necessary distinctions.
- `ground_embed_dim=4`: Might be too large. In competitive play on legal stages, ground is mostly binary (grounded vs airborne, plus a few platform IDs). Could try 2 and see if anything is lost. Low priority — ground prediction isn't a bottleneck.
- `last_attack_embed_dim=8`: Reasonable. 64 vocab, 8 dims. Would increase to 12-16 if we see the model confusing attacks with different properties (a weak jab vs a strong smash getting similar embeddings).

**Experiment size:** 2K games, 2 epochs. Change one embed_dim, compare the specific head's accuracy. These are fast cheap runs.

### Changing normalization scales

**When to change:** A continuous feature's gradients are either dominating training (scale too large, values too big) or being ignored (scale too small, values near zero). Signal: one prediction head improves while another stagnates or gets worse, and the stagnating head's input features have very different magnitudes from the improving head's.

**How to detect:** Log the mean and std of each continuous feature in the first training batch. If one feature has mean 5.0 and another has mean 0.001, the model will prioritize the larger one. Ideally all normalized features should be in roughly (-1, 1) range.

**Current risk areas:**
- `state_age × 0.01`: Frame 100 → 1.0, frame 1 → 0.01. This is a 100:1 dynamic range in one feature. The Exp 1a embedding approach sidesteps this entirely.
- `combo_count × 0.1`: Combo count 1 → 0.1, combo 10 → 1.0. Range is fine.
- Velocities `× 0.05`: These can spike during knockback (speed_y of 3.0+ game units/frame → 0.15 after scaling). Probably fine.

**Don't change scales speculatively.** They interact with learning rate and weight initialization in non-obvious ways. Only tune if a specific head is demonstrably underperforming relative to its feature magnitudes.

### Changing context length (K)

**When to increase (K>10):** The model makes errors on state transitions that depend on events further back than 167ms. Signal: action prediction accuracy is high for simple transitions (A press → attack) but low for complex sequences (wavedash = jump → airdodge → land, which spans ~15 frames). If the context window can't see the jump that initiated the wavedash, it can't predict the landing.

**When to decrease (K<10):** Training is slow and the model isn't using the early context frames. Signal: if you mask out frames [t-10, ..., t-6] (oldest 5 frames) and accuracy barely changes, those frames aren't contributing. Shorter context = faster training and smaller model.

**Cost:** Each frame of context adds 182 dims to the input → 182×512 = +93,184 trunk params per frame added. K=10→K=15 would add ~466K params (+36%). This is a bigger lever than embed_dim changes.

**Phase note:** Context length matters much more for Mamba-2 (Phase 2) than for the MLP. The MLP treats all K frames as a flat concatenation — it can't learn temporal patterns across frames, just correlations in the big flat vector. Mamba-2 processes frames sequentially and can learn "what happened 5 frames ago matters for this transition" explicitly. Increasing K for the MLP has diminishing returns past ~10 frames.

### Changing trunk width (hidden_dim, trunk_dim)

**When to increase:** All heads are still improving at end of training (loss hasn't plateaued), suggesting the model hasn't converged — it needs more capacity. OR: one head is doing well but another is stuck, and they share the same trunk bottleneck. The 256-dim trunk is forcing all game knowledge through a narrow pipe.

**When to decrease:** The model overfits (train loss keeps dropping, val loss goes up). Smaller trunk = less capacity = less overfitting. But first try increasing dropout or adding weight decay before shrinking the trunk — those are cheaper interventions.

**Current status:** At 1.3M params with 22K games of training data, we're unlikely to overfit. The more probable scenario is underfitting (model too small for the task), which is the Phase 2 thesis.

### Changing loss weights

**When to change:** One head's accuracy is excellent while another's is poor, and you want the model to shift attention. Current weights: continuous=1.0, binary=0.5, action=2.0, jumps=0.5. Action is weighted 2x because it's the primary metric.

**Signal to increase a weight:** That head's validation accuracy is low AND its training accuracy is also low (the model isn't even fitting the training data for that target). Increasing the weight forces the model to pay more attention.

**Signal to decrease a weight:** That head's accuracy is near-perfect (>99%) — the model has "solved" it and further training on it is wasted gradient signal. Decrease its weight to free up capacity for harder tasks.

**Caution:** Changing loss weights is the easiest thing to fiddle with and the hardest to get right. It's a balancing act — making action weight 5.0 might improve action accuracy but tank position prediction. Change one weight at a time, by small amounts (e.g., 2.0 → 3.0, not 2.0 → 10.0).

---

## Priority Order

Based on implementation cost and expected signal:

1. **Exp 1a + 2a first** (state_age embed + press events). These are Phase 1 experiments already planned. Run them before any of the features in this doc.
2. **animation_index** — cheapest to add (no re-parsing, one new embedding). Run diagnostic after Exp 4a (mixed characters).
3. **state_flags** — moderate cost (re-parsing, peppi_py investigation). Run diagnostic after Exp 1a + 2a results are in.
4. **Hitbox frame data** — highest cost (data engineering + IP question). Defer to Phase 2 at earliest, likely Phase 3.
