# Batch Teacher-Forced Eval: Error Analysis

**Date:** 2026-02-27
**Model:** mamba2-v3-2k-test-v2 (Mamba-2 4.3M, K=10, 2ep on 2K games)
**Eval:** 5 games, 116,662 frames, MPS

## Summary Numbers

| Metric | Value |
|--------|-------|
| Action accuracy | 96.7% |
| Change accuracy | 68.3% |
| on_ground accuracy | 77.8% |
| facing accuracy | 76.1% |
| Avg position error | 1.02 game units |

The 96.7% headline is inflated — ~85% of frames are continuation frames (same action as last frame). The model just has to predict "keep doing what you're doing." **Change accuracy (68.3%)** is the real number. One in three action transitions is wrong.

## The on_ground Recall Crisis

| | Pred True | Pred False |
|---|---|---|
| **Actually grounded** | 28,122 | **25,670** |
| **Actually airborne** | 245 | 62,625 |

99.1% precision, **52.3% recall**. The model defaults to "airborne." When a character IS on the ground, it misses it nearly half the time. But it almost never falsely claims grounded.

**This is the mechanism behind "characters stop taking visible actions" in autoregressive mode:**
1. Model predicts airborne when character is actually grounded
2. In autoregressive mode, that wrong prediction feeds back as context
3. Context now shows airborne → model keeps predicting airborne
4. Character locks into FALLING/aerial states indefinitely

Crucially, the time-bucket data shows this is NOT a drift problem in teacher-forced mode — accuracy is flat across all 10 time buckets (96.3-96.9%). The model makes the same ~22% on_ground errors every frame at a constant rate. Autoregressive mode just lets them compound.

### Why: It's a Collision Detection Problem

The model doesn't have stage geometry. It has a stage ID embedding — a 4-dimensional learned vector. From that, it must *learn* that FD ground is at y=0, Battlefield platforms are at y=27.2, Yoshi's platforms at y=23.45, etc.

Landing in Melee is a **discontinuity**. y-position approaches a surface, speed_y is negative, and on one exact frame — *snap* — on_ground flips true. The model has to learn `y ≈ 0 AND speed_y < 0 → on_ground`. That's a sharp decision boundary in continuous space, trained with vanilla BCE loss on a dataset where ~55% of frames are airborne. The loss gradient barely cares about nailing the exact landing frame.

Worse: **ECB (Environmental Collision Box) shifts**. Melee's collision detection uses a box that changes shape based on animation. Some moves extend the ECB below the character's nominal y-position, meaning the character lands *earlier* than y would suggest. The model can't see the ECB. y lies to it.

## Facing Has the Same Problem (and It's Coupled)

| | Pred True | Pred False |
|---|---|---|
| **Actually facing right** | 28,766 | **27,788** |
| **Not facing right** | 134 | 59,974 |

99.5% precision, 50.9% recall. Same one-directional bias — defaults to "not facing right."

Facing direction changes are **state-gated** in Melee. On the ground, you can turn by pressing the stick the other way. In the air, facing is mostly locked from the moment you leave the ground. Predicting facing *correctly requires knowing on_ground correctly first*. If the model thinks you're airborne when you're grounded, it predicts facing as locked when it should be changeable.

These two flags are **correlated failures**. Get one wrong, the other follows.

## BAIR vs FAIR: The Beautiful Consequence

FAIR = forward aerial (attack toward facing). BAIR = back aerial (attack behind you). Same button press, distinguished *entirely* by facing direction.

- BAIR change accuracy: **20.9%**
- FAIR change accuracy: **12.7%**

The model gets controller input — it sees the A button and stick direction. But to decide FAIR vs BAIR, it must combine stick direction with facing. Facing has 50% recall. FAIR and BAIR become a coin flip. The numbers bear this out almost exactly.

NAIR (neutral aerial) doesn't depend on facing at all — and it's much easier for the model.

## TURNING: One-Frame Ghost

TURNING exists for exactly one frame in Melee. Standing → press stick backward → TURNING for one frame → immediately into DASHING or another grounded state. It's a state-machine edge that happens to be observable.

- Accuracy: 64.7%
- Change accuracy: **27.8%**

The same input pattern (standing + stick moving backward) could produce TURNING, DASHING directly, WALK_SLOW, or continued STANDING depending on analog stick thresholds and dead zones. These depend on values the model has to learn purely from data.

## The Dead Zone Theory

Melee has analog stick dead zones — below a threshold, tilting the stick does nothing. The thresholds differ by action (the dash-back dead zone is notoriously tight). Our controller encoding is the raw float value. The model has to learn all these thresholds from data, and small encoding errors near the boundary mean it can't reliably distinguish "stick at 0.49 (no action)" from "stick at 0.51 (walk begins)."

This specifically hits WALK_SLOW (33.1% change acc) and TURNING — states that live at the boundary between "nothing happens" and "movement begins."

## Damage Change Accuracy is a Fundamental Wall

24% change accuracy on the damage category. Not a bug — a ceiling.

To predict "you get hit this frame," the model would need the opponent's attack hitbox geometry and active frames, your hurtbox position, and frame-perfect range/timing calculations. We don't encode hitboxes. The model can learn rough correlations ("opponent in FSMASH frame 15 and close → you're about to eat it"), but connect-or-whiff is a sub-pixel calculation in the original engine.

## Category Breakdown

| Category | Count | Accuracy | Change Acc | Pos Error |
|----------|-------|----------|------------|-----------|
| aerial | 27,194 | 97.5% | 81.7% | 1.0 |
| aerial_attack | 15,909 | 96.6% | 53.7% | 1.02 |
| damage | 15,306 | 97.3% | 24.1% | 1.3 |
| special | 12,428 | 98.3% | 82.1% | 0.95 |
| movement | 10,939 | **89.8%** | 55.6% | 0.79 |
| shield_dodge | 8,952 | 96.9% | 78.3% | 0.77 |
| ground_attack | 6,914 | 98.7% | 73.5% | 0.59 |
| idle | 4,209 | 97.2% | 81.9% | 0.49 |
| grab | 3,370 | 97.5% | 78.9% | 0.88 |
| edge | 2,023 | 96.2% | 56.9% | 1.09 |

Movement is the weakest category overall (89.8%), dragged down by TURNING and WALK transitions.

## Position Error Distribution

| Bin | Count | % |
|-----|-------|---|
| [0,1) | 76,854 | 65.9% |
| [1,2) | 30,354 | 26.0% |
| [2,3) | 5,800 | 5.0% |
| [3,5) | 2,715 | 2.3% |
| [5,10) | 848 | 0.7% |
| [10,20) | 62 | 0.1% |
| [20+) | 29 | 0.0% |

91.9% of frames have position error under 2 game units. The model tracks position well. Outliers >10 units are likely stock/respawn or blast zone deaths.

## Invulnerable: Vacuous

0 TP, 0 FP, 0 FN, 116K TN. Either no invulnerability occurred in these 5 games or the encoding isn't capturing it. Invulnerability is brief (respawn, certain specials) — likely just rare.

## Research Note: Frame-1 Moves and Delay Frames

Moves that come out on frame 1 are a known problem in rollback netcode. Slippi handles this with **delay frames** — adding latency to account for input transmission time. From the [Slippi FAQ](https://github.com/project-slippi/slippi-launcher/blob/main/FAQ.md):

> Delay frames are how we account for the time it takes to send an opponent your input. Since we have removed about 1.5 frames of visual delay from Melee, adding 2 frames brings us very close to a CRT experience. Using a 120hz+ monitor removes an additional half frame to bring us in line with CRT.
>
> A single delay frame is equal to 4 buffer in traditional Dolphin netplay. Our recommended default is 2 frame delay (8 buffer). We suggest using 2 frame delay for connections up to 130ms ping.

Our world model faces an analogous problem: predicting frame-1 transitions (TURNING, attack startup) from a context window that doesn't include the triggering frame's result yet. Rollback netcode "solves" this by accepting that frame-1 predictions will be wrong and rolling back to correct them. We might consider a similar philosophy — accept that certain transitions are inherently unpredictable one frame in advance, and design the autoregressive loop to be robust to those errors rather than trying to eliminate them.

No action item yet, but this parallel between rollback netcode and autoregressive world models is worth exploring. Both systems are fundamentally predicting future game state from incomplete information, and both break on the same class of instant-transition events.

## Actionable Fixes

1. **Weighted/focal loss on binary head.** The class imbalance is the proximate cause of the on_ground/facing crisis. Upweight minority class or use focal loss to punish missed landings harder.

2. **Check the ground categorical head.** We encode `ground` (surface ID, 0 = airborne) in the combat context heads. If that prediction is accurate, it's a second signal for on_ground through a different loss function. Worth checking its accuracy.

3. **Enable `press_events`.** Binary "button just pressed this frame" features (already a flag in EncodingConfig, not enabled). Would help with the dead zone problem and aerial attack prediction.

4. **Threshold tuning.** The binary head thresholds at 0.0 (sigmoid 0.5). Lowering to -0.5 or -1.0 would recover on_ground recall at the cost of some precision. Worth testing in autoregressive mode specifically.

5. **BAIR/FAIR resolves itself** if facing gets better.

6. **Damage ceiling (24% change acc) is fine.** Don't chase it.
