# E003: Auxiliary Physics Losses

**Category:** loss
**Target wall:** Impossible state transitions in autoregressive rollout (hallucinated KOs, percent decreasing, positions outside blast zones)
**Target metric:** Fewer physics violations in rollout; secondary: overall val loss

## Hypothesis

The model currently learns physics rules implicitly from data. It sometimes violates them — stocks increase, percent decreases within a stock, characters teleport outside blast zones. These are currently handled by post-hoc clamping in the rollout harness.

Adding soft penalty terms to the training loss teaches the model the rules instead of duct-taping violations after the fact. This is the PINN (Physics-Informed Neural Networks) approach, well-established in the literature.

## Intervention

Add to `metrics.py` training loss:

| Penalty | What it enforces | Implementation |
|---------|-----------------|----------------|
| stock_monotonic | Stocks can only decrease | Penalize predicted Δstocks > 0 |
| percent_monotonic | Damage doesn't decrease within a stock | Penalize predicted Δpercent < 0 (except on respawn) |
| blast_zone_penalty | Positions stay in bounds | Soft penalty for predicted x/y outside stage boundaries |
| shield_range | Shield in [0, 60] | Clamp penalty on decoded shield |
| velocity_consistency | Δposition ≈ velocity | Penalize divergence between position delta and velocity prediction |

## What to measure

- Physics violation rate in 600-frame autoregressive rollouts (count impossible transitions)
- Val loss (should decrease if physics losses are teaching real structure)
- Action/change accuracy (should not degrade — physics losses target continuous heads)

## Run config

Baseline config + auxiliary losses. Weight each penalty at 0.1× (low enough to not dominate the primary loss). 2K games, 2 epochs, A100.

## Risks

- If penalty weights are too high, model trades accuracy for constraint satisfaction. Start low.
- Some "violations" are real game events (percent resets on respawn). Need to mask respawn frames.
- velocity_consistency assumes position delta = velocity, but Melee has hitlag frames where position doesn't change but velocity is stored. May need to mask hitlag frames.
