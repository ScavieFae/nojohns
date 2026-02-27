# Commitment Window Analysis

**Date:** 2026-02-27
**Data:** 50 games sampled from 2K training set, 963 hit interactions

## The Question

How long is a "commitment window" in Melee — from when a player commits to an attack through the victim exiting hitstun? This determines how much temporal context the model needs to see the full consequences of a move.

## Findings

### Full commitment windows (attack commit → hitstun resolution)

| Metric | Frames | Time |
|--------|--------|------|
| Mean | 51.5 | 858ms |
| Median | 38 | 633ms |
| P75 | 64 | 1067ms |
| P90 | 100 | 1667ms |
| P95 | 135 | 2250ms |
| P99 | 262 | 4367ms |
| Max | 622 | 10.4s |

### Breakdown: attack phase vs hitstun phase

**Attack phase** (commit to move → hit connects):
- Mean 15.3 frames, median 8, P95 43

**Hitstun phase** (hit connects → victim exits hitstun):
- Mean 36.2 frames, median 27, P90 80, P95 107

Hitstun is the longer component. High-knockback moves at high percent produce very long hitstun.

### Context window coverage

| K | Time | % of commitment windows covered |
|---|------|--------------------------------|
| **10 (current)** | **167ms** | **14.6%** |
| 20 | 333ms | 28.5% |
| 30 | 500ms | 39.5% |
| 45 | 750ms | 58.9% |
| **60** | **1000ms** | **71.7%** |
| 90 | 1500ms | 87.6% |
| 120 | 2000ms | 93.0% |

### Action duration distribution (all actions, not just attacks)

| Duration | Count | % |
|----------|-------|---|
| 1-5 frames | 42,763 | 47.8% |
| 6-10 frames | 19,493 | 21.8% |
| 11-20 frames | 11,363 | 12.7% |
| 21-30 frames | 7,053 | 7.9% |
| 31-60 frames | 7,937 | 8.9% |
| 61-120 frames | 798 | 0.9% |
| 121+ frames | 99 | 0.1% |

Nearly half of all action runs are ≤5 frames. But the consequential ones (attacks, hitstun) are much longer.

## Implications

1. **K=10 is fundamentally too short for combat context.** The model sees less than 15% of commitment windows. It's predicting state transitions without seeing the interaction that caused them.

2. **K=60 is the natural target.** Covers 71.7% of interactions, equals one second of game time, and is the threshold where the model can see most attack-through-hitstun arcs.

3. **K=90-120 for completeness.** Covers 87-93%. Diminishing returns but captures the heavy hits.

4. **The attack phase is short (median 8 frames).** The model needs to see past the hit into the hitstun to understand the interaction. A context window that captures the attack but not the hitstun is almost worse than one that captures neither — it sees the cause but not the effect.

5. **Combined with focal context:** If the model sees K=60 frames with focal_offset=D, it could see both the attack and the full hitstun resolution for most interactions, while predicting a frame in the middle.
