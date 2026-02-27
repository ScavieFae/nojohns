# E001: Focal Loss on Binary Head

**Category:** loss
**Target wall:** on_ground recall crisis, facing recall crisis
**Target metric:** on_ground recall > 75% (baseline: 52.3%)

## Hypothesis

The binary prediction head (on_ground, facing, invulnerable) is trained with vanilla BCE loss on a dataset where ~55% of frames are airborne. The loss gradient is dominated by correctly predicting the majority class (airborne). Focal loss downweights well-classified examples (easy negatives), forcing the model to focus on the hard cases — landing frames and facing transitions.

## Intervention

Replace `F.binary_cross_entropy_with_logits` in `metrics.py` with focal BCE:

```python
def focal_bce_with_logits(logits, targets, gamma=2.0, alpha=0.75):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()
```

`alpha=0.75` upweights the minority class (grounded/facing-right). `gamma=2.0` is the standard focal loss exponent.

## What to measure

- on_ground precision, recall, F1 (primary)
- facing precision, recall, F1
- Overall action accuracy and change accuracy (should not degrade)
- BAIR/FAIR change accuracy (downstream of facing — should improve if facing improves)
- Autoregressive rollout visual quality (qualitative)

## Run config

Baseline config (`mamba2-medium-gpu.yaml`) with one change: focal BCE on binary head. 2K games, 2 epochs, A100. ~$6.

## Risks

- Aggressive alpha/gamma could flip the bias — model starts falsely predicting grounded, hurting precision. Sweep gamma in {1.0, 2.0, 3.0} if first run is ambiguous.
- Focal loss changes the total loss magnitude. May need to adjust binary loss weight.
