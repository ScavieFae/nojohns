# Model Analysis: all_d21_imitation_v3.pkl

**Date:** 2026-02-02
**Status:** ✅ Successfully loaded and verified

## Model Details

**File:** `all_d21_imitation_v3.pkl`
**Source:** Dropbox link in slippi-ai test suite
**Size:** 40.37 MB
**Parameters:** 10,582,808 (10.6M)
**Format:** JAX/Sonnet parameter tuple (137 arrays)

## Configuration

**Training Type:** Imitation Learning (Phase 1)
**Delay:** 21 frames (~350ms reaction time)
**Network:** `tx_like` (transformer-like architecture)
**Controller Head:** `autoregressive` (generates button presses sequentially)

## Structure

```python
{
    'state': {
        'policy': tuple of 137 numpy arrays (float32)
    },
    'config': {
        'runtime': {...},
        'dataset': {...},
        'network': {'name': 'tx_like'},
        'controller_head': {'name': 'autoregressive'},
        'policy': {'delay': 21, ...},
        'embed': {...},
        ...
    },
    'name_map': {
        # Player names the model was trained on
        'Platinum Player', 'Master Player', 'Diamond Player',
        'Hax', 'Cody', 'Amsa', 'Kodorin', ...
    }
}
```

## Verification Tests

✅ **Pickle load:** Success (pure Python)
✅ **Structure validation:** Correct format
✅ **Parameter count:** 10.6M params (reasonable size)
✅ **Data types:** All float32 (4 bytes per param)
✅ **Size check:** 40.37 MB matches expected size

## What This Model Can Do

Based on the configuration and name:

1. **Multi-character:** Likely trained on all Melee characters
2. **Imitation learning:** Plays in human-like style (not RL-refined)
3. **21-frame delay:** Slightly higher delay than the strongest models (d18)
4. **Test/demo quality:** In the test suite, so not the absolute strongest

This is **perfect for our needs:**
- Good enough to be interesting
- Not so strong it would be overpowered
- Publicly available (in test suite)
- Demonstrates the technology works

## Next Steps

1. Install slippi-ai dependencies
2. Test loading with official loader
3. Try running `scripts/eval_two.py` with this model
4. Build PhillipFighter adapter
5. Test in nojohns local match

## Notes

- Model appears to be trained on top-level players (Hax, Cody, Amsa, etc.)
- This is an imitation model (phase 1) not RL-refined (phase 2)
- Perfect for proof-of-concept before requesting stronger models
- Being in the test suite suggests it's intended for public use
