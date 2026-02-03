# Phillip Integration Research

**Branch:** `phillip-research`
**Goal:** Integrate Phillip neural network AI as a fighter option in nojohns

## TL;DR

**Phillip is totally viable for nojohns!** Here's what we found:

✅ **Good news:**
- Uses libmelee (same as SmashBot) - familiar integration
- Can run locally with Dolphin (not cloud-only)
- Apache 2.0 licensed (compatible with our MIT license)
- 18-frame delay is actually perfect for tournaments (prevents CPU speed advantage)
- Python-based, can import as module

❌ **Challenges:**
- **Model weights not public** - need to ask x_pilot or train our own
- **Python version mismatch** - Phillip uses 3.10-3.11, we're on 3.12 (TensorFlow constraint)
- **libmelee fork** - Phillip uses vladfi1's fork v0.43.0, we use mainline
- **GPU preferred** - CPU inference possible but slower

## What is Phillip?

Phillip is a neural network Melee AI that:
1. Trains on human Slippi replays (imitation learning)
2. Self-improves through reinforcement learning
3. Beats top players like Moky (3-10) and Cody Schwab

**Key feature for us:** 18-frame input delay makes it fair for tournaments (no "faster computer = better AI" problem).

## Architecture

```
┌─────────────────────────────────────┐
│  nojohns (our code)                 │
│  ┌───────────────────────────────┐  │
│  │ PhillipFighter (adapter)      │  │
│  │   implements Fighter protocol │  │
│  └──────────┬────────────────────┘  │
│             │                        │
│             ▼                        │
│  ┌───────────────────────────────┐  │
│  │ slippi-ai (Phillip's code)    │  │
│  │  - Agent class                │  │
│  │  - Neural network policy      │  │
│  │  - Delay buffer (18 frames)   │  │
│  │  - libmelee wrapper           │  │
│  └──────────┬────────────────────┘  │
│             │                        │
└─────────────┼────────────────────────┘
              │
              ▼
      ┌───────────────┐
      │   Dolphin     │
      │   (libmelee)  │
      └───────────────┘
```

## Integration Plan

### Phase 1: Setup & Research (DONE ✅)
- [x] Clone slippi-ai repository
- [x] Explore codebase structure
- [x] Understand dependencies
- [x] Design adapter pattern
- [x] Document findings

### Phase 2: Get Model Weights
- [ ] Option A: Ask x_pilot/vladfi1 for pre-trained models
  - Best for testing quickly
  - Start with basic imitation agents (smaller)
- [ ] Option B: Train our own
  - Would take days-weeks on good GPU
  - Need Slippi replay dataset
  - Community datasets in Slippi Discord

### Phase 3: Environment Setup
- [ ] Create compatible Python environment
  - Downgrade to Python 3.10 or 3.11 (for TensorFlow)
  - Or wait for TensorFlow 3.13 support
- [ ] Install slippi-ai dependencies
  - TensorFlow + TensorFlow Probability
  - dm-sonnet
  - vladfi1's libmelee fork
- [ ] Resolve libmelee version conflict
  - Option A: Use vladfi1's fork everywhere
  - Option B: Make Phillip work with mainline libmelee

### Phase 4: Implement Adapter
- [ ] Create `fighters/phillip/` directory
- [ ] Implement `PhillipFighter` class
  - Wraps Phillip's Agent
  - Implements our Fighter protocol
- [ ] Handle gamestate conversion
- [ ] Handle controller conversion
- [ ] Test locally: Phillip vs SmashBot

### Phase 5: Testing & Tuning
- [ ] Local matches
  - Phillip vs SmashBot
  - Phillip vs CPU
  - Phillip vs Phillip (ditto)
- [ ] Netplay matches
  - Test with delay + netplay lag
  - Tune delay parameter
- [ ] Performance optimization
  - CPU vs GPU inference
  - Async inference
  - Memory footprint

## Files in This Research

- **`claude.md`** - Detailed working notes with all research questions answered
- **`phillip_adapter_poc.py`** - Proof-of-concept adapter code (design sketch)
- **`slippi-ai/`** - Cloned repository of Phillip's code
- **`README.md`** - This file (summary)

## Key Technical Details

### Dependencies
```
Python 3.10-3.11
tensorflow
tensorflow-probability
dm-sonnet
dm-tree
libmelee (vladfi1 fork v0.43.0)
```

### Model Format
- Pickle files (`.pkl`)
- Saved to `experiments/<tag>/latest.pkl` during training
- Contains:
  - Neural network weights
  - Training config
  - Embedding parameters

### How Phillip Works (Each Frame)
1. Receive gamestate from libmelee
2. Add to 18-frame delay buffer
3. Get state from 18 frames ago
4. Embed gamestate into neural network input
5. Run forward pass (inference)
6. Convert network output to controller buttons
7. Send to Dolphin via libmelee

### The 18-Frame Delay
- Intentional training constraint
- Makes reactions more human-like
- ~300ms lag (acceptable for AI, terrible for humans)
- **Perfect for tournaments:** Prevents "faster CPU = unfair advantage"

## Next Steps

1. **Get model weights** - this is the blocker
   - Reach out to x_pilot on Discord/GitHub
   - Explain nojohns tournament use case
   - Ask for basic imitation agents to start

2. **Once we have a model:**
   - Set up Python 3.10/3.11 environment
   - Install slippi-ai + dependencies
   - Implement the adapter
   - Test locally with SmashBot

3. **After local testing works:**
   - Test in netplay
   - Tune performance
   - Document Phillip setup for users
   - Add to fighter registry

## Resources

- **slippi-ai repo:** https://github.com/vladfi1/slippi-ai
- **x_pilot's Twitch:** https://twitch.tv/x_pilot
- **Phillip Discord:** https://discord.gg/hfVTXGu
- **Reddit bounty thread:** https://www.reddit.com/r/SSBM/comments/18jyduo/
- **License:** Apache 2.0 (compatible with MIT)

## Questions?

See `claude.md` for detailed research notes and answered questions.
