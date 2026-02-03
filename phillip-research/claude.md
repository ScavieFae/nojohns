# Phillip Research - Claude Working Notes

**Date Started:** 2026-02-02
**Branch:** phillip-research
**Goal:** Integrate Phillip neural network AI as a fighter in nojohns

## What is Phillip?

Phillip is a neural network-based Melee AI developed by vladfi1 (x_pilot on Twitch).

### Key Characteristics:
- **Training approach:**
  - Phase 1: Imitation learning on human Slippi replays (creates "basic" agents)
  - Phase 2: Self-play reinforcement learning (creates advanced agents like fox_d18_ditto_v3)
- **Input method:** Fed game state info (positions, velocities, animations, etc.) each frame
- **Reaction time:** 18-frame delay (consistent, not perfect reactions)
- **No input reading:** Cannot see opponent's controller inputs
- **Characters available:** Fox, Marth, Puff, Falco, Sheik, Falcon, Peach, Samus, Yoshi
- **Strength:** Beats top players (Moky went 3-10 vs fox_d18_ditto_v2.1)

### Why 18-frame delay is workable:
- ~300ms lag (18 frames * 16.67ms/frame)
- Humans would hate this for playing, but Phillip is trained with this delay
- Phillip doesn't need low latency - it needs **consistent** latency
- Actually helpful for tournament fairness - prevents "fast CPU = better AI" issues

### Current Status:
- Hosted on x_pilot's Twitch channel
- Connect via Slippi direct connect: PHAI#591
- NOT publicly distributed (concerns about misuse)
- Source code: https://github.com/vladfi1/slippi-ai

## Research Questions

### 1. Architecture & Dependencies ✅
- [x] What does Phillip's runtime require? (Python version, TensorFlow/PyTorch, GPU?)
  - **Python 3.10 or 3.11** (tested versions)
  - **TensorFlow** + TensorFlow Probability
  - **dm-sonnet** (DeepMind's TensorFlow library)
  - **libmelee v0.43.0** (custom fork by vladfi1)
  - **Optional GPU** (TensorFlow can run on CPU, but slower)
- [x] Can we run Phillip locally without x_pilot's infrastructure?
  - **YES!** Can run locally via `scripts/eval_two.py`
  - Requires trained model weights (`.pkl` files)
- [x] What are the model file formats and sizes?
  - **Format:** `.pkl` (Python pickle files)
  - **Location:** `experiments/<tag>/latest.pkl` during training
  - Size: Unknown (need to find actual models)
- [x] Does it need specific versions of libmelee or other dependencies?
  - Uses **vladfi1's fork of libmelee** at v0.43.0
  - May have custom modifications not in mainline libmelee

### 2. Integration Points ✅
- [x] How does Phillip interface with Dolphin?
  - Uses **libmelee** (same as SmashBot!)
  - Has custom `dolphin.py` module wrapping libmelee
  - Creates `Agent` objects that read gamestate and output controller inputs
- [x] Does it use libmelee like SmashBot, or a custom interface?
  - **libmelee-based**, but with custom wrappers
  - See `slippi_ai/dolphin.py` and `slippi_ai/eval_lib.py`
- [x] Can we run Phillip + Dolphin on the same machine?
  - **YES!** That's the standard local play mode
  - Example: `python scripts/eval_two.py --p1.type human --p2.ai.path <model>`
- [x] What's the communication protocol between Phillip and the game?
  - libmelee's standard UDP protocol (port 51441)
  - Agent reads gamestate, outputs controller inputs via libmelee Controller class

### 3. Model Access ❓
- [ ] Are the trained model weights publicly available?
  - **NOT in the repo** (would be too large for git)
  - Need to check if x_pilot hosts them somewhere
  - May need to contact vladfi1 or train our own
- [ ] Which agents should we prioritize? (fox_d18_ditto_v3 for Fox?)
  - Start with **any basic imitation agent** for testing
  - Advanced agents: fox_d18_ditto_v3, marth_d18_ditto_v3
- [ ] Can we get basic-* agents for testing before trying advanced ones?
  - Basic agents are just phase 1 (imitation learning)
  - Would be smaller and easier to work with initially
- [ ] Licensing concerns?
  - **Repo is Apache 2.0 licensed** ✅
  - Model weights distribution: unclear, need to verify with x_pilot

### 4. Performance Requirements
- [x] GPU required? If so, what specs?
  - **GPU optional** but recommended for good performance
  - CPU inference possible (eval_lib.disable_gpus() is used in eval)
  - For training: needs good GPU (3080Ti mentioned in README)
- [ ] CPU requirements?
  - Unknown, likely moderate (TensorFlow inference)
- [ ] Memory footprint?
  - Unknown, depends on model size
- [x] Can it run on Apple Silicon (M-series)?
  - **TensorFlow supports Apple Silicon** via tensorflow-metal
  - Should work, but may need special setup

### 5. Implementation Strategy ✅
- [x] Fork slippi-ai and create adapter?
  - **Don't need to fork!** Can use as-is
  - Just need trained model weights
- [x] Run Phillip as separate process, communicate via IPC?
  - **Better approach:** Import slippi_ai as Python module
  - Create adapter that wraps Phillip's Agent in our Fighter protocol
- [x] Modify Phillip to implement our Fighter protocol?
  - **Adapter pattern** is cleanest
  - Phillip has its own architecture, we wrap it
- [x] Test with local matches first, then netplay?
  - **YES** - start with local, netplay should work too

## Next Steps

1. **Clone and explore slippi-ai repository**
   - Understand codebase structure
   - Identify dependencies
   - Find documentation

2. **Research model availability**
   - Check if weights are in repo or downloadable
   - Reach out to vladfi1/x_pilot if needed
   - Look for community forks with models

3. **Understand the interface**
   - How does it connect to Dolphin?
   - What does the game state input look like?
   - What does the action output look like?

4. **Prototype adapter**
   - Create `fighters/phillip/` directory
   - Implement Fighter protocol wrapper
   - Test with local match first

## Resources

- **Source:** https://github.com/vladfi1/slippi-ai
- **Twitch:** https://twitch.tv/x_pilot
- **Reddit bounty thread:** https://www.reddit.com/r/SSBM/comments/18jyduo/
- **Credits in nojohns README:** vladfi1/slippi-ai

## Key Discoveries

### Architecture Deep Dive

From exploring the slippi-ai codebase:

**Entry Point for Evaluation:**
- `scripts/eval_two.py` - main script to run agents
- Supports human vs AI, AI vs AI
- Command: `python scripts/eval_two.py --p1.type human --p2.ai.path <model>`

**Core Modules:**
- `slippi_ai/dolphin.py` - Dolphin wrapper using libmelee
- `slippi_ai/eval_lib.py` - Agent evaluation infrastructure
- `slippi_ai/policies.py` - Neural network policy implementation
- `slippi_ai/networks.py` - Model architecture
- `slippi_ai/embed.py` - Game state embeddings
- `slippi_ai/controller_heads.py` - Maps network outputs to controller inputs

**How It Works:**
1. Agent loads trained model (`.pkl` file)
2. Dolphin launches and connects via libmelee (UDP port 51441)
3. Each frame:
   - Agent receives gamestate from libmelee
   - Gamestate is embedded into neural network input
   - Network outputs action probabilities
   - Controller head converts to button presses
   - Buttons sent to Dolphin via libmelee Controller

**The 18-Frame Delay:**
- Implemented as a **console_delay** parameter in DolphinConfig
- Agent sees gamestate from 18 frames ago
- This is NOT network latency - it's an intentional training constraint
- Makes the AI more human-like (no frame-perfect reactions)

### Integration Approach

**Option 1: Direct Integration (Cleanest)**
1. Add slippi-ai as a dependency to nojohns
2. Create `fighters/phillip/adapter.py`
3. Wrapper class that:
   - Implements our `Fighter` protocol
   - Internally uses Phillip's `Agent` class
   - Handles model loading
4. Model path specified in fighter config

**Option 2: Subprocess (More Isolated)**
1. Run Phillip's eval script as subprocess
2. Communicate via shared Dolphin connection
3. More complex, but keeps dependencies separate

**Recommendation: Option 1**
- Cleaner integration
- Better error handling
- Can reuse our existing Dolphin management
- Easier to debug

### Critical Missing Piece: Model Weights

The biggest blocker is **getting trained model weights**. Options:

1. **Ask x_pilot/vladfi1:**
   - Reach out on Discord or GitHub
   - Ask for basic imitation agents (smaller, easier to share)
   - Explain nojohns tournament use case

2. **Train Our Own:**
   - Would take days-to-weeks on good GPU
   - Need large dataset of Slippi replays
   - Could start with basic imitation learning
   - Community datasets available in Slippi Discord

3. **Community Sources:**
   - Check if anyone else has shared weights
   - Look for Phillip forks with models

## Notes

- The 18-frame delay is actually a FEATURE for our use case - makes it fairer in tournaments
- We don't need the absolute strongest agents - even basic-* would be interesting
- If we can't get models, we could potentially train our own using the framework
- **Phillip uses TensorFlow**, while we're on Python 3.12 - need to check TF compatibility
- **libmelee version mismatch**: Phillip uses v0.43.0 fork, we use mainline - potential issue
