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

### 3. Model Access ✅ (Bad News)
- [x] Are the trained model weights publicly available?
  - **NO - Intentionally not released**
  - Quote from vladfi1: "I am hesitant to release any trained agents as I don't want people using them on ranked/unranked, so at the moment the bot isn't available to play against locally."
  - **Old phillip repo** has some agents in `agents/delay0/` but project is deprecated
  - **New slippi-ai repo** does NOT include weights
- [x] Which agents should we prioritize?
  - Would need to train our own from scratch
  - Start with basic imitation learning (days on good GPU)
- [x] Can we get basic-* agents for testing?
  - **Not publicly available for slippi-ai**
  - Old phillip repo has some delay0 agents (but different architecture)
- [x] Licensing concerns?
  - **Repo is Apache 2.0 licensed** ✅
  - **Model weights:** Developer explicitly not distributing to prevent ranked abuse

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

## How to Contact x_pilot/vladfi1

**Discord (Best Option):**
- Phillip Discord: https://discord.gg/hfVTXGu
- Main Slippi Discord: https://slippi.gg/discord (40k+ members, training data links here)

**GitHub:**
- Profile: https://github.com/vladfi1 (Vlad Firoiu)
- slippi-ai repo: https://github.com/vladfi1/slippi-ai
- Old phillip repo: https://github.com/vladfi1/phillip

**Twitch:**
- Channel: https://twitch.tv/x_pilot (streams almost constantly)
- Bot connect code: PHAI#591

## Resources

- **Source:** https://github.com/vladfi1/slippi-ai
- **Old Project:** https://github.com/vladfi1/phillip (has agents/delay0/ but deprecated)
- **Twitch:** https://twitch.tv/x_pilot
- **Discord:** https://discord.gg/hfVTXGu
- **Reddit bounty thread:** https://www.reddit.com/r/SSBM/comments/18jyduo/
- **Credits in nojohns README:** vladfi1/slippi-ai

## Key Discoveries

### How x_pilot Runs His Twitch Bot

From `scripts/twitchbot.py` (this is what he uses!):

**Model Storage:**
```python
MODELS_PATH = flags.DEFINE_string('models', 'pickled_models', 'Path to models')
```
- Default path: `pickled_models/` directory
- Bot scans this directory and loads ALL `.pkl` files

**Model Loading:**
```python
for model in os.listdir(self._models_path):
    path = os.path.join(self._models_path, model)
    state = saving.load_state_from_disk(path)
    add_agent(self._single_agent(model=model))
```

**Agent Types Created:**
1. **Regular agents** - from any .pkl in models path
2. **Imitation agents** - "basic-*" prefix (phase 1 training)
3. **Auto agents** - "auto-*" prefix (best matchup selection)
4. **Medium agents** - special configured agents

**Key Finding:** x_pilot definitely HAS all the model files in a local `pickled_models/` directory. They're just not distributed publicly.

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

## Options Moving Forward

### Option 1: Request Special Access from vladfi1
**Approach:** Explain nojohns use case (controlled tournament, not ranked abuse)
- **Pros:** Get actual trained models, skip weeks of training
- **Cons:** He may still say no (understandably protective)
- **How:** DM on Discord (https://discord.gg/hfVTXGu) or GitHub issue

**Message draft:**
> Hey! I'm working on "No Johns" - a tournament system for Melee AI fighters. We're integrating SmashBot and wanted to add Phillip as well. Our use case is strictly for controlled tournaments (not ranked/unranked play). Would you be willing to share basic imitation models, or is there any way we could get access? We totally understand if not - happy to credit the work properly and respect any restrictions you'd want.

### Option 2: Use Old phillip Repo Agents
**Approach:** Use the deprecated agents from vladfi1/phillip repo
- **Pros:** Publicly available in `agents/delay0/` directory
- **Cons:**
  - Old architecture (pure RL, not imitation learning)
  - "Subject to bit-rot" per README
  - Different codebase than slippi-ai
  - May not be as strong or human-like
- **Feasibility:** Medium - would need to integrate old phillip instead of slippi-ai

### Option 3: Train Our Own Phillip
**Approach:** Use slippi-ai framework to train from scratch
- **Pros:** Full control, could customize for tournament
- **Cons:**
  - Takes days-weeks on good GPU (3080Ti mentioned)
  - Need large Slippi replay dataset
  - Training costs (GPU time)
  - Significant time investment
- **Datasets:** Available in Slippi Discord (anonymized ranked collections)

### Option 4: Wait for Community Weights
**Approach:** Monitor for someone else releasing weights
- **Pros:** Free, no work needed
- **Cons:** May never happen, could be low quality
- **Likelihood:** Low - vladfi1's concerns about ranked abuse apply to anyone

### Option 5: Focus on Other Neural Fighters
**Approach:** Look for other neural network Melee AIs with available weights
- **Pros:** May find more accessible options
- **Cons:** Phillip is the strongest/most well-known
- **Examples to explore:**
  - Eric Gu's project ([ericyuegu.com/melee-pt1](https://ericyuegu.com/melee-pt1)) - also not releasing weights yet
  - Community forks of Phillip
  - Other RL projects

## Recommendation

**Start with Option 1** - reach out to vladfi1 with our use case. Key points:
- Controlled tournament environment (not public ranked)
- Educational/research project
- Willing to add restrictions (e.g., not connect to ranked)
- Just need basic imitation models to start

If that fails, **Option 2** (old repo agents) as fallback, even though they're deprecated.

## Notes

- The 18-frame delay is actually a FEATURE for our use case - makes it fairer in tournaments
- We don't need the absolute strongest agents - even basic-* would be interesting
- If we can't get models, we could potentially train our own using the framework
- **Phillip uses TensorFlow**, while we're on Python 3.12 - need to check TF compatibility
- **libmelee version mismatch**: Phillip uses v0.43.0 fork, we use mainline - potential issue
- **Developer's concern is valid:** Protecting ranked integrity is important
- **Our use case is different:** Controlled tournaments, not public matchmaking
