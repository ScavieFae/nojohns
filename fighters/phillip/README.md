# Phillip Fighter

Neural network Melee AI fighter adapter for nojohns.

## What is Phillip?

Phillip is a neural network AI developed by vladfi1 that:
1. Learns from human Slippi replays (imitation learning)
2. Self-improves through reinforcement learning
3. Plays in a human-like style with consistent delays
4. Has beaten top players like Moky and Cody Schwab

## Status

✅ **Implementation Complete - Ready for Testing!** ✅

**What works:**
- ✅ Model loading
- ✅ Configuration
- ✅ Agent initialization
- ✅ Gamestate → action integration
- ✅ Controller state conversion
- ✅ Agent lifecycle (start/stop)

**What's TODO:**
- ⚠️ Testing with Dolphin
- ⚠️ Verification against SmashBot
- ⚠️ Performance tuning
- ⚠️ Integration into main fighter registry

## Setup

See `phillip-research/SETUP.md` for detailed setup instructions.

**Quick version:**
```bash
# Create Python 3.11 environment
cd phillip-research
python3.11 -m venv .venv-phillip
source .venv-phillip/bin/activate

# Install slippi-ai
cd slippi-ai
pip install -r requirements.txt
pip install -e .

# Test the model
cd ..
python test_phillip_model.py
```

## Usage

```python
from fighters.phillip import load_phillip

# Load a Phillip model
phillip = load_phillip("all_d21_imitation_v3")

# Use in a match (once integration is complete)
from nojohns.runner import LocalRunner
from fighters.smashbot import load_smashbot

runner = LocalRunner(...)
result = runner.run_match(
    phillip,
    load_smashbot("fox"),
    games=1
)
```

## Available Models

Currently we have:

**all_d21_imitation_v3.pkl** (40.3 MB)
- 10.6M parameters
- 21-frame delay
- Imitation learning (Phase 1)
- Trained on top players
- Public (from test suite)

Located in: `phillip-research/models/`

## How It Works

```
nojohns GameState
      ↓
PhillipFighter (adapter)
      ↓
slippi-ai Agent
      ↓
Neural Network (TensorFlow)
      ↓
Controller Output
      ↓
PhillipFighter (convert)
      ↓
nojohns ControllerState
```

## Integration Challenges

### 1. GameState Format

Phillip expects raw libmelee gamestate, we use our own GameState class.

**Solution:** Add `raw_state` attribute to GameState when using libmelee.

### 2. Agent Control Flow

Phillip's agent is designed to be called each frame and updates a controller directly.

**Solution:** Need to adapt the agent's step() pattern to our act() pattern.

### 3. libmelee Version

Phillip uses vladfi1's fork (v0.43.0), we use mainline libmelee.

**Solution:** Either:
- Use vladfi1's fork everywhere, OR
- Test compatibility and bridge differences

### 4. Python Version

Phillip requires Python 3.10-3.11, nojohns uses 3.12.

**Solution:** Separate venv for Phillip, or subprocess approach.

## Testing

```bash
# Test model loading
cd phillip-research
python test_phillip_model.py

# Test with eval_two.py (slippi-ai's script)
cd slippi-ai
python scripts/eval_two.py \
  --p1.type=human \
  --p2.ai.path=../models/all_d21_imitation_v3.pkl \
  --dolphin.path="/path/to/Slippi Dolphin.app" \
  --dolphin.iso="/path/to/melee.iso"

# Test adapter (once complete)
cd ../..
pytest tests/test_phillip_fighter.py
```

## Architecture Notes

**Phillip's delay buffer:**
- Agent maintains internal 21-frame history
- Each frame, adds current state to buffer
- Inference runs on state from 21 frames ago
- Output is deterministic for same input sequence

**Async inference:**
- Inference runs in background thread
- Reduces frame drops during forward pass
- Recommended for real-time play

**Multi-character models:**
- Some models support all characters
- Character is selected at agent build time
- We may need to specify which character to use

## Next Steps

1. ✅ Model acquired (all_d21_imitation_v3.pkl)
2. ✅ Setup documentation
3. ✅ Test script created
4. ✅ Adapter fully implemented
5. ✅ Agent control flow complete
6. ✅ act() implementation complete
7. ⚠️ Set up Python 3.11 environment
8. ⚠️ Test with Dolphin
9. ⚠️ Test Phillip vs SmashBot
10. ⚠️ Add to fighter registry

## Resources

- **Research:** `phillip-research/claude.md`
- **Setup:** `phillip-research/SETUP.md`
- **Model analysis:** `phillip-research/MODEL_ANALYSIS.md`
- **Source:** https://github.com/vladfi1/slippi-ai
- **Discord:** https://discord.gg/hfVTXGu
