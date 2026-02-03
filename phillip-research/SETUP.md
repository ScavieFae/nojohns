# Phillip Integration Setup Guide

**Goal:** Get Phillip neural network AI working with nojohns

## Prerequisites

- Python 3.10 or 3.11 (NOT 3.12 - TensorFlow constraint)
- macOS with Rosetta 2 (for Dolphin)
- Slippi Dolphin installed
- Melee ISO

## Quick Start

```bash
# From nojohns root
cd phillip-research

# Create Python 3.11 environment (if you have it)
python3.11 -m venv .venv-phillip
source .venv-phillip/bin/activate

# Or use pyenv to get 3.11
# brew install pyenv
# pyenv install 3.11.7
# pyenv local 3.11.7
# python -m venv .venv-phillip
# source .venv-phillip/bin/activate

# Install slippi-ai dependencies
cd slippi-ai
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "from slippi_ai import saving; print('✅ slippi-ai installed')"
```

## Step-by-Step Setup

### 1. Python Version Management

**Option A: If you have Python 3.11 installed**
```bash
python3.11 --version  # Verify you have 3.11.x
python3.11 -m venv phillip-research/.venv-phillip
source phillip-research/.venv-phillip/bin/activate
```

**Option B: Install Python 3.11 with pyenv**
```bash
# Install pyenv if needed
brew install pyenv

# Install Python 3.11
pyenv install 3.11.7

# Set it for this directory
cd phillip-research
pyenv local 3.11.7

# Create venv
python -m venv .venv-phillip
source .venv-phillip/bin/activate
```

**Option C: Use conda/miniconda**
```bash
conda create -n phillip python=3.11
conda activate phillip
```

### 2. Install slippi-ai

```bash
cd phillip-research/slippi-ai

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install slippi-ai in development mode
pip install -e .
```

**Key dependencies installed:**
- `tensorflow` - Neural network framework
- `tensorflow_probability` - Probabilistic modeling
- `dm-sonnet` - DeepMind's TensorFlow library
- `dm-tree` - Tree utilities
- `libmelee` - Melee interface (vladfi1's fork v0.43.0)
- `pandas`, `wandb`, `pyarrow` - Data/logging tools

### 3. Verify Installation

```bash
# Test imports
python -c "
from slippi_ai import saving, eval_lib
import tensorflow as tf
print('✅ All imports successful!')
print(f'TensorFlow version: {tf.__version__}')
"

# Test model loading
python -c "
from slippi_ai import saving
state = saving.load_state_from_disk('../models/all_d21_imitation_v3.pkl')
print('✅ Model loads successfully!')
print(f'Model delay: {state[\"config\"][\"policy\"][\"delay\"]} frames')
"
```

### 4. Test with eval_two.py

```bash
# Run demo script (once we create it)
python ../test_phillip_model.py
```

## Dependency Notes

### TensorFlow on Apple Silicon

If you're on Apple Silicon (M1/M2/M3), TensorFlow should use Metal acceleration:

```bash
pip install tensorflow-metal  # Optional, for GPU acceleration
```

### libmelee Version Conflict

slippi-ai uses vladfi1's fork of libmelee (v0.43.0), while nojohns uses mainline libmelee. For the adapter:

**Option A: Use vladfi1's fork everywhere**
```bash
# In nojohns main venv
pip uninstall libmelee
pip install git+https://github.com/vladfi1/libmelee.git@v0.43.0
```

**Option B: Separate environments**
- Use phillip venv for Phillip adapter
- Keep main venv for rest of nojohns
- Bridge via subprocess or shared Dolphin

**Option C: Check compatibility**
- Test if vladfi1's fork is compatible with nojohns
- Likely just minor changes

## Common Issues

### Issue: "No module named 'tensorflow'"

**Solution:** Make sure you're in the right venv
```bash
which python  # Should show .venv-phillip path
pip list | grep tensorflow
```

### Issue: "Python 3.12 - TensorFlow build fails"

**Solution:** Must use Python 3.10 or 3.11
```bash
deactivate
# Create new venv with 3.11 (see above)
```

### Issue: "Module 'dm-sonnet' not found"

**Solution:** Install from requirements.txt
```bash
pip install dm-sonnet dm-tree
```

### Issue: "ImportError: libmelee version mismatch"

**Solution:** Use vladfi1's fork
```bash
pip install git+https://github.com/vladfi1/libmelee.git@v0.43.0
```

## Environment Variables

For running the bot:
```bash
export DOLPHIN_PATH="/Applications/Slippi Dolphin.app"
export ISO_PATH="/path/to/melee.iso"
```

Or for your machine:
```bash
export DOLPHIN_PATH="/Users/queenmab/Library/Application Support/Slippi Launcher/netplay/Slippi Dolphin.app"
export ISO_PATH="/Users/queenmab/claude-projects/games/melee/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"
```

## Testing Checklist

- [ ] Python 3.11 venv created
- [ ] slippi-ai dependencies installed
- [ ] TensorFlow imports successfully
- [ ] Model loads from pickle
- [ ] eval_lib imports work
- [ ] Test script runs (see test_phillip_model.py)

## Next Steps

Once setup is complete:
1. Run `test_phillip_model.py` to verify model works
2. Test `scripts/eval_two.py` with the model
3. Build PhillipFighter adapter
4. Integrate into nojohns
