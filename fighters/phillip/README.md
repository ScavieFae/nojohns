# Phillip Fighter

Neural network Melee AI adapter for nojohns.

Wraps vladfi1's [slippi-ai](https://github.com/vladfi1/slippi-ai) (Phillip) to work with the nojohns fighter system.

## What is Phillip?

Phillip is a neural network AI that:
- Learns from human Slippi replays (imitation learning)
- Self-improves through reinforcement learning
- Plays human-like with consistent 21-frame delay
- Has beaten top players like Moky and Cody Schwab

## Directory Structure

```
fighters/phillip/
├── phillip_fighter.py    # Main adapter (wraps slippi-ai)
├── __init__.py           # Package exports
├── README.md             # This file
├── requirements.txt      # Python dependencies (TensorFlow, etc.)
├── slippi-ai/            # vladfi1's repo (gitignored, clone on setup)
└── models/               # Model weights (gitignored, download on setup)
    └── all_d21_imitation_v3.pkl
```

## Setup

**Requirements:**
- Python 3.11 (TensorFlow doesn't support 3.12+ yet)
- ~500MB disk space (TensorFlow + model)

**Step 1: Install dependencies**
```bash
# Use Python 3.11
python3.11 -m venv .venv-phillip
source .venv-phillip/bin/activate
pip install -e .
pip install -r fighters/phillip/requirements.txt
```

**Step 2: Clone slippi-ai**
```bash
cd fighters/phillip
git clone https://github.com/vladfi1/slippi-ai.git
cd ../..
```

**Step 3: Download model**
```bash
cd fighters/phillip/models
curl -L 'https://dl.dropbox.com/scl/fi/bppnln3rfktxfdocottuw/all_d21_imitation_v3?rlkey=46yqbsp7vi5222x04qt4npbkq&st=6knz106y&dl=1' \
  -o all_d21_imitation_v3.pkl
cd ../../..
```

## Testing

```bash
# From repo root with Python 3.11 venv active
python test_phillip.py
```

## Available Models

**all_d21_imitation_v3.pkl** (40.3 MB)
- 10.6M parameters
- 21-frame delay
- Imitation learning (Phase 1)
- Trained on top players

## Resources

- **Source:** https://github.com/vladfi1/slippi-ai
- **Discord:** https://discord.gg/hfVTXGu
