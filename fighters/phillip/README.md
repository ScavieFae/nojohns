# Phillip Fighter

Neural network Melee AI adapter for nojohns.

Wraps vladfi1's [slippi-ai](https://github.com/vladfi1/slippi-ai) (Phillip) to work with the nojohns fighter system.

## What is Phillip?

Phillip is a neural network AI that:
- Learns from human Slippi replays (imitation learning)
- Self-improves through reinforcement learning
- Plays human-like with consistent 21-frame delay
- Has beaten top players like Moky and Cody Schwab

## Quick Setup

From the nojohns root:

```bash
# 1. Install TensorFlow dependencies
.venv/bin/pip install -e ".[phillip]"

# 2. Clone slippi-ai
git clone https://github.com/vladfi1/slippi-ai.git docs/phillip-research/slippi-ai
.venv/bin/pip install -e docs/phillip-research/slippi-ai

# 3. Download model weights (~40 MB)
mkdir -p fighters/phillip/models
curl -L -o fighters/phillip/models/all_d21_imitation_v3.pkl \
  'https://dl.dropbox.com/scl/fi/bppnln3rfktxfdocottuw/all_d21_imitation_v3?rlkey=46yqbsp7vi5222x04qt4npbkq&st=6knz106y&dl=1'

# 4. Verify
.venv/bin/python -c "
from nojohns.registry import load_fighter
f = load_fighter('phillip')
print(f'Loaded: {f.metadata.display_name}')
"
```

See [docs/SETUP.md](../../docs/SETUP.md) Step 7 for detailed instructions
and troubleshooting.

## Requirements

- Python 3.12 (same venv as nojohns)
- TensorFlow 2.18.1 (2.20 crashes on macOS ARM)
- tf-keras 2.18.0
- vladfi1's libmelee fork v0.43.0 (installed by default with nojohns)

## Directory Structure

```
fighters/phillip/
├── phillip_fighter.py    # Main adapter (wraps slippi-ai)
├── __init__.py           # Package exports
├── fighter.toml          # Registry manifest
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── models/               # Model weights (gitignored, download on setup)
│   └── all_d21_imitation_v3.pkl
└── (slippi-ai is cloned to docs/phillip-research/slippi-ai)
```

## How It Works

The adapter bridges between nojohns' `Fighter` protocol and slippi-ai's
`Agent` class:

1. **`on_game_start()`**: Builds the agent with `eval_lib.build_agent()`,
   sets up a `DummyController` to capture outputs, and feeds a synthetic
   frame -123 to initialize the agent's parser.
2. **`act()`**: Calls `agent.step(gamestate)`, decodes the controller
   output from the neural net's internal representation back to our
   `ControllerState` format.
3. **`on_game_end()`**: Cleans up the agent.

The 21-frame delay is built into the model — the neural net was trained
with this delay and expects it. This isn't lag; it's a design choice
that makes the AI play more human-like.

## Available Models

**all_d21_imitation_v3.pkl** (40.3 MB)
- 10.6M parameters
- 21-frame delay (imitation learning, Phase 1)
- Trained on top players (Hax, Cody, Amsa, Kodorin)
- All characters supported (Fox primary)
- Apache 2.0 licensed (found in slippi-ai test suite)

Stronger RL-refined models exist but are not publicly available.
Contact vladfi1 via [Discord](https://discord.gg/hfVTXGu) or
[GitHub](https://github.com/vladfi1) for access.

## Known Limitations

- **Fox only** in practice — the model supports multiple characters but
  the adapter hardcodes Fox
- **SDs occasionally** — this is an imitation model, not RL-refined. It
  learned to play like humans, including sometimes running off stage
- **Controller extraction is fragile** — accesses `_agent._agent.embed_controller`
  internal API. May break if slippi-ai refactors

## Resources

- **Source:** https://github.com/vladfi1/slippi-ai
- **Discord:** https://discord.gg/hfVTXGu
- **Twitch:** https://twitch.tv/x_pilot (vladfi1 streams Phillip matches)
