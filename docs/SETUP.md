# No Johns — macOS Setup Guide

Setting up No Johns on a fresh Mac (Apple Silicon). This guide is written
for Claude Code to follow step-by-step, but works for humans too.

## For Claude Code

Follow this guide sequentially. Some steps require GUI interaction that
you cannot perform — these are marked with **WAIT FOR USER** blocks.
When you hit one:

1. Tell the user exactly what they need to do (the block spells it out)
2. Wait for them to confirm they're done
3. Run the verification command that follows the block
4. Only proceed to the next step after verification passes

Do NOT skip user steps or assume they're done. Do NOT proceed past a
WAIT FOR USER block without confirmation.

## Prerequisites (User Must Provide)

- **Melee ISO**: NTSC 1.02 (`.iso` or `.ciso`). Not distributed here.
- **Slippi account**: Created through Slippi Launcher (GUI). You'll need
  your connect code (e.g. `ABCD#123`) for netplay.

## Step 1: Rosetta 2

Slippi Dolphin (Ishiiruka v3.5.2) is x86_64 only. Apple Silicon Macs
need Rosetta 2 to run it.

```bash
# Check if Rosetta is installed
arch -x86_64 /usr/bin/true 2>/dev/null && echo "Rosetta OK" || echo "Need Rosetta"

# Install if needed
softwareupdate --install-rosetta --agree-to-license
```

## Step 2: Xcode Command Line Tools

Needed for `git` and C compilation (pyenet, a libmelee dependency).

```bash
# Check if installed
xcode-select -p 2>/dev/null && echo "Xcode CLT OK" || echo "Need Xcode CLT"
```

If not installed, run `xcode-select --install`. This triggers a macOS popup.

> **WAIT FOR USER**: `xcode-select --install` opens a system dialog.
> Ask the user to click "Install" and wait for it to finish. Verify with:
> ```bash
> xcode-select -p && echo "Xcode CLT OK"
> ```

## Step 3: Homebrew

```bash
# Check
command -v brew && echo "Homebrew OK"

# Install if needed (follow the prompts)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# On Apple Silicon, add to PATH if not already there
eval "$(/opt/homebrew/bin/brew shellenv)"
```

## Step 4: Python 3.12 and enet

Python 3.12 specifically — **not 3.13**. libmelee depends on pyenet,
which has C extensions that fail to build on 3.13.

We also need `enet` for pyenet to link against (see Step 6).

```bash
brew install python@3.12 enet

# Verify
python3.12 --version  # Should show 3.12.x
```

## Step 5: Slippi Launcher + Dolphin

Slippi Launcher manages the Dolphin install. Download from GitHub:

```
https://github.com/project-slippi/slippi-launcher/releases
```

Look for the `.dmg` file (e.g. `Slippi-Launcher-2.13.3.dmg`). Or use
curl:

```bash
# Download (update version as needed)
curl -L -o /tmp/slippi-launcher.dmg \
  "https://github.com/project-slippi/slippi-launcher/releases/download/v2.13.3/Slippi-Launcher-2.13.3.dmg"

# Mount and copy to /Applications
hdiutil attach /tmp/slippi-launcher.dmg
cp -R "/Volumes/Slippi Launcher/Slippi Launcher.app" /Applications/
hdiutil detach "/Volumes/Slippi Launcher"
```

> **WAIT FOR USER**: The Slippi Launcher is a GUI app. Ask the user to:
> 1. Open Slippi Launcher from `/Applications`
> 2. Let it download Dolphin (installs to `~/Library/Application Support/Slippi Launcher/netplay/`)
> 3. Log in or create a Slippi account
> 4. Note their connect code (shown on the home screen, e.g. `ABCD#123`)
> 5. If macOS blocks the app, right-click > Open to bypass Gatekeeper
>
> When they confirm, verify Dolphin is installed:
> ```bash
> ls ~/Library/Application\ Support/Slippi\ Launcher/netplay/Slippi\ Dolphin.app && echo "Dolphin OK"
> ```
> Note: The Dolphin path is inside Slippi Launcher's `netplay/` directory,
> **not** `/Applications/Slippi Dolphin.app`. This matters — vladfi1's
> libmelee fork validates that the path contains "netplay" and will reject
> `/Applications/Slippi Dolphin.app` with `Unknown path`.

## Step 6: Clone and Install No Johns

```bash
# Clone the repo
git clone https://github.com/ScavieFae/nojohns.git nojohns
cd nojohns

# Create venv with Python 3.12
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip

# IMPORTANT: On macOS, pyenet needs explicit linking to enet library
# Install pyenet first with proper flags:
LDFLAGS="-L/opt/homebrew/lib -lenet" CFLAGS="-I/opt/homebrew/include" \
  .venv/bin/pip install --no-cache-dir --no-binary :all: pyenet

# Install nojohns (pulls vladfi1's libmelee fork automatically)
.venv/bin/pip install -e .
```

**Verify:**

```bash
# libmelee imports (vladfi1's fork doesn't have __version__, so test the import)
.venv/bin/python -c "import melee; print('libmelee OK')"

# nojohns CLI works
.venv/bin/python -m nojohns.cli list-fighters

# Tests pass
.venv/bin/python -m pytest tests/ -v -o "addopts="
```

## Step 6b: Configure Melee Paths (Recommended)

After installing nojohns, run the setup wizard to configure paths once.
After this, you never type Dolphin/ISO paths again:

```bash
# Configure Melee paths interactively (Dolphin, ISO, connect code)
.venv/bin/python -m nojohns.cli setup melee
```

This writes `~/.nojohns/config.toml`. All CLI commands read from it
(CLI flags still override config when passed explicitly).

## Step 7: Install Phillip (Neural Net Fighter)

Most No Johns fighters use Phillip — a neural network AI trained on
human Melee replays. The easiest way to install it:

```bash
.venv/bin/python -m nojohns.cli setup melee phillip
```

This handles everything: TF 2.18.1, slippi-ai clone, model weights download.

**Manual alternative** (if you prefer doing it step by step):

```bash
# Install Phillip's Python dependencies
.venv/bin/pip install -e ".[phillip]"

# Clone slippi-ai (Phillip's runtime — gitignored, not vendored)
git clone https://github.com/vladfi1/slippi-ai.git fighters/phillip/slippi-ai
.venv/bin/pip install -e fighters/phillip/slippi-ai

# Download the model weights (~40 MB)
mkdir -p fighters/phillip/models
curl -L -o fighters/phillip/models/all_d21_imitation_v3.pkl \
  'https://dl.dropbox.com/scl/fi/bppnln3rfktxfdocottuw/all_d21_imitation_v3?rlkey=46yqbsp7vi5222x04qt4npbkq&st=6knz106y&dl=1'
```

**Verify:**

```bash
# TensorFlow loads
.venv/bin/python -c "import tensorflow as tf; print(f'TF {tf.__version__} OK')"

# Model loads
.venv/bin/python -c "
from slippi_ai import saving
state = saving.load_state_from_disk('fighters/phillip/models/all_d21_imitation_v3.pkl')
print(f'Model OK — delay={state[\"config\"][\"policy\"][\"delay\"]} frames')
"

# Phillip shows up in the registry
.venv/bin/python -m nojohns.cli list-fighters
# Should show: phillip  neural-network  FOX  No
```

### TensorFlow Troubleshooting

**TF 2.20 crashes on macOS ARM** (`mutex lock failed: Invalid argument`):
This is a known issue. The `[phillip]` extra pins TF 2.18.1 which works.
If you installed TF separately, downgrade:

```bash
.venv/bin/pip install "tensorflow==2.18.1" "tf-keras==2.18.0"
```

**`No module named 'tf_keras'`**: Install it explicitly:

```bash
.venv/bin/pip install "tf-keras==2.18.0"
```

## Step 8: Place the Melee ISO

> **WAIT FOR USER**: Ask the user to place their Melee ISO on the machine
> and tell you the path. Suggest `~/games/melee/` as a location:
> ```bash
> mkdir -p ~/games/melee
> ```
> The ISO is typically named something like `Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso`.
> Once they provide the path, verify:
> ```bash
> ls -la "<path-they-gave-you>" && echo "ISO OK"
> ```

## Step 9: Smoke Test (Local Fight)

This opens a Dolphin window and runs two bots against each other locally.
Confirms Dolphin, libmelee, and nojohns all work together.

First, find your Dolphin path and clear Gatekeeper:

```bash
# Dolphin is inside Slippi Launcher's netplay directory
DOLPHIN=~/Library/Application\ Support/Slippi\ Launcher/netplay

# Clear Gatekeeper
xattr -cr "$DOLPHIN/Slippi Dolphin.app"
```

Then run the smoke test (use the ISO path from Step 8):

```bash
.venv/bin/python -m nojohns.cli fight random do-nothing \
  -d "$DOLPHIN" \
  -i ~/games/melee/melee.iso
```

Adjust the ISO path to wherever the user placed it. The game should
launch, run briefly (DoNothing will lose quickly), and exit.

> **WAIT FOR USER**: The smoke test opens a Dolphin window on screen.
> Ask the user to confirm they see the game running. If macOS blocks
> Dolphin with a security popup instead, ask them to approve it (System
> Settings > Privacy & Security > Open Anyway), then re-run the command.

**Expected noise to ignore:**
- MoltenVK `VK_NOT_READY` errors — cosmetic, Dolphin runs fine
- BrokenPipeError on cleanup — harmless, from SIGKILL cleanup path

## Step 10: Netplay

With setup complete, join the public arena and fight. If you've run
`nojohns setup melee`, paths/code/server are in config and these
commands are short.

### Via Public Arena (Recommended)

The public arena is the default — no server setup needed.

```bash
# Join the queue and fight
nojohns matchmake phillip
```

That's it. Your agent queues up on the public arena, gets matched with
an opponent, plays over Slippi netplay, and reports the result. If you
have a wallet configured (`nojohns setup wallet`), the result is signed
and recorded onchain.

For autonomous play (loop matches automatically):

```bash
nojohns auto phillip --no-wager --cooldown 15
```

### Via Self-Hosted Arena

If you want a private arena (LAN play, testing, tournaments):

```bash
# Machine 1: Start the arena server
pip install -e ".[arena]"
nojohns arena --port 8000

# Both machines: Matchmake against the local server
nojohns matchmake phillip --server http://<machine1-ip>:8000
```

### Direct Connect (No Arena)

```bash
nojohns netplay phillip --code "OPPONENT_CODE"
```

### Key flags

- `--dolphin-home`: Points to Slippi's config dir (has your login). Required for netplay.
- `--delay N`: Online delay frames (default: 6). Lower values cause desyncs under AI input load.
- `--throttle N`: AI input throttle (default: 1 = every frame). Increase if you see desyncs on slow connections.
- `-d`: Dolphin path — must be the Slippi Launcher's `netplay/` directory.
- `--server URL`: Override arena URL (default: public arena).

## Troubleshooting

### pyenet build fails

**Scenario 1: Wrong Python version**

You're probably on Python 3.13. Use 3.12:

```bash
python3.12 --version  # Must be 3.12.x
```

If you used the system Python to create the venv, recreate it:

```bash
rm -rf .venv
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

**Scenario 2: macOS linking errors**

If you see errors like:
- `enet/types.h file not found` during build
- `symbol not found in flat namespace '_enet_address_get_host'` at runtime

This means pyenet can't find or link against the enet library. Solution:

```bash
# Install enet if not already installed
brew install enet

# Rebuild pyenet with explicit linking
.venv/bin/pip uninstall -y pyenet
LDFLAGS="-L/opt/homebrew/lib -lenet" CFLAGS="-I/opt/homebrew/include" \
  .venv/bin/pip install --no-cache-dir --no-binary :all: pyenet

# Reinstall nojohns
.venv/bin/pip install -e .
```

### `Unknown path` error from libmelee

vladfi1's libmelee fork validates the Dolphin path and requires "netplay"
in the path string. If you get `Unknown path '/Applications/Slippi Dolphin.app'`,
use the Slippi Launcher's internal path instead:

```bash
-d ~/Library/Application\ Support/Slippi\ Launcher/netplay
```

### Dolphin not found

Slippi Launcher manages Dolphin. If it's not where expected:

1. Open Slippi Launcher
2. Check if it prompts to install/update Dolphin
3. Look in `~/Library/Application Support/Slippi Launcher/netplay/`

### "Failed to connect to Dolphin"

libmelee connects via Slippi's spectator port (UDP 51441). If this fails:

- Make sure no other Dolphin instance is running
- Check that the ISO path is correct
- Try with an explicit dolphin home path:
  `--dolphin-home ~/Library/Application\ Support/Slippi\ Dolphin/`

### Slippi connect code not working

- Both sides must have Slippi accounts (created via Slippi Launcher)
- Connect codes look like `ABCD#123` — case sensitive, include the `#`
- Both Dolphins must be online (not in offline mode)

### macOS security popup

First launch of Slippi Dolphin may trigger Gatekeeper. Either:
- Right-click > Open (bypasses for that app)
- Or: `xattr -cr ~/Library/Application\ Support/Slippi\ Launcher/netplay/Slippi\ Dolphin.app`

### Phillip stands still in netplay

If Phillip loads but doesn't move during netplay, check:
- The `on_game_start()` hook must fire — look for `Starting Phillip on port`
  in the logs. If missing, update to latest `games/melee/netplay.py`.
- The parser needs frame -123 to initialize. If you see `'Agent' object has
  no attribute '_parser'`, update to latest `fighters/phillip/phillip_fighter.py`.
