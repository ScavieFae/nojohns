# No Johns — macOS Setup Guide

Setting up No Johns on a fresh Mac (Apple Silicon). This guide is written
for Claude Code to follow step-by-step, but works for humans too.

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

## Step 2: Homebrew

```bash
# Check
command -v brew && echo "Homebrew OK"

# Install if needed (follow the prompts)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# On Apple Silicon, add to PATH if not already there
eval "$(/opt/homebrew/bin/brew shellenv)"
```

## Step 3: Python 3.12

Python 3.12 specifically — **not 3.13**. libmelee depends on pyenet,
which has C extensions that fail to build on 3.13.

```bash
brew install python@3.12

# Verify
python3.12 --version  # Should show 3.12.x
```

## Step 4: Slippi Launcher + Dolphin

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

**Then the user must:**

1. Open Slippi Launcher from `/Applications`
2. Let it download Dolphin (installs to `/Applications/Slippi Dolphin.app`)
3. Log in or create a Slippi account
4. Note their connect code (shown on the home screen)

**Verify Dolphin is installed:**

```bash
ls "/Applications/Slippi Dolphin.app" && echo "Dolphin OK"
```

If Dolphin isn't at `/Applications/Slippi Dolphin.app`, the Launcher may
have placed it elsewhere. Check:

```bash
ls ~/Library/Application\ Support/Slippi\ Launcher/netplay/
```

## Step 5: Clone and Install No Johns

```bash
# Clone the repo (update URL when public)
git clone <repo-url> nojohns
cd nojohns

# Create venv with Python 3.12
python3.12 -m venv .venv

# Install nojohns + dependencies (including libmelee)
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
```

**Verify:**

```bash
# libmelee imports
.venv/bin/python -c "import melee; print(f'libmelee {melee.__version__}')"

# nojohns CLI works
.venv/bin/python -m nojohns.cli list-fighters

# Tests pass
.venv/bin/python -m pytest tests/ -v -o "addopts="
```

## Step 6: Place the Melee ISO

The user places their ISO somewhere on the machine. Common location:

```bash
mkdir -p ~/games/melee
# Then copy/move the ISO there
```

## Step 7: Smoke Test (Local Fight)

This opens a Dolphin window and runs two bots against each other locally.
Confirms Dolphin, libmelee, and nojohns all work together.

```bash
.venv/bin/python -m nojohns.cli fight random do-nothing \
  -d "/Applications/Slippi Dolphin.app" \
  -i ~/games/melee/melee.iso
```

Adjust the ISO path to wherever the user placed it. The game should
launch, run briefly (DoNothing will lose quickly), and exit.

**Expected noise to ignore:**
- MoltenVK `VK_NOT_READY` errors — cosmetic, Dolphin runs fine
- BrokenPipeError on cleanup — harmless, from SIGKILL cleanup path

## Step 8: Netplay

With both machines set up, each side runs:

```bash
.venv/bin/python -m nojohns.cli netplay <fighter> \
  --code "<OPPONENT_CONNECT_CODE>" \
  -d "/Applications/Slippi Dolphin.app" \
  -i ~/games/melee/melee.iso
```

Both sides launch Dolphin, navigate to Slippi direct connect, enter the
opponent's code, and play. The fighter handles inputs; Slippi handles
networking.

## Troubleshooting

### pyenet build fails

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

### Dolphin not found

Slippi Launcher manages Dolphin. If it's not at `/Applications/Slippi Dolphin.app`:

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
- Or: `xattr -cr "/Applications/Slippi Dolphin.app"`
