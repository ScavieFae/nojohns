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

## Step 4: Python 3.12

Python 3.12 specifically — **not 3.13**. libmelee depends on pyenet,
which has C extensions that fail to build on 3.13.

```bash
brew install python@3.12

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
> 2. Let it download Dolphin (installs to `/Applications/Slippi Dolphin.app`)
> 3. Log in or create a Slippi account
> 4. Note their connect code (shown on the home screen, e.g. `ABCD#123`)
> 5. If macOS blocks the app, right-click > Open to bypass Gatekeeper
>
> When they confirm, verify with:
> ```bash
> ls "/Applications/Slippi Dolphin.app" && echo "Dolphin OK"
> ```
> If Dolphin isn't there, check:
> ```bash
> ls ~/Library/Application\ Support/Slippi\ Launcher/netplay/
> ```

## Step 6: Clone and Install No Johns

```bash
# Clone the repo
git clone https://github.com/ScavieFae/nojohns.git nojohns
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

## Step 7: Place the Melee ISO

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

## Step 8: Smoke Test (Local Fight)

This opens a Dolphin window and runs two bots against each other locally.
Confirms Dolphin, libmelee, and nojohns all work together.

First, clear Gatekeeper on Dolphin so it can launch without a security popup:

```bash
xattr -cr "/Applications/Slippi Dolphin.app"
```

Then run the smoke test (use the ISO path from Step 7):

```bash
.venv/bin/python -m nojohns.cli fight random do-nothing \
  -d "/Applications/Slippi Dolphin.app" \
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

## Step 9: Netplay

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
