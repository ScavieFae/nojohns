# No Johns — Linux Setup Guide

Setting up No Johns on Linux (x86_64). Linux is actually libmelee's
native platform — fewer workarounds than macOS or Windows.

## Prerequisites (User Must Provide)

- **Melee ISO**: NTSC 1.02 (`.iso` or `.ciso`). Not distributed here.
- **Slippi account**: Created through Slippi Launcher. You'll need your
  connect code (e.g. `ABCD#123`) for netplay.

## Step 1: System Dependencies

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev \
  git build-essential libenet-dev
```

If `python3.12` isn't available in your repos:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

### Fedora

```bash
sudo dnf install -y python3.12 python3-devel git gcc enet-devel
```

### Arch

```bash
sudo pacman -S python git base-devel enet
```

Verify:

```bash
python3.12 --version  # Should show 3.12.x
```

## Step 2: Slippi Launcher + Dolphin

Download the Slippi Launcher AppImage from:

```
https://github.com/project-slippi/slippi-launcher/releases
```

Get the `.AppImage` file (e.g. `Slippi-Launcher-2.13.3-x86_64.AppImage`).

```bash
# Make it executable
chmod +x Slippi-Launcher-*.AppImage

# Run it
./Slippi-Launcher-*.AppImage
```

1. Let it download Dolphin (installs to `~/.config/Slippi Launcher/netplay/`)
2. Log in or create a Slippi account
3. Note your connect code (shown on the home screen)

Verify Dolphin is installed:

```bash
ls ~/.config/Slippi\ Launcher/netplay/
# Should contain the Dolphin binary (squashfs-root/ or similar)
```

**Note:** The exact Dolphin path inside the Slippi Launcher directory
varies between versions. Check what's actually there — you'll need the
path that contains the Dolphin binary for libmelee.

### FUSE requirement

AppImages need FUSE. If you get a FUSE error:

```bash
# Ubuntu/Debian
sudo apt install -y libfuse2

# Or extract instead of running as AppImage
./Slippi-Launcher-*.AppImage --appimage-extract
./squashfs-root/slippi-launcher
```

## Step 3: Clone and Install No Johns

```bash
git clone https://github.com/ScavieFae/nojohns.git
cd nojohns

# Create venv with Python 3.12
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip

# Install nojohns (pulls vladfi1's libmelee fork + pyenet automatically)
.venv/bin/pip install -e .
```

pyenet should build cleanly since `libenet-dev` provides the headers.
If it fails, check that the dev package is installed:

```bash
# Ubuntu/Debian
dpkg -l | grep libenet

# Fedora
rpm -q enet-devel
```

Verify:

```bash
# libmelee imports
.venv/bin/python -c "import melee; print('libmelee OK')"

# nojohns CLI works
.venv/bin/python -m nojohns.cli list-fighters

# Tests pass
.venv/bin/python -m pytest tests/ -v -o "addopts="
```

## Step 3b: Configure Melee Paths

```bash
.venv/bin/python -m nojohns.cli setup melee
```

When prompted:
- **Dolphin path**: the netplay directory inside Slippi Launcher's config
  (e.g. `~/.config/Slippi Launcher/netplay`)
- **ISO path**: wherever you placed your Melee ISO
- **Connect code**: your Slippi code (e.g. `ABCD#123`)

Config is stored in `~/.nojohns/config.toml`.

## Step 4: Install Phillip (Neural Net Fighter)

```bash
.venv/bin/python -m nojohns.cli setup melee phillip
```

Or manually:

```bash
.venv/bin/pip install -e ".[phillip]"

git clone https://github.com/vladfi1/slippi-ai.git fighters/phillip/slippi-ai
.venv/bin/pip install -e fighters/phillip/slippi-ai

mkdir -p fighters/phillip/models
curl -L -o fighters/phillip/models/all_d21_imitation_v3.pkl \
  'https://dl.dropbox.com/scl/fi/bppnln3rfktxfdocottuw/all_d21_imitation_v3?rlkey=46yqbsp7vi5222x04qt4npbkq&st=6knz106y&dl=1'
```

Verify:

```bash
.venv/bin/python -c "import tensorflow as tf; print(f'TF {tf.__version__} OK')"
.venv/bin/python -m nojohns.cli list-fighters
# Should show: phillip
```

## Step 5: Smoke Test (Local Fight)

```bash
.venv/bin/python -m nojohns.cli fight random do-nothing
```

If you haven't run `setup melee`, pass paths explicitly:

```bash
.venv/bin/python -m nojohns.cli fight random do-nothing \
  -d ~/.config/Slippi\ Launcher/netplay \
  -i ~/games/melee/melee.iso
```

The game should launch in a Dolphin window, run briefly, and exit.

**Headless / SSH:** If you're on a headless server or SSH session, Dolphin
needs a display. You can use Xvfb for a virtual framebuffer:

```bash
sudo apt install -y xvfb
xvfb-run .venv/bin/python -m nojohns.cli fight random do-nothing
```

## Step 6: Connect to the Arena

The arena server runs on the host machine. You need the server's IP and port.

```bash
.venv/bin/python -m nojohns.cli matchmake phillip \
  --server http://<arena-ip>:8000
```

Connect code and paths come from config (Step 3b).

## Troubleshooting

### libmelee can't find Dolphin

vladfi1's libmelee fork validates the Dolphin path and expects "netplay"
in the path string. If you get path errors, make sure your `-d` path
points to the directory containing the Dolphin binary inside Slippi
Launcher's config:

```bash
find ~/.config/Slippi\ Launcher -name "*.AppImage" -o -name "dolphin-emu" 2>/dev/null
```

### pyenet build fails

Usually means the enet dev headers aren't installed:

```bash
# Ubuntu/Debian
sudo apt install -y libenet-dev

# Then reinstall
.venv/bin/pip install --no-cache-dir --force-reinstall pyenet
```

### No display (Wayland/X11 issues)

Dolphin needs a display server. On Wayland, you may need to run under
XWayland:

```bash
GDK_BACKEND=x11 .venv/bin/python -m nojohns.cli fight random do-nothing
```

### Connection to arena fails

The arena server must be reachable from your network. If you're on a
different network than the host:
- The host needs to port-forward 8000 (or use a tunnel like ngrok)
- Check with: `curl http://<arena-ip>:8000/health`

## Reporting Issues

When reporting a problem, include:
1. Distro and version (`cat /etc/os-release`)
2. Python version (`python3.12 --version`)
3. The full error output
4. Contents of the Slippi netplay directory

File issues at: https://github.com/ScavieFae/nojohns/issues
