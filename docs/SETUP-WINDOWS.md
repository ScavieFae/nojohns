# No Johns — Windows Setup Guide

Setting up No Johns on a Windows machine. This guide is for external
testers — the project is primarily developed on macOS, so Windows support
has rough edges. Please report issues.

## Status: Experimental

Windows support depends on libmelee's Windows compatibility, which is
partially broken upstream. This guide documents what we know and what
you'll need to work around. If you get stuck, file an issue.

**Known blockers:**
- pyenet needs MSVC build tools + the enet C library (no pre-built wheels)
- vladfi1's libmelee fork is untested on Windows — binary path detection
  may fail (expects `Slippi Dolphin.exe`, launcher may use different name)
- TensorFlow 2.18.1 works on Windows x64 but NOT on ARM

## Prerequisites (User Must Provide)

- **Melee ISO**: NTSC 1.02 (`.iso` or `.ciso`). Not distributed here.
- **Slippi account**: Created through Slippi Launcher. You'll need your
  connect code (e.g. `ABCD#123`) for netplay.

## Step 1: Python 3.12

Download Python 3.12 from [python.org](https://www.python.org/downloads/).
**Not 3.13** — libmelee's pyenet dependency has C extensions that may fail
on 3.13.

During install:
- Check "Add python.exe to PATH"
- Check "Install for all users" (optional but recommended)

Verify in PowerShell:

```powershell
python --version  # Should show 3.12.x
```

If `python` opens the Microsoft Store, use `python3.12` instead, or
disable the app execution alias in Windows Settings > Apps > Advanced
app settings > App execution aliases.

## Step 2: Visual Studio Build Tools

pyenet (a libmelee dependency) has C extensions that need a C compiler.

1. Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. In the installer, select **"Desktop development with C++"**
3. Make sure these are checked:
   - MSVC v143 (or latest)
   - Windows SDK
   - C++ CMake tools

This is a large download (~2 GB). Only the build tools are needed, not
full Visual Studio.

## Step 3: enet C Library

pyenet links against the enet networking library. On Windows you need to
build it from source or find a pre-built binary.

**Option A: vcpkg (recommended)**

```powershell
# Install vcpkg if you don't have it
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat

# Install enet
C:\vcpkg\vcpkg install enet:x64-windows
```

Then set environment variables before installing pyenet:

```powershell
$env:INCLUDE = "C:\vcpkg\installed\x64-windows\include;$env:INCLUDE"
$env:LIB = "C:\vcpkg\installed\x64-windows\lib;$env:LIB"
```

**Option B: Build enet manually**

```powershell
git clone https://github.com/lsalzman/enet.git C:\enet
cd C:\enet
# Open enet.sln in Visual Studio, build Release x64
# Or use CMake:
mkdir build && cd build
cmake .. -A x64
cmake --build . --config Release
```

Then point to it:

```powershell
$env:INCLUDE = "C:\enet\include;$env:INCLUDE"
$env:LIB = "C:\enet\build\Release;$env:LIB"
```

## Step 4: Slippi Launcher + Dolphin

Download Slippi Launcher from:

```
https://github.com/project-slippi/slippi-launcher/releases
```

Get the `.exe` installer (e.g. `Slippi-Launcher-Setup-2.13.3.exe`).

1. Run the installer
2. Open Slippi Launcher
3. Let it download Dolphin (installs to `%APPDATA%\Slippi Launcher\netplay\`)
4. Log in or create a Slippi account
5. Note your connect code (shown on the home screen)

Verify Dolphin is installed:

```powershell
Test-Path "$env:APPDATA\Slippi Launcher\netplay\Slippi Dolphin.exe"
```

**Note:** The Dolphin executable name may vary between Slippi versions.
If `Slippi Dolphin.exe` doesn't exist, check what's actually in the
netplay directory — libmelee needs to find the right binary.

## Step 5: Clone and Install No Johns

```powershell
git clone https://github.com/ScavieFae/nojohns.git
cd nojohns

# Create venv with Python 3.12
python -m venv .venv
.venv\Scripts\pip install --upgrade pip

# Install pyenet (with enet headers/libs available — see Step 3)
.venv\Scripts\pip install --no-cache-dir pyenet

# Install nojohns
.venv\Scripts\pip install -e .
```

Verify:

```powershell
# libmelee imports
.venv\Scripts\python -c "import melee; print('libmelee OK')"

# nojohns CLI works
.venv\Scripts\python -m nojohns.cli list-fighters

# Tests pass
.venv\Scripts\python -m pytest tests/ -v -o "addopts="
```

If pyenet fails to build, double-check that:
- MSVC build tools are installed (Step 2)
- enet headers are on the INCLUDE path (Step 3)
- You're using Python 3.12, not 3.13

## Step 5b: Configure Melee Paths

```powershell
.venv\Scripts\python -m nojohns.cli setup melee
```

When prompted:
- **Dolphin path**: `%APPDATA%\Slippi Launcher\netplay` (use full expanded
  path, e.g. `C:\Users\YourName\AppData\Roaming\Slippi Launcher\netplay`)
- **ISO path**: wherever you placed your Melee ISO
- **Connect code**: your Slippi code (e.g. `ABCD#123`)

Config is stored in `%APPDATA%\nojohns\config.toml`.

## Step 6: Install Phillip (Neural Net Fighter)

```powershell
.venv\Scripts\python -m nojohns.cli setup melee phillip
```

Or manually:

```powershell
.venv\Scripts\pip install -e ".[phillip]"

git clone https://github.com/vladfi1/slippi-ai.git fighters\phillip\slippi-ai
.venv\Scripts\pip install -e fighters\phillip\slippi-ai

mkdir fighters\phillip\models
# Download model weights (~40 MB)
# curl.exe is included in Windows 10+
curl.exe -L -o fighters\phillip\models\all_d21_imitation_v3.pkl `
  "https://dl.dropbox.com/scl/fi/bppnln3rfktxfdocottuw/all_d21_imitation_v3?rlkey=46yqbsp7vi5222x04qt4npbkq&st=6knz106y&dl=1"
```

Verify:

```powershell
.venv\Scripts\python -c "import tensorflow as tf; print(f'TF {tf.__version__} OK')"
.venv\Scripts\python -m nojohns.cli list-fighters
# Should show: phillip
```

## Step 7: Smoke Test (Local Fight)

```powershell
.venv\Scripts\python -m nojohns.cli fight random do-nothing
```

If you haven't run `setup melee`, pass paths explicitly:

```powershell
.venv\Scripts\python -m nojohns.cli fight random do-nothing `
  -d "$env:APPDATA\Slippi Launcher\netplay" `
  -i "C:\path\to\melee.iso"
```

The game should launch in a Dolphin window, run briefly, and exit.

## Step 8: Connect to the Arena

The arena server runs on the host machine (Mattie's Mac). You need the
server's IP address and port.

```powershell
.venv\Scripts\python -m nojohns.cli matchmake phillip `
  --server http://<arena-ip>:8000
```

Connect code and paths come from config (Step 5b). The arena matches
you with another waiting player and starts a Slippi netplay game.

## Troubleshooting

### libmelee can't find Dolphin

vladfi1's libmelee fork validates the Dolphin path and expects "netplay"
in the path string. If you get path errors:

1. Check what's in `%APPDATA%\Slippi Launcher\netplay\`
2. The binary might be named differently — look for `.exe` files
3. If the binary name doesn't match what libmelee expects, you may need
   to rename it or create a symlink

```powershell
# List contents of the netplay directory
Get-ChildItem "$env:APPDATA\Slippi Launcher\netplay"
```

If libmelee hardcodes a wrong binary name, this is a known upstream
issue. File an issue on our repo with the actual filename you see.

### pyenet build fails

Most common cause: missing enet headers. Make sure Step 3 is done and
the `INCLUDE`/`LIB` environment variables point to enet's headers and
libraries.

```powershell
# Check if enet headers are findable
Test-Path "$env:INCLUDE" -ErrorAction SilentlyContinue
```

### TensorFlow issues

TF 2.18.1 works on Windows x64. If you're on ARM Windows, TensorFlow
may not work — you can still use the `random` and `do-nothing` fighters
without it.

### "Failed to connect to Dolphin"

- Make sure no other Dolphin instance is running
- Check that the ISO path is correct (use forward slashes or escaped backslashes)
- Windows Defender/firewall may block the spectator port (UDP 51441) —
  allow it through

### Connection to arena fails

The arena server must be reachable from your network. If you're on a
different network than the host:
- The host needs to port-forward 8000 (or use a tunnel like ngrok)
- Check with: `curl http://<arena-ip>:8000/health`

## Reporting Issues

When reporting a problem, include:
1. Windows version (`winver`)
2. Python version (`python --version`)
3. The full error output
4. Contents of your netplay directory (`Get-ChildItem` output from above)

File issues at: https://github.com/ScavieFae/nojohns/issues
