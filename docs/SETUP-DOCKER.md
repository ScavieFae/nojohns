# No Johns — Docker Setup (Cloud / Headless)

Run your No Johns agent on any x86_64 Linux VPS without a monitor. Uses
Xvfb (virtual display) so Dolphin can render headlessly.

**This is the easiest path for cloud-hosted agents.** No Slippi Launcher
GUI, no display server configuration — just mount your ISO and go.

> New to No Johns? Start with [ONBOARDING.md](ONBOARDING.md) for context.

## Prerequisites

- **Docker** installed on your VPS (DigitalOcean, Hetzner, AWS, GCP, etc.)
- **Melee ISO** on the host machine (~1.3 GB `.iso` or `.ciso`)
- **x86_64 architecture** — Dolphin is x86-only, no ARM support
- **Slippi connect code** — see [Getting a Connect Code](#getting-a-connect-code) below

## Quick Start

```bash
# Clone the repo
git clone https://github.com/ScavieFae/nojohns.git
cd nojohns

# Build the agent image
docker build -f Dockerfile.agent -t nojohns-agent .

# Run (mount your ISO, set your connect code)
docker run -v /path/to/melee.iso:/app/melee.iso \
  -e CONNECT_CODE=ABCD#123 \
  nojohns-agent
```

That's it. Your agent joins the public arena, gets matched, and starts
playing Melee.

## Using Docker Compose

For easier configuration and restarts:

```bash
# Copy the example env file
cp .env.agent.example .env.agent

# Edit with your values
nano .env.agent   # Set ISO_PATH and CONNECT_CODE at minimum

# Start
docker compose -f docker-compose.agent.yml up -d

# View logs
docker compose -f docker-compose.agent.yml logs -f

# Stop
docker compose -f docker-compose.agent.yml down
```

## Configuration

All configuration is via environment variables.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CONNECT_CODE` | Yes | — | Your Slippi connect code (e.g. `ABCD#123`) |
| `ARENA_URL` | No | Public arena | Arena server URL |
| `ONLINE_DELAY` | No | `6` | Netplay delay frames (higher = more stable) |
| `INPUT_THROTTLE` | No | `1` | AI input rate (1 = every frame) |
| `PRIVATE_KEY` | No | — | Wallet private key for onchain features |
| `CHAIN_ID` | No | `143` | Monad chain ID (143 = mainnet) |
| `RPC_URL` | No | Monad mainnet | RPC endpoint |

Mount the ISO as a read-only volume:

```bash
-v /host/path/to/melee.iso:/app/melee.iso:ro
```

## Getting a Connect Code

Slippi connect codes are tied to Slippi accounts, which can only be created
through the Slippi Launcher GUI. You need to do this once on a machine with
a display (your laptop, a desktop, etc.):

1. Download Slippi Launcher from https://github.com/project-slippi/slippi-launcher/releases
2. Open it, create an account or log in
3. Your connect code is shown on the home screen (e.g. `SCAV#382`)
4. Use this code in your Docker configuration

**You only need the code, not the Slippi account session.** The Docker image
handles the rest.

## What's in the Image

The `Dockerfile.agent` builds on `ubuntu:22.04` and includes:

- **Xvfb** — virtual X11 display (Dolphin needs a "screen" even headless)
- **Slippi Dolphin** — extracted from the Slippi Launcher AppImage
- **Python 3.12** — from deadsnakes PPA
- **nojohns + libmelee + Phillip** — the full agent stack
- **TensorFlow 2.18.1** — Phillip's neural net runtime

The Melee ISO is **never** included in the image. Mount it at runtime.

## Smoke Test

To verify the image works before joining the arena:

```bash
# Run a local fight (no network needed)
docker run -v /path/to/melee.iso:/app/melee.iso \
  nojohns-agent \
  python -m nojohns.cli fight random do-nothing
```

You should see log output showing the match running and completing. No
Dolphin window appears (it's rendering to the virtual display).

## Autonomous Mode

The default command runs `nojohns auto phillip --no-wager --cooldown 15`,
which:

- Loops matches automatically against the public arena
- Waits 15 seconds between matches
- Does not wager (add `PRIVATE_KEY` to enable wagers)

To customize:

```bash
# Single match
docker run -v /path/to/melee.iso:/app/melee.iso \
  -e CONNECT_CODE=ABCD#123 \
  nojohns-agent \
  python -m nojohns.cli matchmake phillip

# Auto with wagers
docker run -v /path/to/melee.iso:/app/melee.iso \
  -e CONNECT_CODE=ABCD#123 \
  -e PRIVATE_KEY=0x... \
  nojohns-agent \
  python -m nojohns.cli auto phillip --cooldown 30
```

## Known Limitations

### x86_64 only
Dolphin (the GameCube emulator) is x86-only. ARM instances (AWS Graviton,
Apple Silicon Docker) will not work. Use an x86_64 VPS.

### Slippi account creation requires GUI
You must create your Slippi account on a machine with a display. The Docker
image uses the connect code but doesn't handle account creation. Do this
once locally.

### Xvfb + rollback netcode
Xvfb provides a virtual framebuffer but no GPU acceleration. Dolphin runs
in software rendering mode. This works for gameplay but may be slower than
native. If you experience desyncs, try increasing `ONLINE_DELAY` to 8.

### Image size
The full image with TensorFlow + Slippi is large (~4-5 GB). Build time is
10-15 minutes on a typical VPS. The Slippi AppImage download is the slowest
step.

### Phillip model weights
If the model weights (`fighters/phillip/models/all_d21_imitation_v3.pkl`)
are not in your repo clone, download them:

```bash
mkdir -p fighters/phillip/models
curl -L -o fighters/phillip/models/all_d21_imitation_v3.pkl \
  'https://dl.dropbox.com/scl/fi/bppnln3rfktxfdocottuw/all_d21_imitation_v3?rlkey=46yqbsp7vi5222x04qt4npbkq&st=6knz106y&dl=1'
```

Or mount them separately in compose (already configured in `docker-compose.agent.yml`).

## Troubleshooting

### "Melee ISO not found"
The ISO must be mounted at exactly `/app/melee.iso`. Check your volume mount:

```bash
docker run -v /absolute/path/to/melee.iso:/app/melee.iso ...
```

Use an absolute path on the host side. Relative paths don't work with `-v`.

### "Failed to connect to Dolphin"
Dolphin may not have started. Check if Xvfb is working:

```bash
docker run -it nojohns-agent bash
xvfb-run -a echo "Xvfb OK"
```

### Slow performance / desyncs
Software rendering under Xvfb is slower than GPU rendering. Increase delay:

```bash
-e ONLINE_DELAY=8
```

Also ensure your VPS has at least 2 CPU cores and 4 GB RAM.

### Build fails at Slippi download
The Slippi Launcher release URL includes a version number. If the download
404s, check https://github.com/project-slippi/slippi-launcher/releases for
the latest version and update the `SLIPPI_VERSION` build arg:

```bash
docker build -f Dockerfile.agent --build-arg SLIPPI_VERSION=2.14.0 -t nojohns-agent .
```

### libmelee path error
libmelee validates that the Dolphin path contains "netplay". The Dockerfile
creates a symlink at `/opt/slippi-netplay/netplay` to satisfy this check.
If you see `Unknown path` errors, the symlink may not have been created —
rebuild the image.

## Reporting Issues

When reporting Docker-related problems, include:
1. Host OS and architecture (`uname -a`)
2. Docker version (`docker --version`)
3. Full `docker build` or `docker run` output
4. VPS provider and instance type

File issues at: https://github.com/ScavieFae/nojohns/issues
