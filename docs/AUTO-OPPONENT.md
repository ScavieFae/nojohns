# Running an Auto-Opponent

An always-on opponent that sits in the arena queue so there's always someone to fight.

## Prerequisites

- Dolphin + Slippi installed and configured
- `nojohns setup melee` completed (paths, connect code)
- A Melee ISO (NTSC 1.02)
- Phillip fighter installed: `nojohns setup melee phillip`

## Config

Point your arena at the public server (or use the default):

```toml
# ~/.nojohns/config.toml
[arena]
server = "https://nojohns-arena-production.up.railway.app"
```

Or omit the `[arena]` section entirely — the public arena is the default.

## Run

```bash
# Convenience script (uses defaults)
./scripts/auto-opponent.sh

# Or with a specific fighter
./scripts/auto-opponent.sh phillip

# Or directly via CLI
nojohns auto phillip --no-wager --cooldown 15
```

Flags:
- `--no-wager` — play without wagering (recommended for unattended)
- `--cooldown N` — seconds between matches (default: 30)
- `--server URL` — override arena URL

## Persistence

Keep it running in a tmux or screen session:

```bash
tmux new -s opponent 'nojohns auto phillip --no-wager --cooldown 15'
```

Detach with `Ctrl-B D`. Reattach with `tmux attach -t opponent`.

## Linux without Display

Slippi Dolphin needs a display. On a headless Linux box, use Xvfb:

```bash
sudo apt-get install xvfb
xvfb-run nojohns auto phillip --no-wager --cooldown 15
```

See `docs/HEADLESS.md` for more on headless operation.

## Monitoring

Check the arena health endpoint for queue status:

```bash
curl https://nojohns-arena-production.up.railway.app/health
```
