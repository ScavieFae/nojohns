# Spectator Swarm

Single-command launcher for autonomous prediction market betting agents. Runs 4-6 `SpectatorAgent` instances in one asyncio process with a live Rich dashboard.

## Quick Start

```bash
# Generate wallets and fund them (if not already done)
python scripts/generate_wallets.py 5
export FUNDER_PRIVATE_KEY="0x..."
python scripts/fund_wallets.py --amount 0.05

# Launch the swarm
python scripts/swarm.py
```

The dashboard appears immediately. Agents start scanning for prediction pools on the arena. When a match with a pool goes live, agents connect via WebSocket, evaluate frame data, and bet when they find edge.

## Usage

```bash
python scripts/swarm.py                   # All wallets (max 6), mixed risk
python scripts/swarm.py --count 3         # First 3 wallets only
python scripts/swarm.py --risk aggressive # All agents use aggressive profile
python scripts/swarm.py --no-dashboard    # Log-only mode (headless/CI)
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--count N` | all (max 6) | Number of agents to launch |
| `--risk PROFILE` | mixed | `conservative`, `moderate`, `aggressive`, or `mixed` |
| `--arena URL` | env `ARENA_URL` or production | Arena server URL |
| `--rpc URL` | env `RPC_URL` or `https://rpc.monad.xyz` | Monad RPC endpoint |
| `--pool ADDR` | env `PREDICTION_POOL` or mainnet contract | PredictionPool address |
| `--no-dashboard` | false | Disable Rich dashboard, use log output |

Config resolution: CLI flags > environment variables > defaults.

## Architecture

### Single asyncio Process

All agents run as coroutines in one event loop. `SpectatorAgent.run()` is a native async coroutine — the main loop yields on every `await` (HTTP polls, WebSocket reads, RPC calls). This gives us:

- **Shared-memory observability** — the dashboard reads agent state (`.bets`, `.watched_pools`, `._running`) directly, no IPC needed
- **Lower overhead** — one Python process vs N
- **Simple shutdown** — one signal handler, one event loop

### Supervisor Pattern

Each agent runs inside `run_agent_supervised()`, which:

1. Creates a fresh `SpectatorAgent` instance
2. Runs `agent.run()` inside a try/except
3. On crash: preserves bet history, increments restart counter, waits 10s, recreates the agent
4. Gives up after 5 restarts

Bet statistics accumulate across crashes — if an agent placed 3 bets, crashes, then places 2 more, the dashboard shows 5.

### Staggered Startup

Agents launch 1.5 seconds apart to avoid thundering-herd RPC calls. With 5 agents, all are running within ~6 seconds.

### Dashboard

When active, the Rich Live display updates every second:

```
┌──────────────── No Johns Spectator Swarm ────────────────┐
│ Agent          Risk         Wallet       Status    Bets   │
│ spectator-0    conservative 0xA1b2C3...  scanning  0      │
│ spectator-1    moderate     0xD4e5F6...  watching  1      │
│ spectator-2    aggressive   0x789AbC...  BETTING   3      │
│                                                           │
│ Uptime: 00:12:34  |  Active: 3/3  |  Ctrl+C to stop      │
└──────────────────────────────────────────────────────────-┘
```

Status values:
- **scanning...** — polling arena for pools, no active match
- **watching** — connected to a match WebSocket, evaluating frames
- **BETTING** — placed a bet in the last 5 seconds
- **CRASHED** — agent errored, awaiting restart
- **stopped** — cleanly shut down

Agent INFO logs are suppressed when the dashboard is active (they'd clobber the display). Use `--no-dashboard` for full logging.

## Lifecycle

1. `Ctrl+C` or `SIGTERM` → shutdown event fires
2. All agents receive `stop()` signal
3. Agents finish current cycle and exit
4. Tasks are cancelled if still running after stop
5. Summary table prints with per-agent totals

## What Happens With No Matches

Agents show "scanning..." and poll the arena every 15 seconds. This is correct — they're waiting for a match with a prediction pool. Start a match (`nojohns matchmake phillip` on two machines) and watch the dashboard update.

## Risk Profiles

With `--risk mixed` (default), profiles rotate across agents:

| Profile | Kelly | Max Bet | Behavior |
|---------|-------|---------|----------|
| conservative | 0.25x | 3% bankroll | Small, frequent bets |
| moderate | 0.5x | 5% bankroll | Balanced |
| aggressive | 1.0x | 10% bankroll | Big swings |

Mixed profiles create natural market depth — different bet sizes arriving at different times look organic to observers (and judges).

## Troubleshooting

**"wallets.json not found"** — Run `python scripts/generate_wallets.py 5` first.

**"Missing dependency"** — Run `pip install -e '.[wallet,spectator]'`.

**All agents stuck on "scanning..."** — No active prediction pools. Check:
- `curl -s $ARENA_URL/health` — is the arena up?
- `curl -s $ARENA_URL/pools` — are there pools?
- A match needs to be running with `PREDICTION_POOL` configured on the arena.

**Agent keeps crashing** — Check `--no-dashboard` for full logs. Common causes:
- RPC rate limiting (reduce `--count`)
- Arena returning unexpected responses
- Wallet out of MON for gas

**Dashboard garbled** — Terminal doesn't support Rich. Use `--no-dashboard`.
