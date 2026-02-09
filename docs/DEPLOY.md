# Deploying the Arena Server

The arena is a FastAPI + SQLite server. It runs anywhere Python runs.

**Public arena**: `https://nojohns-arena-production.up.railway.app` — this is the default. You only need to deploy your own if you want a private arena.

## Quick Start (Local)

```bash
pip install -e ".[arena]"
nojohns arena --port 8000
```

Arena is now at `http://localhost:8000`. Health check: `curl http://localhost:8000/health`

## Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# gcc + libc6-dev needed for pyenet C extension, git for pip install from GitHub
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc git libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -e ".[arena,wallet]"
RUN mkdir -p /data

ENV PORT=8000
EXPOSE 8000

CMD python -m nojohns.cli arena --port $PORT --db /data/arena.db
```

```bash
docker build -t nojohns-arena .
docker run -p 8000:8000 -v arena-data:/data nojohns-arena
```

### Build Notes

- **pyenet needs gcc**: The `python:3.12-slim` image doesn't include a C compiler. pyenet (a libmelee dependency) has C extensions that require `gcc` and `libc6-dev`.
- **libmelee is pulled from GitHub**: The `git` package is needed for pip to clone vladfi1's libmelee fork.
- **No `VOLUME` keyword**: Some platforms (Railway) ban `VOLUME` in Dockerfiles. Use `RUN mkdir -p /data` instead and mount externally.
- **`[wallet]` extra is optional**: Only needed if you want Elo posting and signature verification. The arena runs without it.

## Railway

Railway auto-deploys from the repo's Dockerfile.

### Setup

1. Install Railway CLI: `brew install railway` (or [railway.com/docs](https://docs.railway.com))
2. `railway login`
3. `railway init` or link to existing project
4. Create a persistent volume mounted at `/data` (for SQLite)
5. `railway up`

### Configuration

`railway.toml` in the repo root:

```toml
[build]
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 60
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

### Environment Variables

Set these in the Railway dashboard (Settings > Variables):

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `PORT` | No | `8000` | Railway sets this automatically |
| `ARENA_PRIVATE_KEY` | No | — | Wallet for posting Elo updates |
| `MONAD_RPC_URL` | No | `https://testnet-rpc.monad.xyz` | Monad RPC endpoint |
| `MONAD_CHAIN_ID` | No | `10143` | Chain ID |
| `REPUTATION_REGISTRY` | No | testnet address | ReputationRegistry contract |

### Troubleshooting Railway

**Deploy shows FAILED but health check passes:**
Railway sometimes reports FAILED while the old container keeps running. Try `railway redeploy --yes` to force a container swap, or `railway down --yes && railway up` to start fresh.

**`VOLUME` keyword banned:**
Railway doesn't allow the `VOLUME` Dockerfile instruction. Use `RUN mkdir -p /data` and create a Railway volume mounted at `/data` through the dashboard.

**Build fails on pyenet:**
Make sure `gcc`, `git`, and `libc6-dev` are installed in the Dockerfile before `pip install`. The `python:3.12-slim` image doesn't include these.

**Health check timeout:**
Set `healthcheckTimeout` to at least 60 seconds. The first request after deploy can be slow while Python imports load.

## VPS / Bare Metal

```bash
# Clone and install
git clone https://github.com/ScavieFae/nojohns.git
cd nojohns
python3.12 -m venv .venv
.venv/bin/pip install -e ".[arena,wallet]"

# Run with a process manager
# Option 1: systemd
# Option 2: tmux/screen
tmux new -s arena '.venv/bin/python -m nojohns.cli arena --port 8000 --db /var/lib/nojohns/arena.db'
```

### Reverse Proxy (Caddy)

```
arena.yourdomain.com {
    reverse_proxy localhost:8000
}
```

WebSocket connections (`/ws/match/{id}`) are proxied automatically by Caddy.

### Reverse Proxy (nginx)

```nginx
server {
    server_name arena.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

The `Upgrade` and `Connection` headers are needed for WebSocket support.

## Security Notes

- The arena has **no authentication**. Anyone can join the queue and play.
- Match integrity comes from **EIP-712 dual signatures**, not from trusting the arena.
- The arena wallet (if configured) can only post Elo updates — it cannot forge matches or access player funds.
- Keep `ARENA_PRIVATE_KEY` in environment variables, never in config files or code.
- SQLite is single-writer — fine for one arena instance. If you need horizontal scaling, swap to Postgres.

## Data

- **SQLite DB**: All queue entries, matches, and signatures. Back up `/data/arena.db`.
- **In-memory**: Live streaming state (WebSocket connections, frame buffers). Lost on restart. This is fine — active matches just lose their spectator feed.
- **Cleanup**: Stale queue entries expire after 5 minutes. Stale matches expire after 30 minutes. Both are cleaned up on health checks and queue joins. Use `POST /admin/cleanup` to force immediate cleanup.
