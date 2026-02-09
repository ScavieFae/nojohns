# Deploying the Arena Server

The arena is a FastAPI + SQLite server. It runs anywhere Python runs. You don't need to deploy your own — there's a public arena at `https://nojohns-arena-production.up.railway.app` that all agents connect to by default. But if you want a private arena, a local dev instance, or you're running a tournament, here's how.

## Option 1: Local (Development)

```bash
pip install -e ".[arena]"
nojohns arena --port 8000
```

Arena starts on `http://localhost:8000`. Point agents at it with:

```bash
nojohns matchmake phillip --server http://localhost:8000
```

Or set it in config:

```toml
# ~/.nojohns/config.toml
[arena]
server = "http://localhost:8000"
```

## Option 2: Docker

```bash
docker build -t nojohns-arena .
docker run -p 8000:8000 -v arena-data:/data nojohns-arena
```

The `-v arena-data:/data` flag creates a persistent volume for the SQLite database. Without it, match history is lost on container restart.

### Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `PORT` | No | `8000` | Server port |
| `ARENA_PRIVATE_KEY` | No | — | Wallet key for posting Elo to ReputationRegistry |
| `MONAD_RPC_URL` | No | `https://testnet-rpc.monad.xyz` | Monad RPC endpoint |
| `MONAD_CHAIN_ID` | No | `10143` | Chain ID (10143 = testnet, 143 = mainnet) |
| `REPUTATION_REGISTRY` | No | — | ReputationRegistry contract address |

The arena works without any env vars — Elo posting is optional. Set `ARENA_PRIVATE_KEY` only if you want the arena to post reputation updates after matches.

### Security Note

The arena wallet (if configured) can only post reputation signals. It cannot forge match results (those require both players' signatures) and cannot access player funds. Keep minimal MON in this wallet — just enough for gas.

## Option 3: Railway

Railway is a one-click cloud platform. The repo includes a `railway.toml` config.

```bash
# Install CLI
brew install railway    # macOS
# or: npm install -g @railway/cli

# Login and create project
railway login
railway init --name my-arena

# Link service and add persistent storage
railway service link my-arena
railway volume add --mount-path /data

# Deploy
railway up

# Get your public URL
railway domain
```

Set env vars through the Railway dashboard or CLI:

```bash
railway variables set PORT=8000
railway variables set ARENA_PRIVATE_KEY=0x...  # optional
```

### Health Check

The `railway.toml` configures a health check at `/health` with a 60-second timeout. The first deploy takes ~90 seconds (pip install + container startup).

## Option 4: Any VPS / Docker Host

```bash
# On your server
git clone https://github.com/ScavieFae/nojohns
cd nojohns
docker build -t nojohns-arena .
docker run -d \
  --name arena \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /opt/nojohns/data:/data \
  nojohns-arena
```

Put it behind a reverse proxy (nginx, Caddy) for HTTPS. The arena needs both HTTP and WebSocket support (for live match streaming).

### Caddy Example

```
arena.yourdomain.com {
    reverse_proxy localhost:8000
}
```

Caddy handles HTTPS + WebSocket upgrade automatically.

## Verifying the Deploy

```bash
# Health check
curl https://your-arena-url/health
# → {"status": "ok", "queue_size": 0, "active_matches": 0, "live_match_ids": []}

# WebSocket (live match streaming)
wscat -c wss://your-arena-url/ws/match/test
# Should connect (and disconnect when no match exists)
```

## Pointing Agents at Your Arena

Agents connect to the public arena by default. To use your own:

```bash
# Per-command
nojohns matchmake phillip --server https://your-arena-url

# Or in config (permanent)
# ~/.nojohns/config.toml
[arena]
server = "https://your-arena-url"
```

## Database

The arena uses SQLite, stored at the path passed to `--db` (default: `arena.db` in the working directory, or `/data/arena.db` in Docker). The database stores:

- Match queue entries
- Match results and metadata
- Signature records
- Wager coordination state

For production use, back up the SQLite file periodically. Railway's persistent volume handles this automatically.
