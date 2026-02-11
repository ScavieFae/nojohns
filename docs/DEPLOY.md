# Deploy Guide

How changes get from code to live services.

## Services

| Service | Platform | URL | Deploys on push to |
|---------|----------|-----|--------------------|
| Arena server | Railway | `https://nojohns-arena-production.up.railway.app` | `dev` |
| Website | Vercel | nojohns.gg | `main` (ScavieFae) |
| Contracts | Monad mainnet | `contracts/deployments.json` | Manual |

## Arena (Railway)

### Deploy

Push to `dev`. Railway auto-builds the Dockerfile and restarts.

```bash
git push origin dev
# ~60 seconds later:
curl https://nojohns-arena-production.up.railway.app/health
```

### Env vars (Railway dashboard)

Set in Railway dashboard → Settings → Variables. Not in code.

| Variable | Value | Purpose |
|----------|-------|---------|
| `PORT` | `8000` | Auto-set by Railway |
| `ARENA_PRIVATE_KEY` | `<key>` | Pool creation, Elo posting |
| `MONAD_RPC_URL` | `https://rpc.monad.xyz` | Mainnet RPC |
| `MONAD_CHAIN_ID` | `143` | Mainnet |
| `REPUTATION_REGISTRY` | `0x8004BAa17C55a88189AE136b182e5fdA19dE9b63` | Elo signals |
| `PREDICTION_POOL` | `0x33E65E300575D11a42a579B2675A63cb4374598D` | Spectator betting |

If a contract is redeployed (new address), update the env var here and Railway will pick it up on next deploy.

### Logs

```bash
railway logs                    # Railway CLI
# Or: Railway dashboard → Deployments → latest → Logs
```

### Persistent data

SQLite lives at `/data/arena.db` on a Railway volume. Survives redeploys. To wipe: delete the volume in Railway dashboard and redeploy.

### Troubleshooting

- **Deploy FAILED but health check passes**: `railway redeploy --yes` to force container swap
- **Build fails on pyenet**: Dockerfile needs `gcc`, `git`, `libc6-dev` before `pip install`
- **Health check timeout**: Set to 10s+ in `railway.toml`. First request is slow.

## Website (Vercel)

ScavieFae controls the website. Scav does not deploy it.

If the website needs a change (new contract address, chain config), add it to `docs/HANDOFF-SCAVIEFAE.md` and she'll handle it.

**Key file**: `web/src/config.ts` — all contract addresses, chain ID, RPC URL, deploy block.

**SPA routing**: `web/vercel.json` rewrites all routes to `/index.html`.

## Contracts (Foundry)

Already deployed to Monad mainnet (chain 143). Redeployment is rare.

### Current addresses

| Contract | Address |
|----------|---------|
| MatchProof | `0x1CC748475F1F666017771FB49131708446B9f3DF` |
| Wager | `0x8d4D9FD03242410261b2F9C0e66fE2131AE0459d` |
| PredictionPool | `0x33E65E300575D11a42a579B2675A63cb4374598D` |

### If a contract changes (multi-step, coordinate both agents)

1. ScavieFae deploys via `forge script --broadcast`
2. Update `contracts/deployments.json`
3. Update `nojohns/config.py` ChainConfig defaults (Scav)
4. Update `web/src/config.ts` (ScavieFae)
5. Update Railway env vars if address changed (dashboard)
6. Push `dev` (arena) and `main` (website)

## Release to Main

`main` is the public branch. During the hackathon, most work stays on `dev`. When shipping a milestone:

```bash
./scripts/release-to-main.sh
```

Squash-merges `dev` → `main`, strips internal files (handoff docs, `.claude/`). Vercel auto-deploys the website from `main`.

## Self-Hosting the Arena

For operators running their own arena (not using the public one).

### Local

```bash
pip install -e ".[arena]"
nojohns arena --port 8000
```

### Docker

```bash
docker build -t nojohns-arena .
docker run -p 8000:8000 -v arena-data:/data nojohns-arena
```

### Docker build notes

- `python:3.12-slim` needs `gcc` + `libc6-dev` for pyenet C extensions
- `git` needed for pip to clone vladfi1's libmelee fork from GitHub
- No `VOLUME` keyword (Railway bans it) — use `RUN mkdir -p /data` + external mount
- `[wallet]` extra is optional — only needed for Elo posting and signature verification

### VPS / bare metal

```bash
git clone https://github.com/ScavieFae/nojohns.git
cd nojohns
python3.12 -m venv .venv
.venv/bin/pip install -e ".[arena,wallet]"
tmux new -s arena '.venv/bin/python -m nojohns.cli arena --port 8000 --db arena.db'
```

### Reverse proxy

**Caddy** (WebSocket support automatic):
```
arena.yourdomain.com {
    reverse_proxy localhost:8000
}
```

**nginx** (needs WebSocket headers):
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

## Security

- Arena has **no authentication**. Anyone can join and play.
- Match integrity comes from **EIP-712 dual signatures**, not from trusting the arena.
- Arena wallet can only post Elo and create/resolve pools — cannot forge matches or touch player funds.
- Keep `ARENA_PRIVATE_KEY` in env vars, never in code or config files.
- SQLite is single-writer. Fine for one instance. Swap to Postgres for horizontal scaling.

## Quick Reference

```bash
# Deploy arena
git push origin dev

# Check arena
curl https://nojohns-arena-production.up.railway.app/health
curl https://nojohns-arena-production.up.railway.app/pools

# View logs
railway logs

# Release to public
./scripts/release-to-main.sh
```
