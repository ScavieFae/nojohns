# Build Pipeline & Workflow

Development workflow and CI/CD practices for No Johns.

## Environments

| Env | Website | Arena | Chain |
|-----|---------|-------|-------|
| **Local** | localhost:5173 | localhost:8000 | Testnet |
| **Preview** | `*.vercel.app` (per-PR) | Railway | Testnet |
| **Production** | `web-phi-two-18.vercel.app` | Railway | Testnet |

## Local Development

### Website
```bash
cd web
npm run build    # TypeScript check + production build
npm run dev      # Dev server with hot reload
```

### Python (arena, CLI, agents)
```bash
source .venv/bin/activate
pytest tests/ -v -o "addopts="    # Run tests
nojohns arena                      # Local arena server
```

### Contracts
```bash
cd contracts
forge build      # Compile
forge test -vv   # Run tests
```

## Workflow

### Standard Change Flow

```
code → local test → commit → push → CI checks → preview deploy → merge → prod deploy
```

1. **Code locally** — make changes
2. **Test locally** — `npm run build`, `pytest`, manual verification
3. **Commit & push** — GitHub Actions CI runs automatically
4. **Preview deploy** — Vercel creates a preview URL for the PR
5. **Test preview** — verify with real data on testnet
6. **Merge to main** — triggers production deploy

### When to Test Where

| Change Type | Local | Preview | Prod |
|-------------|-------|---------|------|
| UI tweaks | Yes | Optional | After local |
| New data hooks | Yes | Yes | After preview |
| Contract integration | Yes (mock) | Yes (testnet) | After preview |
| Arena API changes | Yes | Yes | Scav's call |
| Pre-demo polish | Yes | Yes | Smoke test after |

## CI/CD

### GitHub Actions

On every push and PR:
- **web**: `npm ci && npm run build` — catches TypeScript errors
- **python**: `pip install && pytest` — catches test regressions

See `.github/workflows/ci.yml`.

### Vercel

- **Preview deploys**: Every PR gets a unique URL (automatic)
- **Production deploys**: On merge to main, or manual `vercel --prod`
- **Env vars**: Set in Vercel dashboard, not committed

### Railway (Arena)

- **Deploy**: `railway up` from repo root
- **Health check**: `https://nojohns-arena-production.up.railway.app/health`
- **Logs**: `railway logs`

## Environment Variables

### Website (Vercel)

| Var | Local (.env) | Production (Vercel) |
|-----|--------------|---------------------|
| `VITE_USE_MOCK_DATA` | `false` | not set (defaults false) |
| `VITE_ARENA_URL` | `http://localhost:8000` | `https://nojohns-arena-production.up.railway.app` |

### Arena (Railway)

| Var | Purpose |
|-----|---------|
| `PORT` | Server port (Railway sets this) |
| `ARENA_PRIVATE_KEY` | Wallet for Elo posting |
| `MONAD_RPC_URL` | Chain RPC |
| `REPUTATION_REGISTRY` | ERC-8004 address |

## Quick Commands

```bash
# Local website
cd web && npm run dev

# Local arena
nojohns arena --port 8000

# Run all tests
pytest tests/ -v -o "addopts="

# Deploy website to prod
cd web && vercel --prod

# Deploy arena to Railway
railway up

# Check arena health
curl https://nojohns-arena-production.up.railway.app/health
```

## Troubleshooting

### "TypeScript errors on deploy but not locally"

Your local `node_modules` may be stale. Run:
```bash
rm -rf node_modules && npm ci && npm run build
```

### "Preview deploy doesn't show real data"

Check that `VITE_USE_MOCK_DATA` is not set (or set to `false`) in Vercel env vars.

### "Live viewer not connecting"

1. Check arena is running: `curl .../health`
2. Check WebSocket URL in browser devtools (should be `wss://...railway.app/ws/match/...`)
3. Check CORS if arena rejects the request

### "Tests pass locally but fail in CI"

CI uses a fresh install. Common causes:
- Missing dependency in `pyproject.toml`
- Test depends on local state (config file, database)
- Flaky test with timing issues
