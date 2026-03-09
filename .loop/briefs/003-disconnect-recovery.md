# Brief: Disconnect Recovery & Admin Controls

**Branch:** scav/disconnect-recovery
**Model:** sonnet

## Goal

Give the operator tools to recover from stuck matches within 30 seconds. Decouple match expiration from pool cancellation — the operator decides what to do, not an automated chain that might silently fail. Protect admin endpoints with auth.

No wagers during the tournament — only prediction pools. Ignore wager logic.

Reference: `arena/server.py` (match expiration, pool lifecycle, existing `/admin/cleanup`), `nojohns/contract.py` (cancelPool).

## Design Decisions

- **Decouple match and pool lifecycle.** Expiring a match does NOT automatically cancel its pool. The operator decides: cancel the pool (refund bettors) or rematch (pool stays open, new match resolves it).
- **Manual pacing = manual recovery.** No automated cascades. Admin panel has independent buttons for each action.
- **Auth via bearer token.** `ADMIN_TOKEN` env var on Railway. All `/admin/*` endpoints check it.
- **Faster default timeout.** 300s (5 min) instead of 1800s. Configurable via `MATCH_TIMEOUT` env var.

## Tasks

1. Add `ADMIN_TOKEN` env var check to all `/admin/*` endpoints in `arena/server.py`. Middleware or dependency that reads `Authorization: Bearer <token>` header, returns 401 without it. Apply to existing `/admin/cleanup` too.

2. Add `MATCH_TIMEOUT` env var (default 300) to `_expire_matches_and_cancel_pools()`. Replace hardcoded 1800.

3. Decouple match expiration from pool cancellation. `_expire_matches_and_cancel_pools()` should only expire matches, not auto-cancel pools. Rename to `_expire_stale_matches()`. Pool cancellation becomes an explicit admin action.

4. Add targeted admin endpoints:
   - `POST /admin/matches/{match_id}/expire` — expire one specific match. Does NOT cancel its pool.
   - `POST /admin/pools/{pool_id}/cancel` — cancel one specific prediction pool onchain, enabling refunds.
   - `POST /admin/matches/{match_id}/rematch` — create a new match with the same two players (read from expired match), queue them. Optionally reassign the existing pool to the new match so bets carry over.
   - `GET /admin/wallet` — return arena wallet address + MON balance. Operator can check this from phone.

5. Add wallet balance logging: log balance after every onchain operation (`_try_create_pool`, `_try_resolve_pool`, `_try_cancel_pool`). Log WARNING if below 0.05 MON. Add balance check at server startup — log the starting balance.

6. Add auth to existing `/admin/cleanup` endpoint.

## Completion Criteria

- [ ] All `/admin/*` endpoints return 401 without valid bearer token
- [ ] Match timeout configurable via `MATCH_TIMEOUT` env var, defaults to 300s
- [ ] Match expiration does NOT auto-cancel prediction pools
- [ ] `POST /admin/matches/{id}/expire` expires a single match
- [ ] `POST /admin/pools/{id}/cancel` cancels a single pool onchain
- [ ] `POST /admin/matches/{id}/rematch` re-queues same players
- [ ] `GET /admin/wallet` returns address + balance
- [ ] Wallet balance logged after onchain ops, WARNING below 0.05 MON

## Verification

- `.venv/bin/python -m pytest tests/ -v -o "addopts="` passes
- `curl -X POST localhost:8000/admin/matches/1/expire` returns 401
- `curl -X POST -H "Authorization: Bearer $TOKEN" localhost:8000/admin/matches/1/expire` returns 200
- `curl -H "Authorization: Bearer $TOKEN" localhost:8000/admin/wallet` returns balance
