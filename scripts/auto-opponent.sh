#!/usr/bin/env bash
# auto-opponent.sh — Run an always-on opponent against the public arena.
# Usage: ./scripts/auto-opponent.sh [fighter]
set -euo pipefail

FIGHTER="${1:-phillip}"
ARENA_URL="${ARENA_URL:-https://nojohns-arena-production.up.railway.app}"

exec .venv/bin/python -m nojohns.cli auto "$FIGHTER" --no-wager --cooldown 15 --server "$ARENA_URL"
