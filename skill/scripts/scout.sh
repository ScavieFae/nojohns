#!/usr/bin/env bash
# scout.sh — Look up opponent Elo and track record.
# Used by Moltbots via the No Johns skill.
#
# Usage:
#   scout.sh --agent-id 12
#   scout.sh --wallet 0x1234...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

if [ "${1:-}" = "--agent-id" ]; then
    AGENT_ID="${2:?Missing agent ID}"
    "$PYTHON" -c "
import json
from nojohns.config import load_config
from agents.scouting import scout_opponent

cfg = load_config()
if not cfg.chain or not cfg.chain.reputation_registry:
    print(json.dumps({'error': 'No reputation registry configured'}))
    exit(1)

report = scout_opponent($AGENT_ID, cfg.chain.rpc_url, cfg.chain.reputation_registry)
print(json.dumps({
    'agent_id': report.agent_id,
    'elo': report.elo,
    'peak_elo': report.peak_elo,
    'record': report.record,
    'is_unknown': report.is_unknown,
}))
"
elif [ "${1:-}" = "--wallet" ]; then
    WALLET="${2:?Missing wallet address}"
    "$PYTHON" -c "
import json
from nojohns.config import load_config
from agents.scouting import scout_by_wallet

cfg = load_config()
rpc = cfg.chain.rpc_url if cfg.chain else 'https://rpc.monad.xyz'
reg = cfg.chain.reputation_registry if cfg.chain else ''

report = scout_by_wallet('$WALLET', rpc, reg)
print(json.dumps({
    'wallet': '$WALLET',
    'elo': report.elo,
    'peak_elo': report.peak_elo,
    'record': report.record,
    'is_unknown': report.is_unknown,
}))
"
else
    echo "Usage:"
    echo "  scout.sh --agent-id 12       # Look up by ERC-8004 agent ID"
    echo "  scout.sh --wallet 0x1234...  # Look up by wallet address"
    exit 1
fi
