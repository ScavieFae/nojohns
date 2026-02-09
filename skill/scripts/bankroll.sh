#!/usr/bin/env bash
# bankroll.sh — Query bankroll state and get Kelly criterion recommendations.
# Used by Moltbots via the No Johns skill.
#
# Usage:
#   bankroll.sh --status                              # Full bankroll snapshot
#   bankroll.sh --kelly --opponent-elo 1400            # Wager recommendation
#   bankroll.sh --kelly --opponent-elo 1400 --our-elo 1540 --risk aggressive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Use project venv if available
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

if [ "${1:-}" = "--status" ]; then
    "$PYTHON" -c "
import json
from nojohns.config import load_config
from agents.bankroll import get_bankroll_state

cfg = load_config()
if not cfg.wallet or not cfg.wallet.address:
    print(json.dumps({'error': 'No wallet configured. Run: nojohns setup wallet'}))
    exit(1)
if not cfg.chain or not cfg.chain.wager:
    print(json.dumps({'error': 'No wager contract configured'}))
    exit(1)

state = get_bankroll_state(cfg.wallet.address, cfg.chain.rpc_url, cfg.chain.wager)
print(json.dumps({
    'address': cfg.wallet.address,
    'balance_mon': state.balance_mon,
    'active_exposure_mon': state.active_exposure_wei / 10**18,
    'available_mon': state.available_mon,
}))
"
elif [ "${1:-}" = "--kelly" ]; then
    shift
    OUR_ELO=1500
    OPP_ELO=1500
    RISK="moderate"
    while [ $# -gt 0 ]; do
        case "$1" in
            --our-elo) OUR_ELO="$2"; shift 2;;
            --opponent-elo) OPP_ELO="$2"; shift 2;;
            --risk) RISK="$2"; shift 2;;
            *) echo "Unknown option: $1"; exit 1;;
        esac
    done
    "$PYTHON" -c "
import json
from agents.bankroll import win_probability_from_elo, kelly_fraction, kelly_wager
from agents.strategy import KellyStrategy, MatchContext, RISK_PROFILES
from agents.scouting import ScoutReport

our_elo = $OUR_ELO
opp_elo = $OPP_ELO
risk = '$RISK'

win_prob = win_probability_from_elo(our_elo, opp_elo)
kf = kelly_fraction(win_prob)
profile = RISK_PROFILES[risk]

# Use 1 MON as reference bankroll for percentage display
ref_bankroll = 10**18
amount = kelly_wager(win_prob, ref_bankroll, profile['multiplier'], profile['max_pct'])

print(json.dumps({
    'our_elo': our_elo,
    'opponent_elo': opp_elo,
    'win_probability': round(win_prob, 4),
    'kelly_fraction': round(kf, 4),
    'risk_profile': risk,
    'recommended_pct': round(amount / ref_bankroll * 100, 2),
    'edge': 'yes' if win_prob > 0.5 else 'no',
}))
"
else
    echo "Usage:"
    echo "  bankroll.sh --status                    # Bankroll snapshot"
    echo "  bankroll.sh --kelly --opponent-elo 1400  # Wager recommendation"
    exit 1
fi
