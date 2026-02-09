#!/usr/bin/env bash
# matchmake.sh — Find and play arena matches.
# Used by Moltbots via the No Johns skill.
#
# Usage:
#   matchmake.sh --fighter phillip
#   matchmake.sh --fighter phillip --auto-wager 0.01

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

FIGHTER=""
WAGER=""

while [ $# -gt 0 ]; do
    case "$1" in
        --fighter) FIGHTER="$2"; shift 2;;
        --auto-wager) WAGER="--wager $2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "$FIGHTER" ]; then
    echo "Usage:"
    echo "  matchmake.sh --fighter phillip"
    echo "  matchmake.sh --fighter phillip --auto-wager 0.01"
    exit 1
fi

# shellcheck disable=SC2086
"$PYTHON" -m nojohns.cli matchmake "$FIGHTER" $WAGER
