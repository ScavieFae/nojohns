# .loop/config.sh — project configuration
PROJECT_NAME="nojohns"
HEARTBEAT_INTERVAL=300        # Idle interval (seconds)
WORKER_COOLDOWN=30            # Between worker iterations
MAX_ITERATIONS=20             # Safety limit per brief
NTFY_TOPIC=""                 # ntfy.sh topic (empty = no push notifications)
VERIFY_CMD=".venv/bin/python -m pytest tests/ -v -o addopts="
GIT_REMOTE="origin"
GIT_MAIN_BRANCH="main"
