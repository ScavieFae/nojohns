FROM python:3.12-slim

WORKDIR /app

# Install build deps: gcc + libc6-dev for pyenet C extension, git for pip install from GitHub
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc git libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -e ".[arena,wallet]"

# Railway volumes mount at /data — no VOLUME keyword (Railway bans it)
RUN mkdir -p /data

ENV PORT=8000
EXPOSE 8000

CMD python -m nojohns.cli arena --port $PORT --db /data/arena.db
