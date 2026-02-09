FROM python:3.12-slim

WORKDIR /app

# gcc is needed to build pyenet (C extension required by libmelee).
# git is needed to pip install libmelee from GitHub.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc git libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -e ".[arena,wallet]"

# Railway manages volumes externally — mount at /data via dashboard/CLI
RUN mkdir -p /data

ENV PORT=8000
EXPOSE 8000

CMD python -m nojohns.cli arena --port $PORT --db /data/arena.db
