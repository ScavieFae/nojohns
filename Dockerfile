FROM python:3.12-slim

WORKDIR /app

# Install git (needed for pip install of libmelee from GitHub)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -e ".[arena,wallet]"

VOLUME /data

ENV PORT=8000
EXPOSE 8000

CMD python -m nojohns.cli arena --port $PORT --db /data/arena.db
