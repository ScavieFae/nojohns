"""
arena/db.py - SQLite storage for the matchmaking server.

All queries go through ArenaDB. One instance per server lifetime,
backed by a single SQLite file (or :memory: for tests).
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any


class ArenaDB:
    """Thin wrapper around SQLite for queue + match storage."""

    def __init__(self, path: str = "arena.db"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS queue (
                id TEXT PRIMARY KEY,
                connect_code TEXT NOT NULL,
                fighter_name TEXT,
                status TEXT DEFAULT 'waiting',
                match_id TEXT,
                created_at TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS matches (
                id TEXT PRIMARY KEY,
                p1_queue_id TEXT NOT NULL,
                p2_queue_id TEXT NOT NULL,
                p1_connect_code TEXT NOT NULL,
                p2_connect_code TEXT NOT NULL,
                status TEXT DEFAULT 'playing',
                p1_result TEXT,
                p2_result TEXT,
                p1_duration REAL,
                p2_duration REAL,
                created_at TEXT,
                completed_at TEXT
            );
            """
        )

    # ------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------

    def add_to_queue(self, connect_code: str, fighter_name: str | None = None) -> str:
        """Add a player to the queue. Returns queue_id.

        If the same connect_code already has a waiting entry, it is cancelled
        first to prevent self-matching from stale entries.
        """
        # Cancel any stale waiting entries for this connect code
        self._conn.execute(
            "UPDATE queue SET status = 'cancelled', updated_at = ? "
            "WHERE connect_code = ? AND status = 'waiting'",
            (_now(), connect_code),
        )

        queue_id = str(uuid.uuid4())
        now = _now()
        self._conn.execute(
            "INSERT INTO queue (id, connect_code, fighter_name, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'waiting', ?, ?)",
            (queue_id, connect_code, fighter_name, now, now),
        )
        self._conn.commit()
        return queue_id

    def get_queue_entry(self, queue_id: str) -> dict[str, Any] | None:
        """Fetch a queue entry by id. Returns dict or None."""
        row = self._conn.execute("SELECT * FROM queue WHERE id = ?", (queue_id,)).fetchone()
        if row is None:
            return None
        # Touch updated_at so the entry doesn't go stale while polling
        self._conn.execute(
            "UPDATE queue SET updated_at = ? WHERE id = ?", (_now(), queue_id)
        )
        self._conn.commit()
        return dict(row)

    def find_waiting_opponent(self, exclude_id: str) -> dict[str, Any] | None:
        """Find the oldest waiting queue entry that isn't us (by ID or connect code)."""
        entry = self._conn.execute(
            "SELECT connect_code FROM queue WHERE id = ?", (exclude_id,)
        ).fetchone()
        if entry is None:
            return None
        row = self._conn.execute(
            "SELECT * FROM queue WHERE status = 'waiting' AND id != ? AND connect_code != ? "
            "ORDER BY created_at ASC LIMIT 1",
            (exclude_id, entry["connect_code"]),
        ).fetchone()
        return dict(row) if row else None

    def update_queue_status(
        self, queue_id: str, status: str, match_id: str | None = None
    ) -> None:
        """Update a queue entry's status (and optionally match_id)."""
        if match_id is not None:
            self._conn.execute(
                "UPDATE queue SET status = ?, match_id = ?, updated_at = ? WHERE id = ?",
                (status, match_id, _now(), queue_id),
            )
        else:
            self._conn.execute(
                "UPDATE queue SET status = ?, updated_at = ? WHERE id = ?",
                (status, _now(), queue_id),
            )
        self._conn.commit()

    def cancel_queue_entry(self, queue_id: str) -> bool:
        """Cancel a queue entry. Returns True if it was waiting."""
        row = self._conn.execute(
            "SELECT status FROM queue WHERE id = ?", (queue_id,)
        ).fetchone()
        if row is None or row["status"] != "waiting":
            return False
        self.update_queue_status(queue_id, "cancelled")
        return True

    def queue_position(self, queue_id: str) -> int:
        """How many waiting entries are ahead of this one (1-indexed)."""
        row = self._conn.execute(
            "SELECT created_at FROM queue WHERE id = ?", (queue_id,)
        ).fetchone()
        if row is None:
            return 0
        count = self._conn.execute(
            "SELECT COUNT(*) FROM queue WHERE status = 'waiting' AND created_at <= ?",
            (row["created_at"],),
        ).fetchone()[0]
        return count

    def expire_stale_entries(self, timeout_seconds: int = 300) -> int:
        """Expire queue entries that haven't been polled recently. Returns count."""
        cutoff = datetime.now(timezone.utc).timestamp() - timeout_seconds
        # Convert cutoff back to ISO for comparison
        from datetime import datetime as dt

        cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat()
        cursor = self._conn.execute(
            "UPDATE queue SET status = 'expired' WHERE status = 'waiting' AND updated_at < ?",
            (cutoff_iso,),
        )
        self._conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Matches
    # ------------------------------------------------------------------

    def create_match(self, p1_entry: dict, p2_entry: dict) -> str:
        """Create a match from two queue entries. Returns match_id."""
        match_id = str(uuid.uuid4())
        now = _now()
        self._conn.execute(
            "INSERT INTO matches (id, p1_queue_id, p2_queue_id, p1_connect_code, p2_connect_code, "
            "status, created_at) VALUES (?, ?, ?, ?, ?, 'playing', ?)",
            (
                match_id,
                p1_entry["id"],
                p2_entry["id"],
                p1_entry["connect_code"],
                p2_entry["connect_code"],
                now,
            ),
        )
        # Update both queue entries
        for entry in [p1_entry, p2_entry]:
            self._conn.execute(
                "UPDATE queue SET status = 'matched', match_id = ?, updated_at = ? WHERE id = ?",
                (match_id, now, entry["id"]),
            )
        self._conn.commit()
        return match_id

    def get_match(self, match_id: str) -> dict[str, Any] | None:
        """Fetch a match by id."""
        row = self._conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)).fetchone()
        return dict(row) if row else None

    def report_result(
        self,
        match_id: str,
        queue_id: str,
        outcome: str,
        duration_seconds: float | None = None,
    ) -> bool:
        """Report one side's result. Returns True if match is now complete."""
        match = self.get_match(match_id)
        if match is None:
            return False

        now = _now()

        if queue_id == match["p1_queue_id"]:
            self._conn.execute(
                "UPDATE matches SET p1_result = ?, p1_duration = ? WHERE id = ?",
                (outcome, duration_seconds, match_id),
            )
        elif queue_id == match["p2_queue_id"]:
            self._conn.execute(
                "UPDATE matches SET p2_result = ?, p2_duration = ? WHERE id = ?",
                (outcome, duration_seconds, match_id),
            )
        else:
            return False

        # Re-fetch to check if both sides reported
        self._conn.commit()
        match = self.get_match(match_id)
        if match["p1_result"] and match["p2_result"]:
            self._conn.execute(
                "UPDATE matches SET status = 'completed', completed_at = ? WHERE id = ?",
                (now, match_id),
            )
            self._conn.commit()
            return True
        return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def queue_size(self) -> int:
        """Number of entries currently waiting."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM queue WHERE status = 'waiting'"
        ).fetchone()[0]

    def active_matches(self) -> int:
        """Number of matches currently playing."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM matches WHERE status = 'playing'"
        ).fetchone()[0]


def _now() -> str:
    """ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()
