"""
arena/server.py - FastAPI matchmaking server for No Johns.

Endpoints:
    POST   /queue/join          Join the matchmaking queue
    GET    /queue/{queue_id}    Poll queue status
    DELETE /queue/{queue_id}    Leave the queue
    POST   /matches/{id}/result Report match result
    GET    /matches/{id}        Get match details
    GET    /health              Server health check
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .db import ArenaDB

logger = logging.getLogger(__name__)

# Global DB instance — set during lifespan
_db: ArenaDB | None = None


def get_db() -> ArenaDB:
    assert _db is not None, "DB not initialized"
    return _db


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db
    db_path = getattr(app.state, "db_path", "arena.db")
    _db = ArenaDB(db_path)
    logger.info(f"Arena DB initialized: {db_path}")
    yield
    _db = None


app = FastAPI(title="No Johns Arena", lifespan=lifespan)


# ======================================================================
# Request/Response Models
# ======================================================================


class JoinRequest(BaseModel):
    connect_code: str
    fighter_name: str | None = None


class JoinResponse(BaseModel):
    queue_id: str
    status: str


class QueueStatusResponse(BaseModel):
    queue_id: str
    status: str
    position: int | None = None
    match_id: str | None = None
    opponent_code: str | None = None


class ResultRequest(BaseModel):
    queue_id: str
    outcome: str
    duration_seconds: float | None = None


class SuccessResponse(BaseModel):
    success: bool


class MatchResponse(BaseModel):
    id: str
    status: str
    p1_connect_code: str
    p2_connect_code: str
    p1_result: str | None = None
    p2_result: str | None = None
    created_at: str | None = None
    completed_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    queue_size: int
    active_matches: int


# ======================================================================
# Endpoints
# ======================================================================


@app.post("/queue/join", response_model=QueueStatusResponse)
def join_queue(req: JoinRequest) -> dict[str, Any]:
    """Join the matchmaking queue. If an opponent is waiting, match immediately."""
    db = get_db()

    # Sweep stale entries on each queue operation
    db.expire_stale_entries()

    queue_id = db.add_to_queue(req.connect_code, req.fighter_name)
    logger.info(f"Joined queue: {req.connect_code} ({req.fighter_name}) -> {queue_id}")

    # Try to match immediately
    opponent = db.find_waiting_opponent(queue_id)
    if opponent:
        entry = db.get_queue_entry(queue_id)
        match_id = db.create_match(opponent, entry)
        logger.info(
            f"Matched! {opponent['connect_code']} vs {req.connect_code} -> {match_id}"
        )

        # Return matched status with opponent info
        return {
            "queue_id": queue_id,
            "status": "matched",
            "match_id": match_id,
            "opponent_code": opponent["connect_code"],
        }

    # No match yet
    position = db.queue_position(queue_id)
    return {
        "queue_id": queue_id,
        "status": "waiting",
        "position": position,
    }


@app.get("/queue/{queue_id}", response_model=QueueStatusResponse)
def poll_queue(queue_id: str) -> dict[str, Any]:
    """Poll queue entry status. Clients call this every 2 seconds."""
    db = get_db()

    # Sweep stale entries
    db.expire_stale_entries()

    entry = db.get_queue_entry(queue_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Queue entry not found")

    if entry["status"] == "matched":
        # Look up opponent code from the match
        match = db.get_match(entry["match_id"])
        if match is None:
            raise HTTPException(status_code=500, detail="Match not found")

        # Figure out which side we are
        if match["p1_queue_id"] == queue_id:
            opponent_code = match["p2_connect_code"]
        else:
            opponent_code = match["p1_connect_code"]

        return {
            "queue_id": queue_id,
            "status": "matched",
            "match_id": entry["match_id"],
            "opponent_code": opponent_code,
        }

    if entry["status"] == "waiting":
        position = db.queue_position(queue_id)
        return {
            "queue_id": queue_id,
            "status": "waiting",
            "position": position,
        }

    # expired, cancelled
    return {
        "queue_id": queue_id,
        "status": entry["status"],
    }


@app.delete("/queue/{queue_id}", response_model=SuccessResponse)
def leave_queue(queue_id: str) -> dict[str, Any]:
    """Leave the matchmaking queue."""
    db = get_db()
    success = db.cancel_queue_entry(queue_id)
    return {"success": success}


@app.post("/matches/{match_id}/result", response_model=SuccessResponse)
def report_result(match_id: str, req: ResultRequest) -> dict[str, Any]:
    """Report one side's match result."""
    db = get_db()

    match = db.get_match(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    completed = db.report_result(
        match_id, req.queue_id, req.outcome, req.duration_seconds
    )
    logger.info(
        f"Result reported for {match_id} by {req.queue_id}: {req.outcome}"
        + (" (match complete)" if completed else "")
    )
    return {"success": True}


@app.get("/matches/{match_id}", response_model=MatchResponse)
def get_match(match_id: str) -> dict[str, Any]:
    """Get match details."""
    db = get_db()
    match = db.get_match(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    return dict(match)


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    """Server health check."""
    db = get_db()
    return {
        "status": "ok",
        "queue_size": db.queue_size(),
        "active_matches": db.active_matches(),
    }
