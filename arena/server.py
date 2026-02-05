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

# Allow the website (and other frontends) to call arena endpoints
from starlette.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================
# Request/Response Models
# ======================================================================


class JoinRequest(BaseModel):
    connect_code: str
    fighter_name: str | None = None
    wallet_address: str | None = None


class JoinResponse(BaseModel):
    queue_id: str
    status: str


class QueueStatusResponse(BaseModel):
    queue_id: str
    status: str
    position: int | None = None
    match_id: str | None = None
    opponent_code: str | None = None
    opponent_wallet: str | None = None


class ResultRequest(BaseModel):
    queue_id: str
    outcome: str
    duration_seconds: float | None = None
    stocks_remaining: int | None = None
    opponent_stocks: int | None = None


class SuccessResponse(BaseModel):
    success: bool


class MatchResponse(BaseModel):
    id: str
    status: str
    p1_connect_code: str
    p2_connect_code: str
    p1_wallet: str | None = None
    p2_wallet: str | None = None
    p1_result: str | None = None
    p2_result: str | None = None
    # Canonical result fields — set by arena when match completes.
    # Both agents sign these exact values for EIP-712 consistency.
    winner_wallet: str | None = None
    loser_wallet: str | None = None
    winner_score: int | None = None  # stocks remaining (per-game, not series)
    loser_score: int | None = None
    result_timestamp: int | None = None  # unix epoch, deterministic
    created_at: str | None = None
    completed_at: str | None = None
    # Wager coordination fields
    wager_status: str | None = None  # 'proposed', 'accepted', 'declined', 'settled'
    wager_amount: int | None = None  # in wei
    wager_id: int | None = None  # onchain wager ID
    wager_proposer: str | None = None  # 'p1' or 'p2'


class SignatureRequest(BaseModel):
    address: str
    signature: str  # hex-encoded 65-byte EIP-712 signature


class SignatureResponse(BaseModel):
    match_id: str
    signatures_received: int
    ready_for_submission: bool


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

    queue_id = db.add_to_queue(req.connect_code, req.fighter_name, req.wallet_address)
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
            "opponent_wallet": opponent.get("wallet_address"),
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
            opponent_wallet = match["p2_wallet"]
        else:
            opponent_code = match["p1_connect_code"]
            opponent_wallet = match["p1_wallet"]

        return {
            "queue_id": queue_id,
            "status": "matched",
            "match_id": entry["match_id"],
            "opponent_code": opponent_code,
            "opponent_wallet": opponent_wallet,
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
        match_id, req.queue_id, req.outcome, req.duration_seconds,
        req.stocks_remaining, req.opponent_stocks,
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


@app.post("/matches/{match_id}/signature", response_model=SignatureResponse)
def submit_signature(match_id: str, req: SignatureRequest) -> dict[str, Any]:
    """Submit an EIP-712 match result signature.

    Each side submits their signature after a match ends. When both
    signatures are collected, the result is ready for onchain submission.
    """
    db = get_db()

    match = db.get_match(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    db.store_signature(match_id, req.address, req.signature)
    sigs = db.get_signatures(match_id)
    count = len(sigs)

    logger.info(
        f"Signature for {match_id} from {req.address} ({count}/2 received)"
    )

    return {
        "match_id": match_id,
        "signatures_received": count,
        "ready_for_submission": count >= 2,
    }


@app.get("/matches/{match_id}/signatures")
def get_signatures(match_id: str) -> dict[str, Any]:
    """Get all signatures collected for a match."""
    db = get_db()

    match = db.get_match(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    sigs = db.get_signatures(match_id)
    return {
        "match_id": match_id,
        "signatures": sigs,
        "signatures_received": len(sigs),
        "ready_for_submission": len(sigs) >= 2,
    }


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    """Server health check."""
    db = get_db()
    return {
        "status": "ok",
        "queue_size": db.queue_size(),
        "active_matches": db.active_matches(),
    }


# ======================================================================
# Wager Coordination Endpoints
# ======================================================================


class WagerProposalRequest(BaseModel):
    queue_id: str
    amount_wei: int
    wager_id: int  # onchain wager ID from proposeWager()


class WagerAcceptRequest(BaseModel):
    queue_id: str


class WagerStatusResponse(BaseModel):
    match_id: str
    wager_status: str | None  # None, 'proposed', 'accepted', 'declined', 'settled'
    wager_amount: int | None
    wager_id: int | None
    wager_proposer: str | None  # 'p1' or 'p2'


@app.post("/matches/{match_id}/wager/propose", response_model=WagerStatusResponse)
def propose_match_wager(match_id: str, req: WagerProposalRequest) -> dict[str, Any]:
    """Propose a wager for a match (after matchmaking, before game starts)."""
    db = get_db()

    match = db.get_match(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    success = db.propose_wager(match_id, req.queue_id, req.amount_wei, req.wager_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot propose wager")

    match = db.get_match(match_id)
    return {
        "match_id": match_id,
        "wager_status": match.get("wager_status"),
        "wager_amount": match.get("wager_amount"),
        "wager_id": match.get("wager_id"),
        "wager_proposer": match.get("wager_proposer"),
    }


@app.post("/matches/{match_id}/wager/accept", response_model=WagerStatusResponse)
def accept_match_wager(match_id: str, req: WagerAcceptRequest) -> dict[str, Any]:
    """Accept a wager proposal."""
    db = get_db()

    match = db.get_match(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    success = db.accept_wager(match_id, req.queue_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot accept wager")

    match = db.get_match(match_id)
    return {
        "match_id": match_id,
        "wager_status": match.get("wager_status"),
        "wager_amount": match.get("wager_amount"),
        "wager_id": match.get("wager_id"),
        "wager_proposer": match.get("wager_proposer"),
    }


@app.post("/matches/{match_id}/wager/decline", response_model=WagerStatusResponse)
def decline_match_wager(match_id: str, req: WagerAcceptRequest) -> dict[str, Any]:
    """Decline a wager proposal (game proceeds without wager)."""
    db = get_db()

    match = db.get_match(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    success = db.decline_wager(match_id, req.queue_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot decline wager")

    match = db.get_match(match_id)
    return {
        "match_id": match_id,
        "wager_status": match.get("wager_status"),
        "wager_amount": match.get("wager_amount"),
        "wager_id": match.get("wager_id"),
        "wager_proposer": match.get("wager_proposer"),
    }


@app.get("/matches/{match_id}/wager", response_model=WagerStatusResponse)
def get_match_wager(match_id: str) -> dict[str, Any]:
    """Get wager status for a match."""
    db = get_db()

    match = db.get_match(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    return {
        "match_id": match_id,
        "wager_status": match.get("wager_status"),
        "wager_amount": match.get("wager_amount"),
        "wager_id": match.get("wager_id"),
        "wager_proposer": match.get("wager_proposer"),
    }
