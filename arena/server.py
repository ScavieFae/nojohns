"""
arena/server.py - FastAPI matchmaking server for No Johns.

Endpoints:
    POST   /queue/join          Join the matchmaking queue
    GET    /queue/{queue_id}    Poll queue status
    DELETE /queue/{queue_id}    Leave the queue
    POST   /matches/{id}/result Report match result
    GET    /matches/{id}        Get match details
    GET    /health              Server health check

Live streaming:
    WS     /ws/match/{id}       WebSocket for live match viewing
    POST   /matches/{id}/frame  Post frame data (client -> arena)
    POST   /matches/{id}/start  Signal match start with player info
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .db import ArenaDB

logger = logging.getLogger(__name__)

# =============================================================================
# Arena Wallet (for posting Elo updates to ReputationRegistry)
# =============================================================================
# SECURITY: Arena wallet key is loaded from env var, NOT config file.
# This wallet is used only for posting Elo updates — it cannot access
# player funds or forge match results (those need player signatures).
# Keep minimal MON in this wallet, just enough for gas.

_arena_account = None  # Loaded lazily


def _get_arena_account():
    """Load arena wallet from ARENA_PRIVATE_KEY env var. Returns None if not set."""
    global _arena_account
    if _arena_account is not None:
        return _arena_account

    key = os.environ.get("ARENA_PRIVATE_KEY")
    if not key:
        return None

    try:
        from eth_account import Account
        _arena_account = Account.from_key(key)
        logger.info(f"Arena wallet loaded: {_arena_account.address}")
        return _arena_account
    except ImportError:
        logger.debug("eth-account not installed — Elo posting disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to load arena wallet: {e}")
        return None


def _post_elo_updates(match: dict):
    """Post Elo updates for both players after match completion.

    Called when both signatures are received. Posts to ReputationRegistry
    on behalf of both players (arena is the reputation authority).

    Silently skips if:
    - Arena wallet not configured
    - Either player doesn't have an agent_id
    - ReputationRegistry not reachable
    """
    account = _get_arena_account()
    if account is None:
        logger.debug("Arena wallet not configured — skipping Elo posting")
        return

    # Get chain config from env
    rpc_url = os.environ.get("MONAD_RPC_URL", "https://testnet-rpc.monad.xyz")
    chain_id = int(os.environ.get("MONAD_CHAIN_ID", "10143"))
    reputation_registry = os.environ.get(
        "REPUTATION_REGISTRY",
        "0x8004B663056A597Dffe9eCcC1965A193B7388713"  # testnet default
    )

    winner_agent_id = None
    loser_agent_id = None
    winner_wallet = match.get("winner_wallet")
    loser_wallet = match.get("loser_wallet")

    # Determine agent_ids from match (p1/p2 mapping)
    if winner_wallet and winner_wallet.lower() == (match.get("p1_wallet") or "").lower():
        winner_agent_id = match.get("p1_agent_id")
        loser_agent_id = match.get("p2_agent_id")
    elif winner_wallet and winner_wallet.lower() == (match.get("p2_wallet") or "").lower():
        winner_agent_id = match.get("p2_agent_id")
        loser_agent_id = match.get("p1_agent_id")

    if winner_agent_id is None and loser_agent_id is None:
        logger.debug("Neither player has agent_id — skipping Elo posting")
        return

    try:
        from nojohns.reputation import (
            get_current_elo,
            calculate_new_elo,
            post_elo_update,
            STARTING_ELO,
        )
    except ImportError:
        logger.debug("reputation module not available")
        return

    # Post Elo for winner
    if winner_agent_id is not None:
        try:
            current = get_current_elo(winner_agent_id, rpc_url, reputation_registry)
            # Use loser's Elo if available, else default
            loser_elo = STARTING_ELO
            if loser_agent_id is not None:
                loser_current = get_current_elo(loser_agent_id, rpc_url, reputation_registry)
                loser_elo = loser_current.elo

            new_elo = calculate_new_elo(current.elo, loser_elo, won=True)
            peak_elo = max(current.peak_elo, new_elo)
            record = f"{current.wins + 1}-{current.losses}"

            tx = post_elo_update(
                winner_agent_id, new_elo, peak_elo, record,
                account, rpc_url, reputation_registry, chain_id
            )
            if tx:
                logger.info(f"Posted Elo for winner agent {winner_agent_id}: {current.elo} → {new_elo}")
        except Exception as e:
            logger.warning(f"Failed to post winner Elo: {e}")

    # Post Elo for loser
    if loser_agent_id is not None:
        try:
            current = get_current_elo(loser_agent_id, rpc_url, reputation_registry)
            # Use winner's Elo if available
            winner_elo = STARTING_ELO
            if winner_agent_id is not None:
                winner_current = get_current_elo(winner_agent_id, rpc_url, reputation_registry)
                winner_elo = winner_current.elo

            new_elo = calculate_new_elo(current.elo, winner_elo, won=False)
            peak_elo = current.peak_elo  # Don't update peak on loss
            record = f"{current.wins}-{current.losses + 1}"

            tx = post_elo_update(
                loser_agent_id, new_elo, peak_elo, record,
                account, rpc_url, reputation_registry, chain_id
            )
            if tx:
                logger.info(f"Posted Elo for loser agent {loser_agent_id}: {current.elo} → {new_elo}")
        except Exception as e:
            logger.warning(f"Failed to post loser Elo: {e}")

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
    agent_id: int | None = None  # ERC-8004 agent ID for Elo tracking


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
    opponent_agent_id: int | None = None  # ERC-8004 agent ID


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
    live_match_ids: list[str] = []


# ======================================================================
# Endpoints
# ======================================================================


@app.post("/queue/join", response_model=QueueStatusResponse)
def join_queue(req: JoinRequest) -> dict[str, Any]:
    """Join the matchmaking queue. If an opponent is waiting, match immediately."""
    db = get_db()

    # Sweep stale entries on each queue operation
    db.expire_stale_entries()

    queue_id = db.add_to_queue(req.connect_code, req.fighter_name, req.wallet_address, req.agent_id)
    logger.info(f"Joined queue: {req.connect_code} ({req.fighter_name}) agent_id={req.agent_id} -> {queue_id}")

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
            "opponent_agent_id": opponent.get("agent_id"),
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

    # When both signatures received, post Elo updates (async, non-blocking)
    if count >= 2:
        try:
            _post_elo_updates(match)
        except Exception as e:
            logger.warning(f"Elo posting failed (non-critical): {e}")

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
    """Server health check. Returns live match IDs for spectator discovery."""
    _manager.sweep_stale()
    db = get_db()
    return {
        "status": "ok",
        "queue_size": db.queue_size(),
        "active_matches": db.active_matches(),
        "live_match_ids": list(_manager.match_info.keys()),
    }


# ======================================================================
# Live Match Streaming
# ======================================================================


class ConnectionManager:
    """Manages WebSocket connections for live match streaming."""

    STALE_MATCH_SECONDS = 30 * 60  # 30 minutes

    def __init__(self):
        # match_id -> list of connected WebSockets
        self.viewers: dict[str, list[WebSocket]] = {}
        # match_id -> last match_start message (for late joiners)
        self.match_info: dict[str, dict] = {}
        # match_id -> timestamp of last activity (for stale cleanup)
        self._last_activity: dict[str, float] = {}

    async def connect(self, match_id: str, websocket: WebSocket):
        await websocket.accept()
        if match_id not in self.viewers:
            self.viewers[match_id] = []
        self.viewers[match_id].append(websocket)
        logger.info(f"Viewer connected to {match_id} ({len(self.viewers[match_id])} total)")

        # Send match_start if we have it (late joiner)
        if match_id in self.match_info:
            try:
                await websocket.send_json(self.match_info[match_id])
            except Exception:
                pass

    def disconnect(self, match_id: str, websocket: WebSocket):
        if match_id in self.viewers:
            if websocket in self.viewers[match_id]:
                self.viewers[match_id].remove(websocket)
            if not self.viewers[match_id]:
                del self.viewers[match_id]
            logger.info(f"Viewer disconnected from {match_id}")

    async def broadcast(self, match_id: str, message: dict):
        """Broadcast message to all viewers of a match."""
        # Track activity for stale cleanup
        self._last_activity[match_id] = time.time()

        # Store match_start for late joiners (BEFORE checking viewers)
        if message.get("type") == "match_start":
            self.match_info[match_id] = message
            logger.info(f"Stored match_info for {match_id}")

        # Clean up match_info on match_end
        if message.get("type") == "match_end":
            self.match_info.pop(match_id, None)

        if match_id not in self.viewers:
            return

        dead = []
        for ws in self.viewers[match_id]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)

        # Remove dead connections
        for ws in dead:
            self.disconnect(match_id, ws)

    def viewer_count(self, match_id: str) -> int:
        return len(self.viewers.get(match_id, []))

    def cleanup_match(self, match_id: str):
        """Clean up all state for a match."""
        self.viewers.pop(match_id, None)
        self.match_info.pop(match_id, None)
        self._last_activity.pop(match_id, None)

    def sweep_stale(self):
        """Remove matches with no activity for STALE_MATCH_SECONDS."""
        now = time.time()
        stale = [
            mid for mid, ts in self._last_activity.items()
            if now - ts > self.STALE_MATCH_SECONDS
        ]
        for mid in stale:
            logger.info(f"Sweeping stale match {mid}")
            self.cleanup_match(mid)


# Global connection manager
_manager = ConnectionManager()


@app.websocket("/ws/match/{match_id}")
async def websocket_match(websocket: WebSocket, match_id: str):
    """WebSocket endpoint for live match viewing.

    Spectators connect here to receive frame data in real-time.
    """
    db = get_db()
    match = db.get_match(match_id)

    if match is None:
        await websocket.close(code=4004, reason="Match not found")
        return

    await _manager.connect(match_id, websocket)

    try:
        # Keep connection alive, wait for disconnect
        while True:
            # We don't expect messages from viewers, but need to handle disconnect
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        _manager.disconnect(match_id, websocket)


class MatchStartRequest(BaseModel):
    """Signal match start with player/stage info."""
    stage_id: int
    players: list[dict]  # [{port, character_id, connect_code, display_name?}, ...]


class FrameRequest(BaseModel):
    """Frame data from client."""
    frame: int
    players: list[dict]  # [{port, x, y, action_state_id, action_frame, facing_direction, percent, stocks}, ...]


class FrameBatchRequest(BaseModel):
    """Batch of frames from client — reduces HTTP round trips over internet."""
    frames: list[FrameRequest]


class GameEndRequest(BaseModel):
    """Game end signal."""
    game_number: int
    winner_port: int
    end_method: str  # "stocks", "timeout", "lras"


class MatchEndRequest(BaseModel):
    """Match end signal."""
    winner_port: int
    final_score: list[int]  # [p1_wins, p2_wins]


@app.post("/matches/{match_id}/stream/start")
async def stream_match_start(match_id: str, req: MatchStartRequest) -> dict[str, Any]:
    """Signal match start — called by client when game begins."""
    logger.info(f"[stream/start] Received for match {match_id}")
    db = get_db()
    match = db.get_match(match_id)
    if match is None:
        logger.warning(f"[stream/start] Match {match_id} not found in DB")
        raise HTTPException(status_code=404, detail="Match not found")

    message = {
        "type": "match_start",
        "matchId": match_id,
        "stageId": req.stage_id,
        "players": [
            {
                "port": p.get("port"),
                "characterId": p.get("character_id"),
                "connectCode": p.get("connect_code"),
                "displayName": p.get("display_name"),
            }
            for p in req.players
        ],
    }

    await _manager.broadcast(match_id, message)
    logger.info(f"Match {match_id} started streaming ({_manager.viewer_count(match_id)} viewers)")

    return {"success": True, "viewers": _manager.viewer_count(match_id)}


@app.post("/matches/{match_id}/stream/frame")
async def stream_frame(match_id: str, req: FrameRequest) -> dict[str, Any]:
    """Stream a frame — called by client every frame (~60fps)."""
    message = {
        "type": "frame",
        "frame": req.frame,
        "players": [
            {
                "port": p.get("port"),
                "x": p.get("x"),
                "y": p.get("y"),
                "actionStateId": p.get("action_state_id"),
                "actionFrame": p.get("action_frame"),
                "facingDirection": p.get("facing_direction"),
                "percent": p.get("percent"),
                "stocks": p.get("stocks"),
                "shieldHealth": p.get("shield_health"),
                "isInvincible": p.get("is_invincible"),
                "isInHitstun": p.get("is_in_hitstun"),
            }
            for p in req.players
        ],
    }

    await _manager.broadcast(match_id, message)
    return {"success": True}


@app.post("/matches/{match_id}/stream/frames")
async def stream_frames_batch(match_id: str, req: FrameBatchRequest) -> dict[str, Any]:
    """Stream a batch of frames — reduces HTTP round trips over internet."""
    for frame_req in req.frames:
        message = {
            "type": "frame",
            "frame": frame_req.frame,
            "players": [
                {
                    "port": p.get("port"),
                    "x": p.get("x"),
                    "y": p.get("y"),
                    "actionStateId": p.get("action_state_id"),
                    "actionFrame": p.get("action_frame"),
                    "facingDirection": p.get("facing_direction"),
                    "percent": p.get("percent"),
                    "stocks": p.get("stocks"),
                    "shieldHealth": p.get("shield_health"),
                    "isInvincible": p.get("is_invincible"),
                    "isInHitstun": p.get("is_in_hitstun"),
                }
                for p in frame_req.players
            ],
        }
        await _manager.broadcast(match_id, message)
    return {"success": True, "frames": len(req.frames)}


@app.post("/matches/{match_id}/stream/game_end")
async def stream_game_end(match_id: str, req: GameEndRequest) -> dict[str, Any]:
    """Signal game end — called when a game in the set ends."""
    message = {
        "type": "game_end",
        "gameNumber": req.game_number,
        "winnerPort": req.winner_port,
        "endMethod": req.end_method,
    }

    await _manager.broadcast(match_id, message)
    logger.info(f"Match {match_id} game {req.game_number} ended (winner: port {req.winner_port})")

    return {"success": True}


@app.post("/matches/{match_id}/stream/end")
async def stream_match_end(match_id: str, req: MatchEndRequest) -> dict[str, Any]:
    """Signal match end — called when the full set is complete."""
    message = {
        "type": "match_end",
        "winnerPort": req.winner_port,
        "finalScore": req.final_score,
    }

    await _manager.broadcast(match_id, message)
    logger.info(f"Match {match_id} ended (score: {req.final_score})")

    # Clean up after a short delay to let viewers receive the message
    async def cleanup():
        await asyncio.sleep(5.0)
        _manager.cleanup_match(match_id)

    asyncio.create_task(cleanup())

    return {"success": True}


@app.get("/matches/{match_id}/viewers")
def get_viewer_count(match_id: str) -> dict[str, Any]:
    """Get number of viewers for a match."""
    return {
        "match_id": match_id,
        "viewers": _manager.viewer_count(match_id),
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
