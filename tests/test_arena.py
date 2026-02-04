"""
tests/test_arena.py - Arena server endpoint tests.

Uses FastAPI's TestClient — no server process needed.
"""

import time

import pytest
from fastapi.testclient import TestClient

from arena.db import ArenaDB
from arena.server import app, get_db


@pytest.fixture
def db():
    """Fresh in-memory DB for each test."""
    return ArenaDB(":memory:")


@pytest.fixture
def client(db):
    """FastAPI test client backed by an in-memory DB."""
    import arena.server as srv

    # Create a bare app without lifespan so it doesn't overwrite _db
    from fastapi import FastAPI

    test_app = FastAPI()
    # Copy all routes from the real app
    for route in app.routes:
        test_app.routes.append(route)

    srv._db = db
    with TestClient(test_app) as c:
        yield c
    srv._db = None


# ======================================================================
# Queue Tests
# ======================================================================


class TestJoinQueue:
    def test_join_returns_queue_id(self, client):
        resp = client.post("/queue/join", json={"connect_code": "SCAV#382"})
        assert resp.status_code == 200
        data = resp.json()
        assert "queue_id" in data
        assert data["status"] in ("waiting", "matched")

    def test_join_single_is_waiting(self, client):
        resp = client.post("/queue/join", json={"connect_code": "SCAV#382"})
        data = resp.json()
        assert data["status"] == "waiting"
        assert data["position"] == 1

    def test_join_two_auto_matches(self, client):
        r1 = client.post(
            "/queue/join",
            json={"connect_code": "SCAV#382", "fighter_name": "random"},
        )
        r2 = client.post(
            "/queue/join",
            json={"connect_code": "SCAV#861", "fighter_name": "random"},
        )
        d1 = r1.json()
        d2 = r2.json()

        # First joiner was waiting
        assert d1["status"] == "waiting"

        # Second joiner triggers the match
        assert d2["status"] == "matched"
        assert d2["opponent_code"] == "SCAV#382"
        assert d2["match_id"] is not None

    def test_join_with_fighter_name(self, client):
        resp = client.post(
            "/queue/join",
            json={"connect_code": "SCAV#382", "fighter_name": "smashbot"},
        )
        assert resp.status_code == 200


class TestPollQueue:
    def test_poll_waiting(self, client):
        join = client.post("/queue/join", json={"connect_code": "SCAV#382"})
        qid = join.json()["queue_id"]

        resp = client.get(f"/queue/{qid}")
        data = resp.json()
        assert data["status"] == "waiting"
        assert data["position"] == 1

    def test_poll_after_match(self, client):
        r1 = client.post("/queue/join", json={"connect_code": "SCAV#382"})
        r2 = client.post("/queue/join", json={"connect_code": "SCAV#861"})

        # Poll the first player — should now be matched
        qid1 = r1.json()["queue_id"]
        resp = client.get(f"/queue/{qid1}")
        data = resp.json()
        assert data["status"] == "matched"
        assert data["opponent_code"] == "SCAV#861"
        assert data["match_id"] is not None

    def test_poll_nonexistent(self, client):
        resp = client.get("/queue/not-a-real-id")
        assert resp.status_code == 404

    def test_poll_cancelled(self, client):
        join = client.post("/queue/join", json={"connect_code": "SCAV#382"})
        qid = join.json()["queue_id"]
        client.delete(f"/queue/{qid}")

        resp = client.get(f"/queue/{qid}")
        data = resp.json()
        assert data["status"] == "cancelled"


class TestLeaveQueue:
    def test_cancel_waiting(self, client):
        join = client.post("/queue/join", json={"connect_code": "SCAV#382"})
        qid = join.json()["queue_id"]

        resp = client.delete(f"/queue/{qid}")
        assert resp.json()["success"] is True

    def test_cancel_nonexistent(self, client):
        resp = client.delete("/queue/not-real")
        assert resp.json()["success"] is False

    def test_cancel_already_matched(self, client):
        """Can't cancel after matching."""
        client.post("/queue/join", json={"connect_code": "SCAV#382"})
        r2 = client.post("/queue/join", json={"connect_code": "SCAV#861"})
        qid2 = r2.json()["queue_id"]

        resp = client.delete(f"/queue/{qid2}")
        assert resp.json()["success"] is False


# ======================================================================
# Match Tests
# ======================================================================


class TestReportResult:
    def _make_match(self, client):
        """Helper: join two players, return (qid1, qid2, match_id)."""
        r1 = client.post("/queue/join", json={"connect_code": "SCAV#382"})
        r2 = client.post("/queue/join", json={"connect_code": "SCAV#861"})
        d2 = r2.json()
        match_id = d2["match_id"]
        return r1.json()["queue_id"], d2["queue_id"], match_id

    def test_single_report(self, client):
        qid1, qid2, match_id = self._make_match(client)

        resp = client.post(
            f"/matches/{match_id}/result",
            json={"queue_id": qid1, "outcome": "COMPLETED", "duration_seconds": 120.5},
        )
        assert resp.json()["success"] is True

        # Match still playing (only one side reported)
        match = client.get(f"/matches/{match_id}").json()
        assert match["status"] == "playing"

    def test_both_report_completes(self, client):
        qid1, qid2, match_id = self._make_match(client)

        client.post(
            f"/matches/{match_id}/result",
            json={"queue_id": qid1, "outcome": "COMPLETED", "duration_seconds": 120.5},
        )
        client.post(
            f"/matches/{match_id}/result",
            json={"queue_id": qid2, "outcome": "COMPLETED", "duration_seconds": 119.8},
        )

        match = client.get(f"/matches/{match_id}").json()
        assert match["status"] == "completed"
        assert match["p1_result"] == "COMPLETED"
        assert match["p2_result"] == "COMPLETED"
        assert match["completed_at"] is not None

    def test_disconnect_report(self, client):
        qid1, qid2, match_id = self._make_match(client)

        client.post(
            f"/matches/{match_id}/result",
            json={"queue_id": qid1, "outcome": "DISCONNECT"},
        )
        client.post(
            f"/matches/{match_id}/result",
            json={"queue_id": qid2, "outcome": "COMPLETED", "duration_seconds": 45.0},
        )

        match = client.get(f"/matches/{match_id}").json()
        assert match["status"] == "completed"
        assert match["p1_result"] == "DISCONNECT"
        assert match["p2_result"] == "COMPLETED"

    def test_report_nonexistent_match(self, client):
        resp = client.post(
            "/matches/not-real/result",
            json={"queue_id": "q1", "outcome": "COMPLETED"},
        )
        assert resp.status_code == 404


class TestGetMatch:
    def test_get_match(self, client):
        client.post("/queue/join", json={"connect_code": "SCAV#382"})
        r2 = client.post("/queue/join", json={"connect_code": "SCAV#861"})
        match_id = r2.json()["match_id"]

        resp = client.get(f"/matches/{match_id}")
        data = resp.json()
        assert data["id"] == match_id
        assert data["p1_connect_code"] == "SCAV#382"
        assert data["p2_connect_code"] == "SCAV#861"
        assert data["status"] == "playing"

    def test_get_nonexistent(self, client):
        resp = client.get("/matches/not-real")
        assert resp.status_code == 404


# ======================================================================
# Stale Expiry
# ======================================================================


class TestStaleExpiry:
    def test_expire_old_entries(self, db):
        """Entries older than timeout get expired."""
        qid = db.add_to_queue("SCAV#382")

        # Manually backdate the updated_at
        db._conn.execute(
            "UPDATE queue SET updated_at = '2020-01-01T00:00:00+00:00' WHERE id = ?",
            (qid,),
        )
        db._conn.commit()

        expired = db.expire_stale_entries(timeout_seconds=300)
        assert expired == 1

        entry = db.get_queue_entry(qid)
        assert entry["status"] == "expired"

    def test_fresh_entries_survive(self, db):
        """Recently polled entries don't expire."""
        db.add_to_queue("SCAV#382")
        expired = db.expire_stale_entries(timeout_seconds=300)
        assert expired == 0


# ======================================================================
# Health
# ======================================================================


class TestHealth:
    def test_health_empty(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["queue_size"] == 0
        assert data["active_matches"] == 0

    def test_health_with_queue(self, client):
        client.post("/queue/join", json={"connect_code": "SCAV#382"})
        resp = client.get("/health")
        assert resp.json()["queue_size"] == 1

    def test_health_with_match(self, client):
        client.post("/queue/join", json={"connect_code": "SCAV#382"})
        client.post("/queue/join", json={"connect_code": "SCAV#861"})
        resp = client.get("/health")
        data = resp.json()
        assert data["queue_size"] == 0
        assert data["active_matches"] == 1


# ======================================================================
# FIFO Matching
# ======================================================================


class TestFIFOMatching:
    def test_third_joiner_waits(self, client):
        """When two are matched, a third should wait."""
        client.post("/queue/join", json={"connect_code": "A#001"})
        client.post("/queue/join", json={"connect_code": "B#002"})
        r3 = client.post("/queue/join", json={"connect_code": "C#003"})
        assert r3.json()["status"] == "waiting"

    def test_four_joiners_two_matches(self, client):
        """Four joiners should produce two separate matches."""
        r1 = client.post("/queue/join", json={"connect_code": "A#001"})
        r2 = client.post("/queue/join", json={"connect_code": "B#002"})
        r3 = client.post("/queue/join", json={"connect_code": "C#003"})
        r4 = client.post("/queue/join", json={"connect_code": "D#004"})

        # r2 matched with r1, r4 matched with r3
        assert r2.json()["status"] == "matched"
        assert r4.json()["status"] == "matched"
        assert r2.json()["match_id"] != r4.json()["match_id"]


# ======================================================================
# Wallet Address Tests
# ======================================================================


class TestWalletAddress:
    def test_join_with_wallet(self, client):
        """Wallet address is accepted on join."""
        resp = client.post("/queue/join", json={
            "connect_code": "SCAV#382",
            "wallet_address": "0xABC123",
        })
        assert resp.status_code == 200

    def test_wallet_returned_on_match(self, client):
        """Opponent wallet is returned when matched."""
        client.post("/queue/join", json={
            "connect_code": "SCAV#382",
            "wallet_address": "0xWALLET_A",
        })
        r2 = client.post("/queue/join", json={
            "connect_code": "SCAV#861",
            "wallet_address": "0xWALLET_B",
        })
        d2 = r2.json()
        assert d2["status"] == "matched"
        assert d2["opponent_wallet"] == "0xWALLET_A"

    def test_wallet_returned_on_poll(self, client):
        """Opponent wallet is returned when polling after match."""
        r1 = client.post("/queue/join", json={
            "connect_code": "SCAV#382",
            "wallet_address": "0xWALLET_A",
        })
        client.post("/queue/join", json={
            "connect_code": "SCAV#861",
            "wallet_address": "0xWALLET_B",
        })

        qid1 = r1.json()["queue_id"]
        resp = client.get(f"/queue/{qid1}")
        data = resp.json()
        assert data["status"] == "matched"
        assert data["opponent_wallet"] == "0xWALLET_B"

    def test_wallet_null_when_not_provided(self, client):
        """Opponent wallet is null if they didn't provide one."""
        client.post("/queue/join", json={"connect_code": "SCAV#382"})
        r2 = client.post("/queue/join", json={
            "connect_code": "SCAV#861",
            "wallet_address": "0xWALLET_B",
        })
        d2 = r2.json()
        assert d2["opponent_wallet"] is None

    def test_wallet_in_match_details(self, client):
        """Match details include wallet addresses."""
        client.post("/queue/join", json={
            "connect_code": "SCAV#382",
            "wallet_address": "0xWALLET_A",
        })
        r2 = client.post("/queue/join", json={
            "connect_code": "SCAV#861",
            "wallet_address": "0xWALLET_B",
        })
        match_id = r2.json()["match_id"]

        match = client.get(f"/matches/{match_id}").json()
        assert match["p1_wallet"] == "0xWALLET_A"
        assert match["p2_wallet"] == "0xWALLET_B"


# ======================================================================
# Signature Tests
# ======================================================================


class TestSignatures:
    def _make_match(self, client):
        """Helper: join two players with wallets, return (qid1, qid2, match_id)."""
        r1 = client.post("/queue/join", json={
            "connect_code": "SCAV#382",
            "wallet_address": "0xADDR_A",
        })
        r2 = client.post("/queue/join", json={
            "connect_code": "SCAV#861",
            "wallet_address": "0xADDR_B",
        })
        d2 = r2.json()
        return r1.json()["queue_id"], d2["queue_id"], d2["match_id"]

    def test_submit_signature(self, client):
        _, _, match_id = self._make_match(client)
        resp = client.post(f"/matches/{match_id}/signature", json={
            "address": "0xADDR_A",
            "signature": "0xdeadbeef",
        })
        data = resp.json()
        assert data["signatures_received"] == 1
        assert data["ready_for_submission"] is False

    def test_two_signatures_ready(self, client):
        _, _, match_id = self._make_match(client)
        client.post(f"/matches/{match_id}/signature", json={
            "address": "0xADDR_A",
            "signature": "0xsig_a",
        })
        resp = client.post(f"/matches/{match_id}/signature", json={
            "address": "0xADDR_B",
            "signature": "0xsig_b",
        })
        data = resp.json()
        assert data["signatures_received"] == 2
        assert data["ready_for_submission"] is True

    def test_get_signatures(self, client):
        _, _, match_id = self._make_match(client)
        client.post(f"/matches/{match_id}/signature", json={
            "address": "0xADDR_A",
            "signature": "0xsig_a",
        })
        client.post(f"/matches/{match_id}/signature", json={
            "address": "0xADDR_B",
            "signature": "0xsig_b",
        })

        resp = client.get(f"/matches/{match_id}/signatures")
        data = resp.json()
        assert data["signatures_received"] == 2
        assert data["ready_for_submission"] is True
        assert len(data["signatures"]) == 2

    def test_get_signatures_nonexistent_match(self, client):
        resp = client.get("/matches/not-real/signatures")
        assert resp.status_code == 404

    def test_signature_upsert(self, client):
        """Resubmitting from same address updates, doesn't duplicate."""
        _, _, match_id = self._make_match(client)
        client.post(f"/matches/{match_id}/signature", json={
            "address": "0xADDR_A",
            "signature": "0xsig_v1",
        })
        resp = client.post(f"/matches/{match_id}/signature", json={
            "address": "0xADDR_A",
            "signature": "0xsig_v2",
        })
        assert resp.json()["signatures_received"] == 1

        sigs = client.get(f"/matches/{match_id}/signatures").json()["signatures"]
        assert len(sigs) == 1
        assert sigs[0]["signature"] == "0xsig_v2"
