"""
ContinuousSpectatorAgent — subclass of SpectatorAgent that places multiple
bets throughout a match instead of betting once and going silent.

Overrides the three gates in SpectatorAgent that enforce bet-once behavior:
    1. evaluate_and_bet() — removes watched_pools check, adds cooldown + max bets
    2. watch_match() — removes the watched_pools gate in the frame loop
    3. bet_on_pool() — doesn't mark pool as "done" after betting

Result: agents re-evaluate every ~5 seconds and place new bets when edge shifts,
creating visible activity throughout the match. Bet sizes are smaller (spread
across max_bets_per_pool) so total exposure stays comparable.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass

from agents.spectator import (
    SpectatorAgent,
    MatchState,
    estimate_win_probability,
    kelly_parimutuel as _kelly_parimutuel,
    get_pool_details,
    is_conflict_of_interest,
    discover_pools,
    fetch_match_data,
    BetRecord,
)


def kelly_parimutuel(
    estimated_prob: float,
    implied_prob: float,
    bankroll_wei: int,
    multiplier: float = 0.5,
    max_pct: float = 0.05,
) -> int:
    """Kelly with a lower min_bet floor for Monad (gas is cheap)."""
    edge = estimated_prob - implied_prob
    if edge <= 0:
        return 0
    fraction = min(edge * multiplier, max_pct)
    amount = int(bankroll_wei * fraction)
    # Monad gas is ~0.00001 MON, so 0.0001 MON bets are fine
    min_bet = int(0.0001 * 10**18)
    if amount < min_bet:
        return 0
    return amount

logger = logging.getLogger(__name__)


class ContinuousSpectatorAgent(SpectatorAgent):
    """SpectatorAgent that re-evaluates and bets throughout the match.

    Args (beyond SpectatorAgent defaults):
        max_bets_per_pool: Cap bets per pool (default 8).
        bet_cooldown: Minimum seconds between bets on same pool (default 15).
    """

    def __init__(
        self,
        *args,
        max_bets_per_pool: int = 20,
        bet_cooldown: float = 12.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_bets_per_pool = max_bets_per_pool
        self.bet_cooldown = bet_cooldown

        # Per-pool tracking
        self._pool_bet_counts: dict[int, int] = {}
        self._pool_last_bet_time: dict[int, float] = {}

        # Each agent evaluates on a slightly different frame cadence (240-360)
        self._eval_interval = random.randint(240, 360)

        # Personality bias: each agent leans slightly toward A or B (±0.08).
        # This prevents all agents from converging to edge=0 at the same
        # market state — they disagree, so some keep betting even after
        # others stop. Creates natural two-sided flow.
        self._prob_bias = random.uniform(-0.08, 0.08)

        # Spread max_pct across bets so total exposure stays comparable,
        # but cap the divisor so small bankrolls still clear the 0.001 MON floor.
        divisor = min(self.max_bets_per_pool / 3, 3.0)
        self.max_pct = self.max_pct / divisor

    def bet_on_pool(self, pool_id: int, bet_on_a: bool, amount_wei: int) -> str | None:
        """Place a bet WITHOUT marking the pool as done."""
        try:
            from nojohns.contract import place_bet

            tx_hash = place_bet(
                pool_id, bet_on_a, amount_wei,
                self.account, self.rpc_url, self.pool_address,
            )
            side = "A" if bet_on_a else "B"
            record = BetRecord(
                pool_id=pool_id,
                side=side,
                amount_wei=amount_wei,
                tx_hash=tx_hash,
                timestamp=time.time(),
            )
            self.bets.append(record)
            # NOTE: deliberately NOT adding to self.watched_pools here
            # so the agent keeps evaluating this pool
            amount_mon = amount_wei / 10**18
            count = self._pool_bet_counts.get(pool_id, 0) + 1
            logger.info(
                f"Bet #{count} on pool {pool_id} side {side}: "
                f"{amount_mon:.4f} MON (tx: {tx_hash})"
            )
            return tx_hash
        except Exception as e:
            logger.warning(f"Failed to bet on pool {pool_id}: {e}")
            return None

    def evaluate_and_bet(self, pool_id: int, state: MatchState, port_a: int, port_b: int) -> bool:
        """Evaluate and bet with cooldown — allows multiple bets per pool."""
        # Max bets check
        count = self._pool_bet_counts.get(pool_id, 0)
        if count >= self.max_bets_per_pool:
            return False

        # Cooldown check (jittered ±5s so agents don't bet in lockstep)
        last_bet = self._pool_last_bet_time.get(pool_id, 0)
        jittered_cooldown = self.bet_cooldown + random.uniform(-5, 5)
        if time.time() - last_bet < jittered_cooldown:
            return False

        # Frame minimum
        if state.frame < self.min_frames_before_bet:
            return False

        # Pool odds from chain (synchronous — blocks briefly, acceptable for 5 agents)
        pool_details = get_pool_details(pool_id, self.rpc_url, self.pool_address)
        if pool_details is None:
            return False
        if pool_details.get("status", 0) != 0:
            return False
        if is_conflict_of_interest(pool_details, self.account.address):
            return False

        # Win probability from frame data + this agent's personality bias
        prob_a = estimate_win_probability(state, port_a, port_b)
        prob_a = max(0.05, min(0.95, prob_a + self._prob_bias))
        implied_a = pool_details.get("impliedProbA", 0.5)
        implied_b = pool_details.get("impliedProbB", 0.5)

        bankroll = self._get_balance()
        if bankroll == 0:
            logger.warning("Zero balance — can't bet")
            return False

        # Kelly sizing
        bet_a = kelly_parimutuel(prob_a, implied_a, bankroll, self.multiplier, self.max_pct)
        bet_b = kelly_parimutuel(1 - prob_a, implied_b, bankroll, self.multiplier, self.max_pct)

        if bet_a > 0 and bet_a >= bet_b:
            edge = prob_a - implied_a
            logger.info(
                f"Pool {pool_id} bet #{count + 1}: "
                f"P(A)={prob_a:.2f} vs market {implied_a:.2f} edge={edge:+.2f}"
            )
            self.bet_on_pool(pool_id, bet_on_a=True, amount_wei=bet_a)
            self._pool_bet_counts[pool_id] = count + 1
            self._pool_last_bet_time[pool_id] = time.time()
            return True
        elif bet_b > 0:
            edge = (1 - prob_a) - implied_b
            logger.info(
                f"Pool {pool_id} bet #{count + 1}: "
                f"P(B)={1 - prob_a:.2f} vs market {implied_b:.2f} edge={edge:+.2f}"
            )
            self.bet_on_pool(pool_id, bet_on_a=False, amount_wei=bet_b)
            self._pool_bet_counts[pool_id] = count + 1
            self._pool_last_bet_time[pool_id] = time.time()
            return True

        return False

    async def watch_match(self, match_id: str, pool_id: int, player_a: str, player_b: str) -> None:
        """Watch match with continuous betting — no watched_pools gate in frame loop."""
        import websockets

        ws_url = f"{self.arena_url.replace('http', 'ws')}/ws/match/{match_id}"
        state = MatchState()
        port_a: int | None = None
        port_b: int | None = None
        # Bail if no meaningful data arrives within this window
        STALE_TIMEOUT = 30.0

        match_data = await asyncio.to_thread(fetch_match_data, self.arena_url, match_id)
        p1_code = (match_data or {}).get("p1_connect_code", "")
        p2_code = (match_data or {}).get("p2_connect_code", "")

        try:
            async with websockets.connect(ws_url, open_timeout=10) as ws:
                logger.info(f"Connected to match {match_id} (pool {pool_id})")
                last_data_time = time.time()

                while True:
                    try:
                        raw_msg = await asyncio.wait_for(ws.recv(), timeout=STALE_TIMEOUT)
                    except asyncio.TimeoutError:
                        logger.info(
                            f"No data from match {match_id} for {STALE_TIMEOUT}s — "
                            "assuming stale, disconnecting"
                        )
                        break

                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type")

                    if msg_type == "match_start":
                        last_data_time = time.time()
                        state.match_started = True
                        for player in msg.get("players", []):
                            code = (player.get("connectCode") or "").upper()
                            port = player.get("port")
                            if port is None:
                                continue
                            if code and code == p1_code.upper():
                                port_a = port
                            elif code and code == p2_code.upper():
                                port_b = port

                        if port_a is None or port_b is None:
                            players = msg.get("players", [])
                            if len(players) >= 2:
                                logger.warning(
                                    f"Could not map ports via connect codes for {match_id}, "
                                    "falling back to position order"
                                )
                                port_a = players[0].get("port")
                                port_b = players[1].get("port")

                    elif msg_type == "frame":
                        last_data_time = time.time()
                        state.update_from_frame(msg)

                        # Evaluate on this agent's cadence — NO watched_pools gate
                        if (
                            port_a is not None
                            and port_b is not None
                            and state.frame % self._eval_interval == 0
                        ):
                            self.evaluate_and_bet(pool_id, state, port_a, port_b)

                    elif msg_type == "game_end":
                        last_data_time = time.time()
                        winner_port = msg.get("winnerPort")
                        if winner_port is not None:
                            state.game_scores[winner_port] = (
                                state.game_scores.get(winner_port, 0) + 1
                            )

                    elif msg_type == "match_end":
                        state.match_ended = True
                        logger.info(f"Match {match_id} ended")
                        break

                    elif msg_type == "ping":
                        continue

        except Exception as e:
            logger.warning(f"WebSocket error for match {match_id}: {e}")

        # NOW mark pool as done — match is over
        self.watched_pools.add(pool_id)
