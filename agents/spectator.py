"""
agents/spectator.py - Spectator agent for prediction market betting.

Discovers live matches with prediction pools, connects to the arena's
WebSocket for frame data, evaluates match state, and places bets using
Kelly criterion adapted for parimutuel pools.

Key design rules:
    - Players must NOT bet on their own match (conflict of interest).
    - Spectators bet on pools for OTHER agents' matches.
    - Players wager directly via the Wager contract instead.

Usage:
    from agents.spectator import SpectatorAgent
    agent = SpectatorAgent(arena_url, account, rpc_url, pool_address)
    await agent.run()  # Autonomous loop
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Match evaluation from frame data
# ============================================================================


@dataclass
class MatchState:
    """Current state of a match derived from frame data."""

    frame: int = 0
    # Per-player state: keyed by port (1-4)
    stocks: dict[int, int] = field(default_factory=dict)
    percent: dict[int, float] = field(default_factory=dict)
    # Map port -> wallet address (from match_start)
    port_to_wallet: dict[int, str] = field(default_factory=dict)
    # Match metadata
    player_a_port: int | None = None
    player_b_port: int | None = None
    game_scores: dict[int, int] = field(default_factory=dict)  # port -> games won
    match_started: bool = False
    match_ended: bool = False

    def update_from_frame(self, frame_data: dict) -> None:
        """Update state from a WebSocket frame message."""
        self.frame = frame_data.get("frame", self.frame)
        for player in frame_data.get("players", []):
            port = player.get("port")
            if port is None:
                continue
            if "stocks" in player and player["stocks"] is not None:
                self.stocks[port] = player["stocks"]
            if "percent" in player and player["percent"] is not None:
                self.percent[port] = player["percent"]

    def update_from_match_start(self, data: dict, player_a_wallet: str, player_b_wallet: str) -> None:
        """Initialize port-to-wallet mapping from match_start message."""
        self.match_started = True
        for player in data.get("players", []):
            port = player.get("port")
            code = player.get("connectCode", "")
            if port is not None:
                self.port_to_wallet[port] = code  # We'll map to wallets below

        # player_a/b wallets come from the pool; we map them to ports
        # by checking the arena match data (p1/p2 mapping)
        # For now, store the wallets and let the caller set ports
        pass

    def stock_advantage(self, port_a: int, port_b: int) -> int:
        """Stocks of port_a minus port_b. Positive = A is ahead."""
        return self.stocks.get(port_a, 0) - self.stocks.get(port_b, 0)

    def percent_advantage(self, port_a: int, port_b: int) -> float:
        """Negative percent of A minus B (lower percent is better). Scaled to [-1, 1]."""
        pct_a = self.percent.get(port_a, 0.0)
        pct_b = self.percent.get(port_b, 0.0)
        # At 0% vs 150%, advantage is large. Cap at 300% diff.
        diff = pct_b - pct_a
        return max(-1.0, min(1.0, diff / 300.0))


def estimate_win_probability(state: MatchState, port_a: int, port_b: int) -> float:
    """Estimate probability that port_a wins, given current match state.

    Simple heuristic for MVP:
    - Stock lead is the dominant signal (each stock ~ 15% swing)
    - Percent advantage is secondary (capped contribution)
    - Game score in a set matters (winning 2-0 in a Bo5 is very different from 0-2)

    Returns probability in [0.05, 0.95] — never fully certain.
    """
    # Base: 50/50
    prob = 0.5

    # Stock advantage: each stock lead shifts ~15%
    stock_adv = state.stock_advantage(port_a, port_b)
    prob += stock_adv * 0.15

    # Percent advantage: smaller signal, ~5% max
    pct_adv = state.percent_advantage(port_a, port_b)
    prob += pct_adv * 0.05

    # Game score advantage (for sets): each game lead shifts ~10%
    games_a = state.game_scores.get(port_a, 0)
    games_b = state.game_scores.get(port_b, 0)
    prob += (games_a - games_b) * 0.10

    # Clamp to [0.05, 0.95]
    return max(0.05, min(0.95, prob))


# ============================================================================
# Kelly criterion for parimutuel pools
# ============================================================================


def kelly_parimutuel(
    estimated_prob: float,
    implied_prob: float,
    bankroll_wei: int,
    multiplier: float = 0.5,  # Half-Kelly by default (spectating is riskier)
    max_pct: float = 0.05,
) -> int:
    """Kelly criterion adapted for parimutuel betting.

    In parimutuel pools, the edge is: estimated_prob - implied_prob.
    Kelly fraction = edge (for parimutuel with proportional payout).

    Args:
        estimated_prob: Our estimate of this side winning.
        implied_prob: Market-implied probability (totalSide / totalPool).
        bankroll_wei: Available bankroll in wei.
        multiplier: Kelly fraction multiplier (0.5 = half-Kelly).
        max_pct: Maximum fraction of bankroll to bet.

    Returns:
        Bet amount in wei. 0 if no edge.
    """
    edge = estimated_prob - implied_prob
    if edge <= 0:
        return 0

    fraction = min(edge * multiplier, max_pct)
    amount = int(bankroll_wei * fraction)

    # Floor: 0.001 MON (gas costs exceed value below this)
    min_bet = int(0.001 * 10**18)
    if amount < min_bet:
        return 0

    return amount


# ============================================================================
# Pool discovery
# ============================================================================


def discover_pools(arena_url: str) -> list[dict[str, Any]]:
    """Query the arena for matches with active prediction pools."""
    import urllib.request

    url = f"{arena_url.rstrip('/')}/pools"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("pools", [])
    except Exception as e:
        logger.warning(f"Failed to discover pools: {e}")
        return []


def get_pool_details(
    pool_id: int, rpc_url: str, pool_address: str
) -> dict[str, Any] | None:
    """Read pool state from chain."""
    try:
        from nojohns.contract import get_pool_odds
        return get_pool_odds(pool_id, rpc_url, pool_address)
    except Exception as e:
        logger.warning(f"Failed to read pool {pool_id}: {e}")
        return None


def is_conflict_of_interest(pool: dict, our_address: str) -> bool:
    """Check if we're a player in this match. Players should wager, not bet on pools."""
    our_addr = our_address.lower()
    player_a = (pool.get("playerA") or "").lower()
    player_b = (pool.get("playerB") or "").lower()
    return our_addr == player_a or our_addr == player_b


# ============================================================================
# Spectator Agent
# ============================================================================


@dataclass
class BetRecord:
    """Track bets we've placed."""
    pool_id: int
    side: str  # "A" or "B"
    amount_wei: int
    tx_hash: str
    timestamp: float


class SpectatorAgent:
    """Autonomous spectator that watches matches and bets on prediction pools.

    Lifecycle:
        1. Poll arena for active pools
        2. For each pool, check conflict of interest
        3. Connect to match WebSocket for frame data
        4. Evaluate match state → estimate win probability
        5. Compare to market odds → bet if edge exists (Kelly)
        6. After match ends, claim payouts

    The agent reuses bankroll utilities from agents.bankroll for balance
    queries and Kelly sizing.
    """

    def __init__(
        self,
        arena_url: str,
        account: Any,  # eth_account.Account
        rpc_url: str,
        pool_address: str,
        multiplier: float = 0.5,  # Half-Kelly
        max_pct: float = 0.05,
        poll_interval: float = 10.0,
        min_frames_before_bet: int = 300,  # ~5 seconds of gameplay
    ):
        self.arena_url = arena_url.rstrip("/")
        self.account = account
        self.rpc_url = rpc_url
        self.pool_address = pool_address
        self.multiplier = multiplier
        self.max_pct = max_pct
        self.poll_interval = poll_interval
        self.min_frames_before_bet = min_frames_before_bet

        # State
        self.bets: list[BetRecord] = []
        self.watched_pools: set[int] = set()  # Pools we've already bet on
        self._running = False

    def _get_balance(self) -> int:
        """Get our available MON balance in wei."""
        try:
            from agents.bankroll import get_mon_balance
            return get_mon_balance(self.account.address, self.rpc_url)
        except Exception as e:
            logger.warning(f"Failed to get balance: {e}")
            return 0

    def bet_on_pool(self, pool_id: int, bet_on_a: bool, amount_wei: int) -> str | None:
        """Place a bet on a prediction pool. Returns tx hash or None."""
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
            self.watched_pools.add(pool_id)
            amount_mon = amount_wei / 10**18
            logger.info(f"Bet {amount_mon:.4f} MON on pool {pool_id} side {side}: {tx_hash}")
            return tx_hash
        except Exception as e:
            logger.warning(f"Failed to bet on pool {pool_id}: {e}")
            return None

    def claim_all(self) -> list[str]:
        """Claim payouts from all resolved pools we bet on. Returns tx hashes."""
        tx_hashes = []
        claimed_pools = set()

        for bet in self.bets:
            if bet.pool_id in claimed_pools:
                continue

            try:
                from nojohns.contract import get_claimable, claim_payout

                claimable = get_claimable(
                    bet.pool_id, self.account.address,
                    self.rpc_url, self.pool_address,
                )
                if claimable > 0:
                    tx = claim_payout(
                        bet.pool_id, self.account,
                        self.rpc_url, self.pool_address,
                    )
                    tx_hashes.append(tx)
                    claimed_pools.add(bet.pool_id)
                    mon = claimable / 10**18
                    logger.info(f"Claimed {mon:.4f} MON from pool {bet.pool_id}: {tx}")
            except Exception as e:
                logger.debug(f"Claim failed for pool {bet.pool_id}: {e}")

        return tx_hashes

    def evaluate_and_bet(self, pool_id: int, state: MatchState, port_a: int, port_b: int) -> bool:
        """Evaluate match state and bet if there's an edge. Returns True if bet placed."""
        if pool_id in self.watched_pools:
            return False  # Already bet on this pool

        if state.frame < self.min_frames_before_bet:
            return False  # Not enough data yet

        # Get pool odds from chain
        pool_details = get_pool_details(pool_id, self.rpc_url, self.pool_address)
        if pool_details is None:
            return False

        # Pool must be open (status 0)
        if pool_details.get("status", 0) != 0:
            return False

        # Conflict of interest check
        if is_conflict_of_interest(pool_details, self.account.address):
            logger.info(f"Skipping pool {pool_id}: we're a player in this match")
            return False

        # Estimate win probability from frame data
        prob_a = estimate_win_probability(state, port_a, port_b)
        implied_a = pool_details.get("impliedProbA", 0.5)
        implied_b = pool_details.get("impliedProbB", 0.5)

        bankroll = self._get_balance()
        if bankroll == 0:
            logger.warning("Zero balance — can't bet")
            return False

        # Check edge on both sides, bet on the one with better edge
        bet_a = kelly_parimutuel(prob_a, implied_a, bankroll, self.multiplier, self.max_pct)
        bet_b = kelly_parimutuel(1 - prob_a, implied_b, bankroll, self.multiplier, self.max_pct)

        if bet_a > 0 and bet_a >= bet_b:
            logger.info(
                f"Pool {pool_id}: P(A)={prob_a:.2f} vs market {implied_a:.2f}, "
                f"edge={prob_a - implied_a:.2f}, betting {bet_a / 10**18:.4f} MON on A"
            )
            self.bet_on_pool(pool_id, bet_on_a=True, amount_wei=bet_a)
            return True
        elif bet_b > 0:
            logger.info(
                f"Pool {pool_id}: P(B)={1 - prob_a:.2f} vs market {implied_b:.2f}, "
                f"edge={1 - prob_a - implied_b:.2f}, betting {bet_b / 10**18:.4f} MON on B"
            )
            self.bet_on_pool(pool_id, bet_on_a=False, amount_wei=bet_b)
            return True

        return False

    async def watch_match(self, match_id: str, pool_id: int, player_a: str, player_b: str) -> None:
        """Connect to a match WebSocket and evaluate for betting."""
        import websockets

        ws_url = f"{self.arena_url.replace('http', 'ws')}/ws/match/{match_id}"
        state = MatchState()

        # We need to figure out which port is playerA vs playerB
        # The arena match data tells us p1_wallet/p2_wallet
        port_a: int | None = None
        port_b: int | None = None

        try:
            async with websockets.connect(ws_url) as ws:
                logger.info(f"Connected to match {match_id} (pool {pool_id})")

                async for raw_msg in ws:
                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type")

                    if msg_type == "match_start":
                        state.match_started = True
                        # Match_start has player info; map ports
                        # We'll use the first two ports as A and B
                        players = msg.get("players", [])
                        if len(players) >= 2:
                            port_a = players[0].get("port")
                            port_b = players[1].get("port")

                    elif msg_type == "frame":
                        state.update_from_frame(msg)

                        # Try to bet periodically (every 300 frames ~ 5s)
                        if (
                            port_a is not None
                            and port_b is not None
                            and state.frame % 300 == 0
                            and pool_id not in self.watched_pools
                        ):
                            self.evaluate_and_bet(pool_id, state, port_a, port_b)

                    elif msg_type == "game_end":
                        winner_port = msg.get("winnerPort")
                        if winner_port is not None:
                            state.game_scores[winner_port] = state.game_scores.get(winner_port, 0) + 1

                    elif msg_type == "match_end":
                        state.match_ended = True
                        logger.info(f"Match {match_id} ended")
                        break

                    elif msg_type == "ping":
                        continue

        except Exception as e:
            logger.warning(f"WebSocket error for match {match_id}: {e}")

    async def run_once(self) -> int:
        """One pass: discover pools, watch matches, bet. Returns number of bets placed."""
        pools = discover_pools(self.arena_url)
        bets_placed = 0

        for pool_info in pools:
            pool_id = pool_info.get("pool_id")
            match_id = pool_info.get("id")
            status = pool_info.get("status")

            if pool_id is None or match_id is None:
                continue

            # Only watch playing matches
            if status != "playing":
                continue

            # Skip pools we already bet on
            if pool_id in self.watched_pools:
                continue

            # Quick conflict of interest check from arena data
            p1_wallet = pool_info.get("p1_wallet", "")
            p2_wallet = pool_info.get("p2_wallet", "")
            if self.account.address.lower() in (
                (p1_wallet or "").lower(),
                (p2_wallet or "").lower(),
            ):
                logger.debug(f"Skipping pool {pool_id}: we're playing")
                continue

            # Watch this match and try to bet
            try:
                await asyncio.wait_for(
                    self.watch_match(match_id, pool_id, p1_wallet or "", p2_wallet or ""),
                    timeout=600,  # 10 minute max per match
                )
            except asyncio.TimeoutError:
                logger.info(f"Timed out watching match {match_id}")

            if pool_id in self.watched_pools:
                bets_placed += 1

        # Try to claim payouts from any resolved pools
        self.claim_all()

        return bets_placed

    async def run(self) -> None:
        """Main loop: discover → watch → bet → claim. Runs until stopped."""
        self._running = True
        logger.info(
            f"Spectator agent started | arena={self.arena_url} | "
            f"wallet={self.account.address} | kelly={self.multiplier}x"
        )

        while self._running:
            try:
                bets = await self.run_once()
                if bets > 0:
                    logger.info(f"Placed {bets} bet(s) this cycle")
            except Exception as e:
                logger.error(f"Spectator cycle error: {e}")

            await asyncio.sleep(self.poll_interval)

    def stop(self):
        """Signal the agent to stop after the current cycle."""
        self._running = False
