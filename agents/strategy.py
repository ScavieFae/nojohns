"""
agents/strategy.py - WagerStrategy protocol and Kelly criterion reference implementation.

The WagerStrategy protocol is like the Fighter protocol — implement your own
or use KellyStrategy out of the box.
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

from agents.bankroll import kelly_wager, win_probability_from_elo
from agents.scouting import ScoutReport

logger = logging.getLogger(__name__)


# ============================================================================
# Data types
# ============================================================================


@dataclass
class SessionStats:
    """Running stats for an auto session. Agents track these however they want."""

    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    total_wagered_wei: int = 0
    net_profit_wei: int = 0
    opponents_faced: set = field(default_factory=set)

    def record_win(self, wager_wei: int = 0):
        self.matches_played += 1
        self.wins += 1
        self.consecutive_losses = 0
        self.total_wagered_wei += wager_wei
        self.net_profit_wei += wager_wei  # won the pot

    def record_loss(self, wager_wei: int = 0):
        self.matches_played += 1
        self.losses += 1
        self.consecutive_losses += 1
        self.total_wagered_wei += wager_wei
        self.net_profit_wei -= wager_wei  # lost our stake

    @property
    def win_rate(self) -> float:
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played


@dataclass
class MatchContext:
    """Everything an agent knows before deciding on a wager."""

    our_elo: int
    opponent: ScoutReport
    bankroll_wei: int  # Available (after exposure)
    session_stats: SessionStats | None = None


@dataclass
class WagerDecision:
    """Output of a strategy's decide() call."""

    should_wager: bool
    amount_wei: int = 0
    amount_mon: float = 0.0
    reasoning: str = ""


# ============================================================================
# Protocol
# ============================================================================


class WagerStrategy(Protocol):
    """Interface for wager decision-making.

    Implement your own or use KellyStrategy. The only method that matters
    is decide() — given match context, return a wager decision.
    """

    def decide(self, context: MatchContext) -> WagerDecision: ...


# ============================================================================
# Risk profiles
# ============================================================================

RISK_PROFILES = {
    "conservative": {"multiplier": 0.5, "max_pct": 0.05},
    "moderate": {"multiplier": 1.0, "max_pct": 0.10},
    "aggressive": {"multiplier": 1.5, "max_pct": 0.25},
}


# ============================================================================
# Reference implementation: Kelly criterion
# ============================================================================


class KellyStrategy:
    """Kelly criterion wager sizing. Reference implementation.

    Agents can use this directly, subclass it, or write their own
    WagerStrategy from scratch.
    """

    def __init__(
        self,
        risk_profile: str = "moderate",
        tilt_threshold: int = 3,
    ):
        if risk_profile not in RISK_PROFILES:
            raise ValueError(
                f"Unknown risk profile: {risk_profile}. "
                f"Choose from: {', '.join(RISK_PROFILES)}"
            )
        self.risk_profile = risk_profile
        self.tilt_threshold = tilt_threshold
        profile = RISK_PROFILES[risk_profile]
        self.multiplier = profile["multiplier"]
        self.max_pct = profile["max_pct"]

    def decide(self, context: MatchContext) -> WagerDecision:
        # 1. Unknown opponent → no wager (gather data first)
        if context.opponent.is_unknown:
            return WagerDecision(
                should_wager=False,
                reasoning="Unknown opponent — playing for experience.",
            )

        # 2. Tilt protection
        if context.session_stats and context.session_stats.consecutive_losses >= self.tilt_threshold:
            return WagerDecision(
                should_wager=False,
                reasoning=f"Tilt protection: {context.session_stats.consecutive_losses} consecutive losses (threshold: {self.tilt_threshold}).",
            )

        # 3. Win probability from Elo
        win_prob = win_probability_from_elo(context.our_elo, context.opponent.elo)

        # 4. No edge → no wager
        if win_prob <= 0.5:
            return WagerDecision(
                should_wager=False,
                reasoning=f"No edge: P(win)={win_prob:.2f} vs Elo {context.opponent.elo}.",
            )

        # 5. Kelly sizing
        amount = kelly_wager(
            win_prob=win_prob,
            bankroll_wei=context.bankroll_wei,
            multiplier=self.multiplier,
            max_pct=self.max_pct,
        )

        if amount == 0:
            return WagerDecision(
                should_wager=False,
                reasoning=f"Wager too small after Kelly sizing (P(win)={win_prob:.2f}, {self.risk_profile}).",
            )

        amount_mon = amount / 10**18

        reasoning = (
            f"Elo {context.our_elo} vs {context.opponent.elo}, "
            f"P(win)={win_prob:.2f}, "
            f"{self.risk_profile} profile, "
            f"wagering {amount_mon:.4f} MON"
        )

        return WagerDecision(
            should_wager=True,
            amount_wei=amount,
            amount_mon=amount_mon,
            reasoning=reasoning,
        )

    def __repr__(self) -> str:
        return f"KellyStrategy(risk={self.risk_profile}, tilt={self.tilt_threshold})"
