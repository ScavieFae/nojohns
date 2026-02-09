"""Tests for agents/strategy.py — KellyStrategy logic, no RPC calls."""

import pytest

from agents.scouting import ScoutReport
from agents.strategy import (
    KellyStrategy,
    MatchContext,
    SessionStats,
    WagerDecision,
    RISK_PROFILES,
)


def _make_context(
    our_elo=1500,
    opp_elo=1500,
    opp_unknown=False,
    bankroll_mon=10.0,
    session=None,
) -> MatchContext:
    """Helper to build a MatchContext for testing."""
    return MatchContext(
        our_elo=our_elo,
        opponent=ScoutReport(
            elo=opp_elo,
            peak_elo=opp_elo,
            record="5-3" if not opp_unknown else "0-0",
            is_unknown=opp_unknown,
            agent_id=42 if not opp_unknown else None,
        ),
        bankroll_wei=int(bankroll_mon * 10**18),
        session_stats=session,
    )


# ============================================================================
# Unknown opponent
# ============================================================================


class TestUnknownOpponent:
    def test_no_wager_against_unknown(self):
        strategy = KellyStrategy(risk_profile="moderate")
        ctx = _make_context(opp_unknown=True)
        decision = strategy.decide(ctx)
        assert not decision.should_wager
        assert "Unknown" in decision.reasoning or "unknown" in decision.reasoning.lower()

    def test_no_wager_against_unknown_any_profile(self):
        for profile in RISK_PROFILES:
            strategy = KellyStrategy(risk_profile=profile)
            ctx = _make_context(opp_unknown=True)
            decision = strategy.decide(ctx)
            assert not decision.should_wager


# ============================================================================
# Tilt protection
# ============================================================================


class TestTiltProtection:
    def test_tilt_after_threshold(self):
        strategy = KellyStrategy(risk_profile="moderate", tilt_threshold=3)
        session = SessionStats(consecutive_losses=3)
        ctx = _make_context(our_elo=1600, opp_elo=1400, session=session)
        decision = strategy.decide(ctx)
        assert not decision.should_wager
        assert "Tilt" in decision.reasoning or "tilt" in decision.reasoning.lower()

    def test_no_tilt_below_threshold(self):
        strategy = KellyStrategy(risk_profile="moderate", tilt_threshold=3)
        session = SessionStats(consecutive_losses=2)
        ctx = _make_context(our_elo=1600, opp_elo=1400, session=session)
        decision = strategy.decide(ctx)
        assert decision.should_wager  # Has edge, under threshold

    def test_no_tilt_without_session(self):
        strategy = KellyStrategy(risk_profile="moderate", tilt_threshold=3)
        ctx = _make_context(our_elo=1600, opp_elo=1400, session=None)
        decision = strategy.decide(ctx)
        assert decision.should_wager  # No session = no tilt check


# ============================================================================
# Edge / no edge
# ============================================================================


class TestEdge:
    def test_even_matchup_no_wager(self):
        strategy = KellyStrategy(risk_profile="moderate")
        ctx = _make_context(our_elo=1500, opp_elo=1500)
        decision = strategy.decide(ctx)
        assert not decision.should_wager

    def test_unfavorable_no_wager(self):
        strategy = KellyStrategy(risk_profile="moderate")
        ctx = _make_context(our_elo=1400, opp_elo=1600)
        decision = strategy.decide(ctx)
        assert not decision.should_wager

    def test_favorable_wagers(self):
        strategy = KellyStrategy(risk_profile="moderate")
        ctx = _make_context(our_elo=1600, opp_elo=1400)
        decision = strategy.decide(ctx)
        assert decision.should_wager
        assert decision.amount_wei > 0
        assert decision.amount_mon > 0

    def test_slightly_favorable(self):
        """Small edge should still produce a wager if bankroll allows."""
        strategy = KellyStrategy(risk_profile="moderate")
        ctx = _make_context(our_elo=1520, opp_elo=1500, bankroll_mon=10.0)
        decision = strategy.decide(ctx)
        # Small edge, but with 10 MON bankroll should be above the 0.001 floor
        # (P(win) ≈ 0.529, Kelly ≈ 0.058, 5.8% of 10 = 0.58 MON)
        assert decision.should_wager


# ============================================================================
# Risk profiles
# ============================================================================


class TestRiskProfiles:
    def test_aggressive_wagers_more(self):
        ctx = _make_context(our_elo=1600, opp_elo=1400)
        conservative = KellyStrategy(risk_profile="conservative").decide(ctx)
        aggressive = KellyStrategy(risk_profile="aggressive").decide(ctx)
        assert aggressive.amount_wei > conservative.amount_wei

    def test_conservative_caps_at_5_pct(self):
        ctx = _make_context(our_elo=1800, opp_elo=1400, bankroll_mon=10.0)
        decision = KellyStrategy(risk_profile="conservative").decide(ctx)
        max_allowed = int(10.0 * 0.05 * 10**18)
        assert decision.amount_wei <= max_allowed

    def test_moderate_caps_at_10_pct(self):
        ctx = _make_context(our_elo=1800, opp_elo=1400, bankroll_mon=10.0)
        decision = KellyStrategy(risk_profile="moderate").decide(ctx)
        max_allowed = int(10.0 * 0.10 * 10**18)
        assert decision.amount_wei <= max_allowed

    def test_aggressive_caps_at_25_pct(self):
        ctx = _make_context(our_elo=1800, opp_elo=1400, bankroll_mon=10.0)
        decision = KellyStrategy(risk_profile="aggressive").decide(ctx)
        max_allowed = int(10.0 * 0.25 * 10**18)
        assert decision.amount_wei <= max_allowed

    def test_invalid_risk_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown risk profile"):
            KellyStrategy(risk_profile="yolo")


# ============================================================================
# Reasoning string
# ============================================================================


class TestReasoning:
    def test_reasoning_includes_elo(self):
        strategy = KellyStrategy(risk_profile="moderate")
        ctx = _make_context(our_elo=1600, opp_elo=1400)
        decision = strategy.decide(ctx)
        assert "1600" in decision.reasoning
        assert "1400" in decision.reasoning

    def test_reasoning_includes_probability(self):
        strategy = KellyStrategy(risk_profile="moderate")
        ctx = _make_context(our_elo=1600, opp_elo=1400)
        decision = strategy.decide(ctx)
        assert "P(win)" in decision.reasoning


# ============================================================================
# SessionStats
# ============================================================================


class TestSessionStats:
    def test_record_win(self):
        s = SessionStats()
        s.record_win(wager_wei=10**18)
        assert s.matches_played == 1
        assert s.wins == 1
        assert s.losses == 0
        assert s.consecutive_losses == 0
        assert s.net_profit_wei == 10**18

    def test_record_loss(self):
        s = SessionStats()
        s.record_loss(wager_wei=10**18)
        assert s.matches_played == 1
        assert s.wins == 0
        assert s.losses == 1
        assert s.consecutive_losses == 1
        assert s.net_profit_wei == -(10**18)

    def test_consecutive_losses_reset_on_win(self):
        s = SessionStats()
        s.record_loss()
        s.record_loss()
        s.record_loss()
        assert s.consecutive_losses == 3
        s.record_win()
        assert s.consecutive_losses == 0

    def test_win_rate(self):
        s = SessionStats()
        s.record_win()
        s.record_win()
        s.record_loss()
        assert s.win_rate == pytest.approx(2 / 3)

    def test_win_rate_empty(self):
        s = SessionStats()
        assert s.win_rate == 0.0
