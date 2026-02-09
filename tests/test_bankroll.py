"""Tests for agents/bankroll.py — pure math, no RPC calls."""

import pytest

from agents.bankroll import (
    BankrollState,
    win_probability_from_elo,
    kelly_fraction,
    kelly_wager,
)


# ============================================================================
# win_probability_from_elo
# ============================================================================


class TestWinProbability:
    def test_equal_elo_is_fifty_fifty(self):
        assert win_probability_from_elo(1500, 1500) == pytest.approx(0.5)

    def test_higher_elo_favored(self):
        p = win_probability_from_elo(1600, 1400)
        assert p > 0.5

    def test_lower_elo_underdog(self):
        p = win_probability_from_elo(1400, 1600)
        assert p < 0.5

    def test_symmetry(self):
        """P(A beats B) + P(B beats A) = 1"""
        p1 = win_probability_from_elo(1600, 1400)
        p2 = win_probability_from_elo(1400, 1600)
        assert p1 + p2 == pytest.approx(1.0)

    def test_200_elo_gap(self):
        """200 Elo gap ≈ 76% win rate (well-known Elo property)."""
        p = win_probability_from_elo(1700, 1500)
        assert p == pytest.approx(0.76, abs=0.01)

    def test_400_elo_gap(self):
        """400 Elo gap ≈ 91% win rate."""
        p = win_probability_from_elo(1900, 1500)
        assert p == pytest.approx(0.91, abs=0.01)

    def test_zero_elo(self):
        """Works with unusual Elo values."""
        p = win_probability_from_elo(0, 0)
        assert p == pytest.approx(0.5)


# ============================================================================
# kelly_fraction
# ============================================================================


class TestKellyFraction:
    def test_no_edge(self):
        """50/50 = no edge = don't bet."""
        assert kelly_fraction(0.5) == pytest.approx(0.0)

    def test_favorable(self):
        """60% win rate → bet 20% of bankroll."""
        assert kelly_fraction(0.6) == pytest.approx(0.2)

    def test_very_favorable(self):
        """70% win rate → bet 40%."""
        assert kelly_fraction(0.7) == pytest.approx(0.4)

    def test_unfavorable(self):
        """40% win rate → negative (don't bet)."""
        assert kelly_fraction(0.4) == pytest.approx(-0.2)

    def test_certain_win(self):
        """100% → bet everything."""
        assert kelly_fraction(1.0) == pytest.approx(1.0)

    def test_certain_loss(self):
        """0% → maximally negative."""
        assert kelly_fraction(0.0) == pytest.approx(-1.0)


# ============================================================================
# kelly_wager
# ============================================================================


class TestKellyWager:
    BANKROLL = 10 * 10**18  # 10 MON

    def test_no_edge_returns_zero(self):
        assert kelly_wager(0.5, self.BANKROLL) == 0

    def test_unfavorable_returns_zero(self):
        assert kelly_wager(0.4, self.BANKROLL) == 0

    def test_favorable_returns_positive(self):
        amount = kelly_wager(0.6, self.BANKROLL)
        assert amount > 0

    def test_cap_enforcement_moderate(self):
        """Even with 90% win rate, moderate cap limits to 10%."""
        amount = kelly_wager(0.9, self.BANKROLL, multiplier=1.0, max_pct=0.10)
        max_amount = int(self.BANKROLL * 0.10)
        assert amount <= max_amount

    def test_cap_enforcement_conservative(self):
        """Conservative: 5% cap."""
        amount = kelly_wager(0.9, self.BANKROLL, multiplier=0.5, max_pct=0.05)
        max_amount = int(self.BANKROLL * 0.05)
        assert amount <= max_amount

    def test_cap_enforcement_aggressive(self):
        """Aggressive: 25% cap."""
        amount = kelly_wager(0.9, self.BANKROLL, multiplier=1.5, max_pct=0.25)
        max_amount = int(self.BANKROLL * 0.25)
        assert amount <= max_amount

    def test_half_kelly_multiplier(self):
        """Half-Kelly should produce roughly half the full-Kelly amount."""
        full = kelly_wager(0.7, self.BANKROLL, multiplier=1.0, max_pct=1.0)
        half = kelly_wager(0.7, self.BANKROLL, multiplier=0.5, max_pct=1.0)
        # Allow some tolerance since floor/cap effects exist
        assert half == pytest.approx(full * 0.5, rel=0.01)

    def test_floor_enforced(self):
        """Wager below 0.001 MON returns 0 (gas would exceed wager)."""
        tiny_bankroll = int(0.005 * 10**18)  # 0.005 MON
        # With 55% edge, Kelly says bet 10% = 0.0005 MON < floor
        amount = kelly_wager(0.55, tiny_bankroll, multiplier=1.0, max_pct=1.0)
        assert amount == 0

    def test_zero_bankroll(self):
        assert kelly_wager(0.7, 0) == 0

    def test_output_is_integer_wei(self):
        amount = kelly_wager(0.65, self.BANKROLL)
        assert isinstance(amount, int)
