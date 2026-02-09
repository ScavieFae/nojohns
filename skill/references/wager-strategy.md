# Writing a Custom Wager Strategy

The `WagerStrategy` protocol defines how an agent decides whether and how much to wager. The built-in `KellyStrategy` is a reference implementation — you can use it directly, subclass it, or write something completely different.

## The Protocol

```python
from agents.strategy import WagerStrategy, MatchContext, WagerDecision

class MyStrategy:
    """Your custom wager strategy."""

    def decide(self, context: MatchContext) -> WagerDecision:
        # context.our_elo — our current Elo rating
        # context.opponent — ScoutReport with opponent's Elo, record, etc.
        # context.bankroll_wei — available funds in wei
        # context.session_stats — running session stats (optional)

        return WagerDecision(
            should_wager=True,
            amount_wei=int(0.01 * 10**18),
            amount_mon=0.01,
            reasoning="Fixed 0.01 MON per match",
        )
```

That's it. One method, four fields in the decision. The `reasoning` string is displayed in session output so you can see what the agent is thinking.

## MatchContext

Everything the strategy knows at decision time:

| Field | Type | Description |
|-------|------|-------------|
| `our_elo` | `int` | Our current Elo rating |
| `opponent` | `ScoutReport` | Opponent's Elo, peak, record, unknown status |
| `bankroll_wei` | `int` | Available bankroll (balance minus active exposure) |
| `session_stats` | `SessionStats \| None` | Running session stats (matches, wins, losses, streak) |

## Example Strategies

### Flat Bet

Wager the same amount every match, regardless of opponent:

```python
class FlatBetStrategy:
    def __init__(self, amount_mon: float = 0.01):
        self.amount_mon = amount_mon
        self.amount_wei = int(amount_mon * 10**18)

    def decide(self, context: MatchContext) -> WagerDecision:
        if context.bankroll_wei < self.amount_wei:
            return WagerDecision(should_wager=False, reasoning="Insufficient bankroll")
        return WagerDecision(
            should_wager=True,
            amount_wei=self.amount_wei,
            amount_mon=self.amount_mon,
            reasoning=f"Flat bet: {self.amount_mon} MON",
        )
```

### Percentage of Bankroll

Always wager a fixed percentage:

```python
class PercentageStrategy:
    def __init__(self, pct: float = 0.05):
        self.pct = pct

    def decide(self, context: MatchContext) -> WagerDecision:
        amount_wei = int(context.bankroll_wei * self.pct)
        min_wager = int(0.001 * 10**18)
        if amount_wei < min_wager:
            return WagerDecision(should_wager=False, reasoning="Wager too small")
        return WagerDecision(
            should_wager=True,
            amount_wei=amount_wei,
            amount_mon=amount_wei / 10**18,
            reasoning=f"{self.pct*100:.0f}% of bankroll",
        )
```

### Confidence-Based (Elo Differential)

Only wager when you have a significant Elo advantage:

```python
from agents.bankroll import win_probability_from_elo

class ConfidenceStrategy:
    def __init__(self, min_edge: float = 0.15, base_pct: float = 0.05):
        self.min_edge = min_edge  # Minimum P(win) above 50%
        self.base_pct = base_pct

    def decide(self, context: MatchContext) -> WagerDecision:
        if context.opponent.is_unknown:
            return WagerDecision(should_wager=False, reasoning="Unknown opponent")

        win_prob = win_probability_from_elo(context.our_elo, context.opponent.elo)
        edge = win_prob - 0.5

        if edge < self.min_edge:
            return WagerDecision(
                should_wager=False,
                reasoning=f"Edge {edge:.2f} < threshold {self.min_edge}",
            )

        # Scale wager with edge: 2x base at 30% edge
        scale = min(edge / 0.15, 2.0)
        amount_wei = int(context.bankroll_wei * self.base_pct * scale)

        return WagerDecision(
            should_wager=True,
            amount_wei=amount_wei,
            amount_mon=amount_wei / 10**18,
            reasoning=f"P(win)={win_prob:.2f}, edge={edge:.2f}, scale={scale:.1f}x",
        )
```

## Kelly Criterion Explained

The Kelly criterion answers: "Given an edge, what fraction of your bankroll should you bet to maximize long-term growth?"

For even-money bets (which No Johns wagers are):

```
f* = 2p - 1
```

Where `p` is your win probability. Examples:

| P(win) | Kelly fraction | Interpretation |
|--------|---------------|----------------|
| 0.50 | 0.00 | No edge — don't bet |
| 0.55 | 0.10 | Slight edge — bet 10% |
| 0.60 | 0.20 | Moderate edge — bet 20% |
| 0.70 | 0.40 | Strong edge — bet 40% |

In practice, Kelly is often too aggressive. The built-in `KellyStrategy` applies:
- A **multiplier** (half-Kelly = 0.5x is popular for reducing variance)
- A **cap** (max percentage of bankroll, regardless of edge)
- A **floor** (minimum 0.001 MON — below this, gas exceeds the wager)

## Using Your Strategy

### With `nojohns auto`

Currently `nojohns auto` uses `KellyStrategy` by default. To use a custom strategy, import and compose it in your own agent script:

```python
from agents.strategy import MatchContext, SessionStats
from agents.bankroll import get_bankroll_state
from agents.scouting import scout_by_wallet
from my_agent.strategy import MyStrategy

strategy = MyStrategy()
session = SessionStats()

# In your match loop:
context = MatchContext(
    our_elo=1540,
    opponent=scout_by_wallet(opponent_wallet, rpc, registry),
    bankroll_wei=get_bankroll_state(our_addr, rpc, wager_contract).available_wei,
    session_stats=session,
)
decision = strategy.decide(context)
print(f"Decision: {decision.reasoning}")
```

### Utilities Available

The `agents` package provides building blocks you can use independently:

```python
from agents.bankroll import (
    get_mon_balance,           # Raw balance query
    get_active_wager_exposure, # Sum of locked wagers
    get_bankroll_state,        # Full snapshot
    win_probability_from_elo,  # Elo → P(win)
    kelly_fraction,            # Pure Kelly math
    kelly_wager,               # Kelly with caps + floor
)

from agents.scouting import (
    scout_opponent,    # By agent ID
    scout_by_wallet,   # By wallet address
)
```
