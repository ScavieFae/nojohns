# Fighter Interface Specification

The Fighter interface is the core abstraction that makes No Johns extensible. Any AI that implements this interface can compete in the arena.

## Quick Reference

```python
from nojohns import Fighter, FighterMetadata, MatchConfig, ControllerState

class MyFighter(Fighter):
    @property
    def metadata(self) -> FighterMetadata:
        return FighterMetadata(name="my-fighter", ...)
    
    def setup(self, match: MatchConfig) -> None:
        # Prepare for match
        pass
    
    def act(self, state: GameState) -> ControllerState:
        # Return inputs for this frame
        return ControllerState()
```

---

## Data Types

### FighterMetadata

Static information about a fighter, used for discovery and compatibility checking.

```python
@dataclass
class FighterMetadata:
    # Identity
    name: str                    # "smashbot" (lowercase, no spaces)
    version: str                 # "1.4.2" (semver)
    display_name: str            # "SmashBot"
    author: str                  # "altf4"
    description: str             # "Rule-based Fox AI..."
    
    # Classification
    fighter_type: str            # "rule-based" | "neural-net" | "hybrid"
    
    # Capabilities
    characters: list[Character]  # [Character.FOX, Character.FALCO]
    stages: list[Stage]          # Empty list = all stages OK
    
    # Requirements
    gpu_required: bool           # True if needs CUDA/GPU
    min_ram_gb: int              # Minimum RAM in gigabytes
    
    # Performance characteristics  
    avg_frame_delay: int         # Frames of input lag (0 for rule-based)
    
    # Source
    repo_url: str | None         # GitHub URL
    weights_required: bool       # True if needs model file
```

### MatchConfig

Provided by the arena when setting up a match.

```python
@dataclass
class MatchConfig:
    # Our assignment
    character: Character         # What we're playing
    port: int                    # Controller port (1-4)
    
    # Match rules
    stage: Stage                 # Current stage
    stocks: int                  # Starting stocks (usually 4)
    time_minutes: int            # Time limit (usually 8)
    
    # Opponent info (if known)
    opponent_port: int           # Their controller port
    opponent_character: Character | None
```

### FighterConfig

Owner-tunable parameters. Fighters declare which params they support.

```python
@dataclass
class FighterConfig:
    # Common params (fighters can ignore if not applicable)
    aggression: float = 0.5          # 0.0-1.0
    recovery_preference: str = "mix"  # "high" | "low" | "mix"
    
    # Fighter-specific params
    extra: dict[str, Any] = field(default_factory=dict)
```

### ControllerState

The fighter's output each frame - what buttons to press.

```python
@dataclass
class ControllerState:
    # Main stick (0.0-1.0, 0.5 is neutral)
    main_x: float = 0.5
    main_y: float = 0.5
    
    # C-stick (0.0-1.0, 0.5 is neutral)
    c_x: float = 0.5
    c_y: float = 0.5
    
    # Triggers (0.0-1.0)
    l_trigger: float = 0.0
    r_trigger: float = 0.0
    
    # Face buttons
    a: bool = False
    b: bool = False
    x: bool = False
    y: bool = False
    z: bool = False
    
    # Start (rarely used in match)
    start: bool = False
```

### MatchResult

Provided after match completion for learning/logging.

```python
@dataclass
class MatchResult:
    won: bool
    stocks_remaining: int
    opponent_stocks: int
    damage_dealt: float
    damage_taken: float
    duration_frames: int
    replay_path: str | None
```

---

## The Fighter Protocol

```python
from typing import Protocol, runtime_checkable
import melee

@runtime_checkable
class Fighter(Protocol):
    """
    The interface every fighter must implement.
    
    Lifecycle:
    1. __init__() - One-time setup (load model, initialize state)
    2. setup(match) - Called before each game
    3. act(state) - Called every frame (~60fps)
    4. on_game_end(result) - Called after each game (optional)
    """
    
    @property
    def metadata(self) -> FighterMetadata:
        """Return static info about this fighter."""
        ...
    
    def setup(self, match: MatchConfig, config: FighterConfig | None = None) -> None:
        """
        Prepare for a game. Called once before game starts.
        
        Use this to:
        - Set character-specific parameters
        - Reset internal state
        - Apply owner configuration
        """
        ...
    
    def act(self, state: melee.GameState) -> ControllerState:
        """
        The core decision function. Called every frame.
        
        Args:
            state: Current game state from libmelee
            
        Returns:
            Controller inputs for this frame
            
        Performance:
            MUST complete in <16ms for real-time play.
            The match runner will skip frames if you're too slow.
        """
        ...
    
    # Optional methods
    
    def on_game_end(self, result: MatchResult) -> None:
        """Called after game ends. Use for learning/logging."""
        ...
    
    def configure(self, config: FighterConfig) -> None:
        """Update configuration between games."""
        ...
```

---

## Base Class

For convenience, we provide a base class with sensible defaults:

```python
from abc import ABC, abstractmethod

class BaseFighter(ABC):
    """
    Convenience base class. Inherit from this or implement Fighter directly.
    """
    
    def __init__(self):
        self._config = FighterConfig()
        self._match: MatchConfig | None = None
    
    @property
    @abstractmethod
    def metadata(self) -> FighterMetadata:
        """Subclasses must define metadata."""
        ...
    
    @abstractmethod
    def act(self, state: melee.GameState) -> ControllerState:
        """Subclasses must implement decision logic."""
        ...
    
    def setup(self, match: MatchConfig, config: FighterConfig | None = None) -> None:
        """Default: store config. Override for custom setup."""
        self._match = match
        if config:
            self._config = config
    
    def on_game_end(self, result: MatchResult) -> None:
        """Default: no-op. Override to learn from games."""
        pass
    
    def configure(self, config: FighterConfig) -> None:
        """Default: store config. Override for validation."""
        self._config = config
```

---

## Example: Minimal Fighter

```python
"""
A fighter that just stands still. Useful for testing.
"""

import melee
from nojohns import BaseFighter, FighterMetadata, ControllerState

class DoNothingFighter(BaseFighter):
    
    @property
    def metadata(self) -> FighterMetadata:
        return FighterMetadata(
            name="do-nothing",
            version="0.0.1",
            display_name="Do Nothing",
            author="nojohns",
            description="Test fighter that stands still",
            fighter_type="rule-based",
            characters=[melee.Character.FOX],
            stages=[],
            gpu_required=False,
            min_ram_gb=1,
            avg_frame_delay=0,
        )
    
    def act(self, state: melee.GameState) -> ControllerState:
        return ControllerState()  # All neutral
```

---

## Example: Simple Aggressive Fighter

```python
"""
A fighter that always runs at the opponent and attacks.
"""

import melee
from nojohns import BaseFighter, FighterMetadata, ControllerState

class AggroFighter(BaseFighter):
    
    @property
    def metadata(self) -> FighterMetadata:
        return FighterMetadata(
            name="aggro",
            version="0.1.0",
            display_name="Aggro Fox",
            author="nojohns",
            description="Runs at opponent, mashes A",
            fighter_type="rule-based",
            characters=[melee.Character.FOX],
            stages=[],
            gpu_required=False,
            min_ram_gb=1,
            avg_frame_delay=0,
        )
    
    def act(self, state: melee.GameState) -> ControllerState:
        ctrl = ControllerState()
        
        # Find ourselves and opponent
        us = state.players.get(self._match.port)
        them = state.players.get(self._match.opponent_port)
        
        if not us or not them:
            return ctrl
        
        # Run toward opponent
        if them.position.x > us.position.x:
            ctrl.main_x = 1.0  # Run right
        else:
            ctrl.main_x = 0.0  # Run left
        
        # Mash A when close
        distance = abs(them.position.x - us.position.x)
        if distance < 20:
            ctrl.a = True
        
        return ctrl
```

---

## Wrapping Existing AIs

Most existing Melee AIs weren't built for this interface. Here's the pattern for wrapping them:

### SmashBot Adapter

```python
"""
Adapter for altf4's SmashBot.
"""

import melee
from nojohns import BaseFighter, FighterMetadata, MatchConfig, FighterConfig, ControllerState

# SmashBot's actual implementation
from smashbot.bot import SmashBot as SmashBotImpl

class SmashBotFighter(BaseFighter):
    
    def __init__(self):
        super().__init__()
        self._bot: SmashBotImpl | None = None
    
    @property
    def metadata(self) -> FighterMetadata:
        return FighterMetadata(
            name="smashbot",
            version="1.4.2",
            display_name="SmashBot",
            author="altf4",
            description="Rule-based AI with frame-perfect execution",
            fighter_type="rule-based",
            characters=[
                melee.Character.FOX,
                melee.Character.FALCO,
                melee.Character.MARTH,
            ],
            stages=[],
            gpu_required=False,
            min_ram_gb=2,
            avg_frame_delay=0,
            repo_url="https://github.com/altf4/SmashBot",
        )
    
    def setup(self, match: MatchConfig, config: FighterConfig | None = None) -> None:
        super().setup(match, config)
        
        # Initialize SmashBot with match parameters
        self._bot = SmashBotImpl(
            port=match.port,
            opponent_port=match.opponent_port,
            character=match.character,
        )
        
        # Apply config if SmashBot supports it
        if config:
            self._apply_config(config)
    
    def act(self, state: melee.GameState) -> ControllerState:
        # SmashBot expects to control a real controller
        # We need to intercept its decisions and convert to ControllerState
        
        # Option 1: If SmashBot has a method that returns actions
        action = self._bot.decide(state)
        return self._action_to_controller(action)
        
        # Option 2: Mock the controller and capture what SmashBot does
        # (depends on SmashBot internals)
    
    def _apply_config(self, config: FighterConfig) -> None:
        # Map our config to SmashBot's parameters
        if hasattr(self._bot, 'aggression'):
            self._bot.aggression = config.aggression
    
    def _action_to_controller(self, action) -> ControllerState:
        # Convert SmashBot's action format to ControllerState
        # Implementation depends on SmashBot internals
        ...
```

---

## Fighter Manifest (fighter.yaml)

Every fighter package includes a YAML manifest:

```yaml
# fighter.yaml

# Identity
name: smashbot
version: 1.4.2
display_name: SmashBot  
author: altf4
description: |
  The original "level 11 CPU". Rule-based AI with frame-perfect
  execution, strong punish game, and aggressive edgeguarding.

# Classification
fighter_type: rule-based  # rule-based | neural-net | hybrid

# Capabilities
characters:
  - FOX
  - FALCO
  - MARTH
  - SHEIK
  
stages: []  # Empty = all tournament legal stages

# Requirements
hardware:
  gpu_required: false
  min_ram_gb: 2
  
# Performance
avg_frame_delay: 0

# Source
repo_url: https://github.com/altf4/SmashBot
weights_required: false

# Installation
install:
  pip:
    - melee>=0.40.0
    - git+https://github.com/altf4/SmashBot.git

# Entry point  
entry_point: nojohns.fighters.smashbot:SmashBotFighter

# Owner-configurable parameters
configurable:
  aggression:
    type: float
    range: [0.0, 1.0]
    default: 0.5
    description: |
      How often to approach vs wait for openings.
      0.0 = very passive, 1.0 = always approaching
      
  recovery_preference:
    type: enum
    values: [high, low, mix]
    default: mix
    description: |
      Preferred recovery height when returning to stage.
      
  edgeguard_depth:
    type: float
    range: [0.0, 1.0]
    default: 0.7
    description: |
      How far offstage to pursue for edgeguards.
      0.0 = stay safe, 1.0 = chase to blast zone
```

---

## Fighter Registry

Fighters are discovered and loaded through the registry:

```python
from nojohns import FighterRegistry

# List available fighters
registry = FighterRegistry()
for fighter in registry.list():
    print(f"{fighter.name} v{fighter.version} by {fighter.author}")

# Load a fighter
SmashBot = registry.load("smashbot")
fighter = SmashBot()

# Load with specific version
Phillip = registry.load("phillip", version="2.3.0")
```

### Registry Sources

1. **Built-in**: Packaged with nojohns (`fighters/` directory)
2. **Installed**: Via pip (`pip install nojohns-fighter-xyz`)
3. **Local**: From `~/.nojohns/fighters/`
4. **Remote**: From No Johns Arena registry (future)

---

## GameState Reference

The `act()` method receives a `melee.GameState`. Key fields:

```python
state.frame                 # Current frame number
state.stage                 # Current stage (Stage enum)
state.menu_state            # Menu/game state (for menu navigation)

state.players               # Dict[int, PlayerState]
state.players[port].character       # Character enum
state.players[port].position.x      # X coordinate
state.players[port].position.y      # Y coordinate  
state.players[port].percent         # Damage percent
state.players[port].stock           # Stocks remaining
state.players[port].facing          # True = facing right
state.players[port].action          # Current action (Action enum)
state.players[port].action_frame    # Frame of current action
state.players[port].invulnerable    # Is invulnerable
state.players[port].on_ground       # Is grounded

state.projectiles           # List of active projectiles
```

See [libmelee documentation](https://libmelee.readthedocs.io/) for complete reference.

---

## Performance Guidelines

### Frame Budget

Melee runs at 60fps = 16.67ms per frame. Your `act()` method should complete well under this.

| Fighter Type | Target Latency | Notes |
|--------------|----------------|-------|
| Rule-based | <1ms | Simple conditionals are fast |
| Neural net (GPU) | <5ms | Batching helps |
| Neural net (CPU) | <10ms | May need optimization |

### Handling Slow Fighters

If a fighter is too slow, the runner will:
1. Log a warning
2. Use the previous frame's inputs
3. Continue (no pause)

Fighters with inherent latency (like Phillip with 18 frame delay) handle this by planning ahead, not by being slow.

### Memory

- Load models once in `__init__`, not in `act()`
- Avoid allocations in the hot path
- Reuse ControllerState objects if needed

---

## Testing Your Fighter

```python
# test_my_fighter.py
import melee
from nojohns import MatchConfig
from my_fighter import MyFighter

def test_metadata():
    fighter = MyFighter()
    meta = fighter.metadata
    assert meta.name == "my-fighter"
    assert len(meta.characters) > 0

def test_act_returns_valid_state():
    fighter = MyFighter()
    fighter.setup(MatchConfig(
        character=melee.Character.FOX,
        port=1,
        stage=melee.Stage.FINAL_DESTINATION,
        stocks=4,
        time_minutes=8,
        opponent_port=2,
    ))
    
    # Create mock gamestate
    state = create_mock_gamestate()
    
    result = fighter.act(state)
    
    assert 0.0 <= result.main_x <= 1.0
    assert 0.0 <= result.main_y <= 1.0
    assert isinstance(result.a, bool)
```

Run with: `pytest test_my_fighter.py`
