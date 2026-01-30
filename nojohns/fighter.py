"""
nojohns/fighter.py - Fighter interface and base classes

This module defines the Fighter protocol that all AI fighters must implement,
along with supporting data types and a convenience base class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# libmelee types
import melee
from melee import Character, Stage


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class FighterMetadata:
    """
    Static information about a fighter.
    
    Used for:
    - Registry discovery
    - Compatibility checking (can this fighter play Fox?)
    - Display in UI
    """
    
    # Identity
    name: str               # Lowercase, no spaces: "smashbot"
    version: str            # Semver: "1.4.2"
    display_name: str       # Human readable: "SmashBot"
    author: str             # Creator: "altf4"
    description: str        # One paragraph description
    
    # Classification
    fighter_type: str       # "rule-based" | "neural-net" | "hybrid"
    
    # Capabilities
    characters: list[Character]  # What characters this fighter can play
    stages: list[Stage] = field(default_factory=list)  # Empty = all stages OK
    
    # Requirements
    gpu_required: bool = False
    min_ram_gb: int = 2
    extra_requirements: dict[str, Any] = field(default_factory=dict)
    
    # Performance
    avg_frame_delay: int = 0  # Inherent input delay in frames
    
    # Source
    repo_url: str | None = None
    weights_required: bool = False  # True if needs model weights file


@dataclass
class MatchConfig:
    """
    Configuration for a specific game, provided by the match runner.
    """
    
    # Our assignment
    character: Character
    port: int  # 1-4
    
    # Match rules
    stage: Stage
    stocks: int = 4
    time_minutes: int = 8
    
    # Opponent info
    opponent_port: int = 2
    opponent_character: Character | None = None


@dataclass
class FighterConfig:
    """
    Owner-tunable parameters.
    
    Fighters declare which parameters they support in their manifest.
    Owners can adjust these between games.
    """
    
    # Common parameters (fighters may ignore if not applicable)
    aggression: float = 0.5  # 0.0 = passive, 1.0 = aggro
    recovery_preference: str = "mix"  # "high" | "low" | "mix"
    
    # Fighter-specific parameters
    extra: dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a parameter, checking extra dict if not a standard param."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key, default)


@dataclass
class ControllerState:
    """
    Controller inputs for a single frame.
    
    All analog values are 0.0 to 1.0, where 0.5 is neutral.
    """
    
    # Main analog stick
    main_x: float = 0.5
    main_y: float = 0.5
    
    # C-stick
    c_x: float = 0.5
    c_y: float = 0.5
    
    # Analog triggers (0.0 = not pressed, 1.0 = full press)
    l_trigger: float = 0.0
    r_trigger: float = 0.0
    
    # Digital buttons
    a: bool = False
    b: bool = False
    x: bool = False
    y: bool = False
    z: bool = False
    start: bool = False
    
    # D-pad (rarely used in competitive play)
    d_up: bool = False
    d_down: bool = False
    d_left: bool = False
    d_right: bool = False
    
    def to_libmelee(self, controller: melee.Controller) -> None:
        """Apply this state to a libmelee Controller."""
        controller.tilt_analog(melee.Button.BUTTON_MAIN, self.main_x, self.main_y)
        controller.tilt_analog(melee.Button.BUTTON_C, self.c_x, self.c_y)
        controller.press_shoulder(melee.Button.BUTTON_L, self.l_trigger)
        controller.press_shoulder(melee.Button.BUTTON_R, self.r_trigger)
        
        for btn, pressed in [
            (melee.Button.BUTTON_A, self.a),
            (melee.Button.BUTTON_B, self.b),
            (melee.Button.BUTTON_X, self.x),
            (melee.Button.BUTTON_Y, self.y),
            (melee.Button.BUTTON_Z, self.z),
            (melee.Button.BUTTON_START, self.start),
            (melee.Button.BUTTON_D_UP, self.d_up),
            (melee.Button.BUTTON_D_DOWN, self.d_down),
            (melee.Button.BUTTON_D_LEFT, self.d_left),
            (melee.Button.BUTTON_D_RIGHT, self.d_right),
        ]:
            if pressed:
                controller.press_button(btn)
            else:
                controller.release_button(btn)


@dataclass
class MatchResult:
    """
    Result of a completed game.
    
    Provided to the fighter after a game ends for learning/logging.
    """
    
    won: bool
    stocks_remaining: int
    opponent_stocks: int
    damage_dealt: float
    damage_taken: float
    duration_frames: int
    replay_path: str | None = None


# ============================================================================
# Fighter Protocol
# ============================================================================

@runtime_checkable
class Fighter(Protocol):
    """
    The interface every fighter must implement.
    
    Lifecycle:
        1. __init__() - One-time setup (load model, etc.)
        2. setup(match, config) - Called before each game
        3. act(state) - Called every frame during game
        4. on_game_end(result) - Called after game (optional)
    
    Example:
        class MyFighter(Fighter):
            @property
            def metadata(self) -> FighterMetadata:
                return FighterMetadata(...)
            
            def setup(self, match: MatchConfig, config: FighterConfig | None = None):
                self.port = match.port
            
            def act(self, state: melee.GameState) -> ControllerState:
                return ControllerState(a=True)  # Mash A
    """
    
    @property
    def metadata(self) -> FighterMetadata:
        """Return static information about this fighter."""
        ...
    
    def setup(self, match: MatchConfig, config: FighterConfig | None = None) -> None:
        """
        Prepare for a game.
        
        Called once before each game starts. Use this to:
        - Store match configuration
        - Reset internal state
        - Apply owner configuration
        
        Args:
            match: Arena-provided match parameters
            config: Owner-provided tuning (optional)
        """
        ...
    
    def act(self, state: melee.GameState) -> ControllerState:
        """
        Decide what inputs to send this frame.
        
        This is called every frame (~60fps). Must be fast!
        Target: <16ms, ideally <5ms.
        
        Args:
            state: Current game state from libmelee
            
        Returns:
            Controller inputs for this frame
        """
        ...
    
    def on_game_end(self, result: MatchResult) -> None:
        """
        Called after a game ends.
        
        Optional - implement for learning or logging.
        
        Args:
            result: Game outcome data
        """
        ...
    
    def configure(self, config: FighterConfig) -> None:
        """
        Update configuration between games.
        
        Optional - implement if your fighter supports runtime tuning.
        
        Args:
            config: New configuration from owner
        """
        ...


# ============================================================================
# Base Class
# ============================================================================

class BaseFighter(ABC):
    """
    Convenience base class with sensible defaults.
    
    Inherit from this to get:
    - Automatic config storage
    - Default no-op for optional methods
    - Type hints for instance variables
    
    You must implement:
    - metadata property
    - act() method
    """
    
    def __init__(self):
        self._config: FighterConfig = FighterConfig()
        self._match: MatchConfig | None = None
    
    @property
    @abstractmethod
    def metadata(self) -> FighterMetadata:
        """Subclasses must define their metadata."""
        ...
    
    @abstractmethod
    def act(self, state: melee.GameState) -> ControllerState:
        """Subclasses must implement decision logic."""
        ...
    
    def setup(self, match: MatchConfig, config: FighterConfig | None = None) -> None:
        """Store match config. Override for custom setup."""
        self._match = match
        if config is not None:
            self._config = config
    
    def on_game_end(self, result: MatchResult) -> None:
        """No-op by default. Override to learn from games."""
        pass
    
    def configure(self, config: FighterConfig) -> None:
        """Store config. Override for validation."""
        self._config = config
    
    # Convenience properties
    
    @property
    def port(self) -> int:
        """Our controller port."""
        if self._match is None:
            raise RuntimeError("setup() not called yet")
        return self._match.port
    
    @property
    def opponent_port(self) -> int:
        """Opponent's controller port."""
        if self._match is None:
            raise RuntimeError("setup() not called yet")
        return self._match.opponent_port
    
    def get_player(self, state: melee.GameState) -> melee.PlayerState | None:
        """Get our PlayerState from a GameState."""
        return state.players.get(self.port)
    
    def get_opponent(self, state: melee.GameState) -> melee.PlayerState | None:
        """Get opponent's PlayerState from a GameState."""
        return state.players.get(self.opponent_port)


# ============================================================================
# Example Fighters
# ============================================================================

class DoNothingFighter(BaseFighter):
    """
    A fighter that does nothing. Useful for testing.
    """
    
    @property
    def metadata(self) -> FighterMetadata:
        return FighterMetadata(
            name="do-nothing",
            version="0.0.1",
            display_name="Do Nothing",
            author="nojohns",
            description="Test fighter that stands still. Will lose every game.",
            fighter_type="rule-based",
            characters=[Character.FOX],
        )
    
    def act(self, state: melee.GameState) -> ControllerState:
        return ControllerState()


class RandomFighter(BaseFighter):
    """
    A fighter that mashes random buttons. Chaotic but fun.
    """
    
    def __init__(self):
        super().__init__()
        import random
        self._random = random
    
    @property
    def metadata(self) -> FighterMetadata:
        return FighterMetadata(
            name="random",
            version="0.0.1",
            display_name="Random Masher",
            author="nojohns",
            description="Mashes random buttons. Surprisingly effective sometimes.",
            fighter_type="rule-based",
            characters=list(Character),  # Can "play" any character
        )
    
    def act(self, state: melee.GameState) -> ControllerState:
        return ControllerState(
            main_x=self._random.random(),
            main_y=self._random.random(),
            a=self._random.random() > 0.7,
            b=self._random.random() > 0.8,
            x=self._random.random() > 0.9,
            l_trigger=self._random.random() if self._random.random() > 0.8 else 0.0,
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Data types
    "FighterMetadata",
    "MatchConfig", 
    "FighterConfig",
    "ControllerState",
    "MatchResult",
    # Protocol
    "Fighter",
    # Base class
    "BaseFighter",
    # Example fighters
    "DoNothingFighter",
    "RandomFighter",
]
