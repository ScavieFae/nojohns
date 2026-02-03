"""
PhillipFighter - Adapter for Phillip neural network AI.

This adapter wraps vladfi1's slippi-ai (Phillip) to work with the nojohns
Fighter protocol.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add slippi-ai to Python path (located alongside this adapter)
SLIPPI_AI_PATH = Path(__file__).parent / 'slippi-ai'
if SLIPPI_AI_PATH.exists():
    sys.path.insert(0, str(SLIPPI_AI_PATH))

try:
    from slippi_ai import eval_lib, saving
    import melee
    SLIPPI_AI_AVAILABLE = True
except ImportError:
    SLIPPI_AI_AVAILABLE = False
    # Define placeholder for type hints
    melee = None

from nojohns.fighter import Fighter, FighterMetadata, MatchConfig, FighterConfig, MatchResult, ControllerState

# Import melee for GameState type
import melee

logger = logging.getLogger(__name__)


@dataclass
class PhillipConfig:
    """Configuration for a Phillip fighter instance."""

    model_path: Path
    """Path to trained model weights (.pkl file)"""

    character: Optional[str] = None
    """
    Character to play as (if model supports multiple).
    If None, uses the model's default/trained character.
    """

    delay: Optional[int] = None
    """
    Input delay in frames (Phillip's reaction time).
    If None, uses the delay from the model config.
    """

    async_inference: bool = True
    """
    Run neural network inference asynchronously.
    Recommended for better performance.
    """

    use_gpu: bool = False
    """
    Use GPU for inference if available.
    Note: May not work well on Apple Silicon.
    """


class PhillipFighter(Fighter):
    """
    Adapter that wraps Phillip's neural network AI.

    This bridges between:
    - Our Fighter interface (act(), on_game_start(), etc.)
    - Phillip's Agent interface (from slippi-ai)

    Key differences handled:
    - Phillip expects libmelee gamestate directly
    - Phillip manages its own delay buffer
    - Phillip's controller output needs conversion
    """

    def __init__(self, config: PhillipConfig):
        """
        Initialize Phillip fighter.

        Args:
            config: Phillip configuration

        Raises:
            ImportError: If slippi-ai is not installed
            FileNotFoundError: If model file doesn't exist
        """
        if not SLIPPI_AI_AVAILABLE:
            raise ImportError(
                "slippi-ai not installed. "
                "See phillip-research/SETUP.md for installation instructions."
            )

        if not config.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {config.model_path}\n"
                f"Download from: phillip-research/claude.md"
            )

        self.config = config
        self._agent: Optional[eval_lib.Agent] = None
        self._port: Optional[int] = None
        self._model_config: Optional[dict] = None

        # Load model config to get metadata
        try:
            state = saving.load_state_from_disk(str(config.model_path))
            self._model_config = state.get('config', {})
            logger.info(f"Loaded Phillip model from {config.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            raise

    @property
    def metadata(self) -> FighterMetadata:
        """Return fighter metadata."""
        model_name = self.config.model_path.stem
        policy_config = self._model_config.get('policy', {}) if self._model_config else {}
        delay = self.config.delay or policy_config.get('delay', 21)

        return FighterMetadata(
            name=f"phillip-{model_name}",
            version="1.0.0",
            display_name=f"Phillip ({model_name})",
            author="vladfi1 (slippi-ai)",
            description=f"Neural network AI trained via imitation learning + RL. Delay: {delay} frames",
            fighter_type="neural-network",
            characters=[melee.Character.FOX],  # Model supports all characters, but we'll start with Fox
        )

    def name(self) -> str:
        """Return fighter name."""
        return self.metadata.display_name

    def description(self) -> str:
        """Return fighter description."""
        return self.metadata.description

    def setup(self, match: MatchConfig, config: FighterConfig | None = None) -> None:
        """
        Prepare for a game. Called by the runner before each game.

        This is different from on_game_start which is called when we detect
        the game actually starting from gamestate.
        """
        # Store match config for later use
        self._match_config = match

    def on_game_start(self, port: int, state: melee.GameState) -> None:
        """
        Called when a game starts.

        Initializes the Phillip agent with the correct port and delay settings.

        Args:
            port: Our controller port (1 or 2)
            state: Initial game state
        """
        self._port = port
        opponent_port = 3 - port  # 1 -> 2, 2 -> 1

        # Get delay from config or model
        policy_config = self._model_config.get('policy', {}) if self._model_config else {}
        model_delay = policy_config.get('delay', 18)
        delay = self.config.delay if self.config.delay is not None else model_delay

        # Phillip's agent expects console_delay to be delay - 1
        # (it adds 1 frame internally for async inference)
        console_delay = max(delay - 1, 0)

        logger.info(
            f"Starting Phillip on port {port} vs port {opponent_port}, "
            f"delay={delay}, console_delay={console_delay}"
        )

        try:
            # Build the agent
            self._agent = eval_lib.build_agent(
                port=port,
                opponent_port=opponent_port,
                console_delay=console_delay,
                path=str(self.config.model_path),
                async_inference=self.config.async_inference,
                # jit_compile=False,  # Disable JIT for debugging
            )

            # Create a dummy controller for the agent to update
            # We don't actually connect to Dolphin with this - we just use it
            # to capture controller state in act()
            class DummyController:
                """Minimal controller that satisfies Agent.set_controller()."""
                def __init__(self, port):
                    self.port = port
                    self.last_action = None

                def press_button(self, button): pass
                def release_button(self, button): pass
                def tilt_analog(self, button, x, y): pass
                def press_shoulder(self, button, amount): pass

            dummy_controller = DummyController(port)
            self._dummy_controller = dummy_controller  # Store reference
            self._agent.set_controller(dummy_controller)

            # Start the agent (initializes internal state)
            self._agent.start()

            logger.info(f"Phillip agent started successfully on port {port}")

        except Exception as e:
            logger.error(f"Failed to start Phillip agent: {e}", exc_info=True)
            raise

    def act(self, state: melee.GameState) -> ControllerState:
        """
        Get Phillip's action for the current gamestate.

        Phillip's agent maintains an internal delay buffer and processes
        the gamestate to output controller inputs.

        Args:
            state: Current melee.GameState from libmelee

        Returns:
            Controller inputs for this frame
        """
        if not self._agent:
            logger.warning("Phillip agent not initialized, returning neutral")
            return ControllerState()

        try:
            # Call agent.step() - note: this also calls send_controller internally
            # but we don't use that, we just want the return value
            sample_outputs = self._agent.step(state)

            # The agent's step() processes and decodes the controller internally
            # We need to access the decoded controller that was sent
            # For now, intercept by checking what was stored in dummy controller
            # or use a simpler approach - just let the agent control directly

            # Extract the raw controller state from sample_outputs
            # sample_outputs.controller_state is the encoded version
            encoded_controller = sample_outputs.controller_state

            # We need to decode it the same way the agent does
            # Looking at agent code: action = utils.map_nt(lambda x: x[0], action)
            # Then: action = self._agent.embed_controller.decode(action)
            import numpy as np
            from slippi_ai import utils

            action = utils.map_nt(lambda x: x[0], encoded_controller)
            decoded_controller = self._agent._agent.embed_controller.decode(action)

            if self._agent.mirror:
                from slippi_ai import mirror_lib
                decoded_controller = mirror_lib.mirror_controller(decoded_controller)

            # Convert to our ControllerState format
            return self._convert_controller(decoded_controller)

        except Exception as e:
            logger.error(f"Phillip agent error: {e}", exc_info=True)
            return ControllerState()

    def _convert_controller(self, phillip_ctrl) -> ControllerState:
        """
        Convert Phillip's Controller NamedTuple to our ControllerState.

        Args:
            phillip_ctrl: slippi_ai.types.Controller NamedTuple

        Returns:
            Our ControllerState with equivalent inputs
        """
        return ControllerState(
            # Analog sticks (Phillip uses 0.5 as neutral, same as us)
            main_x=float(phillip_ctrl.main_stick.x),
            main_y=float(phillip_ctrl.main_stick.y),
            c_x=float(phillip_ctrl.c_stick.x),
            c_y=float(phillip_ctrl.c_stick.y),

            # Triggers (Phillip only uses one 'shoulder' value for L trigger)
            l_trigger=float(phillip_ctrl.shoulder),
            r_trigger=0.0,  # Phillip doesn't separate L/R triggers

            # Buttons
            a=bool(phillip_ctrl.buttons.A),
            b=bool(phillip_ctrl.buttons.B),
            x=bool(phillip_ctrl.buttons.X),
            y=bool(phillip_ctrl.buttons.Y),
            z=bool(phillip_ctrl.buttons.Z),
            d_up=bool(phillip_ctrl.buttons.D_UP),
            # Note: Phillip doesn't use d_down, d_left, d_right, start
            # Those remain False (default)
        )

    def on_game_end(self, result: MatchResult) -> None:
        """
        Called when game ends.

        Clean up Phillip's agent resources.

        Args:
            result: Game result
        """
        if self._agent:
            try:
                logger.info(f"Phillip game ended: {result}")

                # Stop the agent (cleanup internal threads/resources)
                # The agent has async inference threads that need cleanup
                if hasattr(self._agent, 'stop'):
                    self._agent.stop()

            except Exception as e:
                logger.warning(f"Error during Phillip cleanup: {e}")

            finally:
                self._agent = None


def load_phillip(model_name: str = "all_d21_imitation_v3") -> PhillipFighter:
    """
    Convenience function to load a Phillip model by name.

    Args:
        model_name: Model filename (without .pkl extension)

    Returns:
        Configured PhillipFighter instance

    Example:
        >>> phillip = load_phillip("all_d21_imitation_v3")
        >>> # Use in a match
    """
    models_dir = Path(__file__).parent / 'models'
    model_path = models_dir / f'{model_name}.pkl'

    config = PhillipConfig(
        model_path=model_path,
        async_inference=True,
    )

    return PhillipFighter(config)
