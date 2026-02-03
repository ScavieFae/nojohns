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

# Add slippi-ai to Python path
SLIPPI_AI_PATH = Path(__file__).parent.parent.parent / 'phillip-research' / 'slippi-ai'
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

from nojohns.fighter import Fighter, FighterResult, ControllerState

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

    def name(self) -> str:
        """Return fighter name."""
        model_name = self.config.model_path.stem
        return f"Phillip ({model_name})"

    def description(self) -> str:
        """Return fighter description."""
        policy_config = self._model_config.get('policy', {}) if self._model_config else {}
        delay = self.config.delay or policy_config.get('delay', '?')
        network = self._model_config.get('network', {}).get('name', 'unknown') if self._model_config else 'unknown'

        return (
            f"Neural network AI trained via imitation learning + RL. "
            f"Delay: {delay} frames, Network: {network}"
        )

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
            # Phillip's agent works differently from our act() pattern:
            # 1. The agent is called via step(gamestate)
            # 2. It maintains internal delay buffer
            # 3. It updates a controller object directly (via set_controller)
            #
            # We need to:
            # 1. Call agent.step(state) to process the gamestate
            # 2. Read back the controller state from the agent's controller

            # TODO: This needs to be tested with actual slippi-ai code
            # The agent might not have a step() method - need to check eval_lib
            # For now, return neutral until we can test

            logger.debug(f"Phillip act() called for port {self._port}, frame {state.frame}")

            # Placeholder - will implement after testing with slippi-ai
            return ControllerState()

        except Exception as e:
            logger.error(f"Phillip agent error: {e}", exc_info=True)
            return ControllerState()

    def on_game_end(self, result: FighterResult) -> None:
        """
        Called when game ends.

        Clean up Phillip's agent resources.

        Args:
            result: Game result
        """
        if self._agent:
            try:
                # Stop the agent (cleanup internal threads/resources)
                # TODO: Check if Agent has a stop() or cleanup() method
                logger.info(f"Phillip game ended: {result}")

                # The agent might not have an explicit stop method
                # It may clean up automatically or via __del__
                # We'll need to verify this when testing

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
    models_dir = Path(__file__).parent.parent.parent / 'phillip-research' / 'models'
    model_path = models_dir / f'{model_name}.pkl'

    config = PhillipConfig(
        model_path=model_path,
        async_inference=True,
    )

    return PhillipFighter(config)
