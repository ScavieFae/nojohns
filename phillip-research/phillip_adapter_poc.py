"""
Proof-of-concept adapter for integrating Phillip into nojohns.

This shows how we'd wrap Phillip's Agent class to implement our Fighter protocol.

NOTE: This is just a design sketch - won't run without:
1. slippi-ai installed as dependency
2. Trained model weights
3. libmelee version compatibility resolved
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Our nojohns interfaces
from nojohns.fighter import Fighter, FighterResult
from nojohns.runner import ControllerState, GameState

# These would come from slippi-ai (not yet installed)
# from slippi_ai import eval_lib
# from slippi_ai import policies
# from slippi_ai import dolphin as phillip_dolphin


@dataclass
class PhillipConfig:
    """Configuration for a Phillip agent."""

    model_path: Path
    """Path to trained model weights (.pkl file)"""

    character: str = "FOX"
    """Character to play as"""

    delay: int = 18
    """Input delay in frames (Phillip's reaction time)"""

    async_inference: bool = True
    """Run neural network inference asynchronously"""

    use_gpu: bool = False
    """Use GPU for inference (if available)"""


class PhillipFighter(Fighter):
    """
    Adapter that wraps Phillip's neural network AI to implement the Fighter protocol.

    This bridges between:
    - Our Fighter interface (act(), on_game_start(), etc.)
    - Phillip's Agent interface (from slippi-ai)
    """

    def __init__(self, config: PhillipConfig):
        self.config = config
        self._agent: Optional[Any] = None  # phillip's Agent instance
        self._port: Optional[int] = None

    def name(self) -> str:
        return f"Phillip ({self.config.character})"

    def description(self) -> str:
        model_name = self.config.model_path.stem
        return f"Neural network AI trained with imitation + RL. Model: {model_name}"

    def on_game_start(self, port: int, state: GameState) -> None:
        """
        Called when a game starts.

        This is where we'd:
        1. Load the trained model
        2. Initialize Phillip's Agent
        3. Set up the delay buffer (18 frames)
        """
        self._port = port

        # Pseudocode - actual implementation would do:
        # self._agent = eval_lib.build_agent(
        #     port=port,
        #     opponent_port=3 - port,  # 1 -> 2, 2 -> 1
        #     console_delay=self.config.delay,
        #     path=str(self.config.model_path),
        #     async_inference=self.config.async_inference,
        # )
        # self._agent.start()

    def act(self, state: GameState) -> ControllerState:
        """
        Get Phillip's action for the current gamestate.

        The challenge here is converting between:
        - Our GameState representation
        - Phillip's gamestate representation

        Phillip uses libmelee's gamestate directly, so we'd need to either:
        1. Pass through the raw libmelee gamestate, OR
        2. Convert our GameState back to libmelee format (messy)

        Option 1 is cleaner - modify our GameState to include raw libmelee state.
        """
        if not self._agent:
            # Not initialized yet, return neutral
            return ControllerState()

        # Pseudocode:
        # The agent internally maintains a delay buffer and handles timing
        # We just need to call it each frame

        # Phillip's agent would internally:
        # 1. Add current state to delay buffer
        # 2. Get state from 18 frames ago
        # 3. Run neural network inference
        # 4. Convert network output to controller buttons
        # 5. Return controller state

        # phillip_controller = self._agent.get_action(state.raw_libmelee_state)
        # return self._convert_controller(phillip_controller)

        return ControllerState()

    def on_game_end(self, result: FighterResult) -> None:
        """
        Called when game ends.

        Clean up Phillip's agent resources.
        """
        if self._agent:
            # Pseudocode:
            # self._agent.stop()
            pass

    def _convert_controller(self, phillip_ctrl) -> ControllerState:
        """
        Convert Phillip's controller representation to ours.

        Phillip uses libmelee's Controller class, which has:
        - Buttons (A, B, X, Y, Z, L, R, START)
        - Sticks (main_stick, c_stick)
        - Triggers (l_shoulder, r_shoulder)

        We'd map these to our ControllerState.
        """
        # Pseudocode mapping
        return ControllerState(
            # button=phillip_ctrl.button,
            # main_stick=(phillip_ctrl.main_stick[0], phillip_ctrl.main_stick[1]),
            # c_stick=(phillip_ctrl.c_stick[0], phillip_ctrl.c_stick[1]),
            # l_shoulder=phillip_ctrl.l_shoulder,
            # r_shoulder=phillip_ctrl.r_shoulder,
        )


# Example usage (once we have models):
def example_usage():
    """
    How you'd use PhillipFighter in nojohns.
    """

    # Create fighter config
    config = PhillipConfig(
        model_path=Path("models/fox_d18_ditto_v3.pkl"),
        character="FOX",
        delay=18,
    )

    # Create fighter
    phillip = PhillipFighter(config)

    # Use in a match (via our existing runner)
    # from nojohns.runner import LocalRunner
    # runner = LocalRunner(...)
    # result = runner.run_match(phillip, other_fighter)


# Integration checklist:
#
# [ ] Install slippi-ai as dependency
#     - May need Python version downgrade (3.12 -> 3.11 for TensorFlow)
#     - Handle libmelee version conflict (we use mainline, Phillip uses fork)
#
# [ ] Get model weights
#     - Ask x_pilot for basic imitation agents
#     - Or train our own (would take days-weeks)
#
# [ ] Implement adapter
#     - Handle gamestate conversion
#     - Handle controller conversion
#     - Manage Agent lifecycle
#
# [ ] Test locally
#     - Phillip vs SmashBot
#     - Phillip vs CPU
#     - Phillip vs Phillip (ditto)
#
# [ ] Test netplay
#     - May have issues with delay + netplay lag
#     - Need to tune delay parameter for netplay
#
# [ ] Performance tuning
#     - CPU vs GPU inference
#     - Async inference
#     - Batch size (if running multiple Phillip instances)
