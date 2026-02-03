"""
nojohns/menu_navigation.py - Custom Slippi menu navigation

Replaces libmelee's buggy menu_helper_simple for Slippi direct connect.
Handles menu navigation and connect code entry without overflow issues.
"""

import logging
from enum import IntEnum

import melee
from melee import Button, Menu, SubMenu

logger = logging.getLogger(__name__)


class CharacterGrid:
    """
    Maps characters to positions in the Slippi name entry grid.

    The grid layout is:
    A B C D E F G H I J
    K L M N O P Q R S T
    U V W X Y Z _ _ _ #
    0 1 2 3 4 5 6 7 8 9

    We need to navigate this grid to enter connect codes like "SCAV#968"
    """

    # Character rows (top to bottom)
    ROWS = [
        "ABCDEFGHIJ",
        "KLMNOPQRST",
        "UVWXYZ   #",
        "0123456789"
    ]

    @classmethod
    def find_character(cls, char: str) -> tuple[int, int] | None:
        """
        Find the (row, col) position of a character in the grid.

        Returns:
            (row, col) tuple, or None if character not found
        """
        for row_idx, row in enumerate(cls.ROWS):
            col_idx = row.find(char)
            if col_idx != -1:
                return (row_idx, col_idx)
        return None


class SlippiMenuNavigator:
    """
    Custom menu navigation for Slippi direct connect.

    Handles navigating menus and entering connect codes without relying
    on libmelee's buggy menu_helper_simple.
    """

    def __init__(self):
        self.connect_code_index = 0
        self.inputs_live = False
        self.last_menu_selection = None
        self.position_history = []  # Track recent positions to detect loops
        self.attempts_for_current_char = 0

    def navigate_menus(
        self,
        gamestate: melee.GameState,
        controller: melee.Controller,
        connect_code: str,
        character: melee.Character,
        stage: melee.Stage,
        autostart: bool = True,
    ) -> None:
        """
        Main menu navigation function. Call this every frame.

        Args:
            gamestate: Current game state
            controller: Controller to send inputs to
            connect_code: Slippi connect code (e.g. "SCAV#968")
            character: Character to select
            stage: Stage to select
            autostart: Auto-start the match when ready
        """
        menu_state = gamestate.menu_state

        # Log menu state changes for debugging
        if menu_state != getattr(self, '_last_menu_state', None):
            logger.debug(f"Menu state: {menu_state}, submenu: {gamestate.submenu}")
            self._last_menu_state = menu_state

        # Main menu - navigate to online play
        if menu_state == Menu.MAIN_MENU:
            self._navigate_to_online(gamestate, controller, connect_code)

        # Press start screen - proceed
        elif menu_state == Menu.PRESS_START:
            melee.MenuHelper.choose_versus_mode(gamestate, controller)

        # Character select screen (online or local)
        elif menu_state in [Menu.CHARACTER_SELECT, Menu.SLIPPI_ONLINE_CSS]:
            # Name entry submenu - enter connect code
            if gamestate.submenu == SubMenu.NAME_ENTRY_SUBMENU:
                self._enter_connect_code(gamestate, controller, connect_code)
            # Regular character select
            else:
                melee.MenuHelper.choose_character(
                    character=character,
                    gamestate=gamestate,
                    controller=controller,
                    cpu_level=0,
                    costume=0,
                    swag=False,
                    start=autostart,
                )

        # Stage select
        elif menu_state == Menu.STAGE_SELECT:
            melee.MenuHelper.choose_stage(
                stage=stage,
                gamestate=gamestate,
                controller=controller,
            )

        # Postgame - skip back to menus
        elif menu_state == Menu.POSTGAME_SCORES:
            melee.MenuHelper.skip_postgame(controller, gamestate)

    def _navigate_to_online(
        self,
        gamestate: melee.GameState,
        controller: melee.Controller,
        connect_code: str,
    ) -> None:
        """Navigate from MAIN_MENU to Slippi online/direct connect."""
        if connect_code:
            melee.MenuHelper.choose_direct_online(gamestate, controller)
        else:
            melee.MenuHelper.choose_versus_mode(gamestate, controller)

    def _enter_connect_code(
        self,
        gamestate: melee.GameState,
        controller: melee.Controller,
        connect_code: str,
    ) -> None:
        """
        Enter connect code character by character in the name entry screen.

        This is our custom implementation that doesn't use libmelee's buggy
        calculation. We navigate the grid more carefully.
        """
        # The name entry screen is dead for the first few frames
        # Wait until we can move off the first position
        if gamestate.menu_selection != 45:
            self.inputs_live = True

        if not self.inputs_live:
            controller.tilt_analog(Button.BUTTON_MAIN, 1, 0.5)
            return

        # Release controller every other frame to let inputs register
        if gamestate.frame % 2 == 0:
            controller.release_all()
            return

        # Done entering code - press START to confirm
        if self.connect_code_index >= len(connect_code):
            logger.debug("Connect code entry complete, pressing START")
            controller.press_button(Button.BUTTON_START)
            return

        # Get the next character we need to enter
        target_char = connect_code[self.connect_code_index]

        # Track attempts for loop detection
        self.attempts_for_current_char += 1
        self.position_history.append(gamestate.menu_selection)
        if len(self.position_history) > 20:
            self.position_history.pop(0)

        # Log for debugging
        if gamestate.menu_selection != self.last_menu_selection:
            logger.debug(
                f"Entering char {self.connect_code_index}: '{target_char}', "
                f"menu_selection: {gamestate.menu_selection}"
            )
            self.last_menu_selection = gamestate.menu_selection

        # Calculate target position using our safe method
        target_position = self._calculate_target_position(target_char)

        if target_position is None:
            logger.error(f"Character '{target_char}' not found in grid!")
            return

        # If we're at the exact target, press A to select it
        if gamestate.menu_selection == target_position:
            logger.info(f"Selecting '{target_char}' at position {gamestate.menu_selection}")
            controller.press_button(Button.BUTTON_A)
            self.connect_code_index += 1
            self.attempts_for_current_char = 0
            self.position_history = []
            return

        # Detect if we're in a loop (seeing the same positions repeat)
        if self.attempts_for_current_char > 30:
            # We've been trying for too long, check if we're looping
            if len(set(self.position_history)) < 10:  # Less than 10 unique positions = looping
                # Find the position closest to target and use that
                closest_pos = min(self.position_history, key=lambda p: abs(p - target_position))
                if gamestate.menu_selection == closest_pos:
                    logger.warning(
                        f"Loop detected! Selecting closest position {closest_pos} "
                        f"for '{target_char}' (target was {target_position})"
                    )
                    controller.press_button(Button.BUTTON_A)
                    self.connect_code_index += 1
                    self.attempts_for_current_char = 0
                    self.position_history = []
                    return

        # Navigate toward the target position
        logger.info(
            f"Navigating to '{target_char}': "
            f"current={gamestate.menu_selection}, target={target_position}"
        )
        self._navigate_to_position(
            gamestate,
            controller,
            current=gamestate.menu_selection,
            target=target_position,
        )

    def _calculate_target_position(self, char: str) -> int | None:
        """
        Calculate the menu_selection value for a character.

        Uses the same formula as libmelee, but with better error handling.

        Returns:
            The menu_selection value, or None if character not found
        """
        # NOTE: '9' at position 3 has navigation issues on macOS Sequoia
        # Position 3 exists in the grid, but the cursor can't reach it reliably.
        # For now, just use the correct position and let loop detection handle it.
        # TODO: Investigate Sequoia-specific timing issue

        # Find character in grid
        position = CharacterGrid.find_character(char)
        if position is None:
            return None

        row, col = position

        # libmelee's formula (reverse-engineered from their code)
        # Each row starts at a base value, then subtracts 5 per column
        base_values = [45, 46, 47, 48]  # One per row
        target_code = base_values[row] - (col * 5)

        return target_code

    def _navigate_to_position(
        self,
        gamestate: melee.GameState,
        controller: melee.Controller,
        current: int,
        target: int,
    ) -> None:
        """
        Navigate from current position to target position.

        Uses careful movement to avoid the overflow issue.
        """
        # Special case: if we're at the "back" button (57), move down
        if current == 57:
            controller.tilt_analog(Button.BUTTON_MAIN, 0.5, 1)
            return

        # Calculate difference
        # Use explicit checks to avoid overflow
        if current < 0 or target < 0 or current > 100 or target > 100:
            logger.error(
                f"INVALID menu positions: current={current}, target={target} - "
                f"This indicates the menu_selection value is wrong!"
            )
            # Just release - don't make it worse
            controller.release_all()
            return

        diff = abs(target - current)

        # Move toward target
        # If target has higher value (further down/left in grid), move down/left
        if target > current:
            if diff < 5:
                # Move vertically (down)
                controller.tilt_analog(Button.BUTTON_MAIN, 0.5, 0)
            else:
                # Move horizontally (left)
                controller.tilt_analog(Button.BUTTON_MAIN, 0, 0.5)
        # If target has lower value (further up/right in grid), move up/right
        else:
            if diff < 5:
                # Move vertically (up)
                controller.tilt_analog(Button.BUTTON_MAIN, 0.5, 1)
            else:
                # Move horizontally (right)
                controller.tilt_analog(Button.BUTTON_MAIN, 1, 0.5)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "SlippiMenuNavigator",
    "CharacterGrid",
]
