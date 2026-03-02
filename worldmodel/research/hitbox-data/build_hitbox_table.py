#!/usr/bin/env python3
"""Build a hitbox lookup table from meleedb CSV data.

Maps (character_id, action_state_id, state_age) → HitboxProperties
for use in world model encoding.

Data sources:
  - melee_hitboxes.csv from https://github.com/BroccoliRaab/meleedb
  - action_state.json from https://github.com/hohav/ssbm-data

Usage:
    .venv/bin/python worldmodel/research/hitbox-data/build_hitbox_table.py

Output:
    worldmodel/research/hitbox-data/hitbox_table.json
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Character name → libmelee Character ID mapping
# From melee.enums.Character (vladfi1 fork v0.43.0)
# ---------------------------------------------------------------------------
CHAR_NAME_TO_ID = {
    "Mario": 0, "Fox": 1, "Captain Falcon": 2, "Donkey Kong": 3,
    "Kirby": 4, "Bowser": 5, "Link": 6, "Sheik": 7, "Ness": 8,
    "Peach": 9, "Ice Climbers": 10, "Nana": 11, "Pikachu": 12,
    "Samus": 13, "Yoshi": 14, "Jigglypuff": 15, "Mewtwo": 16,
    "Luigi": 17, "Marth": 18, "Zelda": 19, "Young Link": 20,
    "Dr. Mario": 21, "Falco": 22, "Pichu": 23,
    "Mr. Game & Watch": 24, "Ganondorf": 25, "Roy": 26,
}

# meleedb character names don't always match libmelee exactly
MELEEDB_CHAR_FIXUP = {
    "CaptainFalcon": "Captain Falcon",
    "DonkeyKong": "Donkey Kong",
    "IceClimbers": "Ice Climbers",
    "YoungLink": "Young Link",
    "DrMario": "Dr. Mario",
    "GameAndWatch": "Mr. Game & Watch",
    "MrGameAndWatch": "Mr. Game & Watch",
}


def normalize_char_name(meleedb_name: str) -> str | None:
    """Convert meleedb character name to libmelee-compatible name."""
    if meleedb_name in CHAR_NAME_TO_ID:
        return meleedb_name
    if meleedb_name in MELEEDB_CHAR_FIXUP:
        return MELEEDB_CHAR_FIXUP[meleedb_name]
    return None


# ---------------------------------------------------------------------------
# SubactionName → action state ID mapping
# ---------------------------------------------------------------------------

def build_action_state_map(action_state_path: Path) -> dict[str, int]:
    """Build a mapping from normalized action name → action state ID.

    Uses the ssbm-data action_state.json which has:
      Common: { known_values: { "44": { ident: "ATTACK_11" }, ... } }
      Fox:    { known_values: { "341": { ident: "BLASTER_GROUND_STARTUP" }, ... } }
    """
    with open(action_state_path) as f:
        data = json.load(f)

    # Build two maps:
    #   1. normalized_name → id for common actions (shared across characters)
    #   2. (character, normalized_name) → id for character-specific actions
    name_to_id = {}
    char_specific = {}

    for char_key, section in data.items():
        known = section.get("known_values", {})
        for id_str, info in known.items():
            ident = info["ident"]
            action_id = int(id_str)
            if char_key == "Common":
                name_to_id[ident] = action_id
            else:
                char_specific[(char_key, ident)] = action_id

    return name_to_id, char_specific


def subaction_name_to_action_ident(subaction_name: str) -> str | None:
    """Extract action ident from meleedb subactionName.

    Example: PlyFox5K_Share_ACTION_AttackAirF_figatree → ATTACK_AIR_F
             PlyFox5K_Share_ACTION_Attack11_figatree → ATTACK_11
             PlyFox5K_Share_ACTION_AttackS3Hi_figatree → ATTACK_S_3_HI

    The conversion is:
      AttackAirF → ATTACK_AIR_F (insert _ at case/digit boundaries)
    """
    # Extract the action part between _ACTION_ and _figatree
    match = re.search(r"_ACTION_(.+?)_figatree", subaction_name)
    if not match:
        return None

    action_camel = match.group(1)  # e.g., "AttackAirF", "SpecialLwStart"

    # Convert CamelCase to UPPER_SNAKE_CASE
    # 1. Insert _ between lowercase/digit and uppercase: "attackAir" → "attack_Air"
    snake = re.sub(r"([a-z])([A-Z])", r"\1_\2", action_camel)
    # 2. Insert _ between letter and digit: "Attack11" → "Attack_11", "S3" → "S_3"
    snake = re.sub(r"([a-zA-Z])(\d)", r"\1_\2", snake)
    # 3. Insert _ between digit and letter: "3Hi" → "3_Hi", "100Loop" → "100_Loop"
    snake = re.sub(r"(\d)([a-zA-Z])", r"\1_\2", snake)
    # 4. Handle consecutive uppercase: "HiS" → "Hi_S" (but keep "FFW" together)
    snake = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", snake)
    return snake.upper()


# Meleedb uses simplified names for some moves; map to ssbm-data variants
SUBACTION_FIXUPS = {
    # Forward smash without angle → default to S (straight) variant
    "ATTACK_S_4": "ATTACK_S_4_S",
    "ATTACK_S_41": "ATTACK_S_4_S",  # Marth/Roy second hit naming quirk
    # Forward tilt without angle
    "ATTACK_S_3": "ATTACK_S_3_S",
    "ATTACK_S_31": "ATTACK_S_3_S",
}

# Character-specific special moves: meleedb SpecialX names → ssbm-data idents
# Only need this for characters whose specials appear in training data.
# Key: (meleedb_char_name_raw, meleedb_action_ident) → ssbm-data (char_key, ident)
SPECIAL_MOVE_MAP = {
    # Fox — meleedb names both SpecialAirXxx and SpecialXxxAir variants
    ("Fox", "SPECIAL_N_START"): ("Fox", "BLASTER_GROUND_STARTUP"),
    ("Fox", "SPECIAL_N_LOOP"): ("Fox", "BLASTER_GROUND_LOOP"),
    ("Fox", "SPECIAL_N_END"): ("Fox", "BLASTER_GROUND_END"),
    ("Fox", "SPECIAL_AIR_N_START"): ("Fox", "BLASTER_AIR_STARTUP"),
    ("Fox", "SPECIAL_AIR_N_LOOP"): ("Fox", "BLASTER_AIR_LOOP"),
    ("Fox", "SPECIAL_AIR_N_END"): ("Fox", "BLASTER_AIR_END"),
    ("Fox", "SPECIAL_S"): ("Fox", "ILLUSION_GROUND"),
    ("Fox", "SPECIAL_AIR_S"): ("Fox", "ILLUSION_AIR"),
    ("Fox", "SPECIAL_HI_HOLD"): ("Fox", "FIRE_FOX_GROUND_STARTUP"),
    ("Fox", "SPECIAL_AIR_HI_HOLD"): ("Fox", "FIRE_FOX_AIR_STARTUP"),
    ("Fox", "SPECIAL_HI_HOLD_AIR"): ("Fox", "FIRE_FOX_AIR_STARTUP"),  # meleedb alt naming
    ("Fox", "SPECIAL_HI"): ("Fox", "FIRE_FOX_GROUND"),
    ("Fox", "SPECIAL_AIR_HI"): ("Fox", "FIRE_FOX_AIR"),
    ("Fox", "SPECIAL_LW_START"): ("Fox", "REFLECTOR_GROUND_STARTUP"),
    ("Fox", "SPECIAL_AIR_LW_START"): ("Fox", "REFLECTOR_AIR_STARTUP"),
    ("Fox", "SPECIAL_LW_LOOP"): ("Fox", "REFLECTOR_GROUND_LOOP"),
    ("Fox", "SPECIAL_LW_TURN"): ("Fox", "REFLECTOR_GROUND_CHANGE_DIRECTION"),
    # Falco — same air naming variants
    ("Falco", "SPECIAL_N_START"): ("Falco", "BLASTER_GROUND_STARTUP"),
    ("Falco", "SPECIAL_N_LOOP"): ("Falco", "BLASTER_GROUND_LOOP"),
    ("Falco", "SPECIAL_N_END"): ("Falco", "BLASTER_GROUND_END"),
    ("Falco", "SPECIAL_AIR_N_START"): ("Falco", "BLASTER_AIR_STARTUP"),
    ("Falco", "SPECIAL_AIR_N_LOOP"): ("Falco", "BLASTER_AIR_LOOP"),
    ("Falco", "SPECIAL_AIR_N_END"): ("Falco", "BLASTER_AIR_END"),
    ("Falco", "SPECIAL_S"): ("Falco", "PHANTASM_GROUND"),
    ("Falco", "SPECIAL_AIR_S"): ("Falco", "PHANTASM_AIR"),
    ("Falco", "SPECIAL_HI"): ("Falco", "FIRE_BIRD_GROUND"),
    ("Falco", "SPECIAL_AIR_HI"): ("Falco", "FIRE_BIRD_AIR"),
    ("Falco", "SPECIAL_HI_HOLD"): ("Falco", "FIRE_BIRD_GROUND_STARTUP"),
    ("Falco", "SPECIAL_HI_HOLD_AIR"): ("Falco", "FIRE_BIRD_AIR_STARTUP"),
    ("Falco", "SPECIAL_LW_START"): ("Falco", "REFLECTOR_GROUND_STARTUP"),
    ("Falco", "SPECIAL_AIR_LW_START"): ("Falco", "REFLECTOR_AIR_STARTUP"),
    ("Falco", "SPECIAL_LW_LOOP"): ("Falco", "REFLECTOR_GROUND_LOOP"),
    # Marth — ground specials
    ("Marth", "SPECIAL_N_1"): ("Marth", "SHIELD_BREAKER_GROUND_START_CHARGE"),
    ("Marth", "SPECIAL_N_2"): ("Marth", "SHIELD_BREAKER_GROUND_CHARGE_LOOP"),
    ("Marth", "SPECIAL_N_3"): ("Marth", "SHIELD_BREAKER_GROUND_EARLY_RELEASE"),
    ("Marth", "SPECIAL_N_4"): ("Marth", "SHIELD_BREAKER_GROUND_FULLY_CHARGED"),
    ("Marth", "SPECIAL_N_END"): ("Marth", "SHIELD_BREAKER_GROUND_EARLY_RELEASE"),
    ("Marth", "SPECIAL_N_LOOP"): ("Marth", "SHIELD_BREAKER_GROUND_CHARGE_LOOP"),
    ("Marth", "SPECIAL_S_1"): ("Marth", "DANCING_BLADE_1_GROUND"),
    ("Marth", "SPECIAL_S_2_HI"): ("Marth", "DANCING_BLADE_2_UP_GROUND"),
    ("Marth", "SPECIAL_S_2_LW"): ("Marth", "DANCING_BLADE_2_SIDE_GROUND"),
    ("Marth", "SPECIAL_S_3_HI"): ("Marth", "DANCING_BLADE_3_UP_GROUND"),
    ("Marth", "SPECIAL_S_3_S"): ("Marth", "DANCING_BLADE_3_SIDE_GROUND"),
    ("Marth", "SPECIAL_S_3_LW"): ("Marth", "DANCING_BLADE_3_DOWN_GROUND"),
    ("Marth", "SPECIAL_S_4_HI"): ("Marth", "DANCING_BLADE_4_UP_GROUND"),
    ("Marth", "SPECIAL_S_4_S"): ("Marth", "DANCING_BLADE_4_SIDE_GROUND"),
    ("Marth", "SPECIAL_S_4_LW"): ("Marth", "DANCING_BLADE_4_DOWN_GROUND"),
    ("Marth", "SPECIAL_HI"): ("Marth", "DOLPHIN_SLASH_GROUND"),
    ("Marth", "SPECIAL_LW"): ("Marth", "COUNTER_GROUND"),
    ("Marth", "SPECIAL_LW_HIT"): ("Marth", "COUNTER_GROUND_HIT"),
    # Marth — air specials (meleedb uses SpecialAirXxx naming)
    ("Marth", "SPECIAL_AIR_N_1"): ("Marth", "SHIELD_BREAKER_AIR_START_CHARGE"),
    ("Marth", "SPECIAL_AIR_N_2"): ("Marth", "SHIELD_BREAKER_AIR_CHARGE_LOOP"),
    ("Marth", "SPECIAL_AIR_N_3"): ("Marth", "SHIELD_BREAKER_AIR_EARLY_RELEASE"),
    ("Marth", "SPECIAL_AIR_N_4"): ("Marth", "SHIELD_BREAKER_AIR_FULLY_CHARGED"),
    ("Marth", "SPECIAL_AIR_N_END"): ("Marth", "SHIELD_BREAKER_AIR_EARLY_RELEASE"),
    ("Marth", "SPECIAL_AIR_N_LOOP"): ("Marth", "SHIELD_BREAKER_AIR_CHARGE_LOOP"),
    ("Marth", "SPECIAL_AIR_S_1"): ("Marth", "DANCING_BLADE_1_AIR"),
    ("Marth", "SPECIAL_AIR_S_2_HI"): ("Marth", "DANCING_BLADE_2_UP_AIR"),
    ("Marth", "SPECIAL_AIR_S_2_LW"): ("Marth", "DANCING_BLADE_2_SIDE_AIR"),
    ("Marth", "SPECIAL_AIR_S_3_HI"): ("Marth", "DANCING_BLADE_3_UP_AIR"),
    ("Marth", "SPECIAL_AIR_S_3_S"): ("Marth", "DANCING_BLADE_3_SIDE_AIR"),
    ("Marth", "SPECIAL_AIR_S_3_LW"): ("Marth", "DANCING_BLADE_3_DOWN_AIR"),
    ("Marth", "SPECIAL_AIR_S_4_HI"): ("Marth", "DANCING_BLADE_4_UP_AIR"),
    ("Marth", "SPECIAL_AIR_S_4_S"): ("Marth", "DANCING_BLADE_4_SIDE_AIR"),
    ("Marth", "SPECIAL_AIR_S_4_LW"): ("Marth", "DANCING_BLADE_4_DOWN_AIR"),
    ("Marth", "SPECIAL_AIR_HI"): ("Marth", "DOLPHIN_SLASH_AIR"),
    ("Marth", "SPECIAL_AIR_LW"): ("Marth", "COUNTER_AIR"),
    ("Marth", "SPECIAL_AIR_LW_HIT"): ("Marth", "COUNTER_AIR_HIT"),
    # Captain Falcon
    ("CaptainFalcon", "SPECIAL_N"): ("CaptainFalcon", "FALCON_PUNCH_GROUND"),
    ("CaptainFalcon", "SPECIAL_AIR_N"): ("CaptainFalcon", "FALCON_PUNCH_AIR"),
    ("CaptainFalcon", "SPECIAL_S"): ("CaptainFalcon", "RAPTOR_BOOST_GROUND"),
    ("CaptainFalcon", "SPECIAL_S_HIT"): ("CaptainFalcon", "RAPTOR_BOOST_GROUND_HIT"),
    ("CaptainFalcon", "SPECIAL_AIR_S"): ("CaptainFalcon", "RAPTOR_BOOST_AIR"),
    ("CaptainFalcon", "SPECIAL_HI"): ("CaptainFalcon", "FALCON_DIVE_GROUND"),
    ("CaptainFalcon", "SPECIAL_AIR_HI"): ("CaptainFalcon", "FALCON_DIVE_AIR"),
    ("CaptainFalcon", "SPECIAL_HI_CATCH"): ("CaptainFalcon", "FALCON_DIVE_CATCH"),
    ("CaptainFalcon", "SPECIAL_LW"): ("CaptainFalcon", "FALCON_KICK_GROUND"),
    ("CaptainFalcon", "SPECIAL_AIR_LW"): ("CaptainFalcon", "FALCON_KICK_AIR"),
    ("CaptainFalcon", "SPECIAL_AIR_LW_END"): ("CaptainFalcon", "FALCON_KICK_AIR_ENDING_IN_AIR"),
    # Sheik (Ply prefix "Seak" → ssbm key "Sheik")
    ("Sheik", "SPECIAL_N_START"): ("Sheik", "NEEDLE_STORM_GROUND_START_CHARGE"),
    ("Sheik", "SPECIAL_N_LOOP"): ("Sheik", "NEEDLE_STORM_GROUND_CHARGE_LOOP"),
    ("Sheik", "SPECIAL_N_CANCEL"): ("Sheik", "NEEDLE_STORM_GROUND_END_CHARGE"),
    ("Sheik", "SPECIAL_N_SHOOT"): ("Sheik", "NEEDLE_STORM_GROUND_FIRE"),
    ("Sheik", "SPECIAL_S"): ("Sheik", "CHAIN_GROUND_STARTUP"),
    ("Sheik", "SPECIAL_HI_START"): ("Sheik", "VANISH_GROUND_STARTUP"),
    ("Sheik", "SPECIAL_HI"): ("Sheik", "VANISH_GROUND_DISAPPEAR"),
    ("Sheik", "SPECIAL_AIR_HI_START"): ("Sheik", "VANISH_AIR_STARTUP"),
    ("Sheik", "SPECIAL_AIR_HI"): ("Sheik", "VANISH_AIR_DISAPPEAR"),
    ("Sheik", "SPECIAL_LW"): ("Sheik", "TRANSFORM_GROUND"),
    # Peach
    ("Peach", "SPECIAL_S"): ("Peach", "PEACH_BOMBER_GROUND"),
    ("Peach", "SPECIAL_AIR_S"): ("Peach", "PEACH_BOMBER_AIR"),
    ("Peach", "SPECIAL_S_HIT"): ("Peach", "PEACH_BOMBER_GROUND_HIT"),
    ("Peach", "SPECIAL_HI"): ("Peach", "PARASOL_GROUND_START"),
    ("Peach", "SPECIAL_AIR_HI"): ("Peach", "PARASOL_AIR_START"),
    ("Peach", "SPECIAL_LW"): ("Peach", "TOAD_GROUND"),
    ("Peach", "SPECIAL_AIR_LW"): ("Peach", "TOAD_AIR"),
    ("Peach", "SPECIAL_LW_HIT"): ("Peach", "TOAD_GROUND_ATTACK"),
    ("Peach", "SPECIAL_AIR_LW_HIT"): ("Peach", "TOAD_AIR_ATTACK"),
    # Jigglypuff — directional variants for sing/rest
    ("Jigglypuff", "SPECIAL_N"): ("Jigglypuff", "ROLLOUT_GROUND_START_CHARGE_RIGHT"),
    ("Jigglypuff", "SPECIAL_S"): ("Jigglypuff", "POUND_GROUND"),
    ("Jigglypuff", "SPECIAL_AIR_S"): ("Jigglypuff", "POUND_AIR"),
    ("Jigglypuff", "SPECIAL_HI_L"): ("Jigglypuff", "SING_GROUND_LEFT"),
    ("Jigglypuff", "SPECIAL_HI_R"): ("Jigglypuff", "SING_GROUND_LEFT"),
    ("Jigglypuff", "SPECIAL_AIR_HI_L"): ("Jigglypuff", "SING_GROUND_LEFT"),
    ("Jigglypuff", "SPECIAL_AIR_HI_R"): ("Jigglypuff", "SING_GROUND_LEFT"),
    ("Jigglypuff", "SPECIAL_HI"): ("Jigglypuff", "SING_GROUND_LEFT"),
    ("Jigglypuff", "SPECIAL_LW"): ("Jigglypuff", "REST_GROUND"),
    ("Jigglypuff", "SPECIAL_AIR_LW"): ("Jigglypuff", "REST_AIR"),
    ("Jigglypuff", "SPECIAL_LW_L"): ("Jigglypuff", "REST_GROUND"),
    ("Jigglypuff", "SPECIAL_LW_R"): ("Jigglypuff", "REST_GROUND"),
    ("Jigglypuff", "SPECIAL_AIR_LW_L"): ("Jigglypuff", "REST_AIR"),
    ("Jigglypuff", "SPECIAL_AIR_LW_R"): ("Jigglypuff", "REST_AIR"),
    # Samus
    ("Samus", "SPECIAL_HI"): ("Samus", "SCREW_ATTACK_GROUND"),
    ("Samus", "SPECIAL_AIR_HI"): ("Samus", "SCREW_ATTACK_AIR"),
    # Luigi
    ("Luigi", "SPECIAL_HI"): ("Luigi", "SUPER_JUMP_PUNCH_GROUND"),
    ("Luigi", "SPECIAL_AIR_HI"): ("Luigi", "SUPER_JUMP_PUNCH_AIR"),
    ("Luigi", "SPECIAL_LW"): ("Luigi", "CYCLONE_GROUND"),
    ("Luigi", "SPECIAL_AIR_LW"): ("Luigi", "CYCLONE_AIR"),
    ("Luigi", "SPECIAL_S"): ("Luigi", "GREEN_MISSILE_GROUND_STARTUP"),
    # Dr. Mario
    ("DrMario", "SPECIAL_HI"): ("DrMario", "SUPER_JUMP_PUNCH_GROUND"),
    ("DrMario", "SPECIAL_AIR_HI"): ("DrMario", "SUPER_JUMP_PUNCH_AIR"),
    ("DrMario", "SPECIAL_LW"): ("DrMario", "TORNADO_GROUND"),
    ("DrMario", "SPECIAL_AIR_LW"): ("DrMario", "TORNADO_AIR"),
    ("DrMario", "SPECIAL_S"): ("DrMario", "SUPER_SHEET_GROUND"),
    ("DrMario", "SPECIAL_S_AIR"): ("DrMario", "SUPER_SHEET_AIR"),  # meleedb SpecialSAir
    # Mario
    ("Mario", "SPECIAL_HI"): ("Mario", "SUPER_JUMP_PUNCH_GROUND"),
    ("Mario", "SPECIAL_AIR_HI"): ("Mario", "SUPER_JUMP_PUNCH_AIR"),
    ("Mario", "SPECIAL_LW"): ("Mario", "TORNADO_GROUND"),
    ("Mario", "SPECIAL_AIR_LW"): ("Mario", "TORNADO_AIR"),
    ("Mario", "SPECIAL_S"): ("Mario", "CAPE_GROUND"),
    ("Mario", "SPECIAL_S_AIR"): ("Mario", "CAPE_AIR"),  # meleedb SpecialSAir
    # Ganondorf (clone of Falcon with different move names)
    ("Ganondorf", "SPECIAL_N"): ("Ganondorf", "WARLOCK_PUNCH_GROUND"),
    ("Ganondorf", "SPECIAL_AIR_N"): ("Ganondorf", "WARLOCK_PUNCH_AIR"),
    ("Ganondorf", "SPECIAL_S"): ("Ganondorf", "GERUDO_DRAGON_GROUND"),
    ("Ganondorf", "SPECIAL_AIR_S"): ("Ganondorf", "GERUDO_DRAGON_AIR"),
    ("Ganondorf", "SPECIAL_HI"): ("Ganondorf", "DARK_DIVE_GROUND"),
    ("Ganondorf", "SPECIAL_AIR_HI"): ("Ganondorf", "DARK_DIVE_AIR"),
    ("Ganondorf", "SPECIAL_HI_CATCH"): ("Ganondorf", "DARK_DIVE_CATCH"),
    ("Ganondorf", "SPECIAL_LW"): ("Ganondorf", "WIZARDS_FOOT_GROUND"),
    ("Ganondorf", "SPECIAL_AIR_LW_END"): ("Ganondorf", "WIZARDS_FOOT_AIR_ENDING_IN_AIR"),
    # Roy (Emblem — clone of Marth)
    ("Roy", "SPECIAL_N_END"): ("Roy", "FLARE_BLADE_GROUND_EARLY_RELEASE"),
    ("Roy", "SPECIAL_N_LOOP"): ("Roy", "FLARE_BLADE_GROUND_CHARGE_LOOP"),
    ("Roy", "SPECIAL_AIR_N_END"): ("Roy", "FLARE_BLADE_AIR_EARLY_RELEASE"),
    ("Roy", "SPECIAL_AIR_N_LOOP"): ("Roy", "FLARE_BLADE_AIR_CHARGE_LOOP"),
    ("Roy", "SPECIAL_S_1"): ("Roy", "DOUBLE_EDGE_DANCE_1_GROUND"),
    ("Roy", "SPECIAL_AIR_S_1"): ("Roy", "DOUBLE_EDGE_DANCE_1_AIR"),
    ("Roy", "SPECIAL_S_2_HI"): ("Roy", "DOUBLE_EDGE_DANCE_2_UP_GROUND"),
    ("Roy", "SPECIAL_S_2_LW"): ("Roy", "DOUBLE_EDGE_DANCE_2_SIDE_GROUND"),
    ("Roy", "SPECIAL_AIR_S_2_HI"): ("Roy", "DOUBLE_EDGE_DANCE_2_UP_AIR"),
    ("Roy", "SPECIAL_AIR_S_2_LW"): ("Roy", "DOUBLE_EDGE_DANCE_2_SIDE_AIR"),
    ("Roy", "SPECIAL_S_3_HI"): ("Roy", "DOUBLE_EDGE_DANCE_3_UP_GROUND"),
    ("Roy", "SPECIAL_S_3_S"): ("Roy", "DOUBLE_EDGE_DANCE_3_SIDE_GROUND"),
    ("Roy", "SPECIAL_S_3_LW"): ("Roy", "DOUBLE_EDGE_DANCE_3_DOWN_GROUND"),
    ("Roy", "SPECIAL_AIR_S_3_HI"): ("Roy", "DOUBLE_EDGE_DANCE_3_UP_AIR"),
    ("Roy", "SPECIAL_AIR_S_3_S"): ("Roy", "DOUBLE_EDGE_DANCE_3_SIDE_AIR"),
    ("Roy", "SPECIAL_AIR_S_3_LW"): ("Roy", "DOUBLE_EDGE_DANCE_3_DOWN_AIR"),
    ("Roy", "SPECIAL_S_4_HI"): ("Roy", "DOUBLE_EDGE_DANCE_4_UP_GROUND"),
    ("Roy", "SPECIAL_S_4_S"): ("Roy", "DOUBLE_EDGE_DANCE_4_SIDE_GROUND"),
    ("Roy", "SPECIAL_S_4_LW"): ("Roy", "DOUBLE_EDGE_DANCE_4_DOWN_GROUND"),
    ("Roy", "SPECIAL_AIR_S_4_HI"): ("Roy", "DOUBLE_EDGE_DANCE_4_UP_AIR"),
    ("Roy", "SPECIAL_AIR_S_4_S"): ("Roy", "DOUBLE_EDGE_DANCE_4_SIDE_AIR"),
    ("Roy", "SPECIAL_AIR_S_4_LW"): ("Roy", "DOUBLE_EDGE_DANCE_4_DOWN_AIR"),
    ("Roy", "SPECIAL_HI"): ("Roy", "BLAZER_GROUND"),
    ("Roy", "SPECIAL_AIR_HI"): ("Roy", "BLAZER_AIR"),
    ("Roy", "SPECIAL_LW_HIT"): ("Roy", "COUNTER_GROUND_HIT"),
    ("Roy", "SPECIAL_AIR_LW"): ("Roy", "COUNTER_AIR"),
    ("Roy", "SPECIAL_AIR_LW_HIT"): ("Roy", "COUNTER_AIR_HIT"),
    # Link
    ("Link", "SPECIAL_HI"): ("Link", "SPIN_ATTACK_GROUND"),
    ("Link", "SPECIAL_AIR_HI"): ("Link", "SPIN_ATTACK_AIR"),
    # Young Link
    ("YoungLink", "SPECIAL_HI"): ("YoungLink", "SPIN_ATTACK_GROUND"),
    ("YoungLink", "SPECIAL_AIR_HI"): ("YoungLink", "SPIN_ATTACK_AIR"),
    # DK
    ("DonkeyKong", "SPECIAL_N"): ("DonkeyKong", "GIANT_PUNCH_GROUND_CHARGE_STARTUP"),
    ("DonkeyKong", "SPECIAL_AIR_N"): ("DonkeyKong", "GIANT_PUNCH_AIR_CHARGE_STARTUP"),
    ("DonkeyKong", "SPECIAL_S"): ("DonkeyKong", "HEADBUTT_GROUND"),
    ("DonkeyKong", "SPECIAL_AIR_S"): ("DonkeyKong", "HEADBUTT_AIR"),
    ("DonkeyKong", "SPECIAL_HI"): ("DonkeyKong", "SPINNING_KONG_GROUND"),
    ("DonkeyKong", "SPECIAL_AIR_HI"): ("DonkeyKong", "SPINNING_KONG_AIR"),
    ("DonkeyKong", "SPECIAL_LW_LOOP"): ("DonkeyKong", "HAND_SLAP_LOOP"),
    # Bowser
    ("Bowser", "SPECIAL_HI"): ("Bowser", "WHIRLING_FORTRESS_GROUND"),
    ("Bowser", "SPECIAL_AIR_HI"): ("Bowser", "WHIRLING_FORTRESS_AIR"),
    ("Bowser", "SPECIAL_S_START"): ("Bowser", "KOOPA_KLAW_GROUND"),
    ("Bowser", "SPECIAL_AIR_S_START"): ("Bowser", "KOOPA_KLAW_AIR"),
    ("Bowser", "SPECIAL_S_HIT"): ("Bowser", "KOOPA_KLAW_GROUND_GRAB"),
    ("Bowser", "SPECIAL_AIR_S_HIT"): ("Bowser", "KOOPA_KLAW_AIR_GRAB"),
    ("Bowser", "SPECIAL_S_END_F"): ("Bowser", "KOOPA_KLAW_GROUND_THROW_F"),
    ("Bowser", "SPECIAL_AIR_S_END_F"): ("Bowser", "KOOPA_KLAW_AIR_THROW_F"),
    ("Bowser", "SPECIAL_LW_LANDING"): ("Bowser", "BOMB_GROUND_END"),
    # Ness
    ("Ness", "SPECIAL_HI"): ("Ness", "PK_THUNDER_2"),
    # Zelda
    ("Zelda", "SPECIAL_N"): ("Zelda", "NAYRUS_LOVE_GROUND"),
    ("Zelda", "SPECIAL_AIR_N"): ("Zelda", "NAYRUS_LOVE_AIR"),
    ("Zelda", "SPECIAL_HI_START"): ("Zelda", "FARORES_WIND_GROUND"),
    ("Zelda", "SPECIAL_AIR_HI_START"): ("Zelda", "FARORES_WIND_AIR"),
    ("Zelda", "SPECIAL_HI"): ("Zelda", "FARORES_WIND_GROUND_DISAPPEAR"),
    ("Zelda", "SPECIAL_AIR_HI"): ("Zelda", "FARORES_WIND_AIR_DISAPPEAR"),
    ("Zelda", "SPECIAL_AIR_S_END"): ("Zelda", "DINS_FIRE_AIR_EXPLODE"),
    # Kirby
    ("Kirby", "SPECIAL_HI_2"): ("Kirby", "FINAL_CUTTER_GROUND_END"),
    ("Kirby", "SPECIAL_AIR_HI_2"): ("Kirby", "FINAL_CUTTER_AIR_END"),
    ("Kirby", "SPECIAL_S"): ("Kirby", "HAMMER_GROUND"),
    ("Kirby", "SPECIAL_AIR_S"): ("Kirby", "HAMMER_AIR"),
    # Pikachu
    ("Pikachu", "SPECIAL_S"): ("Pikachu", "SKULL_BASH_GROUND"),
    # Mewtwo
    ("Mewtwo", "SPECIAL_S"): ("Mewtwo", "CONFUSION_GROUND"),
    ("Mewtwo", "SPECIAL_AIR_S"): ("Mewtwo", "CONFUSION_AIR"),
    ("Mewtwo", "SPECIAL_N_LOOP"): ("Mewtwo", "SHADOW_BALL_GROUND_CHARGE_LOOP"),
    ("Mewtwo", "SPECIAL_AIR_N_LOOP"): ("Mewtwo", "SHADOW_BALL_AIR_CHARGE_LOOP"),
    ("Mewtwo", "SPECIAL_AIR_N_START"): ("Mewtwo", "SHADOW_BALL_AIR_START"),
    ("Mewtwo", "SPECIAL_AIR_N_END"): ("Mewtwo", "SHADOW_BALL_AIR_FULLY_CHARGED"),
    # Yoshi
    ("Yoshi", "SPECIAL_N_1"): ("Yoshi", "EGG_LAY_GROUND"),
    ("Yoshi", "SPECIAL_AIR_N_1"): ("Yoshi", "EGG_LAY_AIR"),
    ("Yoshi", "SPECIAL_S_LOOP"): ("Yoshi", "EGG_ROLL_GROUND_LOOP"),
}


# Ply prefix → ssbm-data character key (used for action_state.json lookups)
PLY_PREFIX_TO_SSBM_KEY = {
    "Captain": "CaptainFalcon",
    "Clink": "YoungLink",
    "Donkey": "DonkeyKong",
    "Drmario": "DrMario",
    "Emblem": "Roy",
    "Falco": "Falco",
    "Fox": "Fox",
    "Gamewatch": "GameAndWatch",
    "Ganon": "Ganondorf",
    "Kirby": "Kirby",
    "Koopa": "Bowser",
    "Link": "Link",
    "Luigi": "Luigi",
    "Mario": "Mario",
    "Mars": "Marth",
    "Mewtwo": "Mewtwo",
    "Nana": "Nana",
    "Ness": "Ness",
    "Peach": "Peach",
    "Pichu": "Pichu",
    "Pikachu": "Pikachu",
    "Popo": "Popo",
    "Purin": "Jigglypuff",
    "Samus": "Samus",
    "Seak": "Sheik",
    "Yoshi": "Yoshi",
    "Zelda": "Zelda",
}


def extract_char_from_subaction(subaction_name: str) -> str | None:
    """Extract ssbm-data character key from subactionName.

    Example: PlyFox5K_Share_ACTION_... → Fox
             PlyCaptain5K_Share_ACTION_... → CaptainFalcon
             PlyMars5K_Share_ACTION_... → Marth
    """
    match = re.match(r"Ply(\w+?)5K_", subaction_name)
    if not match:
        return None
    prefix = match.group(1)
    return PLY_PREFIX_TO_SSBM_KEY.get(prefix, prefix)


# ---------------------------------------------------------------------------
# Build the table
# ---------------------------------------------------------------------------

def build_hitbox_table(csv_path: Path, action_state_path: Path) -> dict:
    """Build the full hitbox lookup table.

    Returns dict with:
      - table: { "char_id:action_id:frame" → { damage, angle, bkb, kbg, ... } }
      - stats: { matched, unmatched, total_entries }
      - unmatched_actions: list of actions we couldn't map
    """
    common_map, char_specific_map = build_action_state_map(action_state_path)

    # Read CSV
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by (character, subactionName) to handle multi-hitbox moves
    moves = defaultdict(list)
    for row in rows:
        key = (row["character"], row["subactionName"])
        moves[key].append(row)

    table = {}
    matched = 0
    unmatched_actions = set()
    total_frame_entries = 0

    for (char_name_raw, subaction_name), hitbox_rows in moves.items():
        # Resolve character
        char_name = normalize_char_name(char_name_raw)
        if char_name is None:
            continue
        char_id = CHAR_NAME_TO_ID[char_name]

        # Resolve action state ID
        action_ident = subaction_name_to_action_ident(subaction_name)
        if action_ident is None:
            unmatched_actions.add(f"{char_name_raw}:{subaction_name}")
            continue

        # Apply fixups for simplified names
        action_ident = SUBACTION_FIXUPS.get(action_ident, action_ident)

        action_id = common_map.get(action_ident)
        if action_id is None:
            # Try character-specific via ssbm-data
            char_key = extract_char_from_subaction(subaction_name)
            if char_key:
                action_id = char_specific_map.get((char_key, action_ident))

        if action_id is None:
            # Try manual special move map
            char_key_raw = extract_char_from_subaction(subaction_name) or char_name_raw
            special_key = (char_key_raw, action_ident)
            if special_key in SPECIAL_MOVE_MAP:
                target_char, target_ident = SPECIAL_MOVE_MAP[special_key]
                action_id = char_specific_map.get((target_char, target_ident))

        if action_id is None:
            unmatched_actions.add(f"{char_name_raw}:{action_ident}")
            continue

        matched += 1

        # Get total frames and IASA for this move
        total_frames = int(hitbox_rows[0]["totalFrames"])
        iasa_raw = hitbox_rows[0]["iasa"]
        iasa = int(iasa_raw) if iasa_raw and iasa_raw != "NULL" else total_frames

        # Build per-frame hitbox data
        # For each frame, pick the strongest hitbox (highest damage)
        frame_data = {}
        for row in hitbox_rows:
            start = int(row["start"])
            end = int(row["end"])
            damage = float(row["damage"]) if row["damage"] and row["damage"] != "NULL" else 0
            angle = int(row["angle"]) if row["angle"] and row["angle"] != "NULL" else 0
            bkb = int(row["baseKb"]) if row["baseKb"] and row["baseKb"] != "NULL" else 0
            kbg = int(row["kbGrowth"]) if row["kbGrowth"] and row["kbGrowth"] != "NULL" else 0
            size = float(row["size"]) if row["size"] and row["size"] != "NULL" else 0
            element = row["element"] if row["element"] else "normal"

            for frame in range(start, end + 1):
                # Convert meleedb 1-indexed frames to 0-indexed (matching state_age)
                frame_0 = frame - 1
                if frame_0 not in frame_data or damage > frame_data[frame_0]["damage"]:
                    frame_data[frame_0] = {
                        "damage": damage,
                        "angle": angle,
                        "bkb": bkb,
                        "kbg": kbg,
                        "size": size,
                        "element": element,
                    }

        # Write to table: key = "char_id:action_id:state_age"
        # 0-indexed to match Melee's state_age (0 = first frame)
        # Include non-hitbox frames too (is_active=False) up to IASA
        for frame in range(0, iasa):
            key = f"{char_id}:{action_id}:{frame}"
            if frame in frame_data:
                entry = {**frame_data[frame], "is_active": True}
            else:
                entry = {
                    "damage": 0, "angle": 0, "bkb": 0, "kbg": 0,
                    "size": 0, "element": "none", "is_active": False,
                }
            entry["total_frames"] = total_frames
            entry["iasa"] = iasa
            table[key] = entry
            total_frame_entries += 1

    return {
        "table": table,
        "stats": {
            "moves_matched": matched,
            "moves_unmatched": len(unmatched_actions),
            "total_frame_entries": total_frame_entries,
            "unique_characters": len({k.split(":")[0] for k in table}),
        },
        "unmatched_actions": sorted(unmatched_actions),
    }


def fill_gaps_from_training_data(
    table: dict,
    training_dir: str | Path,
    n_games: int = 50,
) -> dict:
    """Fill gaps in the hitbox table using real training data.

    Scans training games to find (char_id, action_id) pairs in attack states
    that aren't in the table, and adds is_active=False entries for them.
    This guarantees 100% coverage.

    Also adds state_age clamping: if a (char_id, action_id) exists but
    the state_age exceeds the table's max frame, extends with is_active=False.

    Args:
        table: The existing hitbox table dict.
        training_dir: Path to parsed training data (games/ directory).
        n_games: Number of games to scan.

    Returns:
        Dict with gap_fills count and overflow_fills count.
    """
    import glob
    import numpy as np

    # Import the game loader
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from worldmodel.data.parse import load_game

    games_dir = Path(training_dir) / "games"
    files = sorted(glob.glob(str(games_dir / "*")))[:n_games]

    # Collect all (char_id, action_id, max_state_age) from training data
    # Only for attack states: 44-69 normals+aerials, 212-227 grabs/throws, 341+ specials
    # Exclude: 70-74 landing lag (no hitbox), 228-240 being-thrown (not attacks)
    seen_pairs: dict[tuple[int, int], int] = {}  # (cid, aid) → max_state_age

    for fp in files:
        try:
            game = load_game(fp)
        except Exception:
            continue

        for player in [game.p0, game.p1]:
            cid = int(player.character[0])
            actions = player.action.astype(int)
            ages = player.state_age.astype(int)

            # Attack filter (excludes landing lag 70-74 and being-thrown 228-240)
            is_attack = (
                ((actions >= 44) & (actions <= 69))
                | ((actions >= 212) & (actions <= 227))
                | (actions >= 341)
            )
            attack_idx = np.where(is_attack)[0]

            for i in attack_idx:
                aid = int(actions[i])
                age = int(ages[i])
                pair = (cid, aid)
                if pair not in seen_pairs or age > seen_pairs[pair]:
                    seen_pairs[pair] = age

    gap_fills = 0
    overflow_fills = 0

    for (cid, aid), max_age in seen_pairs.items():
        # Find what frames exist for this (char, action) in the table
        existing_frames = set()
        for frame in range(0, max_age + 2):  # +2 for safety
            if f"{cid}:{aid}:{frame}" in table:
                existing_frames.add(frame)

        if not existing_frames:
            # Completely missing — add is_active=False for all frames
            for frame in range(0, min(max_age + 1, 121)):  # cap at 120 frames
                key = f"{cid}:{aid}:{frame}"
                table[key] = {
                    "damage": 0, "angle": 0, "bkb": 0, "kbg": 0,
                    "size": 0, "element": "none", "is_active": False,
                    "total_frames": max_age, "iasa": max_age,
                }
                gap_fills += 1
        else:
            # Exists but might have state_age overflow — extend with is_active=False
            max_existing = max(existing_frames)
            if max_age > max_existing:
                for frame in range(max_existing + 1, min(max_age + 1, 301)):
                    key = f"{cid}:{aid}:{frame}"
                    if key not in table:
                        table[key] = {
                            "damage": 0, "angle": 0, "bkb": 0, "kbg": 0,
                            "size": 0, "element": "none", "is_active": False,
                            "total_frames": max_age, "iasa": max_existing,
                        }
                        overflow_fills += 1

    return {"gap_fills": gap_fills, "overflow_fills": overflow_fills}


# ---------------------------------------------------------------------------
# Throw data (not in meleedb — throws use hardcoded game mechanics)
# Source: Melee frame data wiki, ikneedata.com
# ---------------------------------------------------------------------------

# Common throw action state IDs (from ssbm-data Common action_state.json)
# 212: CATCH, 213: CATCH_PULL, 214: CATCH_DASH, 216: CATCH_WAIT
# 217: CATCH_ATTACK (pummel), 218: CATCH_CUT
# 219: THROW_F, 220: THROW_B, 221: THROW_HI, 222: THROW_LW

THROW_DATA = {
    # Fox throws: known frame data
    # (char_id, action_id): {damage, angle, bkb, kbg, total_frames}
    (1, 219): {"damage": 4, "angle": 45, "bkb": 55, "kbg": 105, "total_frames": 25},  # Fox fthrow
    (1, 220): {"damage": 2, "angle": 135, "bkb": 70, "kbg": 50, "total_frames": 30},  # Fox bthrow
    (1, 221): {"damage": 2, "angle": 88, "bkb": 70, "kbg": 60, "total_frames": 30},   # Fox uthrow (famous chaingrab)
    (1, 222): {"damage": 1, "angle": 0, "bkb": 60, "kbg": 80, "total_frames": 30},    # Fox dthrow
    # Falco throws
    (22, 219): {"damage": 4, "angle": 45, "bkb": 60, "kbg": 90, "total_frames": 25},  # Falco fthrow
    (22, 220): {"damage": 2, "angle": 135, "bkb": 70, "kbg": 50, "total_frames": 30},  # Falco bthrow
    (22, 221): {"damage": 2, "angle": 88, "bkb": 70, "kbg": 60, "total_frames": 30},
    (22, 222): {"damage": 1, "angle": 0, "bkb": 60, "kbg": 80, "total_frames": 30},
    # Marth throws
    (18, 219): {"damage": 4, "angle": 45, "bkb": 40, "kbg": 170, "total_frames": 25},  # Marth fthrow
    (18, 220): {"damage": 4, "angle": 45, "bkb": 50, "kbg": 130, "total_frames": 30},  # Marth bthrow
    (18, 221): {"damage": 4, "angle": 88, "bkb": 60, "kbg": 130, "total_frames": 30},  # Marth uthrow
    (18, 222): {"damage": 1, "angle": 0, "bkb": 20, "kbg": 100, "total_frames": 30},
    # Captain Falcon throws
    (2, 219): {"damage": 5, "angle": 45, "bkb": 20, "kbg": 128, "total_frames": 30},  # Falcon fthrow
    (2, 220): {"damage": 5, "angle": 45, "bkb": 50, "kbg": 128, "total_frames": 30},  # Falcon bthrow
    (2, 221): {"damage": 4, "angle": 88, "bkb": 70, "kbg": 100, "total_frames": 35},
    (2, 222): {"damage": 7, "angle": 0, "bkb": 80, "kbg": 40, "total_frames": 30},
    # Sheik throws
    (7, 219): {"damage": 5, "angle": 45, "bkb": 20, "kbg": 110, "total_frames": 25},  # Sheik fthrow
    (7, 220): {"damage": 5, "angle": 130, "bkb": 50, "kbg": 128, "total_frames": 30},  # Sheik bthrow
    (7, 221): {"damage": 3, "angle": 88, "bkb": 55, "kbg": 170, "total_frames": 30},
    (7, 222): {"damage": 2, "angle": 270, "bkb": 40, "kbg": 220, "total_frames": 30},  # Sheik dthrow (chain grab)
    # Peach throws
    (9, 219): {"damage": 2, "angle": 80, "bkb": 70, "kbg": 90, "total_frames": 25},  # Peach fthrow
    (9, 220): {"damage": 2, "angle": 135, "bkb": 70, "kbg": 90, "total_frames": 30},  # Peach bthrow
    (9, 221): {"damage": 2, "angle": 90, "bkb": 90, "kbg": 68, "total_frames": 30},
    (9, 222): {"damage": 0, "angle": 0, "bkb": 0, "kbg": 0, "total_frames": 30},
}


def add_throw_data(table: dict) -> int:
    """Add throw hitbox entries to the table from hardcoded throw data."""
    added = 0
    for (cid, aid), props in THROW_DATA.items():
        tf = props["total_frames"]
        # Throws have a single "hit" frame (the throw release)
        # Approximate: active on frames 4-6 (0-indexed, typical throw release window)
        for frame in range(0, tf):
            key = f"{cid}:{aid}:{frame}"
            if key not in table:
                is_active = 4 <= frame <= 6  # approximate throw release window
                table[key] = {
                    "damage": props["damage"] if is_active else 0,
                    "angle": props["angle"] if is_active else 0,
                    "bkb": props["bkb"] if is_active else 0,
                    "kbg": props["kbg"] if is_active else 0,
                    "size": 0,
                    "element": "normal" if is_active else "none",
                    "is_active": is_active,
                    "total_frames": tf,
                    "iasa": tf,
                }
                added += 1
    return added


def main():
    base = Path(__file__).parent
    csv_path = base / "melee_hitboxes.csv"
    action_state_path = base / "action_state.json"
    output_path = base / "hitbox_table.json"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Download from meleedb first.")
        sys.exit(1)
    if not action_state_path.exists():
        print(f"ERROR: {action_state_path} not found. Download from ssbm-data first.")
        sys.exit(1)

    print("Building hitbox lookup table...")
    result = build_hitbox_table(csv_path, action_state_path)
    table = result["table"]

    stats = result["stats"]
    print(f"  Moves matched:       {stats['moves_matched']}")
    print(f"  Moves unmatched:     {stats['moves_unmatched']}")
    print(f"  Total frame entries: {stats['total_frame_entries']}")
    print(f"  Unique characters:   {stats['unique_characters']}")

    if result["unmatched_actions"]:
        print(f"\n  Unmatched actions ({len(result['unmatched_actions'])}):")
        for action in result["unmatched_actions"][:20]:
            print(f"    {action}")
        if len(result["unmatched_actions"]) > 20:
            print(f"    ... and {len(result['unmatched_actions']) - 20} more")

    # Add throw data
    print("\nAdding throw data...")
    throw_count = add_throw_data(table)
    print(f"  Added {throw_count} throw frame entries")

    # Fill gaps from training data (guarantees 100% coverage)
    training_dir = Path.home() / "claude-projects/nojohns-training/data/parsed-v3-2k"
    if training_dir.exists():
        print(f"\nFilling gaps from training data ({training_dir})...")
        fill_result = fill_gaps_from_training_data(table, training_dir, n_games=50)
        print(f"  Gap fills (new actions):   {fill_result['gap_fills']}")
        print(f"  Overflow fills (extended): {fill_result['overflow_fills']}")
    else:
        print(f"\nSkipping gap fill — training data not found at {training_dir}")

    # Update stats
    result["stats"]["total_frame_entries"] = len(table)
    result["stats"]["unique_characters"] = len({k.split(":")[0] for k in table})

    # Write output
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    size_kb = output_path.stat().st_size / 1024
    print(f"\nWrote {output_path} ({size_kb:.0f} KB)")
    print(f"  Final entry count: {len(table)}")

    # Spot-check: Fox ATTACK_AIR_F (nair = 65, fair = 66), 0-indexed state_age
    print("\n--- Spot check: Fox fair (char=1, action=66) ---")
    for age in range(0, 20):
        key = f"1:66:{age}"
        if key in table:
            entry = table[key]
            active = "ACTIVE" if entry["is_active"] else "      "
            print(f"  age {age:2d}: {active}  dmg={entry['damage']:5.1f}  angle={entry['angle']:3d}  bkb={entry['bkb']:3d}  kbg={entry['kbg']:3d}")

    print("\n--- Spot check: Fox upsmash (char=1, action=63) ---")
    for age in range(0, 25):
        key = f"1:63:{age}"
        if key in table:
            entry = table[key]
            active = "ACTIVE" if entry["is_active"] else "      "
            print(f"  age {age:2d}: {active}  dmg={entry['damage']:5.1f}  angle={entry['angle']:3d}  bkb={entry['bkb']:3d}  kbg={entry['kbg']:3d}")

    # Coverage test
    print("\n--- Coverage test (re-scan training data) ---")
    if training_dir.exists():
        import glob
        import numpy as np
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from worldmodel.data.parse import load_game

        files = sorted(glob.glob(str(training_dir / "games" / "*")))[:20]
        table_keys = set(table.keys())
        total_attack = 0
        total_hit = 0

        for fp in files:
            try:
                game = load_game(fp)
            except Exception:
                continue
            for player in [game.p0, game.p1]:
                cid = int(player.character[0])
                actions = player.action.astype(int)
                ages = player.state_age.astype(int)

                is_attack = (
                    ((actions >= 44) & (actions <= 69))
                    | ((actions >= 212) & (actions <= 227))
                    | (actions >= 341)
                )
                attack_idx = np.where(is_attack)[0]
                total_attack += len(attack_idx)
                for i in attack_idx:
                    key = f"{cid}:{int(actions[i])}:{int(ages[i])}"
                    if key in table_keys:
                        total_hit += 1

        if total_attack > 0:
            print(f"  Attack frames: {total_attack}, hits: {total_hit}, coverage: {total_hit/total_attack*100:.1f}%")


if __name__ == "__main__":
    main()
