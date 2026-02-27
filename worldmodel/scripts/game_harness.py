"""Game logic harness for world model rollouts.

Enforces basic Melee game rules on world model predictions:
- Blast zone KO: stock loss only when position crosses blast zones
- Respawn: teleport to center stage, zero percent/velocities
- Velocity clamping: cap speed components to physically plausible ranges
- Stock protection: override hallucinated stock decrements

The world model predicts physics but sometimes hallucinates KOs when
players are center stage. This harness ensures the demo looks right.
"""

import torch

from worldmodel.model.encoding import EncodingConfig


# Max velocity magnitude per component (game units, pre-scaling).
# Fox's max aerial speed is ~3.6; 5.0 gives comfortable headroom.
MAX_VELOCITY = 5.0


def apply_game_rules(
    next_float: torch.Tensor,
    next_int: torch.Tensor,
    prev_float: torch.Tensor,
    cfg: EncodingConfig,
    blast_zones: dict,
    respawn_x: float = 0.0,
    respawn_y: float = 40.0,
) -> dict:
    """Apply game rules to a predicted frame, mutating tensors in-place.

    Args:
        next_float: (float_per_player * 2 + ...) predicted float tensor
        next_int: (int_per_player * 2 + 1) predicted int tensor
        prev_float: previous frame's float tensor (for stock protection)
        cfg: encoding config
        blast_zones: dict with keys left, right, top, bottom (game units)
        respawn_x: respawn X position (game units)
        respawn_y: respawn Y position (game units)

    Returns:
        dict with event info: {"p0": {...}, "p1": {...}} where each has:
          - "ko": bool — did a KO happen this frame
          - "velocity_clamped": bool — were velocities clamped
          - "stock_protected": bool — was a hallucinated stock loss overridden
    """
    fp = cfg.float_per_player
    cd = cfg.continuous_dim
    bd = cfg.binary_dim

    # Velocity indices
    vel_start = cfg.core_continuous_dim
    vel_end = vel_start + cfg.velocity_dim

    # Dynamics indices (hitlag, stocks, combo)
    dyn_start = vel_end + (0 if cfg.state_age_as_embed else 1)
    stocks_idx = dyn_start + 1  # stocks is second in dynamics block

    events = {}

    for player, offset in [("p0", 0), ("p1", fp)]:
        ev = {"ko": False, "velocity_clamped": False, "stock_protected": False}

        # --- Read position (game units) ---
        x = next_float[offset + 1].item() / cfg.xy_scale
        y = next_float[offset + 2].item() / cfg.xy_scale

        # --- Read stocks ---
        cur_stocks = next_float[offset + stocks_idx].item() / cfg.stocks_scale
        prev_stocks = prev_float[offset + stocks_idx].item() / cfg.stocks_scale

        # --- Blast zone KO check ---
        in_blast = (
            x < blast_zones["left"]
            or x > blast_zones["right"]
            or y > blast_zones["top"]
            or y < blast_zones["bottom"]
        )

        if in_blast and cur_stocks >= 0.5:
            # Legitimate KO: player crossed blast zone
            new_stocks = max(0, round(prev_stocks) - 1)
            next_float[offset + stocks_idx] = new_stocks * cfg.stocks_scale

            # Respawn: teleport, zero percent, zero velocities
            next_float[offset + 0] = 0.0  # percent
            next_float[offset + 1] = respawn_x * cfg.xy_scale  # x
            next_float[offset + 2] = respawn_y * cfg.xy_scale  # y
            next_float[offset + 3] = 60.0 * cfg.shield_scale  # full shield

            # Zero velocities
            for vi in range(vel_start, vel_end):
                next_float[offset + vi] = 0.0

            # Set invulnerable (binary index 1)
            next_float[offset + cd + 1] = 1.0

            # Zero hitlag and combo
            next_float[offset + dyn_start] = 0.0  # hitlag
            next_float[offset + dyn_start + 2] = 0.0  # combo

            ev["ko"] = True

        elif not in_blast and round(cur_stocks) < round(prev_stocks):
            # Stock protection: model hallucinated a stock loss but player
            # isn't at a blast zone. Restore previous stock count.
            next_float[offset + stocks_idx] = prev_float[offset + stocks_idx].clone()
            ev["stock_protected"] = True

        # --- Velocity clamping ---
        clamped = False
        for vi in range(vel_start, vel_end):
            val = next_float[offset + vi].item() / cfg.velocity_scale
            if abs(val) > MAX_VELOCITY:
                next_float[offset + vi] = (
                    MAX_VELOCITY * (1 if val > 0 else -1) * cfg.velocity_scale
                )
                clamped = True
        ev["velocity_clamped"] = clamped

        events[player] = ev

    return events
