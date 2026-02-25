#!/usr/bin/env python3
"""Generate a simulated match by rolling out the world model autoregressively.

Seeds with K real frames from a parsed replay, then predicts forward
using either the replay's controller inputs or scripted inputs.

Supports both MLP and Mamba-2 architectures. Uses config-driven dimensions
so it works with any EncodingConfig (including state_age_as_embed, press_events).

Usage:
    # Rollout using a replay's real inputs (compare model vs reality):
    python -m worldmodel.scripts.rollout \
        --checkpoint checkpoints/v21/best.pt \
        --seed-game data/parsed-v2/games/<md5> \
        --max-frames 500

    # Output as JSON for the visualizer:
    python -m worldmodel.scripts.rollout \
        --checkpoint checkpoints/v21/best.pt \
        --seed-game data/parsed-v2/games/<md5> \
        --max-frames 500 --output rollout.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worldmodel.data.dataset import _encode_game
from worldmodel.data.parse import load_game
from worldmodel.model.encoding import EncodingConfig
from worldmodel.scripts.generate_demo import load_model_from_checkpoint

logger = logging.getLogger(__name__)


def decode_continuous(normalized: torch.Tensor, cfg: EncodingConfig) -> dict:
    """Decode a player's continuous floats back to game units.

    Config-driven: handles both default (13 values) and state_age_as_embed (12 values).
    """
    cd = cfg.continuous_dim
    result = {
        "percent": normalized[0].item() / cfg.percent_scale,
        "x": normalized[1].item() / cfg.xy_scale,
        "y": normalized[2].item() / cfg.xy_scale,
        "shield": normalized[3].item() / cfg.shield_scale,
        "speed_air_x": normalized[4].item() / cfg.velocity_scale,
        "speed_y": normalized[5].item() / cfg.velocity_scale,
        "speed_ground_x": normalized[6].item() / cfg.velocity_scale,
        "speed_attack_x": normalized[7].item() / cfg.velocity_scale,
        "speed_attack_y": normalized[8].item() / cfg.velocity_scale,
    }
    # Dynamics section varies with state_age_as_embed
    if not cfg.state_age_as_embed:
        result["state_age"] = normalized[9].item() / cfg.state_age_scale
        result["hitlag"] = normalized[10].item() / cfg.hitlag_scale
        result["stocks"] = normalized[11].item() / cfg.stocks_scale
        result["combo_count"] = normalized[12].item() / cfg.combo_count_scale
    else:
        result["hitlag"] = normalized[9].item() / cfg.hitlag_scale
        result["stocks"] = normalized[10].item() / cfg.stocks_scale
        result["combo_count"] = normalized[11].item() / cfg.combo_count_scale

    return result


def decode_frame(
    float_frame: torch.Tensor,
    int_frame: torch.Tensor,
    cfg: EncodingConfig,
) -> dict:
    """Decode a full frame tensor back to a visualizer-friendly dict.

    Config-driven: uses cfg.float_per_player and cfg.int_per_player for offsets.
    """
    fp = cfg.float_per_player
    cd = cfg.continuous_dim
    ipp = cfg.int_per_player

    p0_cont = decode_continuous(float_frame[0:cd], cfg)
    p0_cont["facing"] = float_frame[cd].item() > 0.5
    p0_cont["invulnerable"] = float_frame[cd + 1].item() > 0.5
    p0_cont["on_ground"] = float_frame[cd + 2].item() > 0.5

    p1_cont = decode_continuous(float_frame[fp:fp + cd], cfg)
    p1_cont["facing"] = float_frame[fp + cd].item() > 0.5
    p1_cont["invulnerable"] = float_frame[fp + cd + 1].item() > 0.5
    p1_cont["on_ground"] = float_frame[fp + cd + 2].item() > 0.5

    p0_cont["action"] = int_frame[0].item()
    p0_cont["jumps_left"] = int_frame[1].item()
    p0_cont["character"] = int_frame[2].item()

    p1_cont["action"] = int_frame[ipp].item()
    p1_cont["jumps_left"] = int_frame[ipp + 1].item()
    p1_cont["character"] = int_frame[ipp + 2].item()

    return {
        "p0": p0_cont,
        "p1": p1_cont,
        "stage": int_frame[ipp * 2].item(),
    }


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    float_data: torch.Tensor,
    int_data: torch.Tensor,
    cfg: EncodingConfig,
    max_frames: int = 500,
    device: str = "cpu",
) -> list[dict]:
    """Run autoregressive rollout (v2.2 — input-conditioned).

    Seeds with the first K frames from the real data, then predicts forward.
    Controller inputs from the replay are fed to the model as conditioning input,
    so the model predicts "what happens given these button presses."

    Config-driven: works with any EncodingConfig and both MLP/Mamba-2 architectures.

    Args:
        model: Trained world model (FrameStackMLP or FrameStackMamba2).
        float_data: (T, float_per_player*2) — full game float tensors.
        int_data: (T, int_per_frame) — full game int tensors.
        cfg: Encoding config.
        max_frames: Max frames to generate.
        device: Compute device.

    Returns:
        List of decoded frame dicts.
    """
    model.eval()
    K = model.context_len
    fp = cfg.float_per_player
    ipp = cfg.int_per_player
    cd = cfg.continuous_dim
    bd = cfg.binary_dim
    ctrl_start = cd + bd
    ctrl_end = ctrl_start + cfg.controller_dim
    T = min(max_frames + K, float_data.shape[0])

    # Velocity indices (4:9 within each player block)
    vel_start = cfg.core_continuous_dim   # 4
    vel_end = vel_start + cfg.velocity_dim  # 9

    # Dynamics indices: hitlag, stocks, combo_count
    dyn_start = vel_end + (0 if cfg.state_age_as_embed else 1)
    p0_dyn_idx = [dyn_start, dyn_start + 1, dyn_start + 2]
    p1_dyn_idx = [fp + i for i in p0_dyn_idx]

    # State_age index (for rules-based update)
    if not cfg.state_age_as_embed:
        p0_state_age_idx = vel_end  # right after velocity
        p1_state_age_idx = fp + vel_end

    # Start with real seed frames
    sim_floats = float_data[:K].clone()
    sim_ints = int_data[:K].clone()

    frames = []
    # Decode seed frames
    for i in range(K):
        frames.append(decode_frame(sim_floats[i], sim_ints[i], cfg))

    for t in range(K, T):
        # Context window: last K frames
        ctx_f = sim_floats[-K:].unsqueeze(0).to(device)
        ctx_i = sim_ints[-K:].unsqueeze(0).to(device)

        # Controller input for frame t (from replay data)
        if t < float_data.shape[0]:
            ctrl = torch.cat([
                float_data[t, ctrl_start:ctrl_end],
                float_data[t, fp + ctrl_start:fp + ctrl_end],
            ]).unsqueeze(0).to(device)
        else:
            ctrl = torch.zeros(1, cfg.ctrl_conditioning_dim).to(device)

        # Predict frame t's state given the controller input
        preds = model(ctx_f, ctx_i, ctrl)

        # --- Build frame t ---
        curr_float = sim_floats[-1].clone()
        next_float = curr_float.clone()

        # Apply continuous deltas (first 4 of each player block: percent, x, y, shield)
        delta = preds["continuous_delta"][0].cpu()
        next_float[0:4] += delta[0:4]  # p0 deltas
        next_float[fp:fp + 4] += delta[4:8]  # p1 deltas

        # Velocity deltas
        if "velocity_delta" in preds:
            vel_d = preds["velocity_delta"][0].cpu()
            next_float[vel_start:vel_end] += vel_d[0:5]        # p0 velocities
            next_float[fp + vel_start:fp + vel_end] += vel_d[5:10]  # p1 velocities

        # Dynamics (absolute values: hitlag, stocks, combo)
        if "dynamics_pred" in preds:
            dyn = preds["dynamics_pred"][0].cpu()
            for i, idx in enumerate(p0_dyn_idx):
                next_float[idx] = dyn[i]
            for i, idx in enumerate(p1_dyn_idx):
                next_float[idx] = dyn[3 + i]

        # Binary predictions (threshold logits)
        binary = (preds["binary_logits"][0].cpu() > 0).float()
        next_float[cd:cd + bd] = binary[0:3]  # p0 binary
        next_float[fp + cd:fp + cd + bd] = binary[3:6]  # p1 binary

        # Store the controller input in the frame (for context window)
        if t < float_data.shape[0]:
            next_float[ctrl_start:ctrl_end] = float_data[t, ctrl_start:ctrl_end]
            next_float[fp + ctrl_start:fp + ctrl_end] = float_data[t, fp + ctrl_start:fp + ctrl_end]

        # Categorical predictions (argmax)
        next_int = sim_ints[-1].clone()
        next_int[0] = preds["p0_action_logits"][0].cpu().argmax()
        next_int[1] = preds["p0_jumps_logits"][0].cpu().argmax()
        next_int[ipp] = preds["p1_action_logits"][0].cpu().argmax()
        next_int[ipp + 1] = preds["p1_jumps_logits"][0].cpu().argmax()
        # character, stage: carry forward (constant per game)
        # Combat context predictions (argmax)
        if "p0_l_cancel_logits" in preds:
            next_int[3] = preds["p0_l_cancel_logits"][0].cpu().argmax()
            next_int[4] = preds["p0_hurtbox_logits"][0].cpu().argmax()
            next_int[5] = preds["p0_ground_logits"][0].cpu().argmax()
            next_int[6] = preds["p0_last_attack_logits"][0].cpu().argmax()
            next_int[ipp + 3] = preds["p1_l_cancel_logits"][0].cpu().argmax()
            next_int[ipp + 4] = preds["p1_hurtbox_logits"][0].cpu().argmax()
            next_int[ipp + 5] = preds["p1_ground_logits"][0].cpu().argmax()
            next_int[ipp + 6] = preds["p1_last_attack_logits"][0].cpu().argmax()

        # State_age: rules-based (increment if same action, reset if changed)
        if not cfg.state_age_as_embed:
            for sa_idx, act_col in [(p0_state_age_idx, 0), (p1_state_age_idx, ipp)]:
                if next_int[act_col] == sim_ints[-1][act_col]:
                    next_float[sa_idx] += 1.0 * cfg.state_age_scale
                else:
                    next_float[sa_idx] = 0.0

        # Append to simulation
        sim_floats = torch.cat([sim_floats, next_float.unsqueeze(0)], dim=0)
        sim_ints = torch.cat([sim_ints, next_int.unsqueeze(0)], dim=0)

        frame = decode_frame(next_float, next_int, cfg)
        frames.append(frame)

        # Check for KO (stocks < 0.5)
        if frame["p0"]["stocks"] < 0.5 or frame["p1"]["stocks"] < 0.5:
            logger.info("KO detected at frame %d", t)
            break

    return frames


def main():
    parser = argparse.ArgumentParser(description="Roll out world model predictions")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--seed-game", required=True, help="Path to parsed game file for seeding")
    parser.add_argument("--max-frames", type=int, default=500, help="Max frames to generate")
    parser.add_argument("--device", default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: print summary)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load model (handles both architectures and checkpoint formats)
    model, cfg, context_len, arch = load_model_from_checkpoint(
        args.checkpoint, args.device,
    )
    logger.info("Loaded %s model from %s", arch, args.checkpoint)

    # Load and encode seed game
    game = load_game(args.seed_game)
    float_data, int_data = _encode_game(game, cfg)
    logger.info("Seed game: %d frames", game.num_frames)

    # Rollout
    frames = rollout(model, float_data, int_data, cfg,
                     max_frames=args.max_frames, device=args.device)
    logger.info("Generated %d frames", len(frames))

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"frames": frames, "num_frames": len(frames)}, f, indent=2)
        logger.info("Saved to %s", args.output)
    else:
        # Print summary: first, middle, last frame
        for label, idx in [("First", 0), ("Mid", len(frames) // 2), ("Last", -1)]:
            f = frames[idx]
            print(f"{label} (frame {idx if idx >= 0 else len(frames) + idx}):")
            print(f"  P0: x={f['p0']['x']:.1f} y={f['p0']['y']:.1f} "
                  f"pct={f['p0']['percent']:.1f}% stocks={f['p0']['stocks']:.1f} "
                  f"action={f['p0']['action']}")
            print(f"  P1: x={f['p1']['x']:.1f} y={f['p1']['y']:.1f} "
                  f"pct={f['p1']['percent']:.1f}% stocks={f['p1']['stocks']:.1f} "
                  f"action={f['p1']['action']}")


if __name__ == "__main__":
    main()
