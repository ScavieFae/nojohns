#!/usr/bin/env python3
"""Generate a simulated match by rolling out the world model autoregressively.

Seeds with K real frames from a parsed replay, then predicts forward
using either the replay's controller inputs or scripted inputs.

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

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worldmodel.data.dataset import CTRL_DIM, FLOAT_PER_PLAYER, INT_PER_PLAYER, _encode_game
from worldmodel.data.parse import load_game
from worldmodel.model.encoding import EncodingConfig
from worldmodel.model.mlp import FrameStackMLP

logger = logging.getLogger(__name__)


def decode_continuous(normalized: torch.Tensor, cfg: EncodingConfig) -> dict:
    """Decode a player's continuous floats back to game units.

    Input: (13,) — [percent, x, y, shield, 5×vel, state_age, hitlag, stocks, combo_count]
    """
    return {
        "percent": normalized[0].item() / cfg.percent_scale,
        "x": normalized[1].item() / cfg.xy_scale,
        "y": normalized[2].item() / cfg.xy_scale,
        "shield": normalized[3].item() / cfg.shield_scale,
        "speed_air_x": normalized[4].item() / cfg.velocity_scale,
        "speed_y": normalized[5].item() / cfg.velocity_scale,
        "speed_ground_x": normalized[6].item() / cfg.velocity_scale,
        "speed_attack_x": normalized[7].item() / cfg.velocity_scale,
        "speed_attack_y": normalized[8].item() / cfg.velocity_scale,
        "state_age": normalized[9].item() / cfg.state_age_scale,
        "hitlag": normalized[10].item() / cfg.hitlag_scale,
        "stocks": normalized[11].item() / cfg.stocks_scale,
        "combo_count": normalized[12].item() / cfg.combo_count_scale,
    }


def decode_frame(
    float_frame: torch.Tensor,
    int_frame: torch.Tensor,
    cfg: EncodingConfig,
) -> dict:
    """Decode a full frame tensor back to a visualizer-friendly dict.

    Args:
        float_frame: (58,) — [p0: cont(13)+bin(3)+ctrl(13), p1: same]
        int_frame: (15,) — categoricals per player + stage
    """
    FPP = FLOAT_PER_PLAYER  # 29

    p0_cont = decode_continuous(float_frame[0:13], cfg)
    p0_cont["facing"] = float_frame[13].item() > 0.5
    p0_cont["invulnerable"] = float_frame[14].item() > 0.5
    p0_cont["on_ground"] = float_frame[15].item() > 0.5

    p1_cont = decode_continuous(float_frame[FPP:FPP + 13], cfg)
    p1_cont["facing"] = float_frame[FPP + 13].item() > 0.5
    p1_cont["invulnerable"] = float_frame[FPP + 14].item() > 0.5
    p1_cont["on_ground"] = float_frame[FPP + 15].item() > 0.5

    IPP = INT_PER_PLAYER  # 7
    p0_cont["action"] = int_frame[0].item()
    p0_cont["jumps_left"] = int_frame[1].item()
    p0_cont["character"] = int_frame[2].item()

    p1_cont["action"] = int_frame[IPP].item()
    p1_cont["jumps_left"] = int_frame[IPP + 1].item()
    p1_cont["character"] = int_frame[IPP + 2].item()

    return {
        "p0": p0_cont,
        "p1": p1_cont,
        "stage": int_frame[14].item(),
    }


@torch.no_grad()
def rollout(
    model: FrameStackMLP,
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

    Args:
        model: Trained FrameStackMLP (v2.2, input-conditioned).
        float_data: (T, 58) — full game float tensors.
        int_data: (T, 15) — full game int tensors.
        cfg: Encoding config.
        max_frames: Max frames to generate.
        device: Compute device.

    Returns:
        List of decoded frame dicts.
    """
    model.eval()
    K = model.context_len
    FPP = FLOAT_PER_PLAYER  # 29
    T = min(max_frames + K, float_data.shape[0])

    # Start with real seed frames
    sim_floats = float_data[:K].clone()  # (K, 58)
    sim_ints = int_data[:K].clone()  # (K, 15)

    frames = []
    # Decode seed frames
    for i in range(K):
        frames.append(decode_frame(sim_floats[i], sim_ints[i], cfg))

    for t in range(K, T):
        # Context window: last K frames
        ctx_f = sim_floats[-K:].unsqueeze(0).to(device)  # (1, K, 58)
        ctx_i = sim_ints[-K:].unsqueeze(0).to(device)  # (1, K, 15)

        # Controller input for frame t (from replay data)
        if t < float_data.shape[0]:
            ctrl = torch.cat([
                float_data[t, 16:29],            # p0 controller (13)
                float_data[t, FPP + 16:FPP + 29],  # p1 controller (13)
            ]).unsqueeze(0).to(device)  # (1, 26)
        else:
            ctrl = torch.zeros(1, CTRL_DIM).to(device)  # neutral

        # Predict frame t's state given the controller input
        preds = model(ctx_f, ctx_i, ctrl)

        # --- Build frame t ---
        curr_float = sim_floats[-1].clone()
        next_float = curr_float.clone()

        # Apply continuous deltas (first 4 of each player block: percent, x, y, shield)
        delta = preds["continuous_delta"][0].cpu()  # (8,)
        next_float[0:4] += delta[0:4]    # p0 deltas
        next_float[FPP:FPP + 4] += delta[4:8]  # p1 deltas

        # Binary predictions (threshold logits)
        binary = (preds["binary_logits"][0].cpu() > 0).float()  # (6,)
        next_float[13:16] = binary[0:3]  # p0 binary
        next_float[FPP + 13:FPP + 16] = binary[3:6]  # p1 binary

        # Store the controller input in the frame (for context window)
        if t < float_data.shape[0]:
            next_float[16:29] = float_data[t, 16:29]
            next_float[FPP + 16:FPP + 29] = float_data[t, FPP + 16:FPP + 29]

        # Categorical predictions (argmax)
        next_int = sim_ints[-1].clone()
        IPP = INT_PER_PLAYER
        next_int[0] = preds["p0_action_logits"][0].cpu().argmax()
        next_int[1] = preds["p0_jumps_logits"][0].cpu().argmax()
        next_int[IPP] = preds["p1_action_logits"][0].cpu().argmax()
        next_int[IPP + 1] = preds["p1_jumps_logits"][0].cpu().argmax()
        # character, l_cancel, hurtbox, ground, last_attack, stage: carry forward

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

    cfg = EncodingConfig()

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_cfg = checkpoint.get("config", {})
    model = FrameStackMLP(
        cfg=cfg,
        context_len=model_cfg.get("context_len", 10),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        trunk_dim=model_cfg.get("trunk_dim", 256),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    logger.info("Loaded model from %s (epoch %d)", args.checkpoint, checkpoint.get("epoch", -1))

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
