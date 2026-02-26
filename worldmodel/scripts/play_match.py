#!/usr/bin/env python3
"""Two agents playing in the world model.

Policy models (or scripted agents) control players in a world model simulation.
The world model predicts physics, the policies choose actions.

Usage:
    # Two scripted agents (instant, no training needed):
    python -m worldmodel.scripts.play_match \
        --world-model checkpoints/best.pt \
        --seed-game data/parsed-v2/games/<md5> \
        --p0 random --p1 hold-forward \
        --max-frames 600 --output match.json

    # Trained policy agents:
    python -m worldmodel.scripts.play_match \
        --world-model checkpoints/best.pt \
        --seed-game data/parsed-v2/games/<md5> \
        --p0 policy:checkpoints/p0.pt --p1 policy:checkpoints/p1.pt \
        --max-frames 600 --output match.json

    # One trained, one scripted:
    python -m worldmodel.scripts.play_match \
        --world-model checkpoints/best.pt \
        --seed-game data/parsed-v2/games/<md5> \
        --p0 policy:checkpoints/p0.pt --p1 random \
        --max-frames 600 --output match.json

    # Use replay inputs for one player (hybrid test):
    python -m worldmodel.scripts.play_match \
        --world-model checkpoints/best.pt \
        --seed-game data/parsed-v2/games/<md5> \
        --p0 replay --p1 random \
        --max-frames 600 --output match.json

Outputs JSON compatible with viewer.html.
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
from worldmodel.scripts.generate_demo import (
    STAGE_GEOMETRY,
    _build_predicted_player,
    _resolve_character_name,
    load_model_from_checkpoint,
)
from worldmodel.scripts.rollout import clamp_frame, decode_frame

logger = logging.getLogger(__name__)


# --- Agent interfaces ---


class Agent:
    """Base class for agents that produce controller outputs."""

    def get_controller(
        self,
        float_ctx: torch.Tensor,
        int_ctx: torch.Tensor,
        cfg: EncodingConfig,
        t: int,
    ) -> torch.Tensor:
        """Return 13-float controller tensor: [main_x, main_y, c_x, c_y, shoulder, A..D_UP]."""
        raise NotImplementedError


class ReplayAgent(Agent):
    """Replays controller inputs from the original game data."""

    def __init__(self, float_data: torch.Tensor, player: int, cfg: EncodingConfig):
        self.float_data = float_data
        fp = cfg.float_per_player
        cd = cfg.continuous_dim
        bd = cfg.binary_dim
        ctrl_start = cd + bd
        ctrl_end = ctrl_start + cfg.controller_dim
        self.ctrl_slice = slice(
            player * fp + ctrl_start,
            player * fp + ctrl_end,
        )

    def get_controller(self, float_ctx, int_ctx, cfg, t):
        if t < self.float_data.shape[0]:
            return self.float_data[t, self.ctrl_slice].clone()
        return torch.zeros(cfg.controller_dim)


class RandomAgent(Agent):
    """Random controller inputs — stick in random directions, random buttons."""

    def get_controller(self, float_ctx, int_ctx, cfg, t):
        ctrl = torch.zeros(cfg.controller_dim)
        # Random stick positions [0, 1]
        ctrl[0:4] = torch.rand(4)
        # shoulder trigger [0, 1]
        ctrl[4] = torch.rand(1).item()
        # Random buttons (~10% press probability each)
        ctrl[5:13] = (torch.rand(8) > 0.9).float()
        return ctrl


class NoopAgent(Agent):
    """Does nothing — neutral stick, no buttons."""

    def get_controller(self, float_ctx, int_ctx, cfg, t):
        ctrl = torch.zeros(cfg.controller_dim)
        ctrl[0] = 0.5  # neutral main stick x
        ctrl[1] = 0.5  # neutral main stick y
        ctrl[2] = 0.5  # neutral c stick x
        ctrl[3] = 0.5  # neutral c stick y
        return ctrl


class HoldForwardAgent(Agent):
    """Holds forward + A — basic aggressive approach."""

    def get_controller(self, float_ctx, int_ctx, cfg, t):
        ctrl = torch.zeros(cfg.controller_dim)
        ctrl[0] = 1.0  # main stick full right
        ctrl[1] = 0.5  # neutral y
        ctrl[2] = 0.5  # neutral c
        ctrl[3] = 0.5
        # Press A every ~10 frames
        if t % 10 < 3:
            ctrl[5] = 1.0  # A button
        return ctrl


class PolicyAgent(Agent):
    """Trained imitation learning policy."""

    def __init__(self, checkpoint_path: str, cfg: EncodingConfig, device: str = "cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        from worldmodel.model.policy_mlp import PolicyMLP

        state_dict = checkpoint["model_state_dict"]

        hidden_dim = state_dict["trunk.0.weight"].shape[0]
        trunk_dim = state_dict["trunk.3.weight"].shape[0]
        input_dim = state_dict["trunk.0.weight"].shape[1]

        # Infer context_len from input dim and config
        embed_per_player = (
            cfg.action_embed_dim + cfg.jumps_embed_dim + cfg.character_embed_dim
            + cfg.l_cancel_embed_dim + cfg.hurtbox_embed_dim
            + cfg.ground_embed_dim + cfg.last_attack_embed_dim
        )
        if cfg.state_age_as_embed:
            embed_per_player += cfg.state_age_embed_dim
        frame_dim = cfg.float_per_player * 2 + 2 * embed_per_player + cfg.stage_embed_dim
        context_len = input_dim // frame_dim

        self.model = PolicyMLP(
            cfg=cfg,
            context_len=context_len,
            hidden_dim=hidden_dim,
            trunk_dim=trunk_dim,
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

        logger.info(
            "Loaded policy from %s (context=%d, hidden=%d, trunk=%d)",
            checkpoint_path, context_len, hidden_dim, trunk_dim,
        )

    @torch.no_grad()
    def get_controller(self, float_ctx, int_ctx, cfg, t):
        """Run policy model on context, return 13-float controller tensor."""
        ctx_f = float_ctx.unsqueeze(0).to(self.device)
        ctx_i = int_ctx.unsqueeze(0).to(self.device)

        preds = self.model(ctx_f, ctx_i)

        # analog_pred: (5,) — already [0,1] from sigmoid
        # button_logits: (8,) — threshold at 0
        analog = preds["analog_pred"][0].cpu()
        buttons = (preds["button_logits"][0].cpu() > 0).float()

        return torch.cat([analog, buttons])


def make_agent(
    spec: str,
    player: int,
    float_data: torch.Tensor,
    cfg: EncodingConfig,
    device: str,
) -> Agent:
    """Create an agent from a specification string.

    Formats:
        "replay"           — replay original inputs
        "random"           — random controller
        "noop"             — neutral stick, no buttons
        "hold-forward"     — hold right + occasional A
        "policy:<path>"    — trained policy checkpoint
    """
    if spec == "replay":
        return ReplayAgent(float_data, player, cfg)
    elif spec == "random":
        return RandomAgent()
    elif spec == "noop":
        return NoopAgent()
    elif spec == "hold-forward":
        return HoldForwardAgent()
    elif spec.startswith("policy:"):
        return PolicyAgent(spec[7:], cfg=cfg, device=device)
    else:
        raise ValueError(f"Unknown agent spec: {spec!r}. Use: replay, random, noop, hold-forward, policy:<path>")


# --- Play loop ---


@torch.no_grad()
def play_match(
    world_model: torch.nn.Module,
    float_data: torch.Tensor,
    int_data: torch.Tensor,
    cfg: EncodingConfig,
    p0_agent: Agent,
    p1_agent: Agent,
    max_frames: int = 600,
    device: str = "cpu",
    no_early_ko: bool = False,
) -> list[dict]:
    """Run a two-agent match inside the world model.

    Seeds with K real frames, then at each step:
    1. Both agents observe the current state and produce controller inputs
    2. World model receives both controllers and predicts the next frame
    3. Predictions are clamped to valid ranges and fed back as context

    Returns list of decoded frame dicts (compatible with viewer.html).
    """
    world_model.eval()
    K = world_model.context_len
    fp = cfg.float_per_player
    ipp = cfg.int_per_player
    cd = cfg.continuous_dim
    bd = cfg.binary_dim
    ctrl_start = cd + bd
    ctrl_end = ctrl_start + cfg.controller_dim

    # Velocity indices
    vel_start = cfg.core_continuous_dim
    vel_end = vel_start + cfg.velocity_dim

    # Dynamics indices
    dyn_start = vel_end + (0 if cfg.state_age_as_embed else 1)
    p0_dyn_idx = [dyn_start, dyn_start + 1, dyn_start + 2]
    p1_dyn_idx = [fp + i for i in p0_dyn_idx]

    # State_age index
    if not cfg.state_age_as_embed:
        p0_state_age_idx = vel_end
        p1_state_age_idx = fp + vel_end

    T = min(max_frames + K, float_data.shape[0])

    # Seed with real frames
    sim_floats = float_data[:K].clone()
    sim_ints = int_data[:K].clone()

    frames = []
    for i in range(K):
        frame = decode_frame(sim_floats[i], sim_ints[i], cfg)
        frames.append({
            "t": i,
            "source": "seed",
            "actual": frame,
            "predicted": None,
        })

    for t in range(K, T):
        # Context window: last K frames
        ctx_f = sim_floats[-K:]
        ctx_i = sim_ints[-K:]

        # Get controller inputs from both agents
        p0_ctrl = p0_agent.get_controller(ctx_f, ctx_i, cfg, t)
        p1_ctrl = p1_agent.get_controller(ctx_f, ctx_i, cfg, t)

        # Combine into world model conditioning (26 floats)
        next_ctrl = torch.cat([p0_ctrl, p1_ctrl]).unsqueeze(0).to(device)

        # Run world model
        preds = world_model(
            ctx_f.unsqueeze(0).to(device),
            ctx_i.unsqueeze(0).to(device),
            next_ctrl,
        )

        # --- Build next frame from predictions ---
        curr_float = sim_floats[-1].clone()
        next_float = curr_float.clone()

        # Continuous deltas (percent, x, y, shield)
        delta = preds["continuous_delta"][0].cpu()
        next_float[0:4] += delta[0:4]
        next_float[fp:fp + 4] += delta[4:8]

        # Velocity deltas
        if "velocity_delta" in preds:
            vel_d = preds["velocity_delta"][0].cpu()
            next_float[vel_start:vel_end] += vel_d[0:5]
            next_float[fp + vel_start:fp + vel_end] += vel_d[5:10]

        # Dynamics (absolute: hitlag, stocks, combo)
        if "dynamics_pred" in preds:
            dyn = preds["dynamics_pred"][0].cpu()
            for i, idx in enumerate(p0_dyn_idx):
                next_float[idx] = dyn[i]
            for i, idx in enumerate(p1_dyn_idx):
                next_float[idx] = dyn[3 + i]

        # Binary predictions
        binary = (preds["binary_logits"][0].cpu() > 0).float()
        next_float[cd:cd + bd] = binary[0:3]
        next_float[fp + cd:fp + cd + bd] = binary[3:6]

        # Store controller inputs in the frame (for context window)
        next_float[ctrl_start:ctrl_end] = p0_ctrl
        next_float[fp + ctrl_start:fp + ctrl_end] = p1_ctrl

        # Categorical predictions
        next_int = sim_ints[-1].clone()
        next_int[0] = preds["p0_action_logits"][0].cpu().argmax()
        next_int[1] = preds["p0_jumps_logits"][0].cpu().argmax()
        next_int[ipp] = preds["p1_action_logits"][0].cpu().argmax()
        next_int[ipp + 1] = preds["p1_jumps_logits"][0].cpu().argmax()

        # Combat context predictions
        if "p0_l_cancel_logits" in preds:
            next_int[3] = preds["p0_l_cancel_logits"][0].cpu().argmax()
            next_int[4] = preds["p0_hurtbox_logits"][0].cpu().argmax()
            next_int[5] = preds["p0_ground_logits"][0].cpu().argmax()
            next_int[6] = preds["p0_last_attack_logits"][0].cpu().argmax()
            next_int[ipp + 3] = preds["p1_l_cancel_logits"][0].cpu().argmax()
            next_int[ipp + 4] = preds["p1_hurtbox_logits"][0].cpu().argmax()
            next_int[ipp + 5] = preds["p1_ground_logits"][0].cpu().argmax()
            next_int[ipp + 6] = preds["p1_last_attack_logits"][0].cpu().argmax()

        # State_age: rules-based
        if cfg.state_age_as_embed:
            sa_int_idx = cfg.int_per_player - 1
            for sa, act_col in [(sa_int_idx, 0), (ipp + sa_int_idx, ipp)]:
                if next_int[act_col] == sim_ints[-1][act_col]:
                    next_int[sa] = min(int(sim_ints[-1][sa].item()) + 1, cfg.state_age_embed_vocab - 1)
                else:
                    next_int[sa] = 0
        else:
            for sa_idx, act_col in [(p0_state_age_idx, 0), (p1_state_age_idx, ipp)]:
                if next_int[act_col] == sim_ints[-1][act_col]:
                    next_float[sa_idx] += 1.0 * cfg.state_age_scale
                else:
                    next_float[sa_idx] = 0.0

        # Clamp to valid ranges
        next_float = clamp_frame(next_float, cfg)

        # Append to simulation buffer
        sim_floats = torch.cat([sim_floats, next_float.unsqueeze(0)], dim=0)
        sim_ints = torch.cat([sim_ints, next_int.unsqueeze(0)], dim=0)

        # Decode for JSON output
        vel_d = preds["velocity_delta"][0].cpu() if "velocity_delta" in preds else None
        dyn = preds["dynamics_pred"][0].cpu() if "dynamics_pred" in preds else None

        pred_p0 = _build_predicted_player(
            prev_float=curr_float[:fp],
            cont_delta=delta[:4],
            binary=binary[:3],
            action_logits=preds["p0_action_logits"][0].cpu(),
            jumps_logits=preds["p0_jumps_logits"][0].cpu(),
            cfg=cfg,
            vel_delta=vel_d[:5] if vel_d is not None else None,
            dynamics=dyn[:3] if dyn is not None else None,
        )
        pred_p1 = _build_predicted_player(
            prev_float=curr_float[fp:2 * fp],
            cont_delta=delta[4:8],
            binary=binary[3:6],
            action_logits=preds["p1_action_logits"][0].cpu(),
            jumps_logits=preds["p1_jumps_logits"][0].cpu(),
            cfg=cfg,
            vel_delta=vel_d[5:10] if vel_d is not None else None,
            dynamics=dyn[3:6] if dyn is not None else None,
        )

        frames.append({
            "t": t,
            "source": "agent",
            "actual": None,
            "predicted": {"p0": pred_p0, "p1": pred_p1},
        })

        # Check for KO (optional — dynamics head hallucinates stock loss on weak models)
        frame = decode_frame(next_float, next_int, cfg)
        if not no_early_ko and (frame["p0"]["stocks"] < 0.5 or frame["p1"]["stocks"] < 0.5):
            logger.info("KO detected at frame %d", t)
            break

    return frames


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="Two agents playing in the world model")
    parser.add_argument("--world-model", required=True, help="Path to world model checkpoint")
    parser.add_argument("--seed-game", required=True, help="Path to parsed game for seeding")
    parser.add_argument("--p0", default="replay", help="P0 agent: replay, random, noop, hold-forward, policy:<path>")
    parser.add_argument("--p1", default="replay", help="P1 agent: replay, random, noop, hold-forward, policy:<path>")
    parser.add_argument("--max-frames", type=int, default=600, help="Max frames to simulate")
    parser.add_argument("--output", default="match.json", help="Output JSON path")
    parser.add_argument("--device", default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--no-early-ko", action="store_true", help="Don't stop on KO detection")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load world model
    world_model, cfg, context_len, arch = load_model_from_checkpoint(
        args.world_model, args.device,
    )
    logger.info("World model: %s (%s), context=%d", args.world_model, arch, context_len)

    # Load and encode seed game
    game = load_game(args.seed_game)
    float_data, int_data = _encode_game(game, cfg)
    logger.info("Seed game: %d frames", game.num_frames)

    # Create agents
    p0_agent = make_agent(args.p0, player=0, float_data=float_data, cfg=cfg, device=args.device)
    p1_agent = make_agent(args.p1, player=1, float_data=float_data, cfg=cfg, device=args.device)
    logger.info("P0: %s, P1: %s", args.p0, args.p1)

    # Play!
    frames = play_match(
        world_model, float_data, int_data, cfg,
        p0_agent, p1_agent,
        max_frames=args.max_frames,
        device=args.device,
        no_early_ko=args.no_early_ko,
    )
    logger.info("Generated %d frames", len(frames))

    # Build output JSON (viewer-compatible format)
    stage_id = game.stage
    p0_char_id = int(game.p0.character[0])
    p1_char_id = int(game.p1.character[0])

    stage_geo = STAGE_GEOMETRY.get(stage_id, {
        "name": f"Stage {stage_id}",
        "ground_y": 0, "ground_x_range": [-85, 85],
        "platforms": [],
        "blast_zones": {"left": -240, "right": 240, "top": 200, "bottom": -140},
        "camera_bounds": {"left": -160, "right": 160, "top": 100, "bottom": -50},
    })

    output = {
        "meta": {
            "mode": "agent-vs-agent",
            "model_name": Path(args.world_model).parent.name,
            "arch": arch,
            "context_len": context_len,
            "total_frames": len(frames),
            "seed_frames": context_len,
            "p0_agent": args.p0,
            "p1_agent": args.p1,
            "stage": {"id": stage_id, "name": stage_geo["name"]},
            "characters": {
                "p0": {"id": p0_char_id, "name": _resolve_character_name(p0_char_id)},
                "p1": {"id": p1_char_id, "name": _resolve_character_name(p1_char_id)},
            },
        },
        "stage_geometry": stage_geo,
        "frames": frames,
    }

    with open(args.output, "w") as f:
        json.dump(output, f)

    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.1f MB, %d frames)", args.output, size_mb, len(frames))

    # Quick summary
    agent_frames = [f for f in frames if f["source"] == "agent"]
    if agent_frames:
        last = agent_frames[-1]["predicted"]
        logger.info(
            "Final state: P0 stocks=%.0f pct=%.1f%% | P1 stocks=%.0f pct=%.1f%%",
            last["p0"]["stocks"], last["p0"]["percent"],
            last["p1"]["stocks"], last["p1"]["percent"],
        )


if __name__ == "__main__":
    main()
