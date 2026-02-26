#!/usr/bin/env python3
"""Generate demo JSON for the world model viewer.

Runs a trained world model on a replay and produces a self-contained JSON
file that the HTML viewer can load. Two modes:

  teacher-forced: feed real replay data as context every frame, compare
                  predictions to ground truth. Tests "how accurate is the model?"

  autoregressive: seed with K real frames, then feed the model's own predictions
                  back as context. Tests "can it imagine a match?"

Usage:
    # Teacher-forced (compare to ground truth):
    .venv/bin/python -m worldmodel.scripts.generate_demo \
        --checkpoint worldmodel/checkpoints/v22-overnight/best.pt \
        --seed-game ~/claude-projects/nojohns-training/data/parsed-v2/games/<md5> \
        --mode teacher-forced --max-frames 600 --output demo.json

    # Autoregressive (let the model hallucinate):
    .venv/bin/python -m worldmodel.scripts.generate_demo \
        --checkpoint worldmodel/checkpoints/v22-overnight/best.pt \
        --seed-game ~/claude-projects/nojohns-training/data/parsed-v2/games/<md5> \
        --mode autoregressive --max-frames 600 --output demo.json
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worldmodel.data.dataset import _encode_game
from worldmodel.data.parse import load_game
from worldmodel.model.encoding import EncodingConfig
from worldmodel.model.mlp import FrameStackMLP
from worldmodel.model.mamba2 import FrameStackMamba2

logger = logging.getLogger(__name__)


# --- Action / Character name resolution ---

def _resolve_action_name(action_id: int) -> str:
    try:
        from melee.enums import Action
        return Action(action_id).name
    except Exception:
        return f"ACTION_{action_id}"


def _resolve_character_name(char_id: int) -> str:
    try:
        from melee.enums import Character
        return Character(char_id).name
    except Exception:
        _FALLBACK = {
            0: "MARIO", 1: "FOX", 2: "CPTFALCON", 3: "DK", 4: "KIRBY",
            5: "BOWSER", 6: "LINK", 7: "SHEIK", 8: "NESS", 9: "PEACH",
            10: "POPO", 12: "PIKACHU", 13: "SAMUS", 14: "YOSHI",
            15: "JIGGLYPUFF", 16: "MEWTWO", 17: "LUIGI", 18: "MARTH",
            19: "ZELDA", 20: "YOUNGLINK", 21: "DOC", 22: "FALCO",
            23: "PICHU", 24: "GAMEANDWATCH", 25: "GANONDORF", 26: "ROY",
        }
        return _FALLBACK.get(char_id, f"CHAR_{char_id}")


# --- Stage geometry (legal tournament stages) ---

STAGE_GEOMETRY = {
    32: {
        "name": "Final Destination",
        "ground_y": 0, "ground_x_range": [-85.57, 85.57],
        "platforms": [],
        "blast_zones": {"left": -246, "right": 246, "top": 188, "bottom": -140},
        "camera_bounds": {"left": -160, "right": 160, "top": 100, "bottom": -50},
    },
    31: {
        "name": "Battlefield",
        "ground_y": 0, "ground_x_range": [-68.4, 68.4],
        "platforms": [
            {"x_range": [-57.6, -20], "y": 27.2},
            {"x_range": [20, 57.6], "y": 27.2},
            {"x_range": [-18.8, 18.8], "y": 54.4},
        ],
        "blast_zones": {"left": -224, "right": 224, "top": 200, "bottom": -108.8},
        "camera_bounds": {"left": -150, "right": 150, "top": 100, "bottom": -50},
    },
    3: {
        "name": "Pokemon Stadium",
        "ground_y": 0, "ground_x_range": [-87.75, 87.75],
        "platforms": [
            {"x_range": [-55, -25], "y": 25},
            {"x_range": [25, 55], "y": 25},
        ],
        "blast_zones": {"left": -230, "right": 230, "top": 200, "bottom": -111},
        "camera_bounds": {"left": -160, "right": 160, "top": 100, "bottom": -50},
    },
    8: {
        "name": "Yoshi's Story",
        "ground_y": 0, "ground_x_range": [-56, 56],
        "platforms": [
            {"x_range": [-60, -28], "y": 23.45},
            {"x_range": [28, 60], "y": 23.45},
            {"x_range": [-15.75, 15.75], "y": 42},
        ],
        "blast_zones": {"left": -175.7, "right": 173.6, "top": 168, "bottom": -91},
        "camera_bounds": {"left": -120, "right": 120, "top": 80, "bottom": -40},
    },
    28: {
        "name": "Dream Land N64",
        "ground_y": 0, "ground_x_range": [-77.27, 77.27],
        "platforms": [
            {"x_range": [-61.39, -31.73], "y": 30.14},
            {"x_range": [31.73, 63.03], "y": 30.14},
            {"x_range": [-19.02, 19.02], "y": 51.43},
        ],
        "blast_zones": {"left": -255, "right": 255, "top": 250, "bottom": -123},
        "camera_bounds": {"left": -170, "right": 170, "top": 100, "bottom": -50},
    },
    2: {
        "name": "Fountain of Dreams",
        "ground_y": 0, "ground_x_range": [-63.35, 63.35],
        "platforms": [
            {"x_range": [-50.5, -20.5], "y": 27.2},
            {"x_range": [20.5, 50.5], "y": 27.2},
            {"x_range": [-15, 15], "y": 42.75},
        ],
        "blast_zones": {"left": -198.75, "right": 198.75, "top": 202.5, "bottom": -146.25},
        "camera_bounds": {"left": -140, "right": 140, "top": 90, "bottom": -50},
    },
}


# --- Model loading ---

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[torch.nn.Module, EncodingConfig, int, str]:
    """Load a world model from a checkpoint file.

    Handles both old and new checkpoint formats, detects architecture
    (MLP vs Mamba-2) from weight keys, infers hyperparams from weight shapes.

    Returns:
        (model, encoding_config, context_len, arch_name)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # --- Reconstruct EncodingConfig ---
    if "encoding_config" in checkpoint and checkpoint["encoding_config"]:
        cfg = EncodingConfig(**checkpoint["encoding_config"])
        context_len = checkpoint.get("context_len", 10)
    elif "config" in checkpoint and checkpoint["config"]:
        # Old format: mixed keys in "config" dict
        old_cfg = checkpoint["config"]
        context_len = old_cfg.pop("context_len", 10)
        # Filter to only EncodingConfig fields
        enc_fields = {f.name for f in EncodingConfig.__dataclass_fields__.values()}
        enc_kwargs = {k: v for k, v in old_cfg.items() if k in enc_fields}
        cfg = EncodingConfig(**enc_kwargs)
    else:
        cfg = EncodingConfig()
        context_len = 10

    # --- Detect architecture from weight keys ---
    is_mamba2 = any("layers.0.mamba" in k for k in state_dict)

    if is_mamba2:
        d_model = state_dict["frame_proj.weight"].shape[0]
        layer_indices = set()
        for k in state_dict:
            if k.startswith("layers."):
                layer_indices.add(int(k.split(".")[1]))
        n_layers = len(layer_indices)

        d_inner = 2 * d_model  # expand=2 default
        nheads = state_dict["layers.0.mamba.A_log"].shape[0]
        headdim = d_inner // nheads
        conv_dim = state_dict["layers.0.mamba.conv1d.weight"].shape[0]
        d_state = (conv_dim - d_inner) // 2

        model = FrameStackMamba2(
            cfg=cfg,
            context_len=context_len,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            headdim=headdim,
            chunk_size=None,  # sequential for inference
        )
        arch = "mamba2"
        logger.info(
            "Detected Mamba-2: d_model=%d, d_state=%d, n_layers=%d, headdim=%d",
            d_model, d_state, n_layers, headdim,
        )
    else:
        hidden_dim = state_dict["trunk.0.weight"].shape[0]
        trunk_dim = state_dict["trunk.3.weight"].shape[0]
        model = FrameStackMLP(
            cfg=cfg,
            context_len=context_len,
            hidden_dim=hidden_dim,
            trunk_dim=trunk_dim,
        )
        arch = "mlp"
        logger.info(
            "Detected MLP: hidden_dim=%d, trunk_dim=%d",
            hidden_dim, trunk_dim,
        )

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", -1)
    logger.info("Loaded checkpoint: epoch %d, context_len=%d", epoch, context_len)

    return model, cfg, context_len, arch


# --- Frame decoding ---

def _decode_player_state(
    float_data: torch.Tensor,
    int_data: torch.Tensor,
    cfg: EncodingConfig,
) -> dict:
    """Decode one player's encoded tensors back to game-unit values.

    Args:
        float_data: (float_per_player,) — one player's float block
        int_data: (int_per_player,) — one player's int block
    """
    cd = cfg.continuous_dim
    stocks_idx = cd - 2  # stocks is second-to-last continuous value

    return {
        "x": round(float_data[1].item() / cfg.xy_scale, 2),
        "y": round(float_data[2].item() / cfg.xy_scale, 2),
        "percent": round(float_data[0].item() / cfg.percent_scale, 1),
        "shield": round(float_data[3].item() / cfg.shield_scale, 1),
        "stocks": round(float_data[stocks_idx].item() / cfg.stocks_scale),
        "facing_right": bool(float_data[cd].item() > 0.5),
        "on_ground": bool(float_data[cd + 2].item() > 0.5),
        "action": int(int_data[0].item()),
        "action_name": _resolve_action_name(int(int_data[0].item())),
        "jumps_left": int(int_data[1].item()),
    }


def decode_frame(
    float_frame: torch.Tensor,
    int_frame: torch.Tensor,
    cfg: EncodingConfig,
) -> dict:
    """Decode a full encoded frame to a JSON-friendly dict with p0 and p1."""
    fp = cfg.float_per_player
    ipp = cfg.int_per_player
    p0 = _decode_player_state(float_frame[:fp], int_frame[:ipp], cfg)
    p1 = _decode_player_state(float_frame[fp:2 * fp], int_frame[ipp:2 * ipp], cfg)
    return {"p0": p0, "p1": p1}


def _build_predicted_player(
    prev_float: torch.Tensor,
    cont_delta: torch.Tensor,
    binary: torch.Tensor,
    action_logits: torch.Tensor,
    jumps_logits: torch.Tensor,
    cfg: EncodingConfig,
    vel_delta: torch.Tensor | None = None,
    dynamics: torch.Tensor | None = None,
    actual_action: int | None = None,
    actual_x: float | None = None,
    actual_y: float | None = None,
) -> dict:
    """Build a predicted player dict from model outputs.

    Args:
        prev_float: (float_per_player,) — previous frame's float block (for this player)
        cont_delta: (4,) — predicted [percent_delta, x_delta, y_delta, shield_delta]
        binary: (3,) — predicted [facing, invuln, on_ground] as booleans
        action_logits: (action_vocab,) — raw logits
        jumps_logits: (jumps_vocab,) — raw logits
        cfg: encoding config
        vel_delta: (5,) — predicted velocity deltas (optional, from new head)
        dynamics: (3,) — predicted [hitlag, stocks, combo] absolute (optional, from new head)
        actual_action: ground truth action (for action_correct, teacher-forced only)
        actual_x, actual_y: ground truth position (for position_error)
    """
    # Apply delta to previous frame's core continuous values
    pred_percent = (prev_float[0].item() + cont_delta[0].item()) / cfg.percent_scale
    pred_x = (prev_float[1].item() + cont_delta[1].item()) / cfg.xy_scale
    pred_y = (prev_float[2].item() + cont_delta[2].item()) / cfg.xy_scale
    pred_shield = (prev_float[3].item() + cont_delta[3].item()) / cfg.shield_scale

    # Stocks: from dynamics head if available, otherwise carry forward from prev frame
    if dynamics is not None:
        # Clamp to sane ranges (random/untrained heads can output extreme values)
        pred_stocks = max(0, min(4, round(dynamics[1].item() / cfg.stocks_scale)))
        pred_hitlag = round(max(0, min(50, dynamics[0].item() / cfg.hitlag_scale)), 1)
        pred_combo = round(max(0, min(50, dynamics[2].item() / cfg.combo_count_scale)), 1)
    else:
        cd = cfg.continuous_dim
        stocks_idx = cd - 2
        pred_stocks = round(prev_float[stocks_idx].item() / cfg.stocks_scale)
        pred_hitlag = None
        pred_combo = None

    # Action: argmax
    pred_action = int(action_logits.argmax().item())
    pred_jumps = int(jumps_logits.argmax().item())

    # Top-3 actions
    probs = F.softmax(action_logits, dim=-1)
    top3_vals, top3_idx = probs.topk(3)
    top3_actions = [
        {
            "action": int(top3_idx[i].item()),
            "name": _resolve_action_name(int(top3_idx[i].item())),
            "prob": round(top3_vals[i].item(), 4),
        }
        for i in range(3)
    ]

    result = {
        "x": round(pred_x, 2),
        "y": round(pred_y, 2),
        "percent": round(pred_percent, 1),
        "shield": round(pred_shield, 1),
        "stocks": pred_stocks,
        "facing_right": bool(binary[0].item()),
        "on_ground": bool(binary[2].item()),
        "action": pred_action,
        "action_name": _resolve_action_name(pred_action),
        "jumps_left": pred_jumps,
        "top3_actions": top3_actions,
    }

    if vel_delta is not None:
        vel_s = cfg.velocity_scale
        result["velocity"] = {
            "speed_air_x": round((prev_float[4].item() + vel_delta[0].item()) / vel_s, 2),
            "speed_y": round((prev_float[5].item() + vel_delta[1].item()) / vel_s, 2),
            "speed_ground_x": round((prev_float[6].item() + vel_delta[2].item()) / vel_s, 2),
        }

    if pred_hitlag is not None:
        result["hitlag"] = pred_hitlag
        result["combo_count"] = pred_combo

    if actual_action is not None:
        result["action_correct"] = pred_action == actual_action

    if actual_x is not None and actual_y is not None:
        result["position_error"] = round(
            math.sqrt((pred_x - actual_x) ** 2 + (pred_y - actual_y) ** 2), 2
        )

    return result


# --- Prediction modes ---

@torch.no_grad()
def generate_teacher_forced(
    model: torch.nn.Module,
    float_data: torch.Tensor,
    int_data: torch.Tensor,
    cfg: EncodingConfig,
    max_frames: int,
    device: str,
) -> list[dict]:
    """Teacher-forced prediction: real context every frame."""
    K = model.context_len
    fp = cfg.float_per_player
    ipp = cfg.int_per_player
    cd = cfg.continuous_dim
    bd = cfg.binary_dim
    ctrl_start = cd + bd
    ctrl_end = ctrl_start + cfg.controller_dim

    T = min(K + max_frames, float_data.shape[0])
    frames = []

    # Seed frames (no predictions)
    for i in range(K):
        actual = decode_frame(float_data[i], int_data[i], cfg)
        frames.append({
            "t": i,
            "source": "seed",
            "actual": actual,
            "predicted": None,
        })

    # Predicted frames
    for t in range(K, T):
        # Context window (always ground truth)
        ctx_f = float_data[t - K:t].unsqueeze(0).to(device)
        ctx_i = int_data[t - K:t].unsqueeze(0).to(device)

        # Controller input for frame t
        ctrl = torch.cat([
            float_data[t, ctrl_start:ctrl_end],
            float_data[t, fp + ctrl_start:fp + ctrl_end],
        ]).unsqueeze(0).to(device)

        # Run model
        preds = model(ctx_f, ctx_i, ctrl)

        # Decode actual
        actual = decode_frame(float_data[t], int_data[t], cfg)

        # Decode predicted
        delta = preds["continuous_delta"][0].cpu()
        binary = (preds["binary_logits"][0].cpu() > 0).float()
        vel_d = preds["velocity_delta"][0].cpu() if "velocity_delta" in preds else None
        dyn = preds["dynamics_pred"][0].cpu() if "dynamics_pred" in preds else None

        pred_p0 = _build_predicted_player(
            prev_float=float_data[t - 1, :fp],
            cont_delta=delta[:4],
            binary=binary[:3],
            action_logits=preds["p0_action_logits"][0].cpu(),
            jumps_logits=preds["p0_jumps_logits"][0].cpu(),
            cfg=cfg,
            vel_delta=vel_d[:5] if vel_d is not None else None,
            dynamics=dyn[:3] if dyn is not None else None,
            actual_action=actual["p0"]["action"],
            actual_x=actual["p0"]["x"],
            actual_y=actual["p0"]["y"],
        )

        pred_p1 = _build_predicted_player(
            prev_float=float_data[t - 1, fp:2 * fp],
            cont_delta=delta[4:8],
            binary=binary[3:6],
            action_logits=preds["p1_action_logits"][0].cpu(),
            jumps_logits=preds["p1_jumps_logits"][0].cpu(),
            cfg=cfg,
            vel_delta=vel_d[5:10] if vel_d is not None else None,
            dynamics=dyn[3:6] if dyn is not None else None,
            actual_action=actual["p1"]["action"],
            actual_x=actual["p1"]["x"],
            actual_y=actual["p1"]["y"],
        )

        frames.append({
            "t": t,
            "source": "teacher-forced",
            "actual": actual,
            "predicted": {"p0": pred_p0, "p1": pred_p1},
        })

    return frames


@torch.no_grad()
def generate_autoregressive(
    model: torch.nn.Module,
    float_data: torch.Tensor,
    int_data: torch.Tensor,
    cfg: EncodingConfig,
    max_frames: int,
    device: str,
) -> list[dict]:
    """Autoregressive prediction: seed with K real frames, then self-sustain."""
    K = model.context_len
    fp = cfg.float_per_player
    ipp = cfg.int_per_player
    cd = cfg.continuous_dim
    bd = cfg.binary_dim
    ctrl_start = cd + bd
    ctrl_end = ctrl_start + cfg.controller_dim

    T = min(K + max_frames, float_data.shape[0])

    # Start with seed frames
    sim_floats = float_data[:K].clone()
    sim_ints = int_data[:K].clone()

    frames = []
    for i in range(K):
        actual = decode_frame(sim_floats[i], sim_ints[i], cfg)
        frames.append({
            "t": i,
            "source": "seed",
            "actual": actual,
            "predicted": None,
        })

    for t in range(K, T):
        # Context from simulation (model's own predictions after seed)
        ctx_f = sim_floats[-K:].unsqueeze(0).to(device)
        ctx_i = sim_ints[-K:].unsqueeze(0).to(device)

        # Controller input from replay (we always use real inputs)
        if t < float_data.shape[0]:
            ctrl = torch.cat([
                float_data[t, ctrl_start:ctrl_end],
                float_data[t, fp + ctrl_start:fp + ctrl_end],
            ]).unsqueeze(0).to(device)
        else:
            ctrl = torch.zeros(1, cfg.ctrl_conditioning_dim).to(device)

        # Run model
        preds = model(ctx_f, ctx_i, ctrl)

        # Build next frame from predictions
        prev_float = sim_floats[-1].clone()
        next_float = prev_float.clone()

        delta = preds["continuous_delta"][0].cpu()
        next_float[0:4] += delta[0:4]  # p0 core continuous deltas
        next_float[fp:fp + 4] += delta[4:8]  # p1 core continuous deltas

        # Velocity deltas
        vel_d = preds["velocity_delta"][0].cpu() if "velocity_delta" in preds else None
        if vel_d is not None:
            vel_start = cfg.core_continuous_dim
            vel_end_idx = vel_start + cfg.velocity_dim
            next_float[vel_start:vel_end_idx] += vel_d[0:5]
            next_float[fp + vel_start:fp + vel_end_idx] += vel_d[5:10]

        # Dynamics (absolute values: hitlag, stocks, combo)
        dyn = preds["dynamics_pred"][0].cpu() if "dynamics_pred" in preds else None
        if dyn is not None:
            dyn_start = cfg.core_continuous_dim + cfg.velocity_dim + (0 if cfg.state_age_as_embed else 1)
            for i in range(3):
                next_float[dyn_start + i] = dyn[i]
                next_float[fp + dyn_start + i] = dyn[3 + i]

        binary = (preds["binary_logits"][0].cpu() > 0).float()
        next_float[cd:cd + bd] = binary[0:3]  # p0 binary
        next_float[fp + cd:fp + cd + bd] = binary[3:6]  # p1 binary

        # Controller input from replay
        if t < float_data.shape[0]:
            next_float[ctrl_start:ctrl_end] = float_data[t, ctrl_start:ctrl_end]
            next_float[fp + ctrl_start:fp + ctrl_end] = float_data[t, fp + ctrl_start:fp + ctrl_end]

        # Categorical predictions
        next_int = sim_ints[-1].clone()
        next_int[0] = preds["p0_action_logits"][0].cpu().argmax()
        next_int[1] = preds["p0_jumps_logits"][0].cpu().argmax()
        next_int[ipp] = preds["p1_action_logits"][0].cpu().argmax()
        next_int[ipp + 1] = preds["p1_jumps_logits"][0].cpu().argmax()

        # Combat context categoricals (match rollout.py)
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
        if cfg.state_age_as_embed:
            # Integer embedding index: same logic, different storage
            sa_int_idx = cfg.int_per_player - 1  # state_age_int is last int column
            for sa, act_col in [(sa_int_idx, 0), (ipp + sa_int_idx, ipp)]:
                if next_int[act_col] == sim_ints[-1][act_col]:
                    next_int[sa] = min(int(sim_ints[-1][sa].item()) + 1, cfg.state_age_embed_vocab - 1)
                else:
                    next_int[sa] = 0
        else:
            sa_idx = cfg.core_continuous_dim + cfg.velocity_dim  # right after velocity
            for sa, act_col in [(sa_idx, 0), (fp + sa_idx, ipp)]:
                if next_int[act_col] == sim_ints[-1][act_col]:
                    next_float[sa] += 1.0 * cfg.state_age_scale
                else:
                    next_float[sa] = 0.0

        # Append to simulation buffer
        sim_floats = torch.cat([sim_floats, next_float.unsqueeze(0)], dim=0)
        sim_ints = torch.cat([sim_ints, next_int.unsqueeze(0)], dim=0)

        # Decode predicted state for JSON
        pred_p0 = _build_predicted_player(
            prev_float=prev_float[:fp],
            cont_delta=delta[:4],
            binary=binary[:3],
            action_logits=preds["p0_action_logits"][0].cpu(),
            jumps_logits=preds["p0_jumps_logits"][0].cpu(),
            cfg=cfg,
            vel_delta=vel_d[:5] if vel_d is not None else None,
            dynamics=dyn[:3] if dyn is not None else None,
        )
        pred_p1 = _build_predicted_player(
            prev_float=prev_float[fp:2 * fp],
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
            "source": "autoregressive",
            "actual": None,
            "predicted": {"p0": pred_p0, "p1": pred_p1},
        })

    return frames


# --- Summary metrics ---

def compute_summary(frames: list[dict], mode: str) -> dict:
    """Compute aggregate metrics from predicted frames."""
    predicted_frames = [f for f in frames if f["predicted"] is not None]
    if not predicted_frames:
        return {"total_frames_predicted": 0}

    total = len(predicted_frames)
    summary = {"total_frames_predicted": total}

    if mode == "teacher-forced":
        p0_correct = sum(1 for f in predicted_frames if f["predicted"]["p0"].get("action_correct"))
        p1_correct = sum(1 for f in predicted_frames if f["predicted"]["p1"].get("action_correct"))
        summary["p0_action_acc"] = round(p0_correct / total, 4)
        summary["p1_action_acc"] = round(p1_correct / total, 4)

        pos_errors = []
        for f in predicted_frames:
            for p in ["p0", "p1"]:
                err = f["predicted"][p].get("position_error")
                if err is not None:
                    pos_errors.append(err)
        if pos_errors:
            summary["avg_position_error"] = round(sum(pos_errors) / len(pos_errors), 2)

        # Action-change accuracy: only count frames where action changed
        change_correct = 0
        change_total = 0
        for f in predicted_frames:
            t = f["t"]
            # Find previous frame's action
            prev_idx = t - 1
            prev_frame = next((pf for pf in frames if pf["t"] == prev_idx), None)
            if prev_frame is None:
                continue
            prev_actual = prev_frame.get("actual")
            if prev_actual is None:
                continue
            for p in ["p0", "p1"]:
                if prev_actual[p]["action"] != f["actual"][p]["action"]:
                    change_total += 1
                    if f["predicted"][p].get("action_correct"):
                        change_correct += 1
        if change_total > 0:
            summary["action_change_acc"] = round(change_correct / change_total, 4)

    return summary


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Generate world model demo JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--seed-game", required=True, help="Path to parsed game file")
    parser.add_argument("--mode", choices=["teacher-forced", "autoregressive"],
                        default="teacher-forced", help="Prediction mode")
    parser.add_argument("--max-frames", type=int, default=600, help="Max frames to predict")
    parser.add_argument("--output", default="demo.json", help="Output JSON path")
    parser.add_argument("--device", default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load model
    model, cfg, context_len, arch = load_model_from_checkpoint(
        args.checkpoint, args.device,
    )

    # Load and encode game
    game = load_game(args.seed_game)
    float_data, int_data = _encode_game(game, cfg)
    logger.info(
        "Game: %d frames, stage=%d, p0_char=%d, p1_char=%d",
        game.num_frames, game.stage,
        int(game.p0.character[0]), int(game.p1.character[0]),
    )

    # Run prediction
    if args.mode == "teacher-forced":
        frames = generate_teacher_forced(
            model, float_data, int_data, cfg, args.max_frames, args.device,
        )
    else:
        frames = generate_autoregressive(
            model, float_data, int_data, cfg, args.max_frames, args.device,
        )

    # Build output JSON
    checkpoint_name = Path(args.checkpoint).parent.name
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

    summary = compute_summary(frames, args.mode)

    output = {
        "meta": {
            "mode": args.mode,
            "model_name": checkpoint_name,
            "arch": arch,
            "context_len": context_len,
            "total_frames": len(frames),
            "seed_frames": context_len,
            "stage": {"id": stage_id, "name": stage_geo["name"]},
            "characters": {
                "p0": {"id": p0_char_id, "name": _resolve_character_name(p0_char_id)},
                "p1": {"id": p1_char_id, "name": _resolve_character_name(p1_char_id)},
            },
        },
        "stage_geometry": stage_geo,
        "frames": frames,
        "summary": summary,
    }

    with open(args.output, "w") as f:
        json.dump(output, f)

    # Human-readable summary
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.1f MB, %d frames)", args.output, size_mb, len(frames))
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
