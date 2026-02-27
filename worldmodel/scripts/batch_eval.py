#!/usr/bin/env python3
"""Batch teacher-forced evaluation with error analysis.

Runs a trained world model in teacher-forced mode across many full games,
aggregating per-action, per-category, per-time-bucket, and binary flag error
statistics. Produces a compact JSON for the batch_eval_viewer.html.

Teacher-forced mode isolates prediction failures from compounding drift:
the model always sees real context, so any errors are pure prediction failures.

Usage:
    .venv/bin/python -m worldmodel.scripts.batch_eval \
        --checkpoint worldmodel/checkpoints/mamba2-v3-2k-test-v2/best.pt \
        --data-dir ~/claude-projects/nojohns-training/data/game-only-v3-2k \
        --num-games 30 --output worldmodel/batch_eval_results.json
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worldmodel.data.dataset import _encode_game
from worldmodel.data.parse import load_game
from worldmodel.scripts.generate_demo import (
    _build_predicted_player,
    _resolve_action_name,
    _resolve_character_name,
    decode_frame,
    generate_teacher_forced,
    load_model_from_checkpoint,
)
from worldmodel.training.metrics import ACTION_CATEGORIES, _ACTION_TO_CATEGORY, action_name

logger = logging.getLogger(__name__)


# Position error histogram bins
POS_ERROR_BINS = [0, 1, 2, 3, 5, 10, 20, float("inf")]
POS_ERROR_LABELS = ["[0,1)", "[1,2)", "[2,3)", "[3,5)", "[5,10)", "[10,20)", "[20+)"]

NUM_TIME_BUCKETS = 10


def _bin_position_error(error: float) -> int:
    """Return bin index for a position error value."""
    for i in range(len(POS_ERROR_BINS) - 1):
        if error < POS_ERROR_BINS[i + 1]:
            return i
    return len(POS_ERROR_BINS) - 2


def analyze_game(
    model: torch.nn.Module,
    game_path: Path,
    cfg,
    device: str,
    compression: str = "zlib",
) -> dict | None:
    """Run teacher-forced eval on one game, return per-frame error data.

    Returns a dict with aggregated stats for this game, or None on failure.
    """
    try:
        game = load_game(game_path, compression=compression)
    except Exception as e:
        logger.warning("Failed to load %s: %s", game_path, e)
        return None

    float_data, int_data = _encode_game(game, cfg)
    K = model.context_len
    T = float_data.shape[0]

    if T <= K + 1:
        logger.warning("Game too short (%d frames): %s", T, game_path)
        return None

    # Run teacher-forced prediction (full game, no frame cap)
    frames = generate_teacher_forced(
        model, float_data, int_data, cfg, max_frames=T, device=device,
    )

    predicted_frames = [f for f in frames if f["predicted"] is not None]
    if not predicted_frames:
        return None

    num_predicted = len(predicted_frames)
    fp = cfg.float_per_player
    cd = cfg.continuous_dim
    bd = cfg.binary_dim

    # Per-frame error extraction
    per_action_stats = defaultdict(lambda: {"correct": 0, "total": 0, "pos_errors": [],
                                            "change_correct": 0, "change_total": 0})
    time_buckets = [{"action_correct": 0, "action_total": 0,
                     "on_ground_correct": 0, "on_ground_total": 0,
                     "pos_errors": []} for _ in range(NUM_TIME_BUCKETS)]
    binary_confusion = {
        "on_ground": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "facing": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "invulnerable": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }
    pos_error_hist = [0] * len(POS_ERROR_LABELS)
    total_action_correct = 0
    total_change_correct = 0
    total_changes = 0
    all_pos_errors = []

    for i, frame in enumerate(predicted_frames):
        t = frame["t"]
        actual = frame["actual"]
        pred = frame["predicted"]

        # Time bucket (fraction through the predicted portion)
        frac = i / max(num_predicted - 1, 1)
        bucket = min(int(frac * NUM_TIME_BUCKETS), NUM_TIME_BUCKETS - 1)

        for player in ["p0", "p1"]:
            a = actual[player]
            p = pred[player]

            action_id = a["action"]
            pred_action = p["action"]
            is_correct = p.get("action_correct", pred_action == action_id)

            # Per-action stats
            stats = per_action_stats[action_id]
            stats["total"] += 1
            if is_correct:
                stats["correct"] += 1
                total_action_correct += 1

            # Position error
            pos_err = p.get("position_error")
            if pos_err is not None:
                stats["pos_errors"].append(pos_err)
                all_pos_errors.append(pos_err)
                time_buckets[bucket]["pos_errors"].append(pos_err)
                pos_error_hist[_bin_position_error(pos_err)] += 1

            # Was this an action change?
            prev_idx = t - 1
            prev_frame = next((f for f in frames if f["t"] == prev_idx), None)
            if prev_frame and prev_frame.get("actual"):
                prev_action = prev_frame["actual"][player]["action"]
                if prev_action != action_id:
                    stats["change_total"] += 1
                    total_changes += 1
                    if is_correct:
                        stats["change_correct"] += 1
                        total_change_correct += 1

            # Time bucket action accuracy
            time_buckets[bucket]["action_total"] += 1
            if is_correct:
                time_buckets[bucket]["action_correct"] += 1

            # Binary flag confusion matrices
            for flag_name, actual_key, pred_key in [
                ("on_ground", "on_ground", "on_ground"),
                ("facing", "facing_right", "facing_right"),
                ("invulnerable", "invulnerable", "invulnerable"),
            ]:
                actual_val = a.get(actual_key)
                pred_val = p.get(pred_key)
                if actual_val is None or pred_val is None:
                    continue
                cm = binary_confusion[flag_name]
                if actual_val and pred_val:
                    cm["tp"] += 1
                elif not actual_val and pred_val:
                    cm["fp"] += 1
                elif actual_val and not pred_val:
                    cm["fn"] += 1
                else:
                    cm["tn"] += 1

                if flag_name == "on_ground":
                    time_buckets[bucket]["on_ground_total"] += 1
                    if actual_val == pred_val:
                        time_buckets[bucket]["on_ground_correct"] += 1

    # Build game summary
    total_predictions = sum(s["total"] for s in per_action_stats.values())
    game_summary = {
        "game_path": game_path.name,
        "num_frames": T,
        "num_predicted": num_predicted,
        "stage": game.stage,
        "p0_char": int(game.p0.character[0]),
        "p1_char": int(game.p1.character[0]),
        "action_acc": round(total_action_correct / max(total_predictions, 1), 4),
        "change_acc": round(total_change_correct / max(total_changes, 1), 4) if total_changes > 0 else None,
        "num_changes": total_changes,
        "avg_pos_error": round(sum(all_pos_errors) / max(len(all_pos_errors), 1), 2) if all_pos_errors else None,
    }

    return {
        "game_summary": game_summary,
        "per_action_stats": dict(per_action_stats),
        "time_buckets": time_buckets,
        "binary_confusion": binary_confusion,
        "pos_error_hist": pos_error_hist,
        "total_action_correct": total_action_correct,
        "total_predictions": total_predictions,
        "total_change_correct": total_change_correct,
        "total_changes": total_changes,
    }


def aggregate_results(game_results: list[dict], model_info: dict) -> dict:
    """Aggregate per-game results into the final output JSON."""
    # Merge per-action stats across games
    merged_actions = defaultdict(lambda: {"correct": 0, "total": 0, "pos_errors": [],
                                          "change_correct": 0, "change_total": 0})
    merged_time = [{"action_correct": 0, "action_total": 0,
                    "on_ground_correct": 0, "on_ground_total": 0,
                    "pos_errors": []} for _ in range(NUM_TIME_BUCKETS)]
    merged_binary = {
        "on_ground": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "facing": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "invulnerable": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }
    merged_pos_hist = [0] * len(POS_ERROR_LABELS)
    total_correct = 0
    total_preds = 0
    total_change_correct = 0
    total_changes = 0
    per_game = []

    for gr in game_results:
        per_game.append(gr["game_summary"])
        total_correct += gr["total_action_correct"]
        total_preds += gr["total_predictions"]
        total_change_correct += gr["total_change_correct"]
        total_changes += gr["total_changes"]

        for aid_str, stats in gr["per_action_stats"].items():
            aid = int(aid_str) if isinstance(aid_str, str) else aid_str
            m = merged_actions[aid]
            m["correct"] += stats["correct"]
            m["total"] += stats["total"]
            m["pos_errors"].extend(stats["pos_errors"])
            m["change_correct"] += stats["change_correct"]
            m["change_total"] += stats["change_total"]

        for i in range(NUM_TIME_BUCKETS):
            tb = gr["time_buckets"][i]
            mt = merged_time[i]
            mt["action_correct"] += tb["action_correct"]
            mt["action_total"] += tb["action_total"]
            mt["on_ground_correct"] += tb["on_ground_correct"]
            mt["on_ground_total"] += tb["on_ground_total"]
            mt["pos_errors"].extend(tb["pos_errors"])

        for flag in merged_binary:
            for k in ["tp", "fp", "fn", "tn"]:
                merged_binary[flag][k] += gr["binary_confusion"][flag][k]

        for i in range(len(merged_pos_hist)):
            merged_pos_hist[i] += gr["pos_error_hist"][i]

    # --- Build output sections ---

    # by_action: per-action-ID table sorted by error count desc
    by_action = []
    for aid, stats in sorted(merged_actions.items(), key=lambda x: x[1]["total"] - x[1]["correct"], reverse=True):
        acc = stats["correct"] / max(stats["total"], 1)
        avg_pos = sum(stats["pos_errors"]) / max(len(stats["pos_errors"]), 1) if stats["pos_errors"] else None
        change_acc = stats["change_correct"] / max(stats["change_total"], 1) if stats["change_total"] > 0 else None
        category = _ACTION_TO_CATEGORY.get(aid, "other")
        by_action.append({
            "action_id": aid,
            "action_name": action_name(aid),
            "category": category,
            "count": stats["total"],
            "correct": stats["correct"],
            "accuracy": round(acc, 4),
            "errors": stats["total"] - stats["correct"],
            "avg_pos_error": round(avg_pos, 2) if avg_pos is not None else None,
            "change_count": stats["change_total"],
            "change_correct": stats["change_correct"],
            "change_accuracy": round(change_acc, 4) if change_acc is not None else None,
        })

    # by_category: grouped by ACTION_CATEGORIES
    by_category = {}
    all_categorized = set()
    for cat, ids in ACTION_CATEGORIES.items():
        all_categorized |= ids
        cat_correct = sum(merged_actions[aid]["correct"] for aid in ids if aid in merged_actions)
        cat_total = sum(merged_actions[aid]["total"] for aid in ids if aid in merged_actions)
        cat_changes = sum(merged_actions[aid]["change_total"] for aid in ids if aid in merged_actions)
        cat_change_correct = sum(merged_actions[aid]["change_correct"] for aid in ids if aid in merged_actions)
        cat_pos = []
        for aid in ids:
            if aid in merged_actions:
                cat_pos.extend(merged_actions[aid]["pos_errors"])
        if cat_total > 0:
            by_category[cat] = {
                "count": cat_total,
                "correct": cat_correct,
                "accuracy": round(cat_correct / cat_total, 4),
                "change_count": cat_changes,
                "change_accuracy": round(cat_change_correct / max(cat_changes, 1), 4) if cat_changes > 0 else None,
                "avg_pos_error": round(sum(cat_pos) / len(cat_pos), 2) if cat_pos else None,
            }

    # "other" category
    other_correct = sum(merged_actions[aid]["correct"] for aid in merged_actions if aid not in all_categorized)
    other_total = sum(merged_actions[aid]["total"] for aid in merged_actions if aid not in all_categorized)
    if other_total > 0:
        other_pos = []
        for aid in merged_actions:
            if aid not in all_categorized:
                other_pos.extend(merged_actions[aid]["pos_errors"])
        by_category["other"] = {
            "count": other_total,
            "correct": other_correct,
            "accuracy": round(other_correct / other_total, 4),
            "avg_pos_error": round(sum(other_pos) / len(other_pos), 2) if other_pos else None,
        }

    # by_time_bucket
    by_time_bucket = []
    for i, tb in enumerate(merged_time):
        bucket_label = f"{i * 10}-{(i + 1) * 10}%"
        action_acc = tb["action_correct"] / max(tb["action_total"], 1)
        on_ground_acc = tb["on_ground_correct"] / max(tb["on_ground_total"], 1)
        avg_pos = sum(tb["pos_errors"]) / max(len(tb["pos_errors"]), 1) if tb["pos_errors"] else 0
        by_time_bucket.append({
            "bucket": bucket_label,
            "action_accuracy": round(action_acc, 4),
            "on_ground_accuracy": round(on_ground_acc, 4),
            "avg_pos_error": round(avg_pos, 2),
            "num_frames": tb["action_total"],
        })

    # binary_flags: add precision/recall/F1
    binary_flags = {}
    for flag, cm in merged_binary.items():
        tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
        binary_flags[flag] = {
            **cm,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        }

    # position_error_distribution
    pos_error_dist = []
    for i, label in enumerate(POS_ERROR_LABELS):
        pos_error_dist.append({"bin": label, "count": merged_pos_hist[i]})

    # Summary
    all_pos = []
    for stats in merged_actions.values():
        all_pos.extend(stats["pos_errors"])

    summary = {
        "total_games": len(game_results),
        "total_frames": total_preds,
        "action_accuracy": round(total_correct / max(total_preds, 1), 4),
        "change_accuracy": round(total_change_correct / max(total_changes, 1), 4) if total_changes > 0 else None,
        "total_changes": total_changes,
        "avg_pos_error": round(sum(all_pos) / max(len(all_pos), 1), 2) if all_pos else None,
        "on_ground_accuracy": binary_flags["on_ground"]["accuracy"],
    }

    return {
        "meta": {
            **model_info,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_games": len(game_results),
        },
        "summary": summary,
        "by_action": by_action,
        "by_category": by_category,
        "by_time_bucket": by_time_bucket,
        "binary_flags": binary_flags,
        "position_error_distribution": pos_error_dist,
        "per_game": per_game,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch teacher-forced evaluation with error analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data-dir", required=True, help="Dataset dir with meta.json + games/")
    parser.add_argument("--num-games", type=int, default=30, help="Number of games to evaluate")
    parser.add_argument("--output", default="worldmodel/batch_eval_results.json", help="Output JSON path")
    parser.add_argument("--device", default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for game sampling")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load model
    logger.info("Loading model from %s", args.checkpoint)
    model, cfg, context_len, arch = load_model_from_checkpoint(args.checkpoint, args.device)

    # Load meta.json and sample games
    data_dir = Path(args.data_dir)
    meta_path = data_dir / "meta.json"
    with open(meta_path) as f:
        meta_list = json.load(f)

    # Filter to training games
    valid_entries = [e for e in meta_list if e.get("is_training", False)]
    logger.info("Found %d training games in %s", len(valid_entries), data_dir)

    # Sample
    rng = random.Random(args.seed)
    if args.num_games < len(valid_entries):
        sampled = rng.sample(valid_entries, args.num_games)
    else:
        sampled = valid_entries
        rng.shuffle(sampled)

    logger.info("Evaluating %d games (seed=%d)", len(sampled), args.seed)

    model_info = {
        "model_name": Path(args.checkpoint).parent.name,
        "arch": arch,
        "context_len": context_len,
        "checkpoint": str(args.checkpoint),
        "data_dir": str(data_dir),
        "device": args.device,
        "seed": args.seed,
    }

    # Evaluate games one at a time
    game_results = []
    t_start = time.time()
    for i, entry in enumerate(sampled):
        game_path = data_dir / "games" / entry["slp_md5"]
        compression = entry.get("compression", "zlib")

        t_game = time.time()
        result = analyze_game(model, game_path, cfg, args.device, compression)
        elapsed = time.time() - t_game

        if result is None:
            logger.warning("[%d/%d] SKIP %s", i + 1, len(sampled), entry["slp_md5"][:12])
            continue

        gs = result["game_summary"]
        game_results.append(result)
        logger.info(
            "[%d/%d] %s: %d frames, action_acc=%.1f%%, change_acc=%s, pos_err=%.2f (%.1fs)",
            i + 1, len(sampled), entry["slp_md5"][:12],
            gs["num_predicted"],
            gs["action_acc"] * 100,
            f"{gs['change_acc'] * 100:.1f}%" if gs["change_acc"] is not None else "N/A",
            gs["avg_pos_error"] or 0,
            elapsed,
        )

    total_elapsed = time.time() - t_start
    logger.info("Evaluated %d games in %.1f min", len(game_results), total_elapsed / 60)

    if not game_results:
        logger.error("No games evaluated successfully!")
        sys.exit(1)

    # Aggregate and write
    output = aggregate_results(game_results, model_info)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    size_kb = Path(args.output).stat().st_size / 1024
    logger.info("Wrote %s (%.1f KB)", args.output, size_kb)
    logger.info("Summary: action_acc=%.1f%%, change_acc=%s, on_ground_acc=%.1f%%, avg_pos_err=%.2f",
                output["summary"]["action_accuracy"] * 100,
                f"{output['summary']['change_accuracy'] * 100:.1f}%" if output["summary"]["change_accuracy"] else "N/A",
                output["summary"]["on_ground_accuracy"] * 100,
                output["summary"]["avg_pos_error"] or 0)


if __name__ == "__main__":
    main()
