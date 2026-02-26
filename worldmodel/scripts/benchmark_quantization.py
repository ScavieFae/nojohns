"""Benchmark INT8 post-training quantization on a world model checkpoint.

Compares float32 vs INT8 on the same validation set: accuracy, loss, inference speed.

Usage:
    .venv/bin/python -m worldmodel.scripts.benchmark_quantization \
        --checkpoint worldmodel/checkpoints/mamba2-22k-ss-ep1.pt \
        --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
        --max-games 500 -v

    # Quick test with fewer games:
    .venv/bin/python -m worldmodel.scripts.benchmark_quantization \
        --checkpoint worldmodel/checkpoints/mamba2-22k-ss-ep1.pt \
        --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
        --max-games 50 -v
"""

import argparse
import copy
import logging
import time

import torch
import torch.quantization
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def load_model_and_data(checkpoint_path: str, dataset_path: str, max_games: int, device: str):
    """Load model from checkpoint and build validation dataset."""
    from worldmodel.data.dataset import MeleeDataset
    from worldmodel.data.parse import load_games_from_dir
    from worldmodel.model.encoding import EncodingConfig
    from worldmodel.model.mamba2 import FrameStackMamba2
    from worldmodel.training.metrics import LossWeights, MetricsTracker

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Infer config from checkpoint
    cfg_dict = checkpoint.get("encoding_config", checkpoint.get("config", {}).get("encoding", {}))
    cfg = EncodingConfig(**{k: v for k, v in cfg_dict.items() if v is not None}) if cfg_dict else EncodingConfig()

    model_cfg = checkpoint.get("config", {}).get("model", {})
    context_len = model_cfg.get("context_len", 10)
    d_model = model_cfg.get("d_model", 384)
    n_layers = model_cfg.get("n_layers", 4)
    d_state = model_cfg.get("d_state", 64)
    headdim = model_cfg.get("headdim", 64)
    chunk_size = model_cfg.get("chunk_size", 10)

    # Build model
    model = FrameStackMamba2(
        cfg=cfg,
        context_len=context_len,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        headdim=headdim,
        chunk_size=chunk_size,
    )
    model.load_state_dict(state_dict)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params, d_model=%d, n_layers=%d", param_count, d_model, n_layers)

    # Load data
    logger.info("Loading dataset from %s (max_games=%d)...", dataset_path, max_games)
    t0 = time.time()
    games = load_games_from_dir(dataset_path, max_games=max_games)
    dataset = MeleeDataset(games, cfg)
    logger.info("Loaded %d games in %.1fs", dataset.num_games, time.time() - t0)

    # Build val dataset (last 10% of games)
    train_split = 0.9
    val_ds = dataset.get_frame_dataset(context_len=context_len, train=False, train_split=train_split)
    logger.info("Val dataset: %d examples", len(val_ds))

    metrics = MetricsTracker(cfg=cfg, weights=LossWeights())

    return model, val_ds, cfg, metrics


def run_val_pass(model, val_loader, metrics, device, label="float32"):
    """Run full validation and return metrics dict + timing."""
    from worldmodel.training.metrics import EpochMetrics

    model.eval()
    epoch_metrics = EpochMetrics()
    total_frames = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (float_ctx, int_ctx, next_ctrl, float_tgt, int_tgt) in enumerate(val_loader):
            float_ctx = float_ctx.to(device)
            int_ctx = int_ctx.to(device)
            next_ctrl = next_ctrl.to(device)
            float_tgt = float_tgt.to(device)
            int_tgt = int_tgt.to(device)

            predictions = model(float_ctx, int_ctx, next_ctrl)
            _, batch_metrics = metrics.compute_loss(predictions, float_tgt, int_tgt, int_ctx)
            epoch_metrics.update(batch_metrics)
            total_frames += float_ctx.shape[0]

    elapsed = time.time() - t0
    results = epoch_metrics.averaged()
    results["_elapsed"] = elapsed
    results["_frames"] = total_frames
    results["_fps"] = total_frames / elapsed if elapsed > 0 else 0

    logger.info(
        "[%s] %d frames in %.1fs (%.0f fps) | loss=%.4f change_acc=%.4f pos_mae=%.3f",
        label, total_frames, elapsed, results["_fps"],
        results.get("loss/total", 0),
        results.get("metric/action_change_acc", 0),
        results.get("metric/position_mae", 0),
    )
    return results


def quantize_model(model):
    """Apply dynamic INT8 quantization to linear layers."""
    # Use qnnpack on ARM/macOS, fbgemm on x86
    if torch.backends.quantized.engine == "none":
        torch.backends.quantized.engine = "qnnpack"
    logger.info("Quantization engine: %s", torch.backends.quantized.engine)

    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    # Count quantized layers
    n_quantized = sum(
        1 for m in quantized.modules()
        if hasattr(m, 'weight') and hasattr(m.weight, 'qscheme')
    )
    n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    logger.info("Quantized %d/%d linear layers to INT8", n_quantized, n_linear)

    # Size comparison
    orig_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    # For quantized, estimate: INT8 weights + float scales
    quant_size = sum(
        p.numel() * (1 if hasattr(p, 'qscheme') else p.element_size())
        for p in quantized.parameters()
    ) / 1e6
    logger.info("Size: float32=%.1f MB, INT8≈%.1f MB (%.1fx reduction)",
                orig_size, quant_size, orig_size / max(quant_size, 0.1))

    return quantized


def main():
    parser = argparse.ArgumentParser(description="Benchmark INT8 quantization")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to parsed dataset")
    parser.add_argument("--max-games", type=int, default=500, help="Games to load for val")
    parser.add_argument("--batch-size", type=int, default=512, help="Val batch size")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Dynamic quantization only works on CPU
    device = "cpu"
    logger.info("Using device: %s (dynamic quantization requires CPU)", device)

    model, val_ds, cfg, metrics_calc = load_model_and_data(
        args.checkpoint, args.dataset, args.max_games, device
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # --- Float32 baseline ---
    print("\n=== Float32 Baseline ===")
    model_fp32 = model.to(device)
    fp32_results = run_val_pass(model_fp32, val_loader, metrics_calc, device, label="float32")

    # --- INT8 quantized ---
    print("\n=== INT8 Dynamic Quantization ===")
    model_int8 = quantize_model(copy.deepcopy(model))
    model_int8 = model_int8.to(device)
    int8_results = run_val_pass(model_int8, val_loader, metrics_calc, device, label="int8")

    # --- Comparison ---
    print("\n" + "=" * 70)
    print("QUANTIZATION BENCHMARK RESULTS")
    print("=" * 70)

    compare_keys = [
        ("loss/total", "Val Loss", "{:.4f}", False),
        ("metric/action_change_acc", "Change Acc", "{:.4f}", True),
        ("metric/p0_action_acc", "P0 Action Acc", "{:.4f}", True),
        ("metric/position_mae", "Position MAE", "{:.3f}", False),
        ("metric/velocity_mae", "Velocity MAE", "{:.3f}", False),
    ]

    print(f"\n{'Metric':<20} {'Float32':>10} {'INT8':>10} {'Delta':>10} {'Verdict':>10}")
    print("-" * 62)

    all_ok = True
    for key, label, fmt, higher_is_better in compare_keys:
        fp32_val = fp32_results.get(key)
        int8_val = int8_results.get(key)
        if fp32_val is None or int8_val is None:
            continue

        delta = int8_val - fp32_val
        if higher_is_better:
            pct = -100 * delta / max(abs(fp32_val), 1e-8)  # negative delta = degradation
        else:
            pct = 100 * delta / max(abs(fp32_val), 1e-8)  # positive delta = degradation

        if abs(pct) < 1:
            verdict = "OK"
        elif abs(pct) < 3:
            verdict = "MILD"
        elif abs(pct) < 5:
            verdict = "WARN"
        else:
            verdict = "BAD"
            all_ok = False

        print(f"{label:<20} {fmt.format(fp32_val):>10} {fmt.format(int8_val):>10} {pct:>+9.1f}% {verdict:>10}")

    # Speed comparison
    fp32_fps = fp32_results["_fps"]
    int8_fps = int8_results["_fps"]
    speedup = int8_fps / max(fp32_fps, 1)
    print(f"\n{'Inference Speed':<20} {fp32_fps:>9.0f}  {int8_fps:>9.0f}   {speedup:>8.2f}x")
    print(f"{'(frames/sec)':<20}")

    # Overall verdict
    print("\n" + "-" * 62)
    if all_ok:
        print("VERDICT: INT8 quantization looks safe. Train in float32, quantize post-training.")
    else:
        print("VERDICT: INT8 shows >5% degradation. Consider mixed precision or QAT.")
    print("=" * 70)


if __name__ == "__main__":
    main()
