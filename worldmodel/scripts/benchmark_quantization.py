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


def quantize_model(model, force_all=False):
    """Apply dynamic INT8 quantization to linear layers.

    Args:
        model: The model to quantize.
        force_all: If True, force-quantize ALL linear layers including tiny ones
                   that quantize_dynamic normally skips. Required for onchain where
                   all math must be integer.
    """
    # Use qnnpack on ARM/macOS, fbgemm on x86
    if torch.backends.quantized.engine == "none":
        torch.backends.quantized.engine = "qnnpack"
    logger.info("Quantization engine: %s", torch.backends.quantized.engine)

    # Count linear layers in original model before quantization
    n_linear_orig = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))

    if force_all:
        # Force INT8 on every Linear layer by setting qconfig on Linear modules
        # only (NOT on root model — that would propagate to Embeddings).
        # quantize_dynamic may skip tiny layers as a CPU optimization, but for
        # onchain we need uniform INT8 math everywhere.
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.qconfig = torch.quantization.default_dynamic_qconfig
        quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        # Force-wrap any remaining nn.Linear stragglers to DynamicQuantizedLinear
        for name, module in list(quantized.named_modules()):
            if isinstance(module, torch.nn.Linear):
                logger.warning("Layer %s (%s) still float — force-wrapping to DynamicQuantizedLinear",
                               name, tuple(module.weight.shape))
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = quantized if not parent_name else dict(quantized.named_modules())[parent_name]
                quantized_layer = torch.nn.quantized.dynamic.Linear(
                    module.in_features, module.out_features,
                    bias=module.bias is not None, dtype=torch.qint8,
                )
                quantized_layer.set_weight_bias(
                    torch.quantize_per_tensor(module.weight, scale=module.weight.abs().mean().item() / 64,
                                              zero_point=0, dtype=torch.qint8),
                    module.bias,
                )
                setattr(parent, child_name, quantized_layer)
        logger.info("Force-quantized ALL linear layers to INT8 (onchain mode)")
    else:
        quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )

    # Count: remaining nn.Linear in quantized model = layers NOT quantized
    n_linear_remaining = sum(1 for m in quantized.modules() if isinstance(m, torch.nn.Linear))
    n_quantized = n_linear_orig - n_linear_remaining
    logger.info("Quantized %d/%d linear layers to INT8 (%d remaining as float)",
                n_quantized, n_linear_orig, n_linear_remaining)

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


def print_comparison(fp32_results, int8_results, label="INT8"):
    """Print comparison table between float32 and quantized results."""
    compare_keys = [
        ("loss/total", "Val Loss", "{:.4f}", False),
        ("metric/action_change_acc", "Change Acc", "{:.4f}", True),
        ("metric/p0_action_acc", "P0 Action Acc", "{:.4f}", True),
        ("metric/position_mae", "Position MAE", "{:.3f}", False),
        ("metric/velocity_mae", "Velocity MAE", "{:.3f}", False),
    ]

    print(f"\n{'Metric':<20} {'Float32':>10} {label:>10} {'Delta':>10} {'Verdict':>10}")
    print("-" * 62)

    all_ok = True
    for key, metric_label, fmt, higher_is_better in compare_keys:
        fp32_val = fp32_results.get(key)
        int8_val = int8_results.get(key)
        if fp32_val is None or int8_val is None:
            continue

        delta = int8_val - fp32_val
        if higher_is_better:
            pct = -100 * delta / max(abs(fp32_val), 1e-8)
        else:
            pct = 100 * delta / max(abs(fp32_val), 1e-8)

        if abs(pct) < 1:
            verdict = "OK"
        elif abs(pct) < 3:
            verdict = "MILD"
        elif abs(pct) < 5:
            verdict = "WARN"
        else:
            verdict = "BAD"
            all_ok = False

        print(f"{metric_label:<20} {fmt.format(fp32_val):>10} {fmt.format(int8_val):>10} {pct:>+9.1f}% {verdict:>10}")

    # Speed comparison
    fp32_fps = fp32_results["_fps"]
    int8_fps = int8_results["_fps"]
    speedup = int8_fps / max(fp32_fps, 1)
    print(f"\n{'Inference Speed':<20} {fp32_fps:>9.0f}  {int8_fps:>9.0f}   {speedup:>8.2f}x")
    print(f"{'(frames/sec)':<20}")

    print("\n" + "-" * 62)
    if all_ok:
        print(f"VERDICT: {label} quantization looks safe.")
    else:
        print(f"VERDICT: {label} shows >5% degradation on at least one metric.")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Benchmark INT8 quantization")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to parsed dataset")
    parser.add_argument("--max-games", type=int, default=500, help="Games to load for val")
    parser.add_argument("--batch-size", type=int, default=512, help="Val batch size")
    parser.add_argument("--force-all", action="store_true",
                        help="Force-quantize ALL linear layers (onchain mode). "
                             "Default quantize_dynamic skips tiny layers.")
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

    # --- INT8 quantized (default — skips tiny layers) ---
    print("\n=== INT8 Dynamic Quantization (default) ===")
    model_int8 = quantize_model(copy.deepcopy(model), force_all=False)
    model_int8 = model_int8.to(device)
    int8_results = run_val_pass(model_int8, val_loader, metrics_calc, device, label="int8")

    print("\n" + "=" * 70)
    print("QUANTIZATION BENCHMARK — DEFAULT (skips tiny heads)")
    print("=" * 70)
    default_ok = print_comparison(fp32_results, int8_results, label="INT8")

    if args.force_all:
        # --- INT8 force-all (onchain mode — every layer quantized) ---
        print("\n\n=== INT8 Force-All Quantization (onchain mode) ===")
        model_int8_all = quantize_model(copy.deepcopy(model), force_all=True)
        model_int8_all = model_int8_all.to(device)
        int8_all_results = run_val_pass(model_int8_all, val_loader, metrics_calc, device, label="int8-all")

        print("\n" + "=" * 70)
        print("QUANTIZATION BENCHMARK — FORCE ALL (onchain mode, all layers INT8)")
        print("=" * 70)
        force_ok = print_comparison(fp32_results, int8_all_results, label="INT8-ALL")

        # --- Delta between default and force-all ---
        print("\n" + "=" * 70)
        print("DELTA: force-all vs default (cost of quantizing the 4 tiny heads)")
        print("=" * 70)
        print_comparison(int8_results, int8_all_results, label="INT8-ALL")

    print("\n" + "=" * 70)
    if not args.force_all:
        if default_ok:
            print("OVERALL: INT8 quantization looks safe. Train float32, quantize post-training.")
            print("Re-run with --force-all to verify onchain mode (all layers INT8).")
        else:
            print("OVERALL: INT8 shows >5% degradation. Consider mixed precision or QAT.")
    else:
        if default_ok and force_ok:
            print("OVERALL: Full INT8 (all layers) looks safe for onchain. No QAT needed.")
        elif default_ok and not force_ok:
            print("OVERALL: Default INT8 is safe, but force-all degrades. Investigate the 4 tiny heads.")
        else:
            print("OVERALL: INT8 shows significant degradation. Consider QAT.")
    print("=" * 70)


if __name__ == "__main__":
    main()
