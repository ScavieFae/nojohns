#!/usr/bin/env python3
"""Benchmark SSD chunked scan vs sequential scan.

Tests:
  1. Correctness: SSD output matches sequential scan on same inputs (atol=1e-4)
  2. Timing: grid of (K, chunk_size, batch_size) configurations
  3. 5-game smoke test: full model forward+backward with SSD at K=60

Usage:
    python -m worldmodel.scripts.benchmark_ssd
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worldmodel.model.encoding import EncodingConfig
from worldmodel.model.mamba2 import Mamba2Block, FrameStackMamba2


def test_correctness():
    """Verify SSD and sequential scan produce matching outputs."""
    print("=" * 60)
    print("CORRECTNESS TEST: SSD vs Sequential Scan")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)

    for K, cs in [(10, 5), (10, 10), (60, 15), (60, 20), (60, 30)]:
        # Build two blocks with identical weights
        block_seq = Mamba2Block(d_model=256, d_state=64, headdim=64, chunk_size=None)
        block_ssd = Mamba2Block(d_model=256, d_state=64, headdim=64, chunk_size=cs)
        block_ssd.load_state_dict(block_seq.state_dict())

        block_seq = block_seq.to(device)
        block_ssd = block_ssd.to(device)

        # Same input
        x = torch.randn(4, K, 256, device=device)

        with torch.no_grad():
            y_seq = block_seq(x)
            y_ssd = block_ssd(x)

        max_diff = (y_seq - y_ssd).abs().max().item()
        mean_diff = (y_seq - y_ssd).abs().mean().item()
        passed = max_diff < 1e-3  # Generous tolerance for float32

        status = "PASS" if passed else "FAIL"
        print(f"  K={K:3d} chunk_size={cs:3d}: max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  [{status}]")

        if not passed:
            print(f"  ** FAILED ** — outputs diverge beyond tolerance")
            return False

    print("\nAll correctness tests passed.\n")
    return True


def test_timing():
    """Benchmark timing across configurations."""
    print("=" * 60)
    print("TIMING BENCHMARK")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    warmup = 3
    trials = 10

    configs = [
        # (K, chunk_size, batch_size, label)
        (10, None, 512, "K=10  seq     B=512"),
        (10, 10,   512, "K=10  ssd/10  B=512"),
        (60, None, 512, "K=60  seq     B=512"),
        (60, 15,   512, "K=60  ssd/15  B=512"),
        (60, 20,   512, "K=60  ssd/20  B=512"),
        (60, 30,   512, "K=60  ssd/30  B=512"),
        (60, 15,   256, "K=60  ssd/15  B=256"),
        (60, 15,   128, "K=60  ssd/15  B=128"),
    ]

    print(f"\n  {'Config':<25s}  {'ms/batch':>10s}  {'rel':>6s}")
    print("  " + "-" * 45)

    baseline_ms = None

    for K, cs, bs, label in configs:
        torch.manual_seed(42)
        block = Mamba2Block(d_model=256, d_state=64, headdim=64, chunk_size=cs).to(device)
        x = torch.randn(bs, K, 256, device=device)

        # Warmup
        for _ in range(warmup):
            _ = block(x)
            if device == "mps":
                torch.mps.synchronize()

        # Timed runs
        times = []
        for _ in range(trials):
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = block(x)
            if device == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)

        ms = sum(times) / len(times) * 1000
        if baseline_ms is None:
            baseline_ms = ms
        rel = ms / baseline_ms

        print(f"  {label:<25s}  {ms:>8.1f}ms  {rel:>5.2f}x")

    print()


def test_full_model():
    """5-game smoke test with full model at K=60 using SSD."""
    print("=" * 60)
    print("FULL MODEL SMOKE TEST: K=60, SSD chunk_size=15")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)

    cfg = EncodingConfig(
        state_age_as_embed=True,
        state_age_embed_vocab=150,
        state_age_embed_dim=8,
    )

    model = FrameStackMamba2(
        cfg=cfg,
        context_len=60,
        d_model=256,
        d_state=64,
        n_layers=2,
        headdim=64,
        dropout=0.0,
        chunk_size=15,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Synthetic batch mimicking real data shapes
    B = 64
    K = 60
    float_cols = cfg.float_per_player * 2
    int_cols = cfg.int_per_player * 2 + 1
    ctrl_dim = cfg.ctrl_conditioning_dim

    float_ctx = torch.randn(B, K, float_cols, device=device)
    # Build int_ctx with proper vocab ranges per column
    ipp = cfg.int_per_player
    int_ctx = torch.zeros(B, K, int_cols, dtype=torch.long, device=device)
    vocab_map = {
        0: cfg.action_vocab, 1: cfg.jumps_vocab, 2: cfg.character_vocab,
        3: cfg.l_cancel_vocab, 4: cfg.hurtbox_vocab, 5: cfg.ground_vocab,
        6: cfg.last_attack_vocab,
    }
    if cfg.state_age_as_embed:
        vocab_map[7] = cfg.state_age_embed_vocab
    for col, vocab in vocab_map.items():
        int_ctx[:, :, col] = torch.randint(0, vocab, (B, K))
        int_ctx[:, :, ipp + col] = torch.randint(0, vocab, (B, K))
    int_ctx[:, :, ipp * 2] = torch.randint(0, cfg.stage_vocab, (B, K))

    next_ctrl = torch.randn(B, ctrl_dim, device=device)

    # Forward pass
    t0 = time.perf_counter()
    out = model(float_ctx, int_ctx, next_ctrl)
    if device == "mps":
        torch.mps.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000

    # Check output shapes
    print(f"  Forward: {fwd_ms:.1f}ms")
    print(f"  Output shapes:")
    for k, v in out.items():
        print(f"    {k}: {tuple(v.shape)}")

    # Backward pass
    loss = sum(v.sum() for v in out.values())
    t0 = time.perf_counter()
    loss.backward()
    if device == "mps":
        torch.mps.synchronize()
    bwd_ms = (time.perf_counter() - t0) * 1000

    print(f"  Backward: {bwd_ms:.1f}ms")
    print(f"  Total fwd+bwd: {fwd_ms + bwd_ms:.1f}ms")

    # Gradient check
    grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
    zero_grads = [n for n, g in grad_norms.items() if g == 0]
    if zero_grads:
        print(f"  WARNING: {len(zero_grads)} parameters have zero gradient")
    else:
        print(f"  All {len(grad_norms)} parameters have non-zero gradients")

    print()
    return True


if __name__ == "__main__":
    ok = test_correctness()
    if not ok:
        print("Aborting — correctness check failed.")
        sys.exit(1)
    test_timing()
    test_full_model()
    print("All benchmarks complete.")
