"""Modal launcher for Melee world model training.

Uses pre-encoded .pt files for instant data loading (no per-file I/O).

Pre-encode on Modal (no local encode/upload needed):
    .venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode
    .venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode --max-games 2000 --output /encoded-2k.pt

Train:
    .venv/bin/modal run worldmodel/scripts/modal_train.py::train
    .venv/bin/modal run worldmodel/scripts/modal_train.py::train --epochs 5 --encoded-file /encoded-2k.pt

Detached training (returns immediately, check wandb for progress):
    .venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train

Sweep (parallel runs on separate A100s):
    .venv/bin/modal run worldmodel/scripts/modal_train.py::sweep --names "lr3e4,lr1e3"
"""

import modal
import os

app = modal.App("melee-worldmodel")
volume = modal.Volume.from_name("melee-training-data", create_if_missing=True)

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("pyarrow", "pyyaml", "wandb", "numpy")
    .add_local_dir(
        os.path.join(repo_root, "worldmodel"),
        remote_path="/root/nojohns/worldmodel",
    )
)

DATA_VOLUME_PATH = "/data"
CHECKPOINT_DIR = f"{DATA_VOLUME_PATH}/checkpoints"


@app.function(
    volumes={DATA_VOLUME_PATH: volume},
    image=image,
    timeout=600,
)
def check_volume():
    """List volume contents."""
    for root, dirs, files in os.walk(DATA_VOLUME_PATH):
        for f in files:
            path = os.path.join(root, f)
            print(f"  {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


@app.function(
    gpu="H100",
    volumes={DATA_VOLUME_PATH: volume},
    image=image,
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-key")],
)
def train(
    config: str = "worldmodel/experiments/mamba2-medium-gpu.yaml",
    epochs: int = 2,
    run_name: str = "mamba2-first-complete",
    encoded_file: str = "/encoded-2k.pt",
    resume: str = "",
):
    """Train from pre-encoded .pt file — instant data loading.

    Supports multi-GPU via PyTorch DDP when multiple GPUs are available.
    Modal gives us N GPUs in one container; we use torchrun to spawn N processes.
    """
    import subprocess, sys

    n_gpus = int(os.environ.get("NVIDIA_VISIBLE_DEVICES", "0").count(",")) + 1
    try:
        import torch
        n_gpus = torch.cuda.device_count()
    except Exception:
        pass

    if n_gpus > 1:
        # Launch via torchrun for DDP — spawns one process per GPU
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(n_gpus),
            "--master_port", "29500",
            "-m", "worldmodel.scripts.modal_train_worker",
            "--config", config,
            "--epochs", str(epochs),
            "--run-name", run_name,
            "--encoded-file", encoded_file,
        ]
        if resume:
            cmd.extend(["--resume", resume])
        print(f"Launching DDP with {n_gpus} GPUs via torchrun...")
        sys.path.insert(0, "/root/nojohns")
        env = os.environ.copy()
        env["PYTHONPATH"] = "/root/nojohns:" + env.get("PYTHONPATH", "")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"DDP training failed with exit code {result.returncode}")
        return

    # Single-GPU fallback
    _train_single_gpu(config, epochs, run_name, encoded_file, resume)


def _train_single_gpu(config, epochs, run_name, encoded_file, resume):
    """Single-GPU training path (also called by DDP worker for rank 0 setup)."""
    import sys
    import time
    import logging

    sys.path.insert(0, "/root/nojohns")
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch
    import yaml
    import numpy as np

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # Load pre-encoded data — single file or multiple comma-separated files
    encoded_files = [f.strip() for f in encoded_file.split(",")]
    payloads = []
    for ef in encoded_files:
        ep = f"{DATA_VOLUME_PATH}{ef}"
        if not os.path.exists(ep):
            raise FileNotFoundError(
                f"No encoded data at {ep}. "
                "Pre-encode first: modal run worldmodel/scripts/modal_train.py::pre_encode"
            )
        print(f"Loading {ep}...")
        t0 = time.time()
        payloads.append(torch.load(ep, weights_only=False))
        print(f"  Loaded in {time.time() - t0:.1f}s — {payloads[-1]['floats'].shape}")

    if len(payloads) == 1:
        payload = payloads[0]
    else:
        # Merge multiple .pt files
        print(f"Merging {len(payloads)} encoded files...")
        all_floats = torch.cat([p["floats"] for p in payloads])
        all_ints = torch.cat([p["ints"] for p in payloads])
        all_lengths = []
        for p in payloads:
            all_lengths.extend(p["game_lengths"])
        total_games = sum(p["num_games"] for p in payloads)
        payload = {
            "floats": all_floats,
            "ints": all_ints,
            "game_offsets": torch.tensor(np.cumsum([0] + all_lengths)),
            "game_lengths": all_lengths,
            "num_games": total_games,
            "encoding_config": payloads[0].get("encoding_config", {}),
        }
        del payloads, all_floats, all_ints
        import gc; gc.collect()

    print(f"  Floats: {payload['floats'].shape}")
    print(f"  Ints: {payload['ints'].shape}")
    print(f"  Games: {payload['num_games']}")

    # Load config
    config_path = f"/root/nojohns/{config}"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print(f"Config: {config}")

    # Reconstruct MeleeDataset from pre-encoded tensors
    from worldmodel.data.dataset import MeleeDataset
    from worldmodel.model.encoding import EncodingConfig

    # Validate encoding config matches what was used to pre-encode
    # Only compare fields explicitly present in the saved config (not resolved defaults)
    enc_cfg_dict = cfg.get("encoding", {})
    saved_cfg = payload.get("encoding_config", {})
    _TRAINING_ONLY_FIELDS = {"focal_offset", "multi_position", "bidirectional"}
    if saved_cfg:
        # Log feature flags from saved config for visibility
        for flag in ("projectiles", "state_flags", "hitstun"):
            if flag in saved_cfg:
                print(f"  Saved config: {flag}={saved_cfg[flag]}")
        diffs = {}
        for k, saved_v in saved_cfg.items():
            if k in _TRAINING_ONLY_FIELDS or saved_v is None:
                continue
            yaml_v = enc_cfg_dict.get(k)
            if yaml_v is not None and yaml_v != saved_v:
                diffs[k] = (yaml_v, saved_v)
        if diffs:
            raise ValueError(f"Config mismatch! Differences: {diffs}")
    from worldmodel.model.mamba2 import FrameStackMamba2
    from worldmodel.training.metrics import LossWeights
    from worldmodel.training.trainer import Trainer

    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None})

    # Tensor dimension sanity check
    expected_floats = enc_cfg.float_per_player * 2
    actual_floats = payload["floats"].shape[1]
    if actual_floats != expected_floats:
        raise ValueError(
            f"Float tensor width {actual_floats} != expected {expected_floats} from config. "
            f"The .pt file was likely encoded with a different EncodingConfig."
        )

    # Build MeleeDataset from pre-encoded tensors
    dataset = MeleeDataset.from_tensors(
        floats=payload["floats"],
        ints=payload["ints"],
        game_offsets=payload["game_offsets"],
        game_lengths=payload["game_lengths"],
        num_games=payload["num_games"],
        cfg=enc_cfg,
    )

    context_len = cfg["model"]["context_len"]
    train_split = cfg["training"]["train_split"]

    train_ds = dataset.get_frame_dataset(context_len=context_len, train=True, train_split=train_split)
    val_ds = dataset.get_frame_dataset(context_len=context_len, train=False, train_split=train_split)
    print(f"Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    # Build model
    model_cfg = cfg["model"]
    model = FrameStackMamba2(
        cfg=enc_cfg,
        context_len=model_cfg["context_len"],
        d_model=model_cfg.get("d_model", 256),
        d_state=model_cfg.get("d_state", 64),
        n_layers=model_cfg.get("n_layers", 2),
        headdim=model_cfg.get("headdim", 64),
        dropout=model_cfg.get("dropout", 0.1),
        chunk_size=model_cfg.get("chunk_size"),
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} params")

    # wandb
    try:
        import wandb
        wandb.init(
            project="melee-worldmodel",
            name=run_name,
            config={"model": model_cfg, "encoding": enc_cfg_dict, "games": payload["num_games"], "epochs": epochs},
        )
        print(f"wandb: {wandb.run.url}")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"WARNING: wandb init failed: {e}")
        print(f"Training will continue WITHOUT monitoring.")
        print(f"If running --detach, you have NO visibility into this run.")
        print(f"{'='*60}\n")
        wandb = None

    # Train
    loss_cfg = cfg.get("loss_weights", {})
    loss_weights = LossWeights(**loss_cfg) if loss_cfg else None

    save_dir = f"{CHECKPOINT_DIR}/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Resolve resume checkpoint path
    resume_path = None
    if resume:
        resume_path = f"{CHECKPOINT_DIR}/{resume}"
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming from: {resume_path}")

    train_cfg = cfg["training"]

    def commit_checkpoints():
        """Commit checkpoints to Modal volume after each epoch."""
        volume.commit()
        print("Checkpoints committed to volume.")

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=enc_cfg,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        batch_size=train_cfg["batch_size"],
        num_epochs=epochs,
        loss_weights=loss_weights,
        save_dir=save_dir,
        device="cuda",
        rollout_every_n=0,
        num_workers=4,
        resume_from=resume_path,
        scheduled_sampling=train_cfg.get("scheduled_sampling", 0.0),
        ss_noise_scale=train_cfg.get("ss_noise_scale", 0.1),
        ss_anneal_epochs=train_cfg.get("ss_anneal_epochs", 3),
        ss_corrupt_frames=train_cfg.get("ss_corrupt_frames", 3),
        log_interval=train_cfg.get("log_interval"),
        epoch_callback=commit_checkpoints,
    )

    print(f"Training {epochs} epochs on {torch.cuda.get_device_name(0)}...")
    history = trainer.train()

    # Final commit (final.pt)
    volume.commit()
    print("Final checkpoints committed to volume.")

    # Summary
    if history:
        final = history[-1]
        print(f"\n=== DONE ===")
        print(f"Final loss: {final.get('loss/total', 0):.4f}")
        print(f"Action acc: {final.get('metric/p0_action_acc', 0):.3f}")
        print(f"Change acc: {final.get('metric/action_change_acc', 0):.3f}")
        print(f"Pos MAE: {final.get('metric/position_mae', 0):.2f}")
        if 'val_loss/total' in final:
            print(f"Val loss: {final['val_loss/total']:.4f}")

    if wandb and wandb.run:
        wandb.finish()

    volume.commit()
    print("Done. Download checkpoints with:")
    print(f"  .venv/bin/modal volume get melee-training-data /checkpoints ./checkpoints")


@app.function(
    volumes={DATA_VOLUME_PATH: volume},
    image=image,
    timeout=14400,
    cpu=4,
    memory=131072,  # 128GB — holds final ~88GB tensor + one chunk at a time
)
def pre_encode(
    config: str = "worldmodel/experiments/mamba2-medium-gpu.yaml",
    max_games: int = 0,
    output: str = "/encoded.pt",
    chunk_size: int = 2000,
):
    """Pre-encode parsed games on Modal — CPU only, no GPU needed.

    Extracts tar to local NVMe (fast), encodes in chunks to avoid OOM,
    saves final .pt to volume.

    Usage:
        .venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode
        .venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode --max-games 2000
        .venv/bin/modal run --detach worldmodel/scripts/modal_train.py::pre_encode --max-games 22000
    """
    import gc
    import json
    import sys
    import time
    import tarfile
    import logging

    sys.path.insert(0, "/root/nojohns")
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch
    import yaml
    import numpy as np

    # --- Phase 0: Get data onto local NVMe (fast random reads) ---
    tar_path = f"{DATA_VOLUME_PATH}/parsed-v2.tar"
    local_dir = "/tmp/parsed-v2"

    if not os.path.isdir(local_dir):
        if not os.path.exists(tar_path):
            raise FileNotFoundError(
                f"No tar at {tar_path}. "
                "Upload first: modal volume put melee-training-data /tmp/parsed-v2.tar /parsed-v2.tar"
            )
        print(f"Extracting tar to local NVMe...")
        t0 = time.time()
        with tarfile.open(tar_path) as tar:
            tar.extractall("/tmp")
        print(f"Extracted in {time.time() - t0:.1f}s")

    # Load config
    config_path = f"/root/nojohns/{config}"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    enc_cfg_dict = cfg.get("encoding", {})
    from worldmodel.model.encoding import EncodingConfig
    from worldmodel.data.parse import load_game
    from worldmodel.data.dataset import MeleeDataset

    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None})

    # Read meta.json to get game entries (filtered to training-eligible)
    with open(f"{local_dir}/meta.json") as f:
        meta_list = json.load(f)

    entries = [e for e in meta_list if e.get("is_training", False)]
    if max_games > 0:
        entries = entries[:max_games]
    print(f"Will encode {len(entries)} games in chunks of {chunk_size}")

    t_start = time.time()

    # --- Phase 1: Encode chunks, save to /tmp ---
    chunk_dir = "/tmp/encode_chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_meta = []  # (chunk_path, num_frames, game_lengths)
    total_frames = 0
    total_games = 0
    float_width = None
    int_width = None

    for chunk_idx, start in enumerate(range(0, len(entries), chunk_size)):
        chunk_entries = entries[start:start + chunk_size]
        t0 = time.time()

        games = []
        for entry in chunk_entries:
            game_path = f"{local_dir}/games/{entry['slp_md5']}"
            if not os.path.exists(game_path):
                continue
            try:
                g = load_game(game_path, compression=entry.get("compression", "zlib"))
                games.append(g)
            except Exception:
                continue

        if not games:
            continue

        ds = MeleeDataset(games, enc_cfg)

        chunk_path = f"{chunk_dir}/chunk_{chunk_idx:04d}.pt"
        torch.save({"floats": ds.floats, "ints": ds.ints, "lengths": ds.game_lengths}, chunk_path)

        if float_width is None:
            float_width = ds.floats.shape[1]
            int_width = ds.ints.shape[1]

        chunk_meta.append((chunk_path, ds.total_frames, ds.game_lengths))
        total_frames += ds.total_frames
        total_games += ds.num_games

        elapsed = time.time() - t0
        print(
            f"  Chunk {chunk_idx}: {ds.num_games} games, {ds.total_frames:,} frames "
            f"({elapsed:.1f}s) | total: {total_games} games, {total_frames:,} frames"
        )

        del games, ds
        gc.collect()

    print(f"\nPhase 1 done: {total_games} games, {total_frames:,} frames in {time.time() - t_start:.0f}s")

    # --- Phase 2: Pre-allocate final tensors and fill from chunks ---
    print(f"Phase 2: Assembling {total_frames:,} × {float_width} floats, {int_width} ints")

    final_floats = torch.empty(total_frames, float_width)
    final_ints = torch.empty(total_frames, int_width, dtype=torch.int64)
    all_lengths = []
    offset = 0

    for i, (chunk_path, n_frames, lengths) in enumerate(chunk_meta):
        chunk = torch.load(chunk_path, weights_only=False)
        n = chunk["floats"].shape[0]
        final_floats[offset:offset + n] = chunk["floats"]
        final_ints[offset:offset + n] = chunk["ints"]
        all_lengths.extend(lengths)
        offset += n
        os.unlink(chunk_path)  # Free disk as we go
        del chunk
        gc.collect()
        if (i + 1) % 3 == 0 or i == len(chunk_meta) - 1:
            print(f"  Merged {i + 1}/{len(chunk_meta)} chunks ({offset:,}/{total_frames:,} frames)")

    # --- Phase 3: Save to volume ---
    print("Phase 3: Saving to volume...")
    payload = {
        "floats": final_floats,
        "ints": final_ints,
        "game_offsets": torch.tensor(np.cumsum([0] + all_lengths)),
        "game_lengths": all_lengths,
        "num_games": total_games,
        "encoding_config": enc_cfg_dict,  # TODO: save full resolved config (needs dataclasses import)
    }

    output_path = f"{DATA_VOLUME_PATH}{output}"
    torch.save(payload, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    volume.commit()
    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.0f}s. Use with:")
    print(f"  modal run worldmodel/scripts/modal_train.py::train --encoded-file {output}")


@app.function(
    volumes={DATA_VOLUME_PATH: volume},
    image=image,
    timeout=7200,
    cpu=4,
    memory=32768,  # 32GB per worker — one chunk at a time
)
def _encode_chunk(
    chunk_idx: int,
    entry_md5s: list[str],
    entry_compressions: list[str],
    config: str,
    tar_name: str = "parsed-v2.tar",
    chunk_prefix: str = "encode_chunks",
):
    """Encode a single chunk of games. Called in parallel by pre_encode_parallel."""
    import gc
    import sys
    import time
    import tarfile
    import logging

    sys.path.insert(0, "/root/nojohns")
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    import torch
    import yaml

    t_start = time.time()
    print(f"[Chunk {chunk_idx}] Starting — {len(entry_md5s)} games")

    # Extract tar to local NVMe — extract INTO named dir so paths match
    tar_path = f"{DATA_VOLUME_PATH}/{tar_name}"
    tar_stem = tar_name.replace(".tar", "")
    local_dir = f"/tmp/{tar_stem}"
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        t0 = time.time()
        with tarfile.open(tar_path) as tar:
            tar.extractall(local_dir)
        print(f"[Chunk {chunk_idx}] Tar extracted in {time.time() - t0:.1f}s")

    # Load config + encoding
    config_path = f"/root/nojohns/{config}"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    enc_cfg_dict = cfg.get("encoding", {})
    from worldmodel.model.encoding import EncodingConfig
    from worldmodel.data.parse import load_game
    from worldmodel.data.dataset import MeleeDataset

    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None})

    # Load games
    t0 = time.time()
    games = []
    for i, (md5, comp) in enumerate(zip(entry_md5s, entry_compressions)):
        game_path = f"{local_dir}/games/{md5}"
        if not os.path.exists(game_path):
            continue
        try:
            g = load_game(game_path, compression=comp)
            games.append(g)
        except Exception:
            continue
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(entry_md5s) - i - 1) / rate
            print(f"[Chunk {chunk_idx}] Loaded {i + 1}/{len(entry_md5s)} games ({rate:.1f}/s, ETA {eta:.0f}s)")

    print(f"[Chunk {chunk_idx}] Loaded {len(games)}/{len(entry_md5s)} games in {time.time() - t0:.1f}s")

    if not games:
        return {"chunk_idx": chunk_idx, "num_games": 0, "num_frames": 0, "lengths": []}

    # Encode
    t0 = time.time()
    ds = MeleeDataset(games, enc_cfg)
    print(f"[Chunk {chunk_idx}] Encoded {ds.num_games} games, {ds.total_frames:,} frames in {time.time() - t0:.1f}s")

    # Sanitize NaN/Inf (hitstun_remaining can produce NaN from corrupted replays)
    nan_count = torch.isnan(ds.floats).sum().item()
    inf_count = torch.isinf(ds.floats).sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"[Chunk {chunk_idx}] WARNING: {nan_count} NaN, {inf_count} Inf — replacing with 0")
        ds.floats = torch.nan_to_num(ds.floats, nan=0.0, posinf=0.0, neginf=0.0)

    del games
    gc.collect()

    # Save chunk to volume
    chunk_dir = f"{DATA_VOLUME_PATH}/{chunk_prefix}"
    chunk_path = f"{chunk_dir}/chunk_{chunk_idx:04d}.pt"
    os.makedirs(chunk_dir, exist_ok=True)
    torch.save({"floats": ds.floats, "ints": ds.ints, "lengths": ds.game_lengths}, chunk_path)
    volume.commit()

    size_mb = os.path.getsize(chunk_path) / 1e6
    elapsed = time.time() - t_start
    print(f"[Chunk {chunk_idx}] Saved {size_mb:.0f} MB in {elapsed:.0f}s total")

    return {
        "chunk_idx": chunk_idx,
        "num_games": ds.num_games,
        "num_frames": ds.total_frames,
        "lengths": ds.game_lengths,
        "float_width": ds.floats.shape[1],
        "int_width": ds.ints.shape[1],
    }


@app.function(
    volumes={DATA_VOLUME_PATH: volume},
    image=image,
    timeout=7200,
    cpu=2,
    memory=131072,  # 128GB for final tensor assembly
)
def _concat_chunks(
    chunk_results: list[dict],
    output: str,
    enc_cfg_dict: dict,
    chunk_prefix: str = "encode_chunks",
):
    """Concatenate encoded chunks into a single .pt file."""
    import gc
    import sys
    import time
    import logging

    sys.path.insert(0, "/root/nojohns")
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    import torch
    import numpy as np

    # Sort by chunk index, filter empty
    results = sorted([r for r in chunk_results if r["num_games"] > 0], key=lambda r: r["chunk_idx"])

    total_frames = sum(r["num_frames"] for r in results)
    total_games = sum(r["num_games"] for r in results)
    float_width = results[0]["float_width"]
    int_width = results[0]["int_width"]

    print(f"Concatenating {len(results)} chunks: {total_games} games, {total_frames:,} frames")
    print(f"  Tensor shape: ({total_frames:,}, {float_width}) floats, ({total_frames:,}, {int_width}) ints")

    # Pre-allocate final tensors
    final_floats = torch.empty(total_frames, float_width)
    final_ints = torch.empty(total_frames, int_width, dtype=torch.int64)
    all_lengths = []
    offset = 0

    t0 = time.time()
    volume.reload()  # Ensure we see chunks from other workers

    for i, r in enumerate(results):
        chunk_path = f"{DATA_VOLUME_PATH}/{chunk_prefix}/chunk_{r['chunk_idx']:04d}.pt"
        chunk = torch.load(chunk_path, weights_only=False)
        n = chunk["floats"].shape[0]
        final_floats[offset:offset + n] = chunk["floats"]
        final_ints[offset:offset + n] = chunk["ints"]
        all_lengths.extend(r["lengths"])
        offset += n
        del chunk
        gc.collect()
        print(f"  Merged chunk {r['chunk_idx']} ({i + 1}/{len(results)}) — {offset:,}/{total_frames:,} frames")

    print(f"Assembly done in {time.time() - t0:.0f}s")

    # Save final .pt with full resolved encoding config
    print("Saving final .pt...")
    import dataclasses
    from worldmodel.model.encoding import EncodingConfig
    resolved_cfg = dataclasses.asdict(EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None}))

    payload = {
        "floats": final_floats,
        "ints": final_ints,
        "game_offsets": torch.tensor(np.cumsum([0] + all_lengths)),
        "game_lengths": all_lengths,
        "num_games": total_games,
        "encoding_config": resolved_cfg,
    }

    output_path = f"{DATA_VOLUME_PATH}{output}"
    t0 = time.time()
    torch.save(payload, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved: {output_path} ({size_mb:.1f} MB) in {time.time() - t0:.0f}s")

    volume.commit()

    # Cleanup chunks
    for r in results:
        chunk_path = f"{DATA_VOLUME_PATH}/{chunk_prefix}/chunk_{r['chunk_idx']:04d}.pt"
        try:
            os.unlink(chunk_path)
        except FileNotFoundError:
            pass
    volume.commit()

    print(f"Done. Use with:")
    print(f"  modal run worldmodel/scripts/modal_train.py::train --encoded-file {output}")
    return {"total_games": total_games, "total_frames": total_frames, "size_mb": size_mb}


@app.local_entrypoint()
def pre_encode_parallel(
    config: str = "worldmodel/experiments/mamba2-medium-gpu.yaml",
    max_games: int = 0,
    output: str = "/encoded.pt",
    chunk_size: int = 2000,
    dataset: str = "",
    tar_name: str = "parsed-v2.tar",
    super_chunk_size: int = 0,
):
    """Pre-encode in parallel — fans out to N workers, one per chunk.

    For small datasets (<25K games), produces a single .pt file.
    For large datasets, use --super-chunk-size to split into multiple .pt files
    that fit in 128GB RAM for concatenation. E.g. --super-chunk-size 20000
    produces encoded-part-0.pt, encoded-part-1.pt, etc.

    Usage:
        # 12K games, single output:
        .venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode_parallel \\
            --tar-name parsed-v3-12k-gameonly.tar --max-games 12000 --output /encoded-game-v3-12k.pt

        # 100K games, super-chunked:
        .venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode_parallel \\
            --tar-name parsed-v3-ranked-gameonly.tar --super-chunk-size 20000 --output /encoded-v3-ranked
    """
    import json
    import time
    import yaml
    from pathlib import Path

    t0 = time.time()

    # Read meta.json from local dataset dir
    if not dataset:
        for p in [
            Path.home() / "claude-projects/nojohns-training/data/parsed-v2",
            Path("data/parsed-v2"),
        ]:
            if (p / "meta.json").exists():
                dataset = str(p)
                break
    if not dataset or not Path(f"{dataset}/meta.json").exists():
        print("ERROR: Can't find meta.json. Pass --dataset /path/to/parsed-v2")
        return

    print(f"Reading meta.json from {dataset}...")
    with open(f"{dataset}/meta.json") as f:
        meta_list = json.load(f)

    # Use all entries (v3 meta.json is already filtered to GAME-only)
    entries = meta_list
    if max_games > 0:
        entries = entries[:max_games]

    # Load config to pass enc_cfg_dict to concat
    with open(config) as f:
        cfg = yaml.safe_load(f)
    enc_cfg_dict = cfg.get("encoding", {})

    if super_chunk_size > 0:
        # Super-chunk mode: split into groups, each produces a separate .pt
        _run_super_chunked(entries, enc_cfg_dict, config, tar_name, output, chunk_size, super_chunk_size, t0)
    else:
        # Single output mode
        _run_single_encode(entries, enc_cfg_dict, config, tar_name, output, chunk_size, t0)


def _run_single_encode(entries, enc_cfg_dict, config, tar_name, output, chunk_size, t0):
    """Encode all entries into a single .pt file."""
    import time

    chunks = _split_into_chunks(entries, chunk_size)
    chunk_prefix = "encode_chunks"

    print(f"Launching {len(chunks)} parallel workers for {len(entries)} games...")
    print(f"  Chunk size: {chunk_size}, Config: {config}, Tar: {tar_name}")

    t_encode = time.time()
    results = list(_encode_chunk.starmap([
        (i, c["md5s"], c["compressions"], config, tar_name, chunk_prefix)
        for i, c in enumerate(chunks)
    ]))
    encode_time = time.time() - t_encode

    total_games = sum(r["num_games"] for r in results)
    total_frames = sum(r["num_frames"] for r in results)
    print(f"\nEncoding done: {total_games} games, {total_frames:,} frames in {encode_time:.0f}s")

    print("Launching concat...")
    t_concat = time.time()
    final = _concat_chunks.remote(results, output, enc_cfg_dict, chunk_prefix)
    concat_time = time.time() - t_concat

    total_time = time.time() - t0
    print(f"\n=== DONE ===")
    print(f"  Encode: {encode_time:.0f}s ({len(chunks)} parallel workers)")
    print(f"  Concat: {concat_time:.0f}s")
    print(f"  Total:  {total_time:.0f}s")
    print(f"  Output: {final['size_mb']:.0f} MB ({final['total_games']} games, {final['total_frames']:,} frames)")
    print(f"\nTrain with:")
    print(f"  .venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train --encoded-file {output}")


def _run_super_chunked(entries, enc_cfg_dict, config, tar_name, output_prefix, chunk_size, super_chunk_size, t0):
    """Split entries into super-chunks, encode each into a separate .pt file."""
    import time

    # Split entries into super-chunks
    super_chunks = []
    for start in range(0, len(entries), super_chunk_size):
        super_chunks.append(entries[start:start + super_chunk_size])

    print(f"Super-chunk mode: {len(super_chunks)} parts × ~{super_chunk_size} games")
    print(f"  Total games: {len(entries)}, Chunk size: {chunk_size}, Tar: {tar_name}")

    part_files = []
    for sc_idx, sc_entries in enumerate(super_chunks):
        part_output = f"{output_prefix}-part-{sc_idx}.pt"
        chunk_prefix = f"encode_chunks_sc{sc_idx}"

        print(f"\n--- Super-chunk {sc_idx}/{len(super_chunks)} ({len(sc_entries)} games) → {part_output} ---")

        chunks = _split_into_chunks(sc_entries, chunk_size)

        t_encode = time.time()
        results = list(_encode_chunk.starmap([
            (i, c["md5s"], c["compressions"], config, tar_name, chunk_prefix)
            for i, c in enumerate(chunks)
        ]))
        encode_time = time.time() - t_encode

        total_games = sum(r["num_games"] for r in results)
        total_frames = sum(r["num_frames"] for r in results)
        print(f"  Encoded: {total_games} games, {total_frames:,} frames in {encode_time:.0f}s")

        t_concat = time.time()
        final = _concat_chunks.remote(results, part_output, enc_cfg_dict, chunk_prefix)
        concat_time = time.time() - t_concat
        print(f"  Concat: {concat_time:.0f}s → {final['size_mb']:.0f} MB")

        part_files.append(part_output)

    total_time = time.time() - t0
    print(f"\n=== DONE ({len(part_files)} parts) ===")
    print(f"  Total time: {total_time:.0f}s")
    for pf in part_files:
        print(f"  {pf}")
    encoded_files = ",".join(part_files)
    print(f"\nTrain with:")
    print(f"  .venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train --encoded-file \"{encoded_files}\"")


def _split_into_chunks(entries, chunk_size):
    """Split meta entries into chunks for parallel encoding."""
    chunks = []
    for start in range(0, len(entries), chunk_size):
        chunk_entries = entries[start:start + chunk_size]
        chunks.append({
            "md5s": [e["slp_md5"] for e in chunk_entries],
            "compressions": [e.get("compression", "zlib") for e in chunk_entries],
        })
    return chunks


@app.function(
    gpu="A100",
    volumes={DATA_VOLUME_PATH: volume},
    image=image,
    timeout=43200,  # 12h — covers 3x calibrated estimate
    secrets=[modal.Secret.from_name("wandb-key")],
)
def train_policy(
    encoded_file: str = "/encoded-22k.pt",
    epochs: int = 5,
    batch_size: int = 1024,
    predict_player: int = 0,
    run_name: str = "policy-22k",
    resume: str = "",
):
    """Train imitation learning policy on pre-encoded data (T4 GPU).

    Usage:
        .venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train_policy
        .venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train_policy \\
            --encoded-file /encoded-22k.pt --epochs 5 --run-name policy-22k
    """
    import sys
    import time
    import logging

    sys.path.insert(0, "/root/nojohns")
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # Load pre-encoded data — single file or multiple comma-separated files
    import numpy as np
    encoded_files = [f.strip() for f in encoded_file.split(",")]
    payloads = []
    for ef in encoded_files:
        ep = f"{DATA_VOLUME_PATH}{ef}"
        if not os.path.exists(ep):
            raise FileNotFoundError(
                f"No encoded data at {ep}. "
                "Pre-encode first: modal run worldmodel/scripts/modal_train.py::pre_encode"
            )
        print(f"Loading {ep}...")
        t0 = time.time()
        payloads.append(torch.load(ep, weights_only=False))
        print(f"  Loaded in {time.time() - t0:.1f}s — {payloads[-1]['floats'].shape}")

    if len(payloads) == 1:
        payload = payloads[0]
    else:
        print(f"Merging {len(payloads)} encoded files...")
        all_floats = torch.cat([p["floats"] for p in payloads])
        all_ints = torch.cat([p["ints"] for p in payloads])
        all_lengths = []
        for p in payloads:
            all_lengths.extend(p["game_lengths"])
        total_games = sum(p["num_games"] for p in payloads)
        payload = {
            "floats": all_floats,
            "ints": all_ints,
            "game_offsets": torch.tensor(np.cumsum([0] + all_lengths)),
            "game_lengths": all_lengths,
            "num_games": total_games,
            "encoding_config": payloads[0].get("encoding_config", {}),
        }
        del payloads, all_floats, all_ints
        import gc; gc.collect()

    print(f"  Floats: {payload['floats'].shape}")
    print(f"  Ints: {payload['ints'].shape}")
    print(f"  Games: {payload['num_games']}")

    from worldmodel.data.dataset import MeleeDataset
    from worldmodel.data.policy_dataset import PolicyFrameDataset
    from worldmodel.model.encoding import EncodingConfig
    from worldmodel.model.policy_mlp import PolicyMLP
    from worldmodel.training.policy_trainer import PolicyTrainer

    enc_cfg_dict = payload.get("encoding_config", {})
    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None})

    # Build MeleeDataset from pre-encoded tensors (same as world model train)
    dataset = MeleeDataset.from_tensors(
        floats=payload["floats"],
        ints=payload["ints"],
        game_offsets=payload["game_offsets"],
        game_lengths=payload["game_lengths"],
        num_games=payload["num_games"],
        cfg=enc_cfg,
    )

    # Split by game boundaries (90/10)
    train_split = 0.9
    n_games = dataset.num_games
    split_idx = int(n_games * train_split)
    train_range = range(0, split_idx)
    val_range = range(split_idx, n_games)

    context_len = 10
    train_ds = PolicyFrameDataset(dataset, train_range, context_len=context_len, predict_player=predict_player, cfg=enc_cfg)
    val_ds = PolicyFrameDataset(dataset, val_range, context_len=context_len, predict_player=predict_player, cfg=enc_cfg)
    print(f"Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    # Build model
    model = PolicyMLP(
        cfg=enc_cfg,
        context_len=context_len,
        hidden_dim=512,
        trunk_dim=256,
        dropout=0.1,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"PolicyMLP: {param_count:,} params")

    # wandb
    try:
        import wandb
        wandb.init(
            project="melee-worldmodel",
            name=run_name,
            tags=["policy", "imitation"],
            config={
                "model_type": "policy_mlp",
                "params": param_count,
                "context_len": context_len,
                "data": {
                    "num_games_loaded": n_games,
                    "predict_player": predict_player,
                    "encoded_file": encoded_file,
                },
                "training": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": 1e-3,
                },
            },
        )
        print(f"wandb: {wandb.run.url}")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"WARNING: wandb init failed: {e}")
        print(f"Training will continue WITHOUT monitoring.")
        print(f"{'='*60}\n")

    # Checkpoint setup
    save_dir = f"{CHECKPOINT_DIR}/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    resume_path = None
    if resume:
        resume_path = f"{CHECKPOINT_DIR}/{resume}"
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming from: {resume_path}")

    def commit_checkpoints():
        volume.commit()
        print("Checkpoints committed to volume.")

    trainer = PolicyTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=batch_size,
        num_epochs=epochs,
        save_dir=save_dir,
        device="cuda",
        resume_from=resume_path,
        epoch_callback=commit_checkpoints,
        log_interval=1000,
    )

    print(f"Training {epochs} epochs on {torch.cuda.get_device_name(0)}...")
    history = trainer.train()

    volume.commit()
    print("Final checkpoints committed to volume.")

    if history:
        final = history[-1]
        print(f"\n=== DONE ===")
        print(f"Final loss: {final.get('loss/total', 0):.4f}")
        print(f"Stick MAE: {final.get('metric/stick_mae', 0):.4f}")
        btn_pressed = final.get('metric/button_pressed_acc', final.get('val_metric/button_pressed_acc', 0))
        print(f"Button pressed acc: {btn_pressed:.3f}")
        if 'val_loss/total' in final:
            print(f"Val loss: {final['val_loss/total']:.4f}")

    if wandb and wandb.run:
        wandb.finish()

    volume.commit()
    print("Done. Download checkpoint with:")
    print(f"  .venv/bin/modal volume get melee-training-data /checkpoints/{run_name}/best.pt worldmodel/checkpoints/{run_name}/best.pt")


@app.local_entrypoint()
def sweep(
    config: str = "worldmodel/experiments/mamba2-medium-gpu.yaml",
    encoded_file: str = "/encoded.pt",
    epochs: int = 10,
    names: str = "",
):
    """Launch parallel training runs on separate A100s.

    Usage:
        .venv/bin/modal run worldmodel/scripts/modal_train.py::sweep --names "lr3e4,lr1e3,lr3e3"
        .venv/bin/modal run worldmodel/scripts/modal_train.py::sweep --names "run-a,run-b" --epochs 5
    """
    if not names:
        print("ERROR: --names is required (comma-separated run names)")
        print("  Example: --names 'sweep-lr3e4,sweep-lr1e3,sweep-lr3e3'")
        return

    run_names = [n.strip() for n in names.split(",") if n.strip()]
    print(f"Launching {len(run_names)} parallel runs: {run_names}")

    handles = []
    for name in run_names:
        h = train.spawn(
            config=config,
            epochs=epochs,
            run_name=name,
            encoded_file=encoded_file,
        )
        handles.append((name, h))
        print(f"  Spawned: {name}")

    print(f"\nAll {len(handles)} runs launched. Waiting for completion...")
    for name, h in handles:
        h.get()
        print(f"  Done: {name}")
