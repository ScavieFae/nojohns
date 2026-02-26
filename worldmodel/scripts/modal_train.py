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
    gpu="A100",
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
):
    """Train from pre-encoded .pt file — instant data loading."""
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

    # Load pre-encoded data — single file, instant
    encoded_path = f"{DATA_VOLUME_PATH}{encoded_file}"
    if not os.path.exists(encoded_path):
        raise FileNotFoundError(
            f"No encoded data at {encoded_path}. "
            "Pre-encode first: modal run worldmodel/scripts/modal_train.py::pre_encode"
        )

    print(f"Loading {encoded_path}...")
    t0 = time.time()
    payload = torch.load(encoded_path, weights_only=False)
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"  Floats: {payload['floats'].shape}")
    print(f"  Ints: {payload['ints'].shape}")
    print(f"  Games: {payload['num_games']}")

    # Load config
    config_path = f"/root/nojohns/{config}"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print(f"Config: {config}")

    # Validate encoding config matches what was used to pre-encode
    enc_cfg_dict = cfg.get("encoding", {})
    saved_cfg = payload.get("encoding_config", {})
    if saved_cfg and saved_cfg != enc_cfg_dict:
        raise ValueError(
            f"Config mismatch! Encoded with {saved_cfg}, training with {enc_cfg_dict}"
        )

    # Reconstruct MeleeDataset from pre-encoded tensors
    from worldmodel.data.dataset import MeleeDataset
    from worldmodel.model.encoding import EncodingConfig
    from worldmodel.model.mamba2 import FrameStackMamba2
    from worldmodel.training.metrics import LossWeights
    from worldmodel.training.trainer import Trainer

    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None})

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
        print(f"wandb failed: {e}")
        wandb = None

    # Train
    loss_cfg = cfg.get("loss_weights", {})
    loss_weights = LossWeights(**loss_cfg) if loss_cfg else None

    save_dir = f"{CHECKPOINT_DIR}/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=enc_cfg,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        batch_size=cfg["training"]["batch_size"],
        num_epochs=epochs,
        loss_weights=loss_weights,
        save_dir=save_dir,
        device="cuda",
        rollout_every_n=0,
        num_workers=4,
    )

    print(f"Training {epochs} epochs on {torch.cuda.get_device_name(0)}...")
    history = trainer.train()

    # Commit checkpoints
    volume.commit()
    print("Checkpoints committed to volume.")

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
    memory=32768,
)
def pre_encode(
    config: str = "worldmodel/experiments/mamba2-medium-gpu.yaml",
    max_games: int = 0,
    output: str = "/encoded.pt",
):
    """Pre-encode parsed games on Modal — CPU only, no GPU needed.

    Reads raw parquet tar from volume, encodes to tensors, writes .pt back.
    Eliminates the local-encode-then-upload-7GB step.

    Usage:
        .venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode
        .venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode --max-games 2000 --output /encoded-2k.pt
    """
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

    # Extract tar if raw parquet dir doesn't exist yet
    tar_path = f"{DATA_VOLUME_PATH}/parsed-v2.tar"
    extract_dir = f"{DATA_VOLUME_PATH}/parsed-v2"
    if not os.path.isdir(extract_dir):
        if not os.path.exists(tar_path):
            raise FileNotFoundError(
                f"No tar at {tar_path} and no extracted dir at {extract_dir}. "
                "Upload first: modal volume put melee-training-data /tmp/parsed-v2.tar /parsed-v2.tar"
            )
        print(f"Extracting {tar_path}...")
        t0 = time.time()
        with tarfile.open(tar_path) as tar:
            tar.extractall(DATA_VOLUME_PATH)
        print(f"Extracted in {time.time() - t0:.1f}s")
        volume.commit()

    # Load config
    config_path = f"/root/nojohns/{config}"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    enc_cfg_dict = cfg.get("encoding", {})
    from worldmodel.model.encoding import EncodingConfig
    from worldmodel.data.parse import load_games_from_dir
    from worldmodel.data.dataset import MeleeDataset

    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None})

    load_kwargs = {}
    if max_games > 0:
        load_kwargs["max_games"] = max_games

    print(f"Loading games from {extract_dir}...")
    t0 = time.time()
    games = load_games_from_dir(extract_dir, **load_kwargs)
    print(f"Loaded {len(games)} games in {time.time() - t0:.1f}s")

    print("Encoding...")
    t1 = time.time()
    dataset = MeleeDataset(games, enc_cfg)
    print(f"Encoded in {time.time() - t1:.1f}s")
    print(f"  Floats: {dataset.floats.shape}")
    print(f"  Ints: {dataset.ints.shape}")
    print(f"  Total frames: {dataset.total_frames:,}")

    payload = {
        "floats": dataset.floats,
        "ints": dataset.ints,
        "game_offsets": torch.tensor(dataset.game_offsets),
        "game_lengths": dataset.game_lengths,
        "num_games": dataset.num_games,
        "encoding_config": enc_cfg_dict,
    }

    output_path = f"{DATA_VOLUME_PATH}{output}"
    torch.save(payload, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    volume.commit()
    print(f"Committed to volume. Use with:")
    print(f"  modal run worldmodel/scripts/modal_train.py::train --encoded-file {output}")


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
