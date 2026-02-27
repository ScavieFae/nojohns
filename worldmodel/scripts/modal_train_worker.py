"""DDP worker for multi-GPU training on Modal.

Launched by torchrun via modal_train.py::train(). Each process handles one GPU.
Rank 0 does logging, wandb, and checkpoint saving. All ranks do forward/backward.
"""

import argparse
import dataclasses
import gc
import logging
import os
import sys
import time

sys.path.insert(0, "/root/nojohns")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--encoded-file", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    import yaml
    import numpy as np

    # DDP setup
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Only rank 0 logs
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if rank == 0:
        sys.stdout.reconfigure(line_buffering=True)
        print(f"DDP: {world_size} GPUs, rank {rank}")
        print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"CUDA: {torch.version.cuda}")

    # Load data (all ranks load — could optimize with rank 0 broadcast but
    # loading from NVMe is fast enough and simpler)
    DATA_VOLUME_PATH = "/data"
    CHECKPOINT_DIR = f"{DATA_VOLUME_PATH}/checkpoints"

    encoded_files = [f.strip() for f in args.encoded_file.split(",")]
    payloads = []
    for ef in encoded_files:
        ep = f"{DATA_VOLUME_PATH}{ef}"
        if not os.path.exists(ep):
            raise FileNotFoundError(f"No encoded data at {ep}")
        t0 = time.time()
        payloads.append(torch.load(ep, weights_only=False))
        if rank == 0:
            print(f"Loaded {ep} in {time.time() - t0:.1f}s — {payloads[-1]['floats'].shape}")

    if len(payloads) == 1:
        payload = payloads[0]
    else:
        if rank == 0:
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
        gc.collect()

    if rank == 0:
        print(f"  Floats: {payload['floats'].shape}")
        print(f"  Ints: {payload['ints'].shape}")
        print(f"  Games: {payload['num_games']}")

    # Load config
    config_path = f"/root/nojohns/{args.config}"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from worldmodel.data.dataset import MeleeDataset
    from worldmodel.model.encoding import EncodingConfig
    from worldmodel.model.mamba2 import FrameStackMamba2
    from worldmodel.training.metrics import LossWeights
    from worldmodel.training.trainer import Trainer

    # Validate encoding config
    enc_cfg_dict = cfg.get("encoding", {})
    saved_cfg = payload.get("encoding_config", {})
    if saved_cfg:
        resolved_yaml = dataclasses.asdict(EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None}))
        resolved_saved = dataclasses.asdict(EncodingConfig(**{k: v for k, v in saved_cfg.items() if v is not None}))
        if resolved_yaml != resolved_saved:
            diffs = {k: (resolved_yaml.get(k), resolved_saved.get(k))
                     for k in set(list(resolved_yaml) + list(resolved_saved))
                     if resolved_yaml.get(k) != resolved_saved.get(k)}
            raise ValueError(f"Config mismatch! Differences: {diffs}")

    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None})

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
    if rank == 0:
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
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model: {param_count:,} params")

    # wandb — rank 0 only
    wandb = None
    if rank == 0:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(
                project="melee-worldmodel",
                name=args.run_name,
                config={
                    "model": model_cfg,
                    "encoding": enc_cfg_dict,
                    "games": payload["num_games"],
                    "epochs": args.epochs,
                    "ddp_world_size": world_size,
                },
            )
            print(f"wandb: {wandb.run.url}")
        except Exception as e:
            print(f"WARNING: wandb init failed: {e}")
            wandb = None

    # Volume commit callback — rank 0 only
    import modal
    volume = modal.Volume.from_name("melee-training-data")

    def commit_checkpoints():
        if rank == 0:
            volume.commit()
            print("Checkpoints committed to volume.")

    # Train
    loss_cfg = cfg.get("loss_weights", {})
    loss_weights = LossWeights(**loss_cfg) if loss_cfg else None
    save_dir = f"{CHECKPOINT_DIR}/{args.run_name}"
    os.makedirs(save_dir, exist_ok=True)

    resume_path = None
    if args.resume:
        resume_path = f"{CHECKPOINT_DIR}/{args.resume}"
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    train_cfg = cfg["training"]

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=enc_cfg,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        batch_size=train_cfg["batch_size"],
        num_epochs=args.epochs,
        loss_weights=loss_weights,
        save_dir=save_dir if rank == 0 else None,  # only rank 0 saves
        device=f"cuda:{local_rank}",
        rollout_every_n=0,
        num_workers=4,
        resume_from=resume_path,
        scheduled_sampling=train_cfg.get("scheduled_sampling", 0.0),
        ss_noise_scale=train_cfg.get("ss_noise_scale", 0.1),
        ss_anneal_epochs=train_cfg.get("ss_anneal_epochs", 3),
        ss_corrupt_frames=train_cfg.get("ss_corrupt_frames", 3),
        log_interval=train_cfg.get("log_interval"),
        epoch_callback=commit_checkpoints,
        ddp=True,
        ddp_rank=rank,
    )

    if rank == 0:
        print(f"Training {args.epochs} epochs on {world_size}x {torch.cuda.get_device_name(local_rank)}...")

    history = trainer.train()

    # Final cleanup
    if rank == 0:
        volume.commit()
        print("Final checkpoints committed to volume.")

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

        print("Done. Download checkpoints with:")
        print(f"  .venv/bin/modal volume get melee-training-data /checkpoints ./checkpoints")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
