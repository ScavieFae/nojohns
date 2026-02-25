#!/bin/bash
# RunPod setup script for Mamba-2 world model training.
# Run this after SSH-ing into a fresh RunPod pod.
#
# Usage:
#   # On RunPod:
#   bash runpod-setup.sh
#
# Prerequisites:
#   - PyTorch template pod (comes with CUDA, Python 3.10+)
#   - Network volume mounted at /workspace (for data persistence)
#   - wandb API key (set WANDB_API_KEY or run `wandb login`)

set -euo pipefail

echo "=== RunPod Setup for Melee World Model ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not found')"
echo "Python: $(python3 --version)"
echo "CUDA: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'not found')"

# 1. Clone repo (or pull if exists)
cd /workspace
if [ -d "nojohns" ]; then
    echo "=== Repo exists, pulling latest ==="
    cd nojohns && git pull
else
    echo "=== Cloning repo ==="
    git clone https://github.com/shinewave/nojohns.git
    cd nojohns
fi

# 2. Install deps (minimal — just what training needs)
echo "=== Installing dependencies ==="
pip install -q pyarrow pyyaml wandb

# 3. Verify imports work
echo "=== Verifying imports ==="
python3 -c "
from worldmodel.model.encoding import EncodingConfig
from worldmodel.model.mamba2 import FrameStackMamba2
from worldmodel.training.trainer import Trainer
from worldmodel.training.metrics import LossWeights, ActionBreakdown
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
cfg = EncodingConfig(state_age_as_embed=True)
m = FrameStackMamba2(cfg, context_len=10, d_model=384, n_layers=4)
print(f'Model params: {sum(p.numel() for p in m.parameters()):,}')
print('All imports OK')
"

# 4. Check for data
DATA_DIR="/workspace/data/parsed-v2"
if [ -d "$DATA_DIR/games" ]; then
    GAME_COUNT=$(ls "$DATA_DIR/games/" | wc -l)
    echo "=== Found $GAME_COUNT parsed games in $DATA_DIR ==="
else
    echo "=== No data found at $DATA_DIR ==="
    echo "Upload parsed games: rsync -avz --progress /path/to/parsed-v2/ runpod:$DATA_DIR/"
    echo "Or from ScavieFae: rsync -avz queenmab@100.93.8.111:~/claude-projects/nojohns-training/data/parsed-v2/ $DATA_DIR/"
fi

# 5. wandb login
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY" 2>/dev/null
    echo "=== wandb logged in ==="
else
    echo "=== Set WANDB_API_KEY or run: wandb login ==="
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start training:"
echo "  cd /workspace/nojohns"
echo "  python3 -m worldmodel.scripts.train \\"
echo "    --config worldmodel/experiments/mamba2-medium-gpu.yaml \\"
echo "    --dataset /workspace/data/parsed-v2 \\"
echo "    --streaming --buffer-size 2000 \\"
echo "    --run-name mamba2-medium-gpu \\"
echo "    --device cuda"
