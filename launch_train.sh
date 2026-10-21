#!/bin/bash
# =============================================================================
# launch_train.sh — 4-GPU training for MONO depth fine-tuning
#
# Usage:
#   bash launch_train.sh                    # use config file
#   bash launch_train.sh --single           # single GPU
#   bash launch_train.sh --config configs/replica_train.yaml  # custom config
# =============================================================================

set -e  # exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
NUM_GPUS=4
GPUS="0,1,2,3"
CONFIG="configs/replica_train.yaml"
SINGLE_GPU=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --single)  SINGLE_GPU=1; shift ;;
        --config)   CONFIG="$2"; shift 2 ;;
        --gpus)     GPUS="$2"; NUM_GPUS=$(echo "$2" | tr ',' '\n' | wc -l); shift 2 ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------
# Resolve absolute config path
# ------------------------------------------------------------------
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$SCRIPT_DIR/$CONFIG"
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config file not found: $CONFIG"
    exit 1
fi

# ------------------------------------------------------------------
# Validate GPUs
# ------------------------------------------------------------------
if [[ $SINGLE_GPU -eq 0 ]]; then
    AVAILABLE_GPUS=$(nvidia-smi -L | grep "NVIDIA GeForce RTX 3090" | wc -l)
    echo "Available GPUs: $AVAILABLE_GPUS"
    if [[ $NUM_GPUS -gt $AVAILABLE_GPUS ]]; then
        echo "[WARN] Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
        NUM_GPUS=$AVAILABLE_GPUS
        GPUS=$(seq 0 $((NUM_GPUS - 1)) | paste -sd ',' -)
    fi
    export CUDA_VISIBLE_DEVICES="$GPUS"
    echo "Using GPUs: $GPUS"
fi

# ------------------------------------------------------------------
# Extract model name from config for log directory
# ------------------------------------------------------------------
MODEL_NAME=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG')); print(d.get('model_name','unnamed'))")
LOG_DIR=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG')); print(d.get('log_dir','./tmp'))")
echo "Model name: $MODEL_NAME"
echo "Log dir   : $LOG_DIR/$MODEL_NAME"

# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
if [[ $SINGLE_GPU -eq 1 ]]; then
    echo ""
    echo ">>> SINGLE-GPU MODE"
    echo ">>> python train.py --config $CONFIG"
    echo ""
    python train.py --config "$CONFIG"
else
    echo ""
    echo ">>> 4-GPU DDP MODE (torchrun)"
    echo ">>> torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 train.py --config $CONFIG"
    echo ""
    torchrun --nproc_per_node=$NUM_GPUS \
             --master_port=29500 \
             --standalone \
             train.py --config "$CONFIG"
fi
