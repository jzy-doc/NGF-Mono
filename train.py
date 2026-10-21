"""
train.py — supports single-GPU and multi-GPU (DDP) training via torchrun.

Usage:
  # Single GPU:
  python train.py --data_path /data3/jzy/vision/monodepth2/kitti_data \
      --model_name mytrain --num_epochs 30 --batch_size 12 --png

  # 4-GPU (torchrun auto-assigns GPUs via LOCAL_RANK):
  torchrun --nproc_per_node=4 --master_port=29500 train.py \
      --data_path /data3/jzy/vision/monodepth2/kitti_data \
      --model_name mytrain --num_epochs 30 --batch_size 12 --png

  # With YAML config:
  python train.py --config configs/replica_train.yaml
  torchrun --nproc_per_node=4 --master_port=29500 train.py --config configs/replica_train.yaml

torchrun sets LOCAL_RANK / RANK / WORLD_SIZE env vars automatically.
Rank 0 prints progress and saves checkpoints; other ranks train silently.
"""

from __future__ import absolute_import, division, print_function

import os
import yaml
import argparse

import torch
import torch.distributed as dist

from options import LiteMonoOptions
from trainer import Trainer


def main():
    # ------------------------------------------------------------------
    # 1. Parse args (YAML config supported)
    # ------------------------------------------------------------------
    # First pass: only capture --config so we know which YAML to load
    options = LiteMonoOptions()
    _pre, _unparsed = options.parser.parse_known_args()
    config_path = getattr(_pre, 'config', None)

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        # Build namespace from YAML defaults
        base = argparse.Namespace(**yaml_cfg)
        # Re-parse with YAML as defaults: CLI args override YAML
        options2 = LiteMonoOptions()
        options2.parser.set_defaults(**yaml_cfg)
        opts = options2.parser.parse_args()
        # Explicit --config value always wins (may override YAML model_name etc.)
        if config_path:
            opts.config = config_path
    else:
        opts = options.parse()

    # ------------------------------------------------------------------
    # 2. DDP environment
    # ------------------------------------------------------------------
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        is_main = (local_rank == 0)
        torch.cuda.set_device(local_rank)
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    else:
        local_rank = 0
        world_size = 1
        is_main = True

    # ------------------------------------------------------------------
    # 3. Banner (main rank only)
    # ------------------------------------------------------------------
    if is_main:
        gpu_desc = f"GPU {local_rank}" if world_size > 1 else "single GPU"
        print("=" * 60)
        print("MONO Training")
        print("=" * 60)
        print(f"  GPUs        : {gpu_desc}  (world_size={world_size})")
        print(f"  Model name : {opts.model_name}")
        print(f"  Batch size : {opts.batch_size} (per GPU, total={opts.batch_size * max(world_size, 1)})")
        print(f"  Epochs     : {opts.num_epochs}")
        print(f"  Geometric  : {getattr(opts, 'use_geometric_loss', False)}")
        print(f"  Log dir    : {opts.log_dir}/{opts.model_name}")
        print(f"  Config     : {getattr(opts, 'config', 'N/A')}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 4. Run training (one Trainer per process)
    # ------------------------------------------------------------------
    trainer = Trainer(opts, gpu_id=local_rank, world_size=world_size)
    trainer.train()

    # ------------------------------------------------------------------
    # 5. Cleanup
    # ------------------------------------------------------------------
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
