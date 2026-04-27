#!/usr/bin/env python3
"""
scripts/train_lora.py
Command-line interface for LoRA fine-tuning.

Usage
-----
    python scripts/train_lora.py \\
        --model_id stabilityai/stable-diffusion-xl-base-1.0 \\
        --dataset_dir data/my_style/ \\
        --output_dir checkpoints/my_style_lora \\
        --rank 16 \\
        --num_train_epochs 10 \\
        --learning_rate 1e-4
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.finetune import LoRATrainer


def parse_args():
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning for Stable Diffusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--output_dir", default="checkpoints/lora")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank (r).")
    p.add_argument("--alpha", type=int, default=16, help="LoRA alpha.")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    trainer = LoRATrainer(
        model_id=args.model_id,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        rank=args.rank,
        alpha=args.alpha,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=args.max_train_steps,
        save_steps=args.save_steps,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
    )
    trainer.train()
    print(f"\n✅ LoRA training complete → {args.output_dir}")


if __name__ == "__main__":
    main()
