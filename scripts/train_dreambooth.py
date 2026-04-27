#!/usr/bin/env python3
"""
scripts/train_dreambooth.py
Command-line interface for DreamBooth fine-tuning.

Usage
-----
    python scripts/train_dreambooth.py \\
        --model_id stabilityai/stable-diffusion-xl-base-1.0 \\
        --instance_data_dir data/my_dog/ \\
        --instance_prompt "a photo of sks dog" \\
        --class_prompt "a photo of a dog" \\
        --output_dir checkpoints/my_dog \\
        --num_train_epochs 4 \\
        --learning_rate 2e-6 \\
        --mixed_precision fp16
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.finetune import DreamBoothTrainer


def parse_args():
    p = argparse.ArgumentParser(
        description="DreamBooth fine-tuning for Stable Diffusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--instance_data_dir", required=True, help="Folder with your subject images.")
    p.add_argument("--instance_prompt", required=True, help='e.g. "a photo of sks dog"')
    p.add_argument("--class_prompt", default=None, help='e.g. "a photo of a dog"')
    p.add_argument("--class_data_dir", default=None, help="Folder with class images.")
    p.add_argument("--output_dir", default="checkpoints/dreambooth")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-6)
    p.add_argument("--lr_scheduler", default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--num_train_epochs", type=int, default=4)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--train_text_encoder", action="store_true")
    p.add_argument("--no_prior_preservation", action="store_true")
    p.add_argument("--prior_loss_weight", type=float, default=1.0)
    p.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    trainer = DreamBoothTrainer(
        model_id=args.model_id,
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        class_data_dir=args.class_data_dir,
        output_dir=args.output_dir,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=args.max_train_steps,
        train_text_encoder=args.train_text_encoder,
        with_prior_preservation=not args.no_prior_preservation,
        prior_loss_weight=args.prior_loss_weight,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        use_8bit_adam=args.use_8bit_adam,
        seed=args.seed,
    )

    trainer.train()
    print(f"\n✅ DreamBooth training complete → {args.output_dir}")


if __name__ == "__main__":
    main()
