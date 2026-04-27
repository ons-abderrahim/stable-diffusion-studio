#!/usr/bin/env python3
"""
scripts/generate.py
Command-line interface for text-to-image generation.

Usage
-----
    python scripts/generate.py \\
        --prompt "A majestic wolf under the northern lights" \\
        --model stabilityai/stable-diffusion-xl-base-1.0 \\
        --num-images 4 \\
        --steps 30 \\
        --cfg 7.5 \\
        --output output/

    # With a style preset
    python scripts/generate.py \\
        --prompt "Portrait of a samurai warrior" \\
        --style cinematic \\
        --seed 42

    # Load a LoRA adapter
    python scripts/generate.py \\
        --prompt "a photo of sks dog in the park" \\
        --lora checkpoints/my_dog_lora/lora_weights_final \\
        --output output/dog/
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import StableDiffusionGenerator, PromptEngineer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate images from text prompts using Stable Diffusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core
    p.add_argument("--prompt", "-p", required=True, help="Text prompt.")
    p.add_argument("--negative-prompt", "-np", default=None, help="Negative prompt.")
    p.add_argument(
        "--model", "-m",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HuggingFace model ID.",
    )

    # Generation
    p.add_argument("--num-images", "-n", type=int, default=4, help="Number of images.")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=30, help="Denoising steps.")
    p.add_argument("--cfg", type=float, default=7.5, help="CFG / guidance scale.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")
    p.add_argument("--scheduler", default="dpm++", choices=["dpm++", "euler_a"])

    # Prompt helpers
    p.add_argument(
        "--style", default=None,
        help="Apply a style preset: photorealistic | cinematic | fantasy | "
             "anime | oil_painting | watercolor | pixel_art | concept_art",
    )
    p.add_argument(
        "--neg-level", default="medium", choices=["light", "medium", "strict"],
        help="Auto-generate negative prompt at this strictness level.",
    )

    # Adapters
    p.add_argument("--lora", default=None, help="Path to LoRA weights directory.")

    # Output
    p.add_argument("--output", "-o", default="output/", help="Output directory.")
    p.add_argument("--save-grid", action="store_true", default=True, help="Save image grid.")
    p.add_argument("--save-individual", action="store_true", help="Also save individual images.")
    p.add_argument("--prefix", default="generated", help="Filename prefix.")

    # Hardware
    p.add_argument("--device", default=None, help="cuda | cpu (auto-detected by default).")
    p.add_argument("--no-xformers", action="store_true", help="Disable xFormers.")
    p.add_argument("--cpu-offload", action="store_true", help="Enable CPU offloading.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    pe = PromptEngineer()

    # Apply style preset to prompt
    prompt = args.prompt
    if args.style:
        prompt = pe.apply_preset(args.style, prompt)
        logger.info(f"Enhanced prompt: {prompt}")

    # Build negative prompt
    negative_prompt = args.negative_prompt
    if not negative_prompt:
        negative_prompt = pe.get_negative_prompt(level=args.neg_level)
        if args.style:
            preset_neg = pe.get_preset_negative(args.style)
            if preset_neg:
                negative_prompt = f"{negative_prompt}, {preset_neg}"

    # Warn about long prompts
    warning = pe.warn_length(prompt)
    if warning:
        logger.warning(warning)

    # Build generator
    generator = StableDiffusionGenerator(
        model_id=args.model,
        device=args.device,
        scheduler=args.scheduler,
        enable_xformers=not args.no_xformers,
        cpu_offload=args.cpu_offload,
        lora_weights=args.lora,
    )

    logger.info(
        f"Generating {args.num_images} image(s) at {args.width}×{args.height}, "
        f"{args.steps} steps, CFG={args.cfg}"
    )

    images = generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images=args.num_images,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_grid:
        grid_path = output_dir / f"{args.prefix}_grid.png"
        generator.save_grid(images, grid_path)
        print(f"✅ Grid saved → {grid_path}")

    if args.save_individual:
        paths = generator.save_all(images, output_dir, prefix=args.prefix)
        print(f"✅ Saved {len(paths)} images → {output_dir}")


if __name__ == "__main__":
    main()
