#!/usr/bin/env python3
"""
scripts/evaluate.py
Command-line interface for evaluating generated images.

Usage
-----
    python scripts/evaluate.py \\
        --generated_dir output/generated/ \\
        --reference_dir data/reference/ \\
        --metrics fid clip_score is \\
        --prompts_file prompts.txt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.metrics import evaluate_all


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate generated images with FID, CLIP Score, and IS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--generated_dir", required=True, help="Folder of generated images.")
    p.add_argument("--reference_dir", required=True, help="Folder of real reference images.")
    p.add_argument(
        "--metrics", nargs="+",
        default=["fid", "clip_score", "is"],
        choices=["fid", "clip_score", "is"],
    )
    p.add_argument(
        "--prompts_file", default=None,
        help="Text file with one prompt per line (required for CLIP Score).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_json", default=None, help="Save results to a JSON file.")
    return p.parse_args()


def main():
    args = parse_args()

    prompts = None
    if "clip_score" in args.metrics:
        if args.prompts_file:
            prompts = Path(args.prompts_file).read_text().strip().splitlines()
        else:
            logging.warning(
                "CLIP Score requested but --prompts_file not provided. Skipping CLIP."
            )

    results = evaluate_all(
        generated_dir=Path(args.generated_dir),
        reference_dir=Path(args.reference_dir),
        prompts=prompts,
        device=args.device,
    )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📊 Results saved → {out}")


if __name__ == "__main__":
    main()
