"""
src/utils/metrics.py
Evaluation metrics: FID, CLIP Score, and Inception Score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)

_TRANSFORM = transforms.Compose([
    transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

_CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])


def _load_images_from_dir(directory: Path, limit: Optional[int] = None) -> List[Image.Image]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in exts)
    if limit:
        paths = paths[:limit]
    return [Image.open(p).convert("RGB") for p in paths]


# ---------------------------------------------------------------------------
# FID — Fréchet Inception Distance
# ---------------------------------------------------------------------------

def compute_fid(
    generated_dir: Path,
    reference_dir: Path,
    device: str = "cuda",
    batch_size: int = 32,
) -> float:
    """
    Compute FID between generated and reference image sets.

    Lower is better. FID < 10 is excellent; < 30 is good.

    Requires `torchmetrics[image]` or `pytorch-fid`.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        raise ImportError(
            "Install torchmetrics: pip install torchmetrics[image]"
        )

    fid = FrechetInceptionDistance(feature=2048).to(device)

    def _add_images(directory: Path, real: bool) -> None:
        images = _load_images_from_dir(directory)
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([_TRANSFORM(img) for img in batch]).to(device)
            # FID expects uint8 in [0, 255]
            tensors = ((tensors + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            fid.update(tensors, real=real)

    logger.info("Computing FID — loading reference images …")
    _add_images(reference_dir, real=True)
    logger.info("Computing FID — loading generated images …")
    _add_images(generated_dir, real=False)

    score = fid.compute().item()
    logger.info(f"FID: {score:.2f}")
    return score


# ---------------------------------------------------------------------------
# CLIP Score
# ---------------------------------------------------------------------------

def compute_clip_score(
    images: List[Image.Image],
    prompts: List[str],
    model_name: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
    batch_size: int = 32,
) -> float:
    """
    Compute average CLIP similarity between images and their prompts.

    Higher is better. Typical range: 0.20–0.35.

    Parameters
    ----------
    images:
        List of PIL images.
    prompts:
        Corresponding text prompts (one per image).
    """
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")

    assert len(images) == len(prompts), "images and prompts must have equal length"

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="CLIP Score"):
            batch_images = images[i : i + batch_size]
            batch_prompts = prompts[i : i + batch_size]

            inputs = processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            outputs = model(**inputs)
            # Cosine similarity between image and text embeddings
            img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            batch_scores = (img_emb * txt_emb).sum(dim=-1).cpu().tolist()
            scores.extend(batch_scores)

    mean_score = float(np.mean(scores))
    logger.info(f"CLIP Score: {mean_score:.4f}")
    return mean_score


# ---------------------------------------------------------------------------
# Inception Score
# ---------------------------------------------------------------------------

def compute_inception_score(
    images: List[Image.Image],
    device: str = "cuda",
    batch_size: int = 32,
    splits: int = 10,
) -> Tuple[float, float]:
    """
    Compute Inception Score (IS) for a set of generated images.

    Returns (mean, std). Higher is better.
    Typical IS for good SD outputs: 15–30.
    """
    try:
        import torch.nn.functional as F
        from torchvision.models import inception_v3
    except ImportError:
        raise ImportError("torchvision required")

    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    probs_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Inception Score"):
            batch = images[i : i + batch_size]
            tensors = torch.stack([_TRANSFORM(img) for img in batch]).to(device)
            logits = model(tensors)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)

    probs_all = np.concatenate(probs_list, axis=0)
    n = probs_all.shape[0]
    split_size = n // splits

    scores = []
    for k in range(splits):
        part = probs_all[k * split_size : (k + 1) * split_size]
        p_y = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-16) - np.log(p_y + 1e-16))
        scores.append(np.exp(kl.sum(axis=1).mean()))

    mean, std = float(np.mean(scores)), float(np.std(scores))
    logger.info(f"Inception Score: {mean:.2f} ± {std:.2f}")
    return mean, std


# ---------------------------------------------------------------------------
# Convenience: run all metrics
# ---------------------------------------------------------------------------

def evaluate_all(
    generated_dir: Path,
    reference_dir: Path,
    prompts: Optional[List[str]] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Run FID, CLIP Score, and Inception Score in one call.

    Returns a dict with keys: fid, clip_score, is_mean, is_std.
    """
    results: Dict[str, float] = {}

    logger.info("=== Evaluation ===")

    results["fid"] = compute_fid(generated_dir, reference_dir, device=device)

    gen_images = _load_images_from_dir(generated_dir)

    if prompts:
        if len(prompts) != len(gen_images):
            logger.warning(
                f"Prompt count ({len(prompts)}) ≠ image count ({len(gen_images)}). "
                "Truncating to minimum."
            )
            n = min(len(prompts), len(gen_images))
            prompts, gen_images_clip = prompts[:n], gen_images[:n]
        else:
            gen_images_clip = gen_images
        results["clip_score"] = compute_clip_score(gen_images_clip, prompts, device=device)

    is_mean, is_std = compute_inception_score(gen_images, device=device)
    results["is_mean"] = is_mean
    results["is_std"] = is_std

    # Pretty print
    print("\n┌──────────────────────────────────────┐")
    print("│         Evaluation Results           │")
    print("├─────────────────┬────────────────────┤")
    print(f"│ FID ↓           │ {results['fid']:<18.2f} │")
    if "clip_score" in results:
        print(f"│ CLIP Score ↑    │ {results['clip_score']:<18.4f} │")
    print(f"│ IS ↑            │ {is_mean:.1f} ± {is_std:.1f}{'':>10} │")
    print("└─────────────────┴────────────────────┘\n")

    return results
