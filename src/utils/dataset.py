"""
src/utils/dataset.py
Dataset classes for DreamBooth and general text-image fine-tuning.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect_images(directory: Path) -> List[Path]:
    """Recursively collect all image files in a directory."""
    return sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# DreamBooth Dataset
# ---------------------------------------------------------------------------

class DreamBoothDataset(Dataset):
    """
    Dataset for DreamBooth training.

    Yields instance images (your subject) and, optionally, class images
    (generic examples of the subject class) for prior preservation.

    Directory layout
    ----------------
    instance_data_root/
        image_001.jpg
        image_002.png
        ...

    class_data_root/        ← optional, for prior preservation
        class_001.jpg
        ...
    """

    def __init__(
        self,
        instance_data_root: Path,
        instance_prompt: str,
        tokenizer: CLIPTokenizer,
        class_data_root: Optional[Path] = None,
        class_prompt: Optional[str] = None,
        size: int = 512,
        center_crop: bool = True,
        augment: bool = False,
    ) -> None:
        self.instance_images = _collect_images(instance_data_root)
        if not self.instance_images:
            raise ValueError(f"No images found in {instance_data_root}")

        self.instance_prompt = instance_prompt
        self.class_images = _collect_images(class_data_root) if class_data_root else []
        self.class_prompt = class_prompt
        self.tokenizer = tokenizer
        self.use_prior = bool(self.class_images)

        self.image_transforms = _build_transforms(size, center_crop, augment)

    def __len__(self) -> int:
        return max(len(self.instance_images), len(self.class_images) or 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        instance_path = self.instance_images[idx % len(self.instance_images)]
        instance_image = _load_image(instance_path)

        batch: Dict[str, torch.Tensor] = {
            "pixel_values": self.image_transforms(instance_image),
            "input_ids": self._tokenize(self.instance_prompt),
        }

        if self.use_prior and self.class_prompt:
            class_path = self.class_images[idx % len(self.class_images)]
            class_image = _load_image(class_path)
            batch["class_pixel_values"] = self.image_transforms(class_image)
            batch["class_input_ids"] = self._tokenize(self.class_prompt)

            # Stack instance + class along batch dim for prior preservation loss
            batch["pixel_values"] = torch.cat(
                [batch["pixel_values"].unsqueeze(0),
                 batch["class_pixel_values"].unsqueeze(0)], dim=0
            ).squeeze(0)
            batch["input_ids"] = torch.cat(
                [batch["input_ids"].unsqueeze(0),
                 batch["class_input_ids"].unsqueeze(0)], dim=0
            ).squeeze(0)

        return batch

    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]


# ---------------------------------------------------------------------------
# General Text-Image Dataset (for LoRA / Textual Inversion)
# ---------------------------------------------------------------------------

class TextImageDataset(Dataset):
    """
    General-purpose text-image dataset.

    Supports two caption modes:
    1. Sidecar .txt files (image_001.jpg → image_001.txt)
    2. A single captions.txt file mapping filenames to captions

    Directory layout (sidecar mode)
    --------------------------------
    data/
        image_001.jpg
        image_001.txt      ← "a painting of a sunset"
        image_002.png
        image_002.txt
        ...

    Directory layout (captions file mode)
    ---------------------------------------
    data/
        images/
            001.jpg
            002.jpg
        captions.txt       ← "001.jpg|a painting of a sunset"
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizer: CLIPTokenizer,
        captions_file: Optional[Path] = None,
        default_caption: str = "a high quality image",
        size: int = 512,
        center_crop: bool = True,
        augment: bool = False,
        random_flip: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.default_caption = default_caption
        self.image_transforms = _build_transforms(size, center_crop, augment, random_flip)

        if captions_file:
            self.samples = self._load_captions_file(captions_file)
        else:
            self.samples = self._load_sidecar_captions(data_dir)

        if not self.samples:
            raise ValueError(f"No image-caption pairs found in {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, caption = self.samples[idx]
        image = _load_image(image_path)
        pixel_values = self.image_transforms(image)
        input_ids = self._tokenize(caption)
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # ------------------------------------------------------------------

    def _load_sidecar_captions(self, directory: Path) -> List[tuple]:
        samples = []
        for img_path in _collect_images(directory):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                caption = txt_path.read_text(encoding="utf-8").strip()
            else:
                caption = self.default_caption
            samples.append((img_path, caption))
        return samples

    def _load_captions_file(self, captions_file: Path) -> List[tuple]:
        samples = []
        base_dir = captions_file.parent
        for line in captions_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "|" in line:
                filename, caption = line.split("|", 1)
                img_path = base_dir / filename.strip()
                if img_path.exists():
                    samples.append((img_path, caption.strip()))
        return samples

    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> Image.Image:
    """Load an image and convert to RGB."""
    return Image.open(path).convert("RGB")


def _build_transforms(
    size: int,
    center_crop: bool,
    augment: bool,
    random_flip: bool = False,
) -> transforms.Compose:
    ops = []
    ops.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR))

    if center_crop:
        ops.append(transforms.CenterCrop(size))
    else:
        ops.append(transforms.RandomCrop(size))

    if random_flip:
        ops.append(transforms.RandomHorizontalFlip())

    if augment:
        ops += [
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(degrees=5),
        ]

    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    return transforms.Compose(ops)
