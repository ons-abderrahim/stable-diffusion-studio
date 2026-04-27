"""
src/pipeline/generator.py
Core image generation pipeline wrapping HuggingFace Diffusers.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import List, Optional, Union

import torch
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image

from src.utils.visualization import make_image_grid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "stabilityai/stable-diffusion-v1-5": StableDiffusionPipeline,
    "stabilityai/stable-diffusion-2-1": StableDiffusionPipeline,
    "stabilityai/stable-diffusion-xl-base-1.0": StableDiffusionXLPipeline,
    "stabilityai/sdxl-turbo": StableDiffusionXLPipeline,
    "stabilityai/stable-diffusion-3-medium": StableDiffusion3Pipeline,
    "stabilityai/stable-diffusion-3.5-large": StableDiffusion3Pipeline,
}


class StableDiffusionGenerator:
    """
    High-level wrapper around HuggingFace Diffusers pipelines.

    Supports SD 1.5, SD 2.1, SDXL, SDXL-Turbo, SD3, and SD3.5.

    Example
    -------
    >>> gen = StableDiffusionGenerator("stabilityai/stable-diffusion-xl-base-1.0")
    >>> images = gen.generate("A serene Japanese garden at sunrise")
    >>> gen.save_grid(images, "output/garden.png")
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        scheduler: str = "dpm++",
        enable_xformers: bool = True,
        enable_attention_slicing: bool = False,
        cpu_offload: bool = False,
        torch_compile: bool = False,
        lora_weights: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device == "cuda" else torch.float32

        logger.info(f"Loading model: {model_id} on {self.device} ({self.dtype})")

        pipe_cls = SUPPORTED_MODELS.get(model_id, DiffusionPipeline)
        self.pipe = pipe_cls.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )

        # Optional LoRA adapter
        if lora_weights:
            self.pipe.load_lora_weights(lora_weights)
            logger.info(f"Loaded LoRA weights from {lora_weights}")

        # Scheduler
        self._set_scheduler(scheduler)

        # Memory optimizations
        if not cpu_offload:
            self.pipe.to(self.device)
        else:
            self.pipe.enable_model_cpu_offload()

        if enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("xFormers memory-efficient attention enabled.")
            except Exception:
                logger.warning("xFormers not available; skipping.")

        if enable_attention_slicing:
            self.pipe.enable_attention_slicing()

        if torch_compile:
            self.pipe.unet = torch.compile(
                self.pipe.unet, mode="reduce-overhead", fullgraph=True
            )

        logger.info("Pipeline ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images: int = 1,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        prompt_2: Optional[str] = None,  # SDXL second encoder
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate images from a text prompt.

        Parameters
        ----------
        prompt:
            Text description of the desired image.
        negative_prompt:
            Things to avoid in the image.
        num_images:
            Number of images to generate.
        height, width:
            Output resolution (must be multiples of 8).
        num_inference_steps:
            Denoising steps. More → higher quality, slower.
        guidance_scale:
            CFG scale. Higher → closer to prompt, less diversity.
        seed:
            Random seed for reproducibility. None = random.

        Returns
        -------
        List of PIL Images.
        """
        generators = self._make_generators(seed, num_images)

        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generators,
            **kwargs,
        )

        # SDXL supports a second text encoder
        if prompt_2 and isinstance(self.pipe, StableDiffusionXLPipeline):
            call_kwargs["prompt_2"] = prompt_2

        output = self.pipe(**call_kwargs)
        return output.images

    def generate_variations(
        self,
        prompt: str,
        seeds: List[int],
        **kwargs,
    ) -> List[Image.Image]:
        """Generate the same prompt with multiple seeds for comparison."""
        images = []
        for seed in seeds:
            imgs = self.generate(prompt, seed=seed, num_images=1, **kwargs)
            images.extend(imgs)
        return images

    def interpolate_prompts(
        self,
        prompt_start: str,
        prompt_end: str,
        steps: int = 6,
        seed: int = 42,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Linearly interpolate between two prompt embeddings to create
        a smooth visual transition.
        """
        images = []
        for i in range(steps):
            alpha = i / (steps - 1)
            # Blend at the embedding level for a smooth transition
            blended = self._blend_prompts(prompt_start, prompt_end, alpha)
            imgs = self.generate(blended, seed=seed, num_images=1, **kwargs)
            images.extend(imgs)
        return images

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def save_grid(
        self,
        images: List[Image.Image],
        path: Union[str, Path],
        cols: Optional[int] = None,
    ) -> None:
        """Save images as a grid PNG."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = cols or math.ceil(math.sqrt(len(images)))
        grid = make_image_grid(images, cols=cols)
        grid.save(path)
        logger.info(f"Grid saved → {path}")

    def save_all(
        self,
        images: List[Image.Image],
        output_dir: Union[str, Path],
        prefix: str = "image",
    ) -> List[Path]:
        """Save each image individually."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, img in enumerate(images):
            p = output_dir / f"{prefix}_{i:04d}.png"
            img.save(p)
            paths.append(p)
        logger.info(f"Saved {len(images)} images to {output_dir}")
        return paths

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _set_scheduler(self, name: str) -> None:
        scheduler_map = {
            "dpm++": DPMSolverMultistepScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
        }
        cls = scheduler_map.get(name, DPMSolverMultistepScheduler)
        self.pipe.scheduler = cls.from_config(self.pipe.scheduler.config)
        logger.info(f"Scheduler: {cls.__name__}")

    def _make_generators(
        self, seed: Optional[int], n: int
    ) -> List[torch.Generator]:
        generators = []
        for i in range(n):
            g = torch.Generator(device=self.device)
            g.manual_seed(seed + i if seed is not None else random.randint(0, 2**32))
            generators.append(g)
        return generators

    def _blend_prompts(self, p1: str, p2: str, alpha: float) -> str:
        """Simple token-level blend: return p1 for alpha=0, p2 for alpha=1."""
        if alpha < 0.5:
            weight = 1 - 2 * alpha
            return f"({p1}:{weight:.2f}) ({p2}:{1 - weight:.2f})"
        else:
            weight = 2 * alpha - 1
            return f"({p1}:{1 - weight:.2f}) ({p2}:{weight:.2f})"

    def __repr__(self) -> str:
        return (
            f"StableDiffusionGenerator(model={self.model_id!r}, "
            f"device={self.device!r}, dtype={self.dtype})"
        )
