"""
src/finetune/dreambooth.py
DreamBooth fine-tuning for Stable Diffusion using HuggingFace Diffusers.

Reference: https://arxiv.org/abs/2208.12242
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from src.utils.dataset import DreamBoothDataset

logger = logging.getLogger(__name__)


class DreamBoothTrainer:
    """
    Fine-tune a Stable Diffusion model on a custom subject using DreamBooth.

    DreamBooth trains the full UNet (and optionally the text encoder) to
    associate a rare token (e.g., "sks") with your subject, while using
    class images to prevent language drift.

    Example
    -------
    >>> trainer = DreamBoothTrainer(
    ...     model_id="stabilityai/stable-diffusion-xl-base-1.0",
    ...     instance_data_dir="data/my_dog/",
    ...     instance_prompt="a photo of sks dog",
    ...     class_prompt="a photo of a dog",
    ...     output_dir="checkpoints/my_dog",
    ... )
    >>> trainer.train(num_train_epochs=4)
    """

    def __init__(
        self,
        model_id: str,
        instance_data_dir: str,
        instance_prompt: str,
        class_prompt: Optional[str] = None,
        class_data_dir: Optional[str] = None,
        output_dir: str = "checkpoints/dreambooth",
        resolution: int = 512,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-6,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        num_train_epochs: int = 4,
        max_train_steps: Optional[int] = None,
        train_text_encoder: bool = False,
        with_prior_preservation: bool = True,
        prior_loss_weight: float = 1.0,
        num_class_images: int = 200,
        mixed_precision: str = "fp16",
        gradient_checkpointing: bool = True,
        use_8bit_adam: bool = False,
        seed: int = 42,
    ) -> None:
        self.model_id = model_id
        self.instance_data_dir = Path(instance_data_dir)
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.class_data_dir = Path(class_data_dir) if class_data_dir else None
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps
        self.train_text_encoder = train_text_encoder
        self.with_prior_preservation = with_prior_preservation
        self.prior_loss_weight = prior_loss_weight
        self.num_class_images = num_class_images
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.use_8bit_adam = use_8bit_adam
        self.seed = seed

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight_dtype = (
            torch.float16 if mixed_precision == "fp16" else torch.float32
        )

    # ------------------------------------------------------------------

    def train(self, num_train_epochs: Optional[int] = None) -> None:
        """Run the full DreamBooth training loop."""
        if num_train_epochs:
            self.num_train_epochs = num_train_epochs

        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.seed)

        # Load components
        tokenizer, text_encoder, vae, unet, noise_scheduler = self._load_components()

        # Freeze / unfreeze
        vae.requires_grad_(False)
        if not self.train_text_encoder:
            text_encoder.requires_grad_(False)

        if self.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        # Optimizer
        params_to_optimize = (
            list(unet.parameters()) + list(text_encoder.parameters())
            if self.train_text_encoder
            else unet.parameters()
        )
        optimizer = self._build_optimizer(params_to_optimize)

        # Dataset & loader
        dataset = DreamBoothDataset(
            instance_data_root=self.instance_data_dir,
            instance_prompt=self.instance_prompt,
            class_data_root=self.class_data_dir if self.with_prior_preservation else None,
            class_prompt=self.class_prompt,
            tokenizer=tokenizer,
            size=self.resolution,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=2,
        )

        # Scheduler
        total_steps = (
            self.max_train_steps
            or math.ceil(len(dataloader) / self.gradient_accumulation_steps)
            * self.num_train_epochs
        )
        lr_sched = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=total_steps,
        )

        # Move models
        unet.to(self.device, dtype=self.weight_dtype)
        vae.to(self.device, dtype=self.weight_dtype)
        text_encoder.to(self.device, dtype=self.weight_dtype)

        logger.info(f"Training DreamBooth for {total_steps} steps …")

        global_step = 0
        for epoch in range(self.num_train_epochs):
            unet.train()
            if self.train_text_encoder:
                text_encoder.train()

            epoch_loss = 0.0
            for step, batch in enumerate(dataloader):
                with torch.cuda.amp.autocast(enabled=(self.mixed_precision == "fp16")):
                    loss = self._training_step(
                        batch, unet, vae, text_encoder, noise_scheduler
                    )
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()
                epoch_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad()
                    global_step += 1

                if self.max_train_steps and global_step >= self.max_train_steps:
                    break

            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{self.num_train_epochs} — loss: {avg_loss:.4f}")

        self._save(unet, text_encoder, tokenizer)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_components(self):
        tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder"
        )
        vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        return tokenizer, text_encoder, vae, unet, noise_scheduler

    def _build_optimizer(self, params):
        if self.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(params, lr=self.learning_rate)
            except ImportError:
                logger.warning("bitsandbytes not installed; falling back to AdamW.")
        return torch.optim.AdamW(params, lr=self.learning_rate)

    def _training_step(self, batch, unet, vae, text_encoder, noise_scheduler):
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.weight_dtype)
        input_ids = batch["input_ids"].to(self.device)

        # Encode images to latent space
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Text embeddings
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Predict noise
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Compute loss (instance + optional prior preservation)
        if self.with_prior_preservation:
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(noise, 2, dim=0)
            instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
            loss = instance_loss + self.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        return loss

    def _save(self, unet, text_encoder, tokenizer) -> None:
        from diffusers import StableDiffusionPipeline

        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            unet=unet,
            text_encoder=text_encoder if self.train_text_encoder else None,
        )
        pipeline.save_pretrained(self.output_dir)
        logger.info(f"Model saved → {self.output_dir}")
