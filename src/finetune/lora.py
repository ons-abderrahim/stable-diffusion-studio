"""
src/finetune/lora.py
LoRA (Low-Rank Adaptation) fine-tuning for Stable Diffusion.

Reference: https://arxiv.org/abs/2106.09685
Much lighter than DreamBooth: trains only ~1% of parameters,
producing ~50-200 MB adapters instead of full model checkpoints.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from src.utils.dataset import TextImageDataset

logger = logging.getLogger(__name__)


class LoRATrainer:
    """
    Fine-tune SD with LoRA for lightweight, portable style adapters.

    Injects low-rank matrices into the UNet's attention layers.
    The base model stays frozen; only the small adapter is trained.

    Example
    -------
    >>> trainer = LoRATrainer(
    ...     model_id="stabilityai/stable-diffusion-xl-base-1.0",
    ...     dataset_dir="data/my_style/",
    ...     output_dir="checkpoints/my_style_lora",
    ...     rank=16,
    ... )
    >>> trainer.train(num_train_epochs=10)
    """

    def __init__(
        self,
        model_id: str,
        dataset_dir: str,
        output_dir: str = "checkpoints/lora",
        rank: int = 16,
        alpha: int = 16,
        target_modules: Optional[list] = None,
        resolution: int = 512,
        train_batch_size: int = 2,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-4,
        lr_scheduler: str = "cosine",
        lr_warmup_steps: int = 100,
        num_train_epochs: int = 10,
        max_train_steps: Optional[int] = None,
        mixed_precision: str = "fp16",
        save_steps: int = 500,
        seed: int = 42,
    ) -> None:
        self.model_id = model_id
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or [
            "to_q", "to_k", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
        ]
        self.resolution = resolution
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps
        self.mixed_precision = mixed_precision
        self.save_steps = save_steps
        self.seed = seed

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight_dtype = (
            torch.float16 if mixed_precision == "fp16" else torch.float32
        )

    # ------------------------------------------------------------------

    def train(self, num_train_epochs: Optional[int] = None) -> None:
        """Run the LoRA training loop."""
        if num_train_epochs:
            self.num_train_epochs = num_train_epochs

        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.seed)

        tokenizer, text_encoder, vae, unet, noise_scheduler = self._load_components()

        # Inject LoRA into UNet
        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=0.1,
            bias="none",
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

        # Freeze everything else
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        vae.to(self.device, dtype=self.weight_dtype)
        text_encoder.to(self.device, dtype=self.weight_dtype)
        unet.to(self.device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, unet.parameters()),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        dataset = TextImageDataset(
            data_dir=self.dataset_dir,
            tokenizer=tokenizer,
            size=self.resolution,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=2,
        )

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

        logger.info(f"LoRA training for {total_steps} steps (rank={self.rank})")

        global_step = 0
        scaler = torch.cuda.amp.GradScaler(enabled=(self.mixed_precision == "fp16"))

        for epoch in range(self.num_train_epochs):
            unet.train()
            epoch_loss = 0.0

            for step, batch in enumerate(dataloader):
                with torch.cuda.amp.autocast(enabled=(self.mixed_precision == "fp16")):
                    loss = self._training_step(
                        batch, unet, vae, text_encoder, noise_scheduler
                    )
                    loss = loss / self.gradient_accumulation_steps

                scaler.scale(loss).backward()
                epoch_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, unet.parameters()), 1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    lr_sched.step()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step % self.save_steps == 0 and global_step > 0:
                    self._save_lora(unet, step=global_step)

                if self.max_train_steps and global_step >= self.max_train_steps:
                    break

            avg_loss = epoch_loss / len(dataloader)
            logger.info(
                f"Epoch {epoch + 1}/{self.num_train_epochs} — loss: {avg_loss:.4f}"
            )

        self._save_lora(unet, step="final")
        logger.info("LoRA training complete.")

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

    def _training_step(self, batch, unet, vae, text_encoder, noise_scheduler):
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.weight_dtype)
        input_ids = batch["input_ids"].to(self.device)

        latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = text_encoder(input_ids)[0]
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

    def _save_lora(self, unet, step: str | int) -> None:
        save_path = self.output_dir / f"lora_weights_{step}"
        unet.save_pretrained(save_path)
        logger.info(f"LoRA checkpoint saved → {save_path}")
