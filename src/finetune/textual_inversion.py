"""
src/finetune/textual_inversion.py
Textual Inversion — learn a new concept as an embedding token.

Reference: https://arxiv.org/abs/2208.01618
Trains only a new token embedding (~few KB) while the model stays frozen.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from src.utils.dataset import TextImageDataset

logger = logging.getLogger(__name__)


class TextualInversionTrainer:
    """
    Learn a new text embedding token for a custom concept.

    The model weights remain frozen — only the new token vector is trained.
    Produces a tiny (~few KB) .pt embedding file that can be dropped into
    any compatible SD model.

    Example
    -------
    >>> trainer = TextualInversionTrainer(
    ...     model_id="stabilityai/stable-diffusion-2-1",
    ...     placeholder_token="<my-style>",
    ...     initializer_token="painting",
    ...     data_dir="data/my_style/",
    ... )
    >>> trainer.train(max_steps=3000)
    """

    def __init__(
        self,
        model_id: str,
        placeholder_token: str,
        initializer_token: str,
        data_dir: str,
        output_dir: str = "checkpoints/textual_inversion",
        num_vectors: int = 1,
        resolution: int = 512,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 5e-4,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        max_steps: int = 3000,
        save_steps: int = 500,
        mixed_precision: str = "fp16",
        seed: int = 42,
    ) -> None:
        self.model_id = model_id
        self.placeholder_token = placeholder_token
        self.initializer_token = initializer_token
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.num_vectors = num_vectors
        self.resolution = resolution
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.mixed_precision = mixed_precision
        self.seed = seed

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight_dtype = torch.float16 if mixed_precision == "fp16" else torch.float32

    # ------------------------------------------------------------------

    def train(self, max_steps: Optional[int] = None) -> None:
        if max_steps:
            self.max_steps = max_steps

        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.seed)

        tokenizer, text_encoder, vae, unet, noise_scheduler = self._load_components()
        placeholder_token_id = self._add_placeholder_token(tokenizer, text_encoder)

        # Freeze everything except the token embedding table
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        vae.to(self.device, dtype=self.weight_dtype)
        unet.to(self.device, dtype=self.weight_dtype)
        text_encoder.to(self.device)

        optimizer = torch.optim.AdamW(
            text_encoder.get_input_embeddings().parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        dataset = TextImageDataset(
            data_dir=self.data_dir,
            tokenizer=tokenizer,
            default_caption=f"a photo of {self.placeholder_token}",
            size=self.resolution,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=2,
        )

        lr_sched = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.max_steps,
        )

        logger.info(
            f"Textual Inversion: training token '{self.placeholder_token}' "
            f"for {self.max_steps} steps …"
        )

        global_step = 0
        # Keep original embeddings to prevent drift of existing tokens
        orig_embeds = (
            text_encoder.get_input_embeddings().weight.data.clone()
        )

        text_encoder.train()
        while global_step < self.max_steps:
            for batch in dataloader:
                with torch.cuda.amp.autocast(enabled=(self.mixed_precision == "fp16")):
                    loss = self._training_step(
                        batch, unet, vae, text_encoder, noise_scheduler
                    )
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_sched.step()
                    optimizer.zero_grad()

                    # Restore all embeddings except the new token
                    with torch.no_grad():
                        index_no_updates = torch.ones(
                            len(tokenizer), dtype=torch.bool
                        )
                        index_no_updates[placeholder_token_id] = False
                        text_encoder.get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds[index_no_updates]

                global_step += 1

                if global_step % self.save_steps == 0:
                    self._save_embedding(
                        text_encoder, tokenizer, placeholder_token_id, step=global_step
                    )
                    logger.info(f"Step {global_step}/{self.max_steps} — loss: {loss.item():.4f}")

                if global_step >= self.max_steps:
                    break

        self._save_embedding(
            text_encoder, tokenizer, placeholder_token_id, step="final"
        )
        logger.info("Textual Inversion training complete.")

    # ------------------------------------------------------------------

    def _load_components(self):
        tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        return tokenizer, text_encoder, vae, unet, noise_scheduler

    def _add_placeholder_token(self, tokenizer, text_encoder) -> int:
        num_added = tokenizer.add_tokens(self.placeholder_token)
        if num_added == 0:
            raise ValueError(f"Token '{self.placeholder_token}' already exists in tokenizer.")

        token_ids = tokenizer.encode(self.initializer_token, add_special_tokens=False)
        init_token_id = token_ids[0]
        placeholder_token_id = tokenizer.convert_tokens_to_ids(self.placeholder_token)

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[init_token_id].clone()

        logger.info(
            f"Added token '{self.placeholder_token}' (id={placeholder_token_id}), "
            f"initialized from '{self.initializer_token}'."
        )
        return placeholder_token_id

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
        encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=self.weight_dtype)
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

    def _save_embedding(self, text_encoder, tokenizer, token_id: int, step) -> None:
        learned_embeds = (
            text_encoder.get_input_embeddings().weight[token_id].detach().cpu()
        )
        learned_embeds_dict = {self.placeholder_token: learned_embeds}
        save_path = self.output_dir / f"learned_embeds_{step}.pt"
        torch.save(learned_embeds_dict, save_path)
        logger.info(f"Embedding saved → {save_path}")
