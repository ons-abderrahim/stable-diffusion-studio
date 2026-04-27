# Fine-Tuning Guide

This guide explains when and how to use each fine-tuning method in Stable Diffusion Studio.

---

## Overview: Which Method Should I Use?

| Goal | Method | Training Data | Time | Storage |
|---|---|---|---|---|
| Teach a specific person/object | **DreamBooth** | 5–30 images | 30–90 min | ~6 GB |
| Learn a new art style | **LoRA** | 50–500 images | 30–120 min | ~100 MB |
| Add a concept/token | **Textual Inversion** | 3–20 images | 20–60 min | < 1 MB |

---

## DreamBooth

DreamBooth fine-tunes the full UNet to associate a rare token (e.g., `sks`) with your subject.

### When to use
- Teaching the model a specific dog, face, product, or unique object
- You want strong, consistent subject representation

### Dataset preparation
- **5–30 images** of your subject
- Varied backgrounds, angles, and lighting
- Square-cropped, 512×512 or higher
- Avoid duplicates or near-duplicates

```
data/my_dog/
    photo_001.jpg    # different angle
    photo_002.jpg    # outdoor
    photo_003.png    # closeup
    ...
```

### Training

```bash
python scripts/train_dreambooth.py \
  --model_id stabilityai/stable-diffusion-xl-base-1.0 \
  --instance_data_dir data/my_dog/ \
  --instance_prompt "a photo of sks dog" \
  --class_prompt "a photo of a dog" \
  --output_dir checkpoints/my_dog \
  --num_train_epochs 4 \
  --learning_rate 2e-6 \
  --mixed_precision fp16
```

### Inference with fine-tuned model

```python
from src.pipeline import StableDiffusionGenerator

gen = StableDiffusionGenerator("checkpoints/my_dog")
images = gen.generate("a photo of sks dog playing in the snow")
```

### Tips
- Use `--with_prior_preservation` (default) to avoid language drift
- If overfitting (exact photos replicated), reduce `--num_train_epochs`
- Use `--train_text_encoder` for stronger concept binding (requires more VRAM)

---

## LoRA

LoRA injects small trainable matrices into the attention layers. The base model stays frozen.

### When to use
- Learning an art style or aesthetic
- Portability: one base model + many LoRA adapters
- Limited VRAM (< 8 GB)

### Dataset preparation
- **50–500 images** matching your target style
- Caption each image with a `.txt` sidecar file
- Consistent aesthetic is more important than quantity

```
data/my_style/
    painting_001.jpg
    painting_001.txt    # "a landscape oil painting, impressionist style"
    painting_002.png
    painting_002.txt    # "a portrait with loose brushwork, warm palette"
```

### Training

```bash
python scripts/train_lora.py \
  --model_id stabilityai/stable-diffusion-xl-base-1.0 \
  --dataset_dir data/my_style/ \
  --output_dir checkpoints/my_style_lora \
  --rank 16 \
  --num_train_epochs 10 \
  --learning_rate 1e-4
```

### Using a LoRA at inference

```python
gen = StableDiffusionGenerator(
    "stabilityai/stable-diffusion-xl-base-1.0",
    lora_weights="checkpoints/my_style_lora/lora_weights_final",
)
images = gen.generate("a serene forest in my style")
```

### Rank selection

| Rank | Expressiveness | File size | VRAM |
|---|---|---|---|
| 4 | Low | ~20 MB | Minimal |
| 8 | Moderate | ~40 MB | Low |
| 16 | Good | ~80 MB | Moderate |
| 32 | High | ~160 MB | Higher |
| 64 | Very high | ~320 MB | High |

Start with `rank=16` and adjust based on results.

---

## Textual Inversion

Learns a single new embedding vector `<token>` representing a concept. The entire model is frozen.

### When to use
- Extremely constrained storage (< 1 MB)
- Adding a small stylistic nudge
- Quick experiments

### Dataset preparation
- **3–20 images** of your concept
- Works best with a very focused concept

### Training

```python
from src.finetune import TextualInversionTrainer

trainer = TextualInversionTrainer(
    model_id="stabilityai/stable-diffusion-2-1",
    placeholder_token="<my-style>",
    initializer_token="painting",   # closest existing concept
    data_dir="data/my_style/",
    max_steps=3000,
)
trainer.train()
```

### Using the learned embedding

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
).to("cuda")

pipe.load_textual_inversion("checkpoints/textual_inversion/learned_embeds_final.pt")

image = pipe("A portrait rendered in <my-style>").images[0]
```

---

## Hardware Requirements

| Method | Min VRAM | Recommended | Mixed Precision |
|---|---|---|---|
| DreamBooth (SD 1.5) | 8 GB | 16 GB | fp16 |
| DreamBooth (SDXL) | 16 GB | 24 GB | fp16 + grad checkpointing |
| LoRA (SD 1.5) | 6 GB | 12 GB | fp16 |
| LoRA (SDXL) | 8 GB | 16 GB | fp16 |
| Textual Inversion | 6 GB | 8 GB | fp16 |

For machines with less VRAM:
- Enable `--gradient_checkpointing`
- Reduce `--train_batch_size` to 1
- Enable `--use_8bit_adam` (requires `bitsandbytes`)
- Use `--cpu_offload` for inference
