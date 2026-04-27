# stable-diffusion-studio
Generate images from text prompts using Stable Diffusion. Optionally fine-tune on a custom dataset.(Diffusion models, prompt engineering, fine-tuning.)


# 🎨 Stable Diffusion Studio

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Diffusers-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Stars](https://img.shields.io/github/stars/yourusername/stable-diffusion-studio?style=flat-square)

**Generate stunning images from text prompts using Stable Diffusion — with full fine-tuning support on custom datasets.**

[**📖 Docs**](#documentation) · [**🚀 Quick Start**](#quick-start) · [**🎛️ Fine-Tuning**](#fine-tuning) · [**🧪 Notebooks**](#notebooks) · [**🤝 Contributing**](#contributing)

---

<img src="assets/banner.png" alt="Stable Diffusion Studio Banner" width="100%"/>

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ **Text-to-Image** | Generate high-quality images from text prompts using SD 1.5, SD 2.1, SDXL, SD3 |
| 🎨 **Prompt Engineering** | Built-in prompt templates, negative prompts, and style presets |
| 🔧 **Fine-Tuning** | DreamBooth, LoRA, and Textual Inversion on custom datasets |
| ⚡ **Fast Inference** | CUDA acceleration, `torch.compile`, attention slicing & xFormers |
| 🖥️ **Gradio UI** | Interactive web UI for generation and fine-tuning |
| 📦 **Model Hub** | Seamless integration with HuggingFace `stabilityai` models |
| 🧪 **Evaluation** | FID, CLIP score, and IS metrics out of the box |
| 🐳 **Docker** | Ready-to-run containerized environment |

---

## 🗂️ Repository Structure

```
stable-diffusion-studio/
│
├── src/
│   ├── pipeline/               # Inference pipeline
│   │   ├── __init__.py
│   │   ├── generator.py        # Core image generation class
│   │   ├── prompt_utils.py     # Prompt engineering utilities
│   │   └── schedulers.py       # Noise scheduler configs
│   │
│   ├── finetune/               # Fine-tuning modules
│   │   ├── __init__.py
│   │   ├── dreambooth.py       # DreamBooth trainer
│   │   ├── lora.py             # LoRA trainer
│   │   └── textual_inversion.py
│   │
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       ├── dataset.py          # Custom dataset loaders
│       ├── metrics.py          # FID / CLIP / IS evaluation
│       └── visualization.py    # Grid & comparison plots
│
├── configs/
│   ├── generation_default.yaml
│   ├── dreambooth_config.yaml
│   └── lora_config.yaml
│
├── notebooks/
│   ├── 01_text_to_image.ipynb
│   ├── 02_prompt_engineering.ipynb
│   ├── 03_dreambooth_finetune.ipynb
│   └── 04_lora_finetune.ipynb
│
├── scripts/
│   ├── generate.py             # CLI generation script
│   ├── train_dreambooth.py     # CLI DreamBooth training
│   ├── train_lora.py           # CLI LoRA training
│   └── evaluate.py             # CLI evaluation script
│
├── tests/
│   ├── test_generator.py
│   ├── test_prompt_utils.py
│   └── test_metrics.py
│
├── app.py                      # Gradio web UI
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── setup.py
└── .env.example
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stable-diffusion-studio.git
cd stable-diffusion-studio

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install xFormers for memory-efficient attention
pip install xformers
```

### 2. Generate Your First Image

```python
from src.pipeline import StableDiffusionGenerator

generator = StableDiffusionGenerator(
    model_id="stabilityai/stable-diffusion-3.5-large",
    device="cuda"
)

images = generator.generate(
    prompt="A breathtaking mountain landscape at golden hour, photorealistic, 8k",
    negative_prompt="blurry, low quality, distorted",
    num_images=4,
    height=1024,
    width=1024,
    num_inference_steps=28,
    guidance_scale=7.5,
)

generator.save_grid(images, "output/my_landscape.png")
```

### 3. CLI Usage

```bash
# Basic generation
python scripts/generate.py \
  --prompt "A cyberpunk cityscape at night, neon lights, rain" \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --num-images 4 \
  --output output/

# Using a preset style
python scripts/generate.py \
  --prompt "Portrait of a wise old wizard" \
  --style fantasy \
  --steps 30 \
  --cfg 8.0
```

### 4. Launch the Gradio UI

```bash
python app.py
# → Open http://localhost:7860
```

---

## 🎨 Supported Models

All models are sourced from [stabilityai on HuggingFace](https://huggingface.co/stabilityai):

| Model | ID | Resolution | Notes |
|---|---|---|---|
| SD 1.5 | `stabilityai/stable-diffusion-v1-5` | 512×512 | Lightweight, fast |
| SD 2.1 | `stabilityai/stable-diffusion-2-1` | 768×768 | Better composition |
| SDXL Base | `stabilityai/stable-diffusion-xl-base-1.0` | 1024×1024 | High fidelity |
| SDXL Turbo | `stabilityai/sdxl-turbo` | 512×512 | Real-time (1–4 steps) |
| SD3 Medium | `stabilityai/stable-diffusion-3-medium` | 1024×1024 | Multimodal transformer |
| SD3.5 Large | `stabilityai/stable-diffusion-3.5-large` | 1024×1024 | Best quality |

---

## 🎛️ Fine-Tuning

### DreamBooth — Teach the model a custom subject

```bash
python scripts/train_dreambooth.py \
  --model_id stabilityai/stable-diffusion-xl-base-1.0 \
  --instance_data_dir data/my_subject/ \
  --instance_prompt "a photo of sks dog" \
  --class_prompt "a photo of a dog" \
  --output_dir checkpoints/my_dog_model \
  --num_train_epochs 4 \
  --train_batch_size 1 \
  --learning_rate 2e-6 \
  --mixed_precision fp16
```

### LoRA — Lightweight fine-tuning (~100MB instead of 6GB)

```bash
python scripts/train_lora.py \
  --model_id stabilityai/stable-diffusion-xl-base-1.0 \
  --dataset_dir data/my_style/ \
  --output_dir checkpoints/my_style_lora \
  --rank 16 \
  --num_train_epochs 10 \
  --learning_rate 1e-4
```

### Textual Inversion — Learn a new concept token

```python
from src.finetune import TextualInversionTrainer

trainer = TextualInversionTrainer(
    model_id="stabilityai/stable-diffusion-2-1",
    placeholder_token="<my-style>",
    initializer_token="painting",
    data_dir="data/my_style/",
)
trainer.train(max_steps=3000, save_steps=500)
```

---

## 🧪 Prompt Engineering

The `PromptEngineer` class provides utilities for crafting effective prompts:

```python
from src.pipeline.prompt_utils import PromptEngineer

pe = PromptEngineer()

# Build a structured prompt
prompt = pe.build(
    subject="a regal cat",
    style="oil painting",
    lighting="dramatic rembrandt lighting",
    quality_tags=["masterpiece", "highly detailed", "4k"],
    artist_refs=["John William Waterhouse"],
)
# → "a regal cat, oil painting, dramatic rembrandt lighting, masterpiece, highly detailed, 4k, in the style of John William Waterhouse"

# Load a style preset
prompt = pe.apply_preset("cinematic", base_prompt="A lone astronaut on Mars")

# Generate negative prompt
neg = pe.get_negative_prompt(level="strict")
```

### Built-in Style Presets

| Preset | Description |
|---|---|
| `photorealistic` | DSLR-quality, sharp, lifelike |
| `cinematic` | Film grain, dramatic lighting, widescreen |
| `fantasy` | Epic, magical, highly detailed environments |
| `anime` | Cel-shaded, vibrant colors, Studio Ghibli-esque |
| `oil_painting` | Textured, classical, rich tones |
| `pixel_art` | 16-bit, retro game aesthetic |
| `watercolor` | Soft washes, organic edges |

---

## 📊 Evaluation

```bash
# Compute FID, CLIP score and IS on a generated set
python scripts/evaluate.py \
  --generated_dir output/generated/ \
  --reference_dir data/reference/ \
  --metrics fid clip_score is
```

```
┌──────────────────────────────────────┐
│         Evaluation Results           │
├─────────────────┬────────────────────┤
│ FID ↓           │ 18.43              │
│ CLIP Score ↑    │ 0.312              │
│ IS ↑            │ 22.7 ± 1.4         │
└─────────────────┴────────────────────┘
```

---

## 🐳 Docker

```bash
# Build and run with GPU support
docker compose up --build

# Or manually
docker build -t sd-studio .
docker run --gpus all -p 7860:7860 sd-studio
```

---

## ⚙️ Configuration

All parameters are configurable via YAML. Example `configs/generation_default.yaml`:

```yaml
model:
  id: stabilityai/stable-diffusion-xl-base-1.0
  dtype: float16
  device: cuda

generation:
  num_inference_steps: 30
  guidance_scale: 7.5
  height: 1024
  width: 1024
  num_images_per_prompt: 1
  seed: 42

optimization:
  enable_xformers: true
  enable_attention_slicing: true
  torch_compile: false
  cpu_offload: false
```

---

## 📖 Documentation

- [**Installation Guide**](docs/installation.md)
- [**Prompt Engineering Guide**](docs/prompt_engineering.md)
- [**Fine-Tuning Guide**](docs/finetuning.md)
- [**API Reference**](docs/api_reference.md)
- [**Model Comparison**](docs/model_comparison.md)
- [**Troubleshooting**](docs/troubleshooting.md)

---

## 🧪 Notebooks

| Notebook | Description | Open |
|---|---|---|
| `01_text_to_image.ipynb` | Basic generation walkthrough | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| `02_prompt_engineering.ipynb` | Advanced prompting techniques | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| `03_dreambooth_finetune.ipynb` | Full DreamBooth tutorial | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| `04_lora_finetune.ipynb` | LoRA fine-tuning on custom data | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |

---

## 🛠️ Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Format code
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Type checking
mypy src/
```

---

## 🤝 Contributing

Contributions are warmly welcomed! Please read our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 🙏 Acknowledgements

- [Stability AI](https://stability.ai) for the incredible open-source models
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) for the foundational library
- [CompVis](https://github.com/CompVis) for the original Latent Diffusion Models paper

---

<div align="center">
  <sub>Built with ❤️ · Give it a ⭐ if you find it useful!</sub>
</div>
