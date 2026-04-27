"""
app.py
Gradio web UI for Stable Diffusion Studio.

Launch with:
    python app.py
Then open http://localhost:7860
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import gradio as gr

from src.pipeline import PromptEngineer, StableDiffusionGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_generator: Optional[StableDiffusionGenerator] = None
_pe = PromptEngineer()

MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo",
    "stabilityai/stable-diffusion-3.5-large",
    "stabilityai/stable-diffusion-3-medium",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-v1-5",
]

STYLE_PRESETS = ["none"] + _pe.list_presets()
NEG_LEVELS = ["light", "medium", "strict"]


def _get_generator(model_id: str) -> StableDiffusionGenerator:
    global _generator
    if _generator is None or _generator.model_id != model_id:
        logger.info(f"Loading model: {model_id}")
        _generator = StableDiffusionGenerator(
            model_id=model_id,
            enable_xformers=True,
        )
    return _generator


# ---------------------------------------------------------------------------
# Generation handler
# ---------------------------------------------------------------------------

def generate_images(
    prompt: str,
    negative_prompt: str,
    model_id: str,
    style_preset: str,
    neg_level: str,
    num_images: int,
    height: int,
    width: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    use_random_seed: bool,
) -> List:
    if not prompt.strip():
        return [], "⚠️ Please enter a prompt."

    # Apply style preset
    final_prompt = prompt
    if style_preset != "none":
        final_prompt = _pe.apply_preset(style_preset, prompt)

    # Build negative prompt
    final_neg = negative_prompt
    if not negative_prompt.strip():
        final_neg = _pe.get_negative_prompt(level=neg_level)
        if style_preset != "none":
            preset_neg = _pe.get_preset_negative(style_preset)
            if preset_neg:
                final_neg = f"{final_neg}, {preset_neg}"

    seed_val = None if use_random_seed else int(seed)

    try:
        gen = _get_generator(model_id)
        images = gen.generate(
            prompt=final_prompt,
            negative_prompt=final_neg,
            num_images=int(num_images),
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=float(cfg_scale),
            seed=seed_val,
        )
        info = (
            f"✅ Generated {len(images)} image(s)\n"
            f"📝 Prompt: {final_prompt}\n"
            f"🚫 Negative: {final_neg}\n"
            f"🔢 Seed: {seed_val if seed_val is not None else 'random'}"
        )
        return images, info
    except Exception as e:
        logger.exception("Generation failed")
        return [], f"❌ Error: {e}"


# ---------------------------------------------------------------------------
# Build the prompt with the PromptEngineer helper
# ---------------------------------------------------------------------------

def build_prompt(subject, style, lighting, quality_tags_str, artist_refs_str):
    quality_tags = [t.strip() for t in quality_tags_str.split(",") if t.strip()]
    artist_refs = [a.strip() for a in artist_refs_str.split(",") if a.strip()]
    prompt = _pe.build(
        subject=subject,
        style=style or None,
        lighting=lighting or None,
        quality_tags=quality_tags,
        artist_refs=artist_refs,
    )
    return prompt


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif"],
).set(
    body_background_fill="#0f0f13",
    body_text_color="#e2e2e8",
    block_background_fill="#1a1a24",
    block_border_color="#2e2e3e",
    input_background_fill="#12121a",
    button_primary_background_fill="*primary_500",
)


def build_ui():
    with gr.Blocks(theme=THEME, title="🎨 Stable Diffusion Studio") as demo:
        gr.Markdown(
            """
# 🎨 Stable Diffusion Studio
**Generate stunning images from text prompts · Fine-tune on custom datasets**
"""
        )

        with gr.Tabs():
            # ── Tab 1: Generate ──────────────────────────────────────────
            with gr.TabItem("🖼️ Generate"):
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A majestic wolf under the northern lights, photorealistic, 8k …",
                            lines=3,
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt (leave blank for auto)",
                            placeholder="blurry, low quality, distorted …",
                            lines=2,
                        )

                        with gr.Row():
                            style_preset = gr.Dropdown(
                                label="Style Preset",
                                choices=STYLE_PRESETS,
                                value="none",
                            )
                            neg_level = gr.Dropdown(
                                label="Neg. Prompt Level",
                                choices=NEG_LEVELS,
                                value="medium",
                            )

                        model_id = gr.Dropdown(
                            label="Model",
                            choices=MODELS,
                            value=MODELS[0],
                        )

                        with gr.Row():
                            num_images = gr.Slider(1, 8, value=4, step=1, label="# Images")
                            steps = gr.Slider(1, 100, value=30, step=1, label="Steps")

                        with gr.Row():
                            height = gr.Slider(256, 1536, value=1024, step=64, label="Height")
                            width = gr.Slider(256, 1536, value=1024, step=64, label="Width")

                        with gr.Row():
                            cfg_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="CFG Scale")
                            seed = gr.Number(value=42, label="Seed", precision=0)
                            use_random_seed = gr.Checkbox(value=False, label="Random seed")

                        generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        gallery = gr.Gallery(
                            label="Generated Images",
                            columns=2,
                            height=600,
                            object_fit="contain",
                        )
                        info_box = gr.Textbox(label="Generation Info", lines=4, interactive=False)

                generate_btn.click(
                    fn=generate_images,
                    inputs=[
                        prompt, negative_prompt, model_id, style_preset, neg_level,
                        num_images, height, width, steps, cfg_scale, seed, use_random_seed,
                    ],
                    outputs=[gallery, info_box],
                )

            # ── Tab 2: Prompt Builder ─────────────────────────────────────
            with gr.TabItem("✏️ Prompt Builder"):
                gr.Markdown("### Build a structured prompt from components")
                with gr.Row():
                    with gr.Column():
                        pb_subject = gr.Textbox(label="Subject", placeholder="a regal lion")
                        pb_style = gr.Textbox(label="Style", placeholder="oil painting")
                        pb_lighting = gr.Textbox(label="Lighting", placeholder="golden hour")
                        pb_quality = gr.Textbox(
                            label="Quality Tags (comma-separated)",
                            placeholder="masterpiece, highly detailed, 4k",
                        )
                        pb_artists = gr.Textbox(
                            label="Artist References (comma-separated)",
                            placeholder="Greg Rutkowski, Alphonse Mucha",
                        )
                        pb_btn = gr.Button("Build Prompt", variant="secondary")
                    with gr.Column():
                        pb_output = gr.Textbox(
                            label="Generated Prompt",
                            lines=5,
                            interactive=True,
                            show_copy_button=True,
                        )
                        gr.Markdown("#### Style Presets Available")
                        gr.Markdown(
                            ", ".join(f"`{p}`" for p in _pe.list_presets())
                        )

                pb_btn.click(
                    fn=build_prompt,
                    inputs=[pb_subject, pb_style, pb_lighting, pb_quality, pb_artists],
                    outputs=pb_output,
                )

            # ── Tab 3: Fine-Tune Info ─────────────────────────────────────
            with gr.TabItem("🔧 Fine-Tuning Guide"):
                gr.Markdown("""
## Fine-Tuning Methods

### DreamBooth
Teach the model a specific subject using 5–30 images.

```bash
python scripts/train_dreambooth.py \\
  --model_id stabilityai/stable-diffusion-xl-base-1.0 \\
  --instance_data_dir data/my_subject/ \\
  --instance_prompt "a photo of sks dog" \\
  --class_prompt "a photo of a dog" \\
  --output_dir checkpoints/my_dog \\
  --num_train_epochs 4
```

---

### LoRA
Lightweight fine-tuning producing a ~100 MB adapter.

```bash
python scripts/train_lora.py \\
  --model_id stabilityai/stable-diffusion-xl-base-1.0 \\
  --dataset_dir data/my_style/ \\
  --output_dir checkpoints/my_style_lora \\
  --rank 16 \\
  --num_train_epochs 10
```

---

### Textual Inversion
Learn a new concept token (~few KB).

```python
from src.finetune import TextualInversionTrainer
trainer = TextualInversionTrainer(
    model_id="stabilityai/stable-diffusion-2-1",
    placeholder_token="<my-style>",
    initializer_token="painting",
    data_dir="data/my_style/",
)
trainer.train(max_steps=3000)
```
""")

        gr.Markdown(
            """
---
<div align="center">
Built with ❤️ using <a href="https://huggingface.co/stabilityai">Stability AI</a> models
& <a href="https://github.com/huggingface/diffusers">🤗 Diffusers</a>
</div>
"""
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
