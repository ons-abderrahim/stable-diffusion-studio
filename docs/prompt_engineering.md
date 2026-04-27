# Prompt Engineering Guide

Effective prompting is one of the most powerful levers for controlling Stable Diffusion output quality and style. This guide covers best practices built into Stable Diffusion Studio.

---

## Anatomy of a Great Prompt

A well-structured SD prompt typically follows this order:

```
[Subject] [Style] [Lighting] [Composition] [Quality Tags] [Artist References]
```

**Example:**
```
a regal white wolf standing on a mountain peak,
oil painting, dramatic golden hour lighting,
rule of thirds composition,
masterpiece, highly detailed, 4k,
in the style of Greg Rutkowski
```

---

## The PromptEngineer Class

```python
from src.pipeline.prompt_utils import PromptEngineer

pe = PromptEngineer()

# Structured build
prompt = pe.build(
    subject="a medieval castle",
    style="cinematic photography",
    lighting="dramatic fog, twilight",
    quality_tags=["highly detailed", "8k", "photorealistic"],
    artist_refs=["Ansel Adams"],
)

# Check token count (CLIP limit ≈ 77 tokens)
warning = pe.warn_length(prompt)
```

---

## Style Presets

Apply a preset to instantly add style-appropriate quality tags:

```python
prompt = pe.apply_preset("cinematic", "A lone astronaut on Mars")
# → "A lone astronaut on Mars, cinematic photography, film grain, 35mm, ..."

prompt = pe.apply_preset("anime", "A young hero in a magical forest")
# → "A young hero in a magical forest, anime style, cel shading, ..."
```

| Preset | Best For |
|---|---|
| `photorealistic` | DSLR-quality portraits, landscapes |
| `cinematic` | Dramatic scenes, film-style imagery |
| `fantasy` | Epic environments, creatures, magic |
| `anime` | Characters, vibrant scenes |
| `oil_painting` | Classical art, portraits |
| `watercolor` | Soft nature scenes, whimsical art |
| `pixel_art` | Retro games, icons |
| `concept_art` | Environment design, professional illustration |

---

## Negative Prompts

Use negative prompts to tell the model what to *avoid*:

```python
# Three strictness levels
neg = pe.get_negative_prompt("light")   # Minimal
neg = pe.get_negative_prompt("medium")  # Default
neg = pe.get_negative_prompt("strict")  # Comprehensive anatomy/quality fixes

# Add custom terms
neg = pe.get_negative_prompt("medium", extra=["text overlay", "watermark"])
```

---

## Prompt Weighting

Control the influence of specific tokens using `(token:weight)` syntax:

```python
# Emphasize a concept
pe.emphasize("dramatic lighting", weight=1.4)
# → "(dramatic lighting:1.4)"

# De-emphasize
pe.de_emphasize("smile", weight=0.6)
# → "(smile:0.6)"

# Blend two prompts
pe.blend("summer forest", "winter tundra", weight_a=0.3)
# → "(summer forest:0.30) AND (winter tundra:0.70)"
```

**Manual weighting in prompts:**
```
(beautiful:1.3) mountain landscape, (fog:0.8), highly detailed
```

---

## Common Patterns

### Portrait Photography
```
professional headshot of a confident woman, natural soft lighting,
studio background, DSLR, f/1.8, sharp focus, photorealistic, 8k
```

### Landscape
```
aerial view of the Swiss Alps at sunrise, dramatic volumetric light,
snow-capped peaks, crystal clear sky, photorealistic, 8k, hyperdetailed
```

### Character Art
```
paladin warrior in ornate golden armor, fantasy concept art,
ArtStation trending, dynamic pose, dramatic back-lighting,
highly detailed, sharp focus, by Artgerm
```

### Abstract / Artistic
```
surrealist dreamscape, melting clocks, infinite desert,
Salvador Dali inspired, oil painting, rich textures, symbolic
```

---

## CLIP Token Limit

CLIP tokenizes text with a maximum of **77 tokens**. Tokens beyond this are silently ignored.

```python
# Automatically warns if prompt is too long
warning = pe.warn_length(your_prompt)

# Rough token estimate
count = pe.count_tokens(your_prompt)
```

**Tips to stay within the limit:**
- Remove filler words ("a very very beautiful")
- Use commas instead of full sentences
- Prioritize the most important descriptors first

---

## Model-Specific Advice

| Model | Resolution | Guidance Scale | Steps |
|---|---|---|---|
| SD 1.5 | 512×512 | 7–8 | 20–30 |
| SD 2.1 | 768×768 | 7–9 | 25–35 |
| SDXL | 1024×1024 | 7–8 | 25–35 |
| SDXL Turbo | 512×512 | 0 (CFG disabled) | 1–4 |
| SD3 / SD3.5 | 1024×1024 | 5–7 | 28 |

SDXL and SD3 support a second text encoder — use `prompt_2` for complementary style descriptors.
