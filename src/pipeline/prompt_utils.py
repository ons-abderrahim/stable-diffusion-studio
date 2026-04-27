"""
src/pipeline/prompt_utils.py
Prompt engineering helpers for Stable Diffusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------

STYLE_PRESETS: Dict[str, Dict[str, List[str]]] = {
    "photorealistic": {
        "positive": [
            "photorealistic", "DSLR", "sharp focus", "8k resolution",
            "highly detailed", "natural lighting", "f/1.8",
        ],
        "negative": [
            "painting", "illustration", "cartoon", "anime", "blurry",
            "overexposed", "underexposed",
        ],
    },
    "cinematic": {
        "positive": [
            "cinematic photography", "film grain", "35mm", "dramatic lighting",
            "anamorphic lens", "movie still", "color graded", "shallow depth of field",
        ],
        "negative": ["amateur", "flat lighting", "no depth"],
    },
    "fantasy": {
        "positive": [
            "epic fantasy", "magical atmosphere", "highly detailed", "concept art",
            "ArtStation trending", "Craig Mullins", "vibrant colors",
        ],
        "negative": ["modern", "realistic", "photography", "low detail"],
    },
    "anime": {
        "positive": [
            "anime style", "cel shading", "vibrant colors", "Studio Ghibli",
            "clean linework", "detailed background",
        ],
        "negative": ["3d render", "photorealistic", "western cartoon"],
    },
    "oil_painting": {
        "positive": [
            "oil painting", "classical", "textured canvas", "impasto technique",
            "rich colors", "Old Masters style", "gallery quality",
        ],
        "negative": ["digital art", "photography", "flat"],
    },
    "watercolor": {
        "positive": [
            "watercolor painting", "soft washes", "delicate edges",
            "paper texture", "loose brushwork", "luminous",
        ],
        "negative": ["sharp edges", "photorealistic", "digital flat"],
    },
    "pixel_art": {
        "positive": [
            "pixel art", "16-bit", "retro game aesthetic",
            "isometric", "clear outlines", "limited palette",
        ],
        "negative": ["photorealistic", "blurry", "high resolution anti-aliased"],
    },
    "concept_art": {
        "positive": [
            "concept art", "environment design", "ArtStation", "keyframe",
            "detailed", "professional illustration",
        ],
        "negative": ["amateur", "rough sketch", "unfinished"],
    },
}


NEGATIVE_PROMPT_LEVELS = {
    "light": [
        "blurry", "low quality", "watermark",
    ],
    "medium": [
        "blurry", "low quality", "watermark", "text", "signature",
        "deformed", "distorted", "disfigured", "bad anatomy",
    ],
    "strict": [
        "blurry", "low quality", "watermark", "text", "signature",
        "deformed", "distorted", "disfigured", "bad anatomy",
        "extra limbs", "missing limbs", "fused fingers", "too many fingers",
        "long neck", "ugly", "duplicate", "morbid", "mutilated",
        "out of frame", "extra fingers", "mutated hands", "poorly drawn hands",
        "poorly drawn face", "mutation", "bad proportions", "gross proportions",
        "oversaturated", "overexposed", "jpeg artifacts", "noise",
    ],
}


# ---------------------------------------------------------------------------
# Dataclass for structured prompt building
# ---------------------------------------------------------------------------

@dataclass
class PromptComponents:
    subject: str
    style: Optional[str] = None
    lighting: Optional[str] = None
    color_palette: Optional[str] = None
    composition: Optional[str] = None
    quality_tags: List[str] = field(default_factory=list)
    artist_refs: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main utility class
# ---------------------------------------------------------------------------

class PromptEngineer:
    """
    Utility class for crafting high-quality Stable Diffusion prompts.

    Example
    -------
    >>> pe = PromptEngineer()
    >>> prompt = pe.build(
    ...     subject="a majestic lion",
    ...     style="oil painting",
    ...     lighting="golden hour",
    ...     quality_tags=["masterpiece", "4k"],
    ... )
    >>> neg = pe.get_negative_prompt("strict")
    """

    # ------------------------------------------------------------------
    # Building prompts
    # ------------------------------------------------------------------

    def build(
        self,
        subject: str,
        style: Optional[str] = None,
        lighting: Optional[str] = None,
        color_palette: Optional[str] = None,
        composition: Optional[str] = None,
        quality_tags: Optional[List[str]] = None,
        artist_refs: Optional[List[str]] = None,
        extra: Optional[List[str]] = None,
    ) -> str:
        """
        Assemble a structured prompt from components.

        Ordering follows the common best-practice:
        subject → style → lighting → composition → quality → artist refs
        """
        parts: List[str] = [subject]

        if style:
            parts.append(style)
        if lighting:
            parts.append(lighting)
        if color_palette:
            parts.append(color_palette)
        if composition:
            parts.append(composition)
        if quality_tags:
            parts.extend(quality_tags)
        if artist_refs:
            parts.extend(f"in the style of {a}" for a in artist_refs)
        if extra:
            parts.extend(extra)

        return ", ".join(p.strip() for p in parts if p.strip())

    def from_components(self, components: PromptComponents) -> str:
        """Build a prompt from a PromptComponents dataclass."""
        return self.build(
            subject=components.subject,
            style=components.style,
            lighting=components.lighting,
            color_palette=components.color_palette,
            composition=components.composition,
            quality_tags=components.quality_tags,
            artist_refs=components.artist_refs,
            extra=components.extra,
        )

    # ------------------------------------------------------------------
    # Style presets
    # ------------------------------------------------------------------

    def apply_preset(self, preset_name: str, base_prompt: str) -> str:
        """
        Apply a named style preset to an existing prompt.

        Parameters
        ----------
        preset_name:
            One of: photorealistic, cinematic, fantasy, anime,
            oil_painting, watercolor, pixel_art, concept_art.
        base_prompt:
            Your subject/scene description.

        Returns
        -------
        Enhanced prompt string.
        """
        preset = STYLE_PRESETS.get(preset_name)
        if not preset:
            available = list(STYLE_PRESETS.keys())
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available: {available}"
            )
        tags = preset.get("positive", [])
        return f"{base_prompt}, {', '.join(tags)}"

    def list_presets(self) -> List[str]:
        """Return all available style preset names."""
        return list(STYLE_PRESETS.keys())

    def get_preset_negative(self, preset_name: str) -> str:
        """Get the negative prompt for a style preset."""
        preset = STYLE_PRESETS.get(preset_name, {})
        return ", ".join(preset.get("negative", []))

    # ------------------------------------------------------------------
    # Negative prompts
    # ------------------------------------------------------------------

    def get_negative_prompt(
        self,
        level: str = "medium",
        extra: Optional[List[str]] = None,
    ) -> str:
        """
        Get a standard negative prompt.

        Parameters
        ----------
        level:
            "light", "medium", or "strict".
        extra:
            Additional custom negative terms to append.
        """
        base = NEGATIVE_PROMPT_LEVELS.get(level, NEGATIVE_PROMPT_LEVELS["medium"])
        combined = list(base) + (extra or [])
        return ", ".join(combined)

    # ------------------------------------------------------------------
    # Prompt weighting (A1111 / compel syntax)
    # ------------------------------------------------------------------

    @staticmethod
    def emphasize(text: str, weight: float = 1.3) -> str:
        """Wrap a token/phrase with emphasis weighting: (text:1.3)."""
        return f"({text}:{weight:.1f})"

    @staticmethod
    def de_emphasize(text: str, weight: float = 0.7) -> str:
        """Reduce the influence of a token: (text:0.7)."""
        return f"({text}:{weight:.1f})"

    @staticmethod
    def blend(prompt_a: str, prompt_b: str, weight_a: float = 0.5) -> str:
        """
        Blend two prompts using AND syntax.
        weight_a controls how much weight goes to prompt_a.
        """
        weight_b = round(1.0 - weight_a, 2)
        return f"({prompt_a}:{weight_a:.2f}) AND ({prompt_b}:{weight_b:.2f})"

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def count_tokens(prompt: str) -> int:
        """Rough token count (whitespace split). CLIP limit is ~77 tokens."""
        return len(prompt.split())

    @staticmethod
    def warn_length(prompt: str, limit: int = 77) -> Optional[str]:
        """Return a warning if the prompt likely exceeds the CLIP token limit."""
        count = PromptEngineer.count_tokens(prompt)
        if count > limit:
            return (
                f"⚠️  Prompt has ~{count} tokens (CLIP limit ≈ {limit}). "
                "Text beyond the limit will be ignored."
            )
        return None
