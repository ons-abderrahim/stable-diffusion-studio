"""
tests/test_prompt_utils.py
Unit tests for PromptEngineer.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.prompt_utils import PromptEngineer, STYLE_PRESETS, NEGATIVE_PROMPT_LEVELS


@pytest.fixture
def pe():
    return PromptEngineer()


class TestBuild:
    def test_subject_only(self, pe):
        result = pe.build(subject="a lion")
        assert result == "a lion"

    def test_full_components(self, pe):
        result = pe.build(
            subject="a lion",
            style="oil painting",
            lighting="golden hour",
            quality_tags=["masterpiece", "4k"],
            artist_refs=["Rembrandt"],
        )
        assert "a lion" in result
        assert "oil painting" in result
        assert "golden hour" in result
        assert "masterpiece" in result
        assert "4k" in result
        assert "Rembrandt" in result

    def test_no_empty_parts(self, pe):
        result = pe.build(subject="cat", style=None, lighting="")
        assert ",," not in result
        assert result.startswith("cat")


class TestApplyPreset:
    def test_all_presets_work(self, pe):
        for preset in STYLE_PRESETS:
            result = pe.apply_preset(preset, "a forest")
            assert "a forest" in result
            assert len(result) > len("a forest")

    def test_invalid_preset_raises(self, pe):
        with pytest.raises(ValueError, match="Unknown preset"):
            pe.apply_preset("nonexistent_preset", "a cat")

    def test_list_presets_returns_all(self, pe):
        presets = pe.list_presets()
        assert set(presets) == set(STYLE_PRESETS.keys())


class TestNegativePrompts:
    def test_levels_all_exist(self, pe):
        for level in ["light", "medium", "strict"]:
            result = pe.get_negative_prompt(level=level)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_strict_longer_than_light(self, pe):
        light = pe.get_negative_prompt("light")
        strict = pe.get_negative_prompt("strict")
        assert len(strict) > len(light)

    def test_extra_terms_appended(self, pe):
        result = pe.get_negative_prompt("light", extra=["my_custom_term"])
        assert "my_custom_term" in result


class TestWeighting:
    def test_emphasize(self):
        result = PromptEngineer.emphasize("dragon", 1.5)
        assert result == "(dragon:1.5)"

    def test_de_emphasize(self):
        result = PromptEngineer.de_emphasize("fog", 0.5)
        assert result == "(fog:0.5)"

    def test_blend(self):
        result = PromptEngineer.blend("forest", "desert", 0.5)
        assert "forest" in result
        assert "desert" in result


class TestTokenCount:
    def test_count_tokens(self):
        prompt = "a beautiful mountain lake at sunrise"
        assert PromptEngineer.count_tokens(prompt) == 7

    def test_warn_length_under_limit(self):
        short = "a cat"
        assert PromptEngineer.warn_length(short) is None

    def test_warn_length_over_limit(self):
        long_prompt = " ".join(["word"] * 80)
        warning = PromptEngineer.warn_length(long_prompt)
        assert warning is not None
        assert "CLIP" in warning
